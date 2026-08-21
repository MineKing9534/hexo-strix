"""HTTP-level tests for the admin model-management API.

Exercises the authenticated /admin/models endpoints (list / add / remove /
set-default) against a real ThreadingHTTPServer backed by a no-op-bot
GameManager and a ModelManager whose build_variant is a stub (no real model
loading). Also covers the persistence file round-trip.
"""
import http.client
import json
import threading
import http.server
from contextlib import contextmanager

import pytest

from hexo_a0.serving.game import GameManager
from hexo_a0.serving.app import make_handler_class, ModelManager


def _noop_mgr():
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None,
        recorder=None,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="x.pt",
        model_label="test",
        difficulty_sims={"standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )


def _stub_build_variant(model_id, label, path):
    return {
        "id": model_id, "label": label, "path": path, "step": None,
        "bot_turn_fn": lambda rec: None, "analyze_ctx": object(),
    }


def _real_build_variant(model_id, label, path):
    """build_variant that actually loads the safetensors via hexo-infer."""
    from hexo_a0.serving.model import load_native_model
    model, mc, ckpt = load_native_model(path)
    return {
        "id": model_id, "label": label, "path": path, "step": None,
        "bot_turn_fn": lambda rec: None, "analyze_ctx": object(),
    }


@contextmanager
def _admin_server(admin_token="secret-token", persist_path=None, models_dir=None,
                  build_variant=None):
    mgr = _noop_mgr()
    analysis_contexts = {}
    mm = ModelManager(mgr, analysis_contexts, build_variant or _stub_build_variant,
                      persist_path, models_dir=models_dir)
    mm.restore()
    Handler = make_handler_class(mgr, admin_token=admin_token, url_prefix="",
                                 analyze_ctx=analysis_contexts, model_manager=mm)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port, mgr
    finally:
        srv.shutdown()
        srv.server_close()


def _req(port, method, path, body=None, token="secret-token"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"X-Admin-Token": token}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
        conn.request(method, path, body=json.dumps(body), headers=hdrs)
    else:
        conn.request(method, path, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except Exception:
        return resp.status, raw


def _multipart(fields):
    """Build a multipart/form-data body from {name: (filename, content)}."""
    boundary = "----hexo-test-boundary"
    parts = []
    for name, (filename, content) in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        if filename is not None:
            parts.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        else:
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _upload(port, label, filename, data, token="secret-token"):
    content_type, body = _multipart({
        "label": (None, label.encode()),
        "file": (filename, data),
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/admin/models/upload", body=body, headers={
        "X-Admin-Token": token,
        "Content-Type": content_type,
    })
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except Exception:
        return resp.status, raw


def _tiny_pt_checkpoint(tmp_path, steps=123):
    """A real trainer.py-style .pt checkpoint (embedded model_config)."""
    import dataclasses
    import torch
    from hexo_a0.gen_parity_fixtures import tiny_model_and_config
    model, mc = tiny_model_and_config()
    path = tmp_path / "tiny_champion.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": dataclasses.asdict(mc),
        "game_config": {"win_length": 6, "placement_radius": 4, "max_moves": 300},
        "train_steps": steps,
    }, path)
    return str(path)


def test_list_models_requires_token():
    with _admin_server() as (port, _):
        status, _ = _req(port, "GET", "/admin/models", token="nope")
        assert status == 404


def test_list_models_returns_default():
    with _admin_server() as (port, _):
        status, body = _req(port, "GET", "/admin/models")
        assert status == 200
        assert body["default"] == "default"
        assert [m["id"] for m in body["models"]] == ["default"]
        assert body["models"][0]["is_default"] is True


def test_add_model(tmp_path):
    model_file = tmp_path / "preview.safetensors"
    model_file.write_bytes(b"")
    with _admin_server() as (port, mgr):
        status, body = _req(port, "POST", "/admin/models",
                            {"label": "Preview", "path": str(model_file)})
        assert status == 200
        ids = [m["id"] for m in body["models"]]
        assert ids == ["default", "preview"]
        assert "preview" in mgr.model_variants


def test_add_model_rejects_non_safetensors(tmp_path):
    model_file = tmp_path / "preview.pt"
    model_file.write_bytes(b"")
    with _admin_server() as (port, _):
        status, body = _req(port, "POST", "/admin/models",
                            {"label": "Preview", "path": str(model_file)})
        assert status == 400
        assert "safetensors" in body["error"]


def test_add_model_rejects_missing_file():
    with _admin_server() as (port, _):
        status, body = _req(port, "POST", "/admin/models",
                            {"label": "Preview", "path": "/nonexistent/x.safetensors"})
        assert status == 400


def test_add_model_rejects_duplicate_id(tmp_path):
    model_file = tmp_path / "preview.safetensors"
    model_file.write_bytes(b"")
    with _admin_server() as (port, _):
        _req(port, "POST", "/admin/models", {"label": "Preview", "path": str(model_file)})
        status, body = _req(port, "POST", "/admin/models",
                            {"label": "Preview!", "path": str(model_file)})
        assert status == 400
        assert "already exists" in body["error"]


def test_set_default_and_remove(tmp_path):
    model_file = tmp_path / "preview.safetensors"
    model_file.write_bytes(b"")
    with _admin_server() as (port, mgr):
        _req(port, "POST", "/admin/models", {"label": "Preview", "path": str(model_file)})
        status, body = _req(port, "POST", "/admin/models/default", {"id": "preview"})
        assert status == 200
        assert body["default"] == "preview"
        assert mgr.default_model_id == "preview"
        # removing the current default is refused
        status, body = _req(port, "POST", "/admin/models/remove", {"id": "preview"})
        assert status == 400
        # re-default then remove
        _req(port, "POST", "/admin/models/default", {"id": "default"})
        status, body = _req(port, "POST", "/admin/models/remove", {"id": "preview"})
        assert status == 200
        assert [m["id"] for m in body["models"]] == ["default"]


def test_remove_default_model_refused():
    with _admin_server() as (port, _):
        status, body = _req(port, "POST", "/admin/models/remove", {"id": "default"})
        assert status == 400
        assert "cannot be removed" in body["error"]


def test_remove_unknown_model():
    with _admin_server() as (port, _):
        status, body = _req(port, "POST", "/admin/models/remove", {"id": "nope"})
        assert status == 400


def test_persistence_round_trip(tmp_path):
    model_file = tmp_path / "preview.safetensors"
    model_file.write_bytes(b"")
    persist = tmp_path / "models.json"
    with _admin_server(persist_path=str(persist)) as (port, _):
        _req(port, "POST", "/admin/models", {"label": "Preview", "path": str(model_file)})
        _req(port, "POST", "/admin/models/default", {"id": "preview"})
    # A fresh server restores the persisted variant + default selection.
    with _admin_server(persist_path=str(persist)) as (port, mgr):
        status, body = _req(port, "GET", "/admin/models")
        assert status == 200
        assert body["default"] == "preview"
        assert [m["id"] for m in body["models"]] == ["default", "preview"]
        assert mgr.default_model_id == "preview"


def test_upload_safetensors(tmp_path):
    models_dir = tmp_path / "models"
    with _admin_server(models_dir=str(models_dir)) as (port, mgr):
        status, body = _upload(port, "Uploaded", "model.safetensors", b"fake-safetensors-bytes")
        assert status == 200
        ids = [m["id"] for m in body["models"]]
        assert ids == ["default", "uploaded"]
        # stored under models_dir as <id>.safetensors, marked browser-local
        assert (models_dir / "uploaded.safetensors").read_bytes() == b"fake-safetensors-bytes"
        uploaded = next(m for m in body["models"] if m["id"] == "uploaded")
        assert uploaded["local"] is True


def test_upload_pt_is_converted(tmp_path):
    models_dir = tmp_path / "models"
    ckpt = _tiny_pt_checkpoint(tmp_path, steps=456)
    with _admin_server(models_dir=str(models_dir)) as (port, mgr):
        status, body = _upload(port, "Converted", "champion.pt", open(ckpt, "rb").read())
        assert status == 200
        ids = [m["id"] for m in body["models"]]
        assert ids == ["default", "converted"]
        # the .pt was converted to a .safetensors in models_dir
        out = models_dir / "converted.safetensors"
        assert out.exists()
        assert not (models_dir / "converted.uploading.pt").exists()
        converted = next(m for m in body["models"] if m["id"] == "converted")
        assert converted["local"] is True


def test_upload_pt_rejected_when_torch_missing(tmp_path, monkeypatch):
    import importlib.util
    real_find_spec = importlib.util.find_spec
    def fake_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    models_dir = tmp_path / "models"
    with _admin_server(models_dir=str(models_dir)) as (port, _):
        status, body = _upload(port, "Converted", "champion.pt", b"fake-pt-bytes")
        assert status == 400
        assert "torch is not installed" in body["error"]
        assert "safetensors" in body["error"]
        # no stray files left behind
        assert not (models_dir / "converted.safetensors").exists()


def test_upload_rejects_bad_extension(tmp_path):
    with _admin_server(models_dir=str(tmp_path / "models")) as (port, _):
        status, body = _upload(port, "Bad", "model.onnx", b"whatever")
        assert status == 400
        assert "safetensors" in body["error"]


def test_upload_rejects_missing_file(tmp_path):
    with _admin_server(models_dir=str(tmp_path / "models")) as (port, _):
        content_type, body = _multipart({"label": (None, b"OnlyLabel")})
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/admin/models/upload", body=body, headers={
            "X-Admin-Token": "secret-token", "Content-Type": content_type})
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        assert resp.status == 400
        assert "missing model file" in json.loads(raw)["error"]


def test_upload_requires_token(tmp_path):
    with _admin_server(models_dir=str(tmp_path / "models")) as (port, _):
        status, _ = _upload(port, "X", "model.safetensors", b"x", token="nope")
        assert status == 404


def test_upload_pt_without_model_config_fails(tmp_path):
    import torch
    models_dir = tmp_path / "models"
    raw = tmp_path / "raw.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2, 2)}}, raw)
    with _admin_server(models_dir=str(models_dir)) as (port, _):
        status, body = _upload(port, "Raw", "raw.pt", open(raw, "rb").read())
        assert status == 400
        assert "convert" in body["error"]
        # no stray files left behind
        assert not (models_dir / "raw.safetensors").exists()


def test_upload_rejects_incompatible_architecture(tmp_path):
    # A legacy model whose value_head was renamed to q_head (so it is neither a
    # valid value-head model nor a valid axis-relational KLENT model) can't be
    # loaded by the browser engine; the upload must fail with a clear 400, not a 500.
    import dataclasses
    from hexo_a0.gen_parity_fixtures import tiny_model_and_config
    from hexo_a0.export import save_safetensors
    model, mc = tiny_model_and_config()
    sd = {k.replace("value_head", "q_head"): v for k, v in model.state_dict().items()}
    src = tmp_path / "source_klent.safetensors"
    save_safetensors(sd, dataclasses.asdict(mc), 123, "klent.pt", src)
    models_dir = tmp_path / "models"
    with _admin_server(models_dir=str(models_dir),
                       build_variant=_real_build_variant) as (port, _):
        status, body = _upload(port, "klent", "klent.safetensors", src.read_bytes())
        assert status == 400
        assert "not loadable for browser inference" in body["error"]
        # no stray file left behind
        assert not (models_dir / "klent.safetensors").exists()


def test_upload_accepts_klent_architecture(tmp_path):
    # A real KLENT checkpoint (axis-relational + Q-head) is now loadable by the
    # browser engine; the upload must succeed (regression for the KLENT 500).
    import dataclasses
    from hexo_a0.gen_parity_fixtures import tiny_klent_model_and_config
    from hexo_a0.export import save_safetensors
    model, mc = tiny_klent_model_and_config()
    src = tmp_path / "klent.safetensors"
    save_safetensors(model.state_dict(), dataclasses.asdict(mc), 123, "klent.pt", src)
    models_dir = tmp_path / "models"
    with _admin_server(models_dir=str(models_dir),
                       build_variant=_real_build_variant) as (port, _):
        status, body = _upload(port, "klent", "klent.safetensors", src.read_bytes())
        assert status == 200
        assert (models_dir / "klent.safetensors").exists()
        uploaded = next(m for m in body["models"] if m["id"] == "klent")
        assert uploaded["local"] is True
