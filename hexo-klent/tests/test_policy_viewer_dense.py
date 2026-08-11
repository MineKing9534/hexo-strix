import dataclasses
import importlib.util
import json
import math
import threading
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path

import pytest
import torch

from hexo_klent.config import AlgorithmConfig, KlentModelConfig
from hexo_klent.mcts_adapter import KlentMCTSAdapter
from hexo_klent.model import (
    DenseAxisKlentNet,
    HexAxialCNNKlentNet,
    HexCNNKlentNet,
    HexD6DilatedCNNKlentNet,
    HexDilatedCNNKlentNet,
    PersistentRayKlentNet,
    make_klent_net,
)


def _load_policy_viewer():
    path = Path(__file__).parents[2] / "scripts" / "policy_viewer.py"
    spec = importlib.util.spec_from_file_location(
        "policy_viewer_dense_test",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/policy_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_dense_config(
    architecture: str = "dense_axis",
) -> KlentModelConfig:
    return KlentModelConfig(
        architecture=architecture,
        dense_ray_radius=5,
        ray_channels=4,
        ray_update_hidden=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        policy_hidden=8,
        q_hidden=4,
        cnn_dilations=(
            [1]
            if architecture in {"hex_dilated_cnn", "hex_d6_dilated_cnn"}
            else []
        ),
        pre_norm=True,
        dropout=0.0,
        graph_type="axis",
        prune_empty_edges=True,
        threat_features=True,
        relative_stone_encoding=True,
        axis_relational=True,
        axis_window=8,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )


@pytest.mark.parametrize(
    ("architecture", "network_type"),
    [
        ("dense_axis", DenseAxisKlentNet),
        ("persistent_ray_axis", PersistentRayKlentNet),
        ("hex_axial_cnn", HexAxialCNNKlentNet),
        ("hex_dilated_cnn", HexDilatedCNNKlentNet),
        ("hex_d6_dilated_cnn", HexD6DilatedCNNKlentNet),
    ],
)
def test_policy_viewer_bridges_dense_klent_checkpoint_everywhere(
    tmp_path,
    architecture,
    network_type,
    monkeypatch,
):
    config = _tiny_dense_config(architecture)
    algorithm = AlgorithmConfig()
    network = make_klent_net(config)
    assert isinstance(network, network_type)
    with torch.no_grad():
        q_output = (
            network.q_head[-1]
            if isinstance(network, HexCNNKlentNet)
            else network.q_head.fc2
        )
        q_output.bias.fill_(math.atanh(0.25))

    checkpoint = tmp_path / "dense-klent.pt"
    torch.save(
        {
            "format": "hexo-klent-v1",
            "iteration": 7,
            "model_state_dict": network.state_dict(),
            "model_config": dataclasses.asdict(config),
            "config": {"algorithm": dataclasses.asdict(algorithm)},
        },
        checkpoint,
    )

    viewer = _load_policy_viewer()
    model, loaded_config = viewer._load_model(str(checkpoint))

    assert isinstance(model, KlentMCTSAdapter)
    assert loaded_config.architecture == architecture

    analysis = viewer._analyze(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert len(analysis["legal"]) == 6
    assert sum(analysis["probs"]) == pytest.approx(1.0)
    assert analysis["raw_q"] == pytest.approx([0.25] * 6)
    assert analysis["value"] == pytest.approx(0.25)
    assert analysis["node_info"]

    mcts_configs = []
    make_mcts_config = viewer._hexo_rs.MCTSConfig

    def capture_mcts_config(*args, **kwargs):
        mcts_configs.append(kwargs)
        return make_mcts_config(*args, **kwargs)

    monkeypatch.setattr(viewer._hexo_rs, "MCTSConfig", capture_mcts_config)
    searched = viewer._analyze(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
        mcts_sims=4,
        mcts_m_actions=2,
        disable_forcing_solver=True,
    )
    assert len(searched["q_hat"]) == len(searched["legal"])
    assert searched["improved_policy"] is not None
    assert mcts_configs[-1]["disable_forcing_solver"] is True

    move = viewer._ai_move(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert move["move"] in analysis["legal"]
    assert move["value"] == pytest.approx(0.25)

    trajectory = viewer._analyze_trajectory(
        [[0, 0], [1, 0], [0, 1]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert trajectory["evaluated_prefixes"] == 3
    assert [entry["value"] for entry in trajectory["trajectory"]] == (
        pytest.approx([0.25, 0.25, 0.25])
    )


def test_policy_viewer_resolves_klent_checkpoint_directory_and_final(tmp_path):
    viewer = _load_policy_viewer()
    output_dir = tmp_path / "dense-run"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_000005.pt").touch()
    (checkpoint_dir / "final.pt").touch()

    resolved = viewer._checkpoint_dir_from_config(
        {"run": {"output_dir": str(output_dir)}},
        None,
    )

    assert resolved == str(checkpoint_dir)
    assert viewer._list_checkpoints(resolved) == [
        str(checkpoint_dir / "checkpoint_000005.pt"),
        str(checkpoint_dir / "final.pt"),
    ]
    assert (
        viewer._checkpoint_dir_from_config(
            {"run": {"output_dir": "ignored"}},
            "explicit/checkpoints",
        )
        == "explicit/checkpoints"
    )


def test_policy_viewer_resolves_periodic_distillation_epoch_checkpoints(tmp_path):
    viewer = _load_policy_viewer()
    output_dir = tmp_path / "d6-distillation"
    epoch_6 = (
        output_dir
        / "distillation_epochs"
        / "epoch_0006"
        / "checkpoints"
        / "checkpoint_000000.pt"
    )
    epoch_12 = (
        output_dir
        / "distillation_epochs"
        / "epoch_0012"
        / "checkpoints"
        / "checkpoint_000000.pt"
    )
    epoch_6.parent.mkdir(parents=True)
    epoch_12.parent.mkdir(parents=True)
    epoch_6.touch()
    epoch_12.touch()

    source, checkpoints = viewer._resolve_checkpoint_source(str(output_dir))

    assert source == str(output_dir.resolve())
    assert checkpoints == [str(epoch_6.resolve()), str(epoch_12.resolve())]
    assert viewer._directory_checkpoint_count(output_dir) == 2
    assert viewer._list_checkpoint_sources([source]) == checkpoints


def test_policy_viewer_adds_runtime_checkpoint_sources_across_runs(tmp_path):
    viewer = _load_policy_viewer()

    run_a = tmp_path / "family-a" / "run-1"
    checkpoints_a = run_a / "checkpoints"
    checkpoints_a.mkdir(parents=True)
    checkpoint_a = checkpoints_a / "checkpoint_000005.pt"
    checkpoint_a.touch()

    checkpoints_b = tmp_path / "family-b" / "checkpoints"
    checkpoints_b.mkdir(parents=True)
    checkpoint_b = checkpoints_b / "checkpoint_000005.pt"
    final_b = checkpoints_b / "final.pt"
    checkpoint_b.touch()
    final_b.touch()

    source_a, found_a = viewer._resolve_checkpoint_source(str(run_a))
    source_b, found_b = viewer._resolve_checkpoint_source(str(checkpoints_b))

    assert source_a == str(run_a.resolve())
    assert found_a == [str(checkpoint_a.resolve())]
    assert source_b == str(checkpoints_b.resolve())
    assert found_b == [str(checkpoint_b.resolve()), str(final_b.resolve())]
    assert viewer._list_checkpoint_sources([source_a, source_b]) == [
        str(checkpoint_a.resolve()),
        str(checkpoint_b.resolve()),
        str(final_b.resolve()),
    ]

    # Runtime sources are refreshed, so a trainer can publish another
    # checkpoint without restarting the viewer or re-adding the directory.
    checkpoint_a_new = checkpoints_a / "checkpoint_000010.pt"
    checkpoint_a_new.touch()
    assert viewer._list_checkpoint_sources([source_a]) == [
        str(checkpoint_a.resolve()),
        str(checkpoint_a_new.resolve()),
    ]


def test_policy_viewer_runtime_source_accepts_one_checkpoint_and_rejects_empty_dir(
    tmp_path,
):
    viewer = _load_policy_viewer()
    checkpoint = tmp_path / "one-off.pt"
    checkpoint.touch()

    source, found = viewer._resolve_checkpoint_source(str(checkpoint))
    assert source == str(checkpoint.resolve())
    assert found == [str(checkpoint.resolve())]

    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(ValueError, match="No checkpoint"):
        viewer._resolve_checkpoint_source(str(empty))


def test_policy_viewer_directory_browser_defaults_to_runs_symlink(tmp_path):
    viewer = _load_policy_viewer()
    actual_runs = tmp_path / "actual-runs"
    checkpoint_dir = actual_runs / "family-a" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_000007.pt").touch()
    (actual_runs / "empty-family").mkdir()
    runs_link = tmp_path / "runs"
    runs_link.symlink_to(actual_runs, target_is_directory=True)

    listing = viewer._browse_directories(default_root=str(runs_link))

    assert listing["path"] == str(runs_link.absolute())
    assert listing["checkpoint_count"] == 0
    assert {
        entry["name"]: entry["checkpoint_count"]
        for entry in listing["directories"]
    } == {"empty-family": 0, "family-a": 1}


def test_policy_viewer_runtime_source_http_endpoint(tmp_path, monkeypatch):
    viewer = _load_policy_viewer()
    checkpoint_dir = tmp_path / "another-run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_000042.pt"
    checkpoint.touch()

    ai_move_args = {}

    def fake_ai_move(**kwargs):
        ai_move_args.update(kwargs)
        return {"move": [1, 0], "value": 0.0, "prob": 1.0, "terminal": False}

    monkeypatch.setattr(viewer, "_ai_move", fake_ai_move)
    viewer.Handler.checkpoint_sources = []
    server = HTTPServer(("127.0.0.1", 0), viewer.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(
            "POST",
            "/checkpoint_sources",
            body=json.dumps({"path": str(checkpoint_dir)}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["source"] == str(checkpoint_dir.resolve())
        assert payload["checkpoints"] == [str(checkpoint.resolve())]

        connection.request("GET", "/checkpoints")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == [str(checkpoint.resolve())]

        connection.request(
            "GET",
            f"/browse_directories?path={checkpoint_dir.parent}",
        )
        response = connection.getresponse()
        listing = json.loads(response.read())
        assert response.status == 200
        assert listing["path"] == str(checkpoint_dir.parent.absolute())
        assert listing["checkpoint_count"] == 1
        assert listing["directories"] == [
            {
                "name": "checkpoints",
                "path": str(checkpoint_dir.absolute()),
                "checkpoint_count": 1,
            }
        ]

        connection.request(
            "POST",
            "/ai_move",
            body=json.dumps(
                {
                    "moves": [[0, 0]],
                    "checkpoint": "unused.pt",
                    "mcts_sims": 4,
                    "mcts_m_actions": 2,
                    "disable_forcing_solver": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["move"] == [1, 0]
        assert ai_move_args["disable_forcing_solver"] is True
        assert 'id="play-use-forcing" type="checkbox" checked' in viewer.HTML
        assert "disable_forcing_solver: !useForcing" in viewer.HTML
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
