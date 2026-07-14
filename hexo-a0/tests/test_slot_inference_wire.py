"""--slot-inference (plan-A3 revised task 5): MSG_FORWARD_STATES served by the
A2 slot backend while MSG_FORWARD stays on the legacy model.

With the flag, states requests bypass the legacy graph rebuild entirely: the
wire's canonical int32 HexKeys feed ``slot_graph.build_slot_batch_from_keys``
directly (keys-input fast path — no per-stone unpack/repack), the checkpoint
is converted once at startup via ``slot_model_from_legacy`` (legacy GINE only;
anything else must FAIL AT STARTUP with a clear message), and the [B, N]
padded ``forward_padded`` logits are gathered back to the per-graph
legal-order layout the states response promises — with the FNV-1a legal-order
hashes recomputed from the slot path's OWN ordering so the guard stays honest.

The slot forward is a different kernel path than the legacy PyG-scriptable
model, so cross-mode outputs are compared as softmax-over-legal + values at
A2's proven tolerance (atol=1e-5 / rtol=1e-4), NOT byte-identically. The
no-flag byte-identity contract is covered by test_wire_states_parity.py and
must stay green.

Also covered: the zero-graph capability probe, in-band ERROR semantics
(builder-flag mismatch, win_length mismatch, the A2 [B, N, S, H] activation
memory guard) leaving the server alive, and the direct
build_slot_batch_from_keys == build_slot_batch tensor equality (incl. the aux
legal ordering == engine ``legal_moves()`` order).

The e2e fixture is a LEGACY HeXONet GINE checkpoint (training-export key
format) — the server's legacy model loads it via load_from_hexonet and the
slot model via HeXONet + slot_model_from_legacy, i.e. both wire modes run off
the same weights. Servers run CPU + --no-compile (torch.compile of the slot
model mirrors the legacy compile policy in production; too slow for CPU
tests).
"""

import dataclasses
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import hexo_rs  # noqa: F401  (fixture games need the engine)

from test_wire_states_parity import (
    FLAG_PRUNE,
    SMALL,
    _collate_graph_body,
    _e2e_games,
    _expected_legal_hash,
    _pack_hexkey,
    _read_forward_response,
    _read_states_response,
    _states_body,
    _write_header,
)

from hexo_a0.inference_server import (
    MAGIC,
    MSG_FORWARD,
    MSG_FORWARD_STATES,
    MSG_RELOAD,
    MSG_SHUTDOWN,
    VERSION,
    StatesRequestError,
    _pad_slot_batch,
    _round_pow2,
    _states_slot_node_counts,
    _validate_states_slot,
)

WIN_LENGTH = SMALL["win_length"]  # 5 -> slot count 6 * (5 - 1) = 24
NODE_DIM = 8
BUILDER_KWARGS = dict(prune_empty_edges=True)


# ---------------------------------------------------------------------------
# Fixture checkpoint + server spawn
# ---------------------------------------------------------------------------

def _make_hexonet_ckpt(conv_type: str) -> str:
    """Tiny LEGACY HeXONet checkpoint (training-export key format), with the
    embedded model_config the server derives its states builder flags from."""
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet

    cfg = ModelConfig(
        hidden_dim=32, num_layers=2, num_heads=4, conv_type=conv_type,
        policy_hidden=16, value_hidden=16, graph_type="axis",
        prune_empty_edges=True,
    )
    torch.manual_seed(1234)
    model = HeXONet(cfg)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {"model": model.state_dict(),
             "model_config": {"prune_empty_edges": True}},
            f.name,
        )
        return f.name


def _spawn_server(ckpt_path: str, *extra: str, conv_type: str = "gine"):
    cmd = [
        sys.executable, "-m", "hexo_a0.inference_server",
        "--checkpoint", ckpt_path,
        "--hidden-dim", "32", "--num-layers", "2", "--num-heads", "4",
        "--policy-hidden", "16", "--value-hidden", "16",
        "--graph-type", "axis", "--conv-type", conv_type,
        "--device", "cpu", "--no-compile", "--node-dim", str(NODE_DIM),
        *extra,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_ready(proc, timeout: float = 60.0) -> None:
    """Poll the child's stderr NON-BLOCKINGLY until READY.

    A plain ``proc.stderr.readline()`` blocks indefinitely when the child is
    alive but silent, so the deadline would never fire. Reading the raw fd in
    non-blocking mode lets the loop honour ``timeout`` regardless of the child.
    """
    import os

    fd = proc.stderr.fileno()
    os.set_blocking(fd, False)
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            chunk = b""
        if chunk:
            buf += chunk
            if b"READY" in buf:
                return
        elif proc.poll() is not None:
            # Exited; drain anything still buffered and check once more.
            try:
                buf += os.read(fd, 65536)
            except (BlockingIOError, OSError):
                pass
            if b"READY" in buf:
                return
            raise RuntimeError(
                f"server died before READY: {buf.decode(errors='replace')}"
            )
        else:
            time.sleep(0.05)
    raise TimeoutError("inference server didn't send READY in time")


@pytest.fixture(scope="module")
def slot_server():
    """The actual inference server with --slot-inference (GINE ckpt, CPU)."""
    ckpt_path = _make_hexonet_ckpt("gine")
    proc = _spawn_server(
        ckpt_path, "--slot-inference", "--win-length", str(WIN_LENGTH),
    )
    try:
        _wait_ready(proc)
        yield proc
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write(struct.pack("<IBB", MAGIC, VERSION, MSG_SHUTDOWN))
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        Path(ckpt_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Keys-input fast path: build_slot_batch_from_keys == build_slot_batch
# ---------------------------------------------------------------------------

def test_build_slot_batch_from_keys_matches_states_path():
    """The wire's int32 HexKeys fed directly to the builder must produce a
    bit-identical SlotBatch to the unpacked-states path, and the aux legal
    ordering must be the engine's legal_moves() order (the response contract)."""
    from hexo_a0.slot_graph import (
        SlotBatch,
        SlotBuilderConfig,
        build_slot_batch,
        build_slot_batch_from_keys,
        unpack,
    )

    games = _e2e_games()
    cfg = SlotBuilderConfig(
        win_length=SMALL["win_length"],
        placement_radius=SMALL["placement_radius"],
        prune_empty_edges=True,
    )
    states = [
        ([(q, r, p) for (q, r), p in g.placed_stones()],
         g.current_player(), g.moves_remaining_this_turn())
        for g in games
    ]
    ref = build_slot_batch(states, cfg)

    keyed = []
    for g in games:
        stones = g.placed_stones()
        p1 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P1"]
        p2 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P2"]
        keyed.append((p1, p2, 0 if g.current_player() == "P1" else 1,
                      g.moves_remaining_this_turn()))
    got, aux = build_slot_batch_from_keys(keyed, cfg, return_aux=True)

    for field in dataclasses.fields(SlotBatch):
        a, b = getattr(got, field.name), getattr(ref, field.name)
        assert a.dtype == b.dtype and a.shape == b.shape, field.name
        assert torch.equal(a, b), f"SlotBatch.{field.name} differs"

    assert aux.legal_counts.tolist() == [g.legal_move_count() for g in games]
    lq, lr = unpack(aux.legal_keys)
    coords = list(zip(lq.tolist(), lr.tolist()))
    offset = 0
    for g in games:
        n = g.legal_move_count()
        assert coords[offset:offset + n] == [tuple(m) for m in g.legal_moves()], (
            "aux legal ordering != engine legal_moves() order"
        )
        offset += n


# ---------------------------------------------------------------------------
# End-to-end: slot states backend vs legacy graph mode on one server
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_e2e_slot_states_agrees_with_graph_mode(slot_server):
    """probe ACK -> MSG_FORWARD (legacy model) -> MSG_FORWARD_STATES (slot
    backend), same positions: same legal counts, softmax-over-legal + values
    within A2 tolerance, FNV hashes matching legal_moves() order."""
    proc = slot_server
    games = _e2e_games()

    # 1. capability probe still ACKs (bypasses all slot guards).
    probe = bytearray()
    _write_header(probe, MSG_FORWARD_STATES)
    probe.extend(struct.pack("<IBBIBB", 0, 0, 0, 0, 0xFF, 99))
    proc.stdin.write(bytes(probe))
    proc.stdin.flush()
    assert _read_states_response(proc.stdout) == ("probe_ack",)

    # 2. graph mode (legacy model, code path untouched by the flag).
    graph_req = bytearray()
    _write_header(graph_req, MSG_FORWARD)
    graph_req.extend(_collate_graph_body(games, NODE_DIM, **BUILDER_KWARGS))
    proc.stdin.write(bytes(graph_req))
    proc.stdin.flush()
    g_logits, g_counts, g_values = _read_forward_response(proc.stdout)

    # 3. states mode -> slot backend, same positions.
    states_req = bytearray()
    _write_header(states_req, MSG_FORWARD_STATES)
    states_req.extend(_states_body(games, SMALL, FLAG_PRUNE, NODE_DIM))
    proc.stdin.write(bytes(states_req))
    proc.stdin.flush()
    resp = _read_states_response(proc.stdout)
    assert resp[0] == "ok", f"slot states request failed: {resp}"
    _, s_logits, s_counts, s_values, s_hashes = resp

    counts = np.frombuffer(g_counts, dtype=np.int32)
    assert counts.tolist() == [g.legal_move_count() for g in games]
    assert np.frombuffer(s_counts, dtype=np.int32).tolist() == counts.tolist()

    # Different kernel path (slot decomposition vs PyG-scriptable GINE) — NOT
    # byte-identical; the contract is A2's proven tolerance on the move
    # distribution (softmax over legal, per graph) and values.
    gl = torch.tensor(np.frombuffer(g_logits, dtype=np.float32).copy())
    sl = torch.tensor(np.frombuffer(s_logits, dtype=np.float32).copy())
    assert gl.shape == sl.shape
    offset = 0
    for c in counts.tolist():
        torch.testing.assert_close(
            torch.softmax(sl[offset:offset + c], dim=0),
            torch.softmax(gl[offset:offset + c], dim=0),
            atol=1e-5, rtol=1e-4,
        )
        offset += c
    torch.testing.assert_close(
        torch.tensor(np.frombuffer(s_values, dtype=np.float32).copy()),
        torch.tensor(np.frombuffer(g_values, dtype=np.float32).copy()),
        atol=1e-5, rtol=1e-4,
    )

    # FNV order guard, recomputed by the slot path from its own legal
    # ordering — must equal the client's legal_moves()-order hash.
    assert s_hashes == [_expected_legal_hash(g) for g in games]


def _oversized_states_body(game) -> bytes:
    """A states body that blows the default 4096 MiB activation budget via the
    SERVER-COMPUTED geometry (PY-1/R3C-W1: the guard must not trust the wire's
    num_legal, and it derives an EXACT node count from the stones). At the max
    placement_radius=64 the union of the stones' disks spans tens of thousands
    of cells, so 64 copies × that many nodes × 24 slots × 32 hidden × bytes × 3
    is tens of GiB — rejected before any tensor exists. The wire num_legal is
    set to a LIE (0) to prove it is ignored by the estimate."""
    stones = game.placed_stones()
    p1 = [c for c, p in stones if p == "P1"]
    p2 = [c for c, p in stones if p == "P2"]
    cur = 0 if game.current_player() == "P1" else 1
    n_copies = 64
    big_radius = 64  # max allowed; huge per-stone disk-cell bound
    buf = bytearray()
    buf.extend(struct.pack(
        "<IBBIBB", n_copies, SMALL["win_length"], big_radius,
        SMALL["max_moves"], FLAG_PRUNE, NODE_DIM,
    ))
    for _ in range(n_copies):
        buf.extend(struct.pack(
            "<HHBBH", len(p1), len(p2), cur,
            game.moves_remaining_this_turn(), 0,
        ))
        for q, r in p1 + p2:
            buf.extend(struct.pack("<i", _pack_hexkey(q, r)))
    return bytes(buf)


@pytest.mark.timeout(60)
def test_e2e_slot_error_paths_keep_server_alive(slot_server):
    """Every slot-backend validation failure answers in-band and leaves the
    server serving both wire modes."""
    proc = slot_server
    games = _e2e_games()

    def send_states(body: bytes):
        req = bytearray()
        _write_header(req, MSG_FORWARD_STATES)
        req.extend(body)
        proc.stdin.write(bytes(req))
        proc.stdin.flush()
        return _read_states_response(proc.stdout)

    # a) builder-flag mismatch (checkpoint says prune) -> in-band ERROR.
    resp = send_states(_states_body(games, SMALL, 0x00, NODE_DIM))
    assert resp[0] == "error" and "builder_flags" in resp[1]
    assert proc.poll() is None, "server died on a slot states error"

    # b) win_length mismatch vs --win-length -> in-band ERROR (the slot
    # model's edge tables are built for a fixed win_length).
    wrong_wl = dict(SMALL, win_length=WIN_LENGTH + 1)
    resp = send_states(_states_body(games, wrong_wl, FLAG_PRUNE, NODE_DIM))
    assert resp[0] == "error" and "win_length" in resp[1]
    assert proc.poll() is None

    # c) A2 memory sharp edge: over-budget activation estimate -> in-band
    # ERROR from the wire-claimed sizes alone (never an allocation/OOM).
    resp = send_states(_oversized_states_body(games[0]))
    assert resp[0] == "error" and "activation budget" in resp[1]
    assert proc.poll() is None

    # d) the server still answers both wire modes afterwards.
    resp = send_states(_states_body(games, SMALL, FLAG_PRUNE, NODE_DIM))
    assert resp[0] == "ok"
    assert resp[4] == [_expected_legal_hash(g) for g in games]

    graph_req = bytearray()
    _write_header(graph_req, MSG_FORWARD)
    graph_req.extend(_collate_graph_body(games, NODE_DIM, **BUILDER_KWARGS))
    proc.stdin.write(bytes(graph_req))
    proc.stdin.flush()
    _logits, counts, _values = _read_forward_response(proc.stdout)
    assert np.frombuffer(counts, dtype=np.int32).tolist() == [
        g.legal_move_count() for g in games
    ]


# ---------------------------------------------------------------------------
# Startup failures: clear errors BEFORE serving, never mid-request
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
def test_slot_inference_rejects_gatv2_checkpoint_at_startup():
    """An unsupported architecture (gatv2 — outside the A2 coverage boundary)
    must kill the server at startup with a clear message, not at request time."""
    ckpt_path = _make_hexonet_ckpt("gatv2")
    proc = _spawn_server(
        ckpt_path, "--slot-inference", "--win-length", str(WIN_LENGTH),
        conv_type="gatv2",
    )
    try:
        # Inner communicate timeout kept well under the outer pytest timeout
        # (180s) so a hang surfaces as a clear communicate TimeoutExpired with
        # the child's captured stderr, not an opaque outer-timeout kill.
        _out, err = proc.communicate(timeout=150)
        # exit(2) + the clean FATAL line — the startup guard catches Exception
        # (not just ValueError), so ANY startup failure routes here instead of a
        # raw traceback; this arch-mismatch ValueError is the concrete case.
        assert proc.returncode == 2
        assert b"READY" not in err, "server served despite unsupported arch"
        assert b"FATAL: --slot-inference startup check failed" in err, err.decode()
        assert b"gine" in err, err.decode()
    finally:
        Path(ckpt_path).unlink(missing_ok=True)


@pytest.mark.timeout(120)
def test_slot_inference_requires_win_length():
    """--slot-inference without --win-length is a startup usage error."""
    ckpt_path = _make_hexonet_ckpt("gine")
    proc = _spawn_server(ckpt_path, "--slot-inference")
    try:
        # Inner timeout ≤ outer (120s) − 30s: a hang surfaces as a clear
        # TimeoutExpired with captured stderr, not an opaque outer-timeout kill.
        _out, err = proc.communicate(timeout=90)
        assert proc.returncode != 0
        assert b"--win-length" in err, err.decode()
    finally:
        Path(ckpt_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _validate_states_slot guards (PY-1 budget, PY-2 radius) — pure, no server
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def _slot_args(**over):
    base = dict(
        prune_empty_edges=True, threat_features=False,
        relative_stone_encoding=False, node_dim=NODE_DIM, win_length=WIN_LENGTH,
        hidden_dim=32, slot_activation_budget_mb=4096.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _slot_frame(graphs, *, win_length=WIN_LENGTH, placement_radius=4,
                builder_flags=FLAG_PRUNE, node_dim=NODE_DIM):
    """graphs: list of (p1_keys, p2_keys, current_player, moves_remaining, num_legal)."""
    return (len(graphs), win_length, placement_radius, 200, builder_flags,
            node_dim, graphs)


def test_validate_states_slot_accepts_small_request():
    args = _slot_args()
    g = ([_pack_hexkey(0, 0)], [], 0, 2, 1)
    win_length, placement_radius, graphs = _validate_states_slot(
        _slot_frame([g], placement_radius=4), args
    )
    assert (win_length, placement_radius) == (WIN_LENGTH, 4)
    assert graphs == [g]


@pytest.mark.parametrize("cur", [2, 255], ids=["cur2", "cur255"])
def test_validate_states_slot_rejects_out_of_range_current_player(cur):
    """A current_player byte outside {0, 1} (P1/P2) is an in-band ERROR on the
    slot path, never a mid-request crash — mirrors the legacy states-path guard
    (test_states_invalid_current_player_is_inband_error)."""
    args = _slot_args()
    g = ([_pack_hexkey(0, 0)], [], cur, 2, 1)
    with pytest.raises(StatesRequestError, match="current_player"):
        _validate_states_slot(_slot_frame([g]), args)


@pytest.mark.parametrize("radius", [0, 65, 200])
def test_validate_states_slot_rejects_out_of_range_radius(radius):
    """PY-2: an out-of-band placement_radius is an in-band ERROR (never a disk
    meshgrid alloc / OOM). Bound matches hexo_rs.GameConfig's 1..=64."""
    args = _slot_args()
    g = ([_pack_hexkey(0, 0)], [], 0, 2, 1)
    with pytest.raises(StatesRequestError, match="placement_radius"):
        _validate_states_slot(_slot_frame([g], placement_radius=radius), args)


def test_validate_states_slot_budget_ignores_wire_num_legal():
    """PY-1: the activation-budget guard must NOT trust the wire's num_legal (an
    under-report would slip past then OOM in the build). A batch of many stones
    at a large radius blows a tiny budget even though the wire lies num_legal=0."""
    args = _slot_args(slot_activation_budget_mb=1.0)  # tiny budget
    many = [_pack_hexkey(q, 0) for q in range(50)]
    g = (many, [], 0, 2, 0)  # wire num_legal LIED to 0
    with pytest.raises(StatesRequestError, match="activation budget"):
        _validate_states_slot(_slot_frame([g], placement_radius=8), args)


def test_validate_states_slot_budget_passes_when_geometry_is_small():
    """Control: the SAME lie is harmless when the server-bounded geometry is
    small — proves the budget rejection above is driven by the server bound,
    not merely by the tiny --slot-activation-budget-mb."""
    args = _slot_args(slot_activation_budget_mb=1.0)
    g = ([_pack_hexkey(0, 0)], [], 0, 2, 0)  # 1 stone, radius 2
    win_length, placement_radius, graphs = _validate_states_slot(
        _slot_frame([g], placement_radius=2), args
    )
    assert graphs == [g]


def test_states_slot_node_counts_dedupes_disk_overlap():
    """R3C-W1: the node count is the EXACT |union of stone disks| (stones +
    legal), not the loose zero-overlap bound stones·disk_cells. A single stone
    at radius 2 has exactly 19 cells in its disk (1 + 6 + 12); two adjacent
    stones' disks OVERLAP, so the pair's count is far below 2·19."""
    one = _states_slot_node_counts([([_pack_hexkey(0, 0)], [], 0, 2, 0)], 2)
    assert one == [19], one
    two = _states_slot_node_counts(
        [([_pack_hexkey(0, 0)], [_pack_hexkey(1, 0)], 0, 2, 0)], 2
    )
    assert 19 < two[0] < 2 * 19, ("adjacent disks must overlap", two)


def test_validate_states_slot_budget_uses_padded_pow2_shape():
    """PY2-W1: the budget must estimate on the PADDED [B, N] the forward actually
    allocates (both dims rounded to powers of two via _round_pow2), not the raw
    counts. pow2 bucketing can ~4x the dense [B, N, S, H] tensor, so a raw-count
    estimate under-reports and could still OOM. Pick a budget the UNPADDED
    estimate clears but the PADDED one (what _handle_states_slot builds) exceeds.
    R3C-W1: node count is the EXACT disk-union size; element size defaults to
    4 B (CPU f32) and the ×3 builder-intermediate margin is part of the formula.
    """
    radius, hidden = 2, 32
    num_slots = 6 * (WIN_LENGTH - 1)
    g = ([_pack_hexkey(0, 0)], [], 0, 2, 0)
    n_max_est = _states_slot_node_counts([g], radius)[0]  # exact = 19
    unpadded = 1 * n_max_est * num_slots * hidden * 4 * 3
    padded = (
        _round_pow2(1) * _round_pow2(n_max_est, floor=8) * num_slots * hidden * 4 * 3
    )
    assert padded > unpadded, "test needs the pad to actually enlarge the alloc"

    # Budget strictly between the two: a raw-count guard would pass, the padded
    # (correct) guard must reject.
    mid_args = _slot_args(
        hidden_dim=hidden,
        slot_activation_budget_mb=(unpadded + padded) / 2 / 2**20,
    )
    with pytest.raises(StatesRequestError, match="activation budget"):
        _validate_states_slot(_slot_frame([g], placement_radius=radius), mid_args)

    # Control: a budget above the padded estimate accepts the same request.
    ok_args = _slot_args(
        hidden_dim=hidden, slot_activation_budget_mb=padded / 2**20 * 1.1,
    )
    _win, _rad, graphs = _validate_states_slot(
        _slot_frame([g], placement_radius=radius), ok_args
    )
    assert graphs == [g]


def _random_radius8_graphs(n: int, plies: int = 30, seed: int = 2024):
    """`n` real random-legal games (~`plies` stones each) at radius 8, encoded
    as slot-wire graph tuples with the wire num_legal set to a LIE (0) to prove
    the budget estimate never trusts it."""
    import random

    cfg = hexo_rs.GameConfig(6, 8, 300)
    rng = random.Random(seed)
    graphs = []
    for _ in range(n):
        game = hexo_rs.GameState(cfg)
        for _ in range(plies):
            if game.is_terminal():
                break
            game.apply_move(*rng.choice(game.legal_moves()))
        stones = game.placed_stones()
        p1 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P1"]
        p2 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P2"]
        cur = 0 if game.current_player() == "P1" else 1
        graphs.append((p1, p2, cur, game.moves_remaining_this_turn(), 0))
    return graphs


def _slot_est_mb(graphs, radius, win_length, hidden, node_count_fn):
    """Recompute the guard's padded-pow2 estimate (CPU f32, ×3) for a batch,
    using ``node_count_fn(graphs, radius)`` for the per-graph node counts —
    lets a test compare the new exact estimate against the old zero-overlap one.
    """
    from hexo_a0.inference_server import _round_pow2

    n_max = max(node_count_fn(graphs, radius))
    num_slots = 6 * (win_length - 1)
    return (
        _round_pow2(len(graphs)) * _round_pow2(n_max, floor=8)
        * num_slots * hidden * 4 * 3 / 2**20
    )


def test_validate_states_slot_passes_realistic_production_batch():
    """R3C-W1: the guard must not fatal ordinary production batches. A batch of
    real random-legal radius-8 games (~30 stones each, hidden 128, win_length 6)
    must comfortably clear the DEFAULT 4096 MiB budget, and the new exact
    disk-union node count must be far below the old zero-overlap bound
    (stones·disk_cells, which ignores inter-stone disk overlap)."""
    graphs = _random_radius8_graphs(8)
    args = _slot_args(win_length=6, hidden_dim=128, slot_activation_budget_mb=4096.0)
    frame = (len(graphs), 6, 8, 300, FLAG_PRUNE, NODE_DIM, graphs)
    _win, _rad, got = _validate_states_slot(frame, args)  # must NOT raise
    assert got == graphs

    disk_cells = 3 * 8 * 8 + 3 * 8 + 1  # r=8 zero-overlap cells/stone
    exact = max(_states_slot_node_counts(graphs, 8))
    old_bound = max((len(p1) + len(p2)) * (1 + disk_cells) for p1, p2, *_ in graphs)
    assert exact < old_bound, ("disk-union dedup must beat the zero-overlap bound",
                               exact, old_bound)


def test_validate_states_slot_unblocks_batch_the_old_bound_rejected():
    """R3C-W1: the exact estimate UNBLOCKS batches the old zero-overlap bound
    rejected. For a 16-graph radius-8 batch the OLD estimate blows 4096 MiB
    while the NEW (exact disk-union) estimate clears it — the guard accepts it."""
    _old_node_counts = lambda gs, r: [  # noqa: E731 (old zero-overlap bound)
        (len(p1) + len(p2)) * (1 + 3 * r * r + 3 * r + 1) for p1, p2, *_ in gs
    ]
    graphs = _random_radius8_graphs(16, seed=5)
    old_mb = _slot_est_mb(graphs, 8, 6, 128, _old_node_counts)
    new_mb = _slot_est_mb(graphs, 8, 6, 128, _states_slot_node_counts)
    assert old_mb > 4096.0 > new_mb, (old_mb, new_mb)

    args = _slot_args(win_length=6, hidden_dim=128, slot_activation_budget_mb=4096.0)
    frame = (len(graphs), 6, 8, 300, FLAG_PRUNE, NODE_DIM, graphs)
    _win, _rad, got = _validate_states_slot(frame, args)  # must NOT raise
    assert got == graphs


def test_validate_states_slot_rejects_genuinely_huge_batch():
    """R3C-W1: a genuinely oversized request (dozens of full radius-8 graphs in
    ONE batch — far beyond edge-budgeted batching) still rejects before any
    allocation, even with the exact node count."""
    args = _slot_args(win_length=6, hidden_dim=128, slot_activation_budget_mb=4096.0)
    huge = _random_radius8_graphs(64, seed=7)
    huge_frame = (len(huge), 6, 8, 300, FLAG_PRUNE, NODE_DIM, huge)
    with pytest.raises(StatesRequestError, match="activation budget"):
        _validate_states_slot(huge_frame, args)


# ---------------------------------------------------------------------------
# Slot padding buckets (PY-4): few compiled shapes under dynamic=False
# ---------------------------------------------------------------------------

def test_round_pow2_buckets_nearby_sizes_together():
    assert _round_pow2(1) == 1
    assert _round_pow2(9) == _round_pow2(15) == _round_pow2(16) == 16
    assert _round_pow2(17) == 32
    assert _round_pow2(3, floor=8) == 8
    assert _round_pow2(100, floor=8) == 128


def test_nearby_sizes_bucket_to_same_padded_shape():
    """PY-4: two requests with nearby node counts round to the SAME padded [B, N]
    shape, so torch.compile(dynamic=False) sees a handful of shapes, not one per
    request."""
    assert _round_pow2(9, floor=8) == _round_pow2(11, floor=8) == 16
    assert _round_pow2(3) == _round_pow2(4) == 4  # nearby B → same bucket


def test_pad_slot_batch_shapes_and_real_region_preserved():
    from hexo_a0.slot_graph import SlotBuilderConfig, build_slot_batch_from_keys

    games = _e2e_games()
    cfg = SlotBuilderConfig(
        win_length=SMALL["win_length"],
        placement_radius=SMALL["placement_radius"],
        prune_empty_edges=True,
    )
    keyed = []
    for g in games:
        stones = g.placed_stones()
        p1 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P1"]
        p2 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P2"]
        keyed.append((p1, p2, 0 if g.current_player() == "P1" else 1,
                      g.moves_remaining_this_turn()))
    batch = build_slot_batch_from_keys(keyed, cfg)
    b, n = batch.node_mask.shape
    b_to, n_to = _round_pow2(b), _round_pow2(n, floor=8)
    assert b_to >= b and n_to >= n

    padded = _pad_slot_batch(batch, b_to, n_to)
    assert padded.node_mask.shape == (b_to, n_to)
    assert padded.x.shape == (b_to, n_to, batch.x.shape[2])
    # Real region byte-identical; pad region inert (masks False, no legal cols).
    assert torch.equal(padded.node_mask[:b, :n], batch.node_mask)
    assert torch.equal(padded.x[:b, :n], batch.x)
    assert torch.equal(padded.legal_mask[:b, :n], batch.legal_mask)
    assert not padded.node_mask[b:, :].any()
    assert not padded.node_mask[:, n:].any()
    assert not padded.legal_mask[b:, :].any() and not padded.legal_mask[:, n:].any()
    assert not padded.filled[b:, :, :].any() and not padded.filled[:, n:, :].any()
    # A no-op pad (already at the target) returns the same object.
    assert _pad_slot_batch(padded, b_to, n_to) is padded


# ---------------------------------------------------------------------------
# Reload staging (PY-3): a NACKed reload must not desync args from live models
# ---------------------------------------------------------------------------

def _make_incompatible_ckpt() -> str:
    """A checkpoint that torch.load's fine (so its embedded model_config —
    prune_empty_edges=False — gets merged into args) but whose weights FAIL to
    load into the server's model (hidden_dim 64 vs the server's 32). The reload
    must NACK with prune_empty_edges already flipped mid-_load_model, exercising
    the args rollback."""
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet

    cfg = ModelConfig(
        hidden_dim=64, num_layers=2, num_heads=4, conv_type="gine",
        policy_hidden=16, value_hidden=16, graph_type="axis",
        prune_empty_edges=False,
    )
    torch.manual_seed(4321)
    model = HeXONet(cfg)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {"model": model.state_dict(),
             "model_config": {"prune_empty_edges": False}},
            f.name,
        )
        return f.name


def _send_reload(proc, path: str) -> None:
    pb = path.encode("utf-8")
    proc.stdin.write(
        struct.pack("<IBB", MAGIC, VERSION, MSG_RELOAD)
        + struct.pack("<I", len(pb))
        + pb
    )
    proc.stdin.flush()


def _read_reload_ack(proc) -> bool:
    def rd(n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = proc.stdout.read(n - len(data))
            if not chunk:
                raise EOFError("server closed stdout before reload ACK")
            data += chunk
        return data

    magic, ver, mt = struct.unpack("<IBB", rd(6))
    assert (magic, ver, mt) == (MAGIC, VERSION, MSG_RELOAD), (magic, ver, mt)
    return rd(1)[0] != 0


@pytest.mark.timeout(180)
def test_slot_reload_nack_leaves_args_consistent():
    """PY-3: a failed MSG_RELOAD must roll back the args it mutated so the
    still-live models stay consistent. Without the rollback, the incompatible
    ckpt's embedded prune_empty_edges=False leaks into args and a subsequent
    (correct) states request fails the builder-flags cross-check."""
    good = _make_hexonet_ckpt("gine")
    bad = _make_incompatible_ckpt()
    proc = _spawn_server(good, "--slot-inference", "--win-length", str(WIN_LENGTH))
    try:
        _wait_ready(proc)
        games = _e2e_games()

        def send_states(flags=FLAG_PRUNE):
            req = bytearray()
            _write_header(req, MSG_FORWARD_STATES)
            req.extend(_states_body(games, SMALL, flags, NODE_DIM))
            proc.stdin.write(bytes(req))
            proc.stdin.flush()
            return _read_states_response(proc.stdout)

        assert send_states()[0] == "ok", "baseline states request must work"

        _send_reload(proc, bad)
        assert _read_reload_ack(proc) is False, "incompatible reload must NACK"
        assert proc.poll() is None, "a NACKed reload must not kill the server"

        # The identical, correct request must STILL succeed — proving args
        # (prune_empty_edges, node dim, schema) were rolled back to the live model.
        resp = send_states()
        assert resp[0] == "ok", f"post-NACK request failed (args desynced?): {resp}"
        assert resp[4] == [_expected_legal_hash(g) for g in games]
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write(struct.pack("<IBB", MAGIC, VERSION, MSG_SHUTDOWN))
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        Path(good).unlink(missing_ok=True)
        Path(bad).unlink(missing_ok=True)
