"""A3 wire-states parity: MSG_FORWARD_STATES vs MSG_FORWARD (Tasks 2 + 3).

SCOPE (R3T-W1): these are TRANSPORT + COLLATION parity tests, not independent
feature-correctness tests. Both wire paths feed the SAME shared Rust lean axis
builder — the graph side via per-graph ``hexo_rs.game_to_axis_graph_raw`` and
the states side via the server's ``axis_states_to_batch`` rebuild — so the node
features and edges are identical BY DESIGN (that shared builder is exactly what
guarantees bit-identical model inputs across the two wires). What this file
pins is that the two wire framings collate to the same tensors: per-graph
feature concatenation, edge src/dst node offsets, edge_attr concatenation,
per-graph legal/stone masks, int32 batch indices, the stone_mask scatter, and
the padded/ghost layout. The builder's feature CORRECTNESS is covered by the
builder's own suites (hexo-rs graph tests + the Python graph tests), not here.

Task 2 (tensor parity, no subprocess): a states-mode request must produce
BIT-IDENTICAL ``_prepare_tensors`` output — every tensor: x, edge_index,
edge_attr, batch, legal_mask/idx, stone_mask/idx, stone_batch, node counts,
padding/ghost layout — to a graph-mode MSG_FORWARD body for the same
positions.

The graph-mode side is collated in this file exactly the way the real Rust
client encoder (``SubprocessModel::forward_graphs`` in
hexo-rs/hexo-mcts/src/inference_subprocess.rs) lays out the wire body:
features concatenated per graph, edge src/dst with per-graph node offsets,
edge_attr concatenated, per-graph legal masks, per-graph stone masks, then
int32 batch indices. (The true golden-byte capture from the Rust encoder
itself is plan Task 5; no cargo here.)

Positions are REAL: random legal games played via hexo_rs, covering mid-turn
(moves_remaining 1 and 2), near-terminal (one ply before game end), small
and large boards, with the flag matrix {prune, threat, relative} at least
all-on (production), all-off, and threat-only — in multi-graph batches.

Task 3 (end-to-end, subprocess): spin up the actual inference server (tiny
CPU checkpoint, same fixture pattern as tests/test_inference_server.py) and
assert byte-identical logits/values between MSG_FORWARD and
MSG_FORWARD_STATES responses for the same positions, correct FNV-1a
legal-coord hashes (recomputed client-side from ``legal_moves()`` order),
the zero-graph capability-probe round trip, and that in-band ERROR responses
leave the server alive for subsequent requests.
"""

import io
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import hexo_rs
from hexo_a0.inference_server import (
    MAGIC,
    MSG_FORWARD,
    MSG_FORWARD_STATES,
    MSG_SHUTDOWN,
    STATES_STATUS_ERROR,
    STATES_STATUS_OK,
    STATES_STATUS_PROBE_ACK,
    VERSION,
    _prepare_tensors,
    _read_forward_body,
    _read_forward_states_body,
)

# ---------------------------------------------------------------------------
# Real positions: random legal play via hexo_rs
# ---------------------------------------------------------------------------

# Small board: tight placement radius keeps graphs tiny; low max_moves means
# random play reaches a terminal state quickly (near-terminal class).
SMALL = dict(win_length=5, placement_radius=2, max_moves=40)
# Large board: production-like placement radius -> big dilation area, large
# legal-move sets, thousands of nodes per graph.
LARGE = dict(win_length=6, placement_radius=8, max_moves=300)

# builder-flag bits (must mirror the server's encoding)
FLAG_PRUNE = 0x01
FLAG_THREAT = 0x02
FLAG_RELATIVE = 0x04

# (flags, builder kwargs, node_dim) — the required matrix: production
# all-on, all-off, threat-only, relative-only.
FLAG_MATRIX = {
    "all_off": (0x00, dict(), 8),
    "threat_only": (
        FLAG_THREAT, dict(threat_features=True), 12),
    "relative_only": (
        FLAG_RELATIVE, dict(relative_stones=True), 7),
    "all_on": (
        FLAG_PRUNE | FLAG_THREAT | FLAG_RELATIVE,
        dict(prune_empty_edges=True, threat_features=True, relative_stones=True),
        11,
    ),
}


def _cfg(spec: dict) -> "hexo_rs.GameConfig":
    return hexo_rs.GameConfig(
        spec["win_length"], spec["placement_radius"], spec["max_moves"]
    )


def _snapshot(game, cfg):
    """Immutable copy of a live (mutated-in-place) game via from_state."""
    return hexo_rs.GameState.from_state(
        game.placed_stones(),
        game.current_player(),
        game.moves_remaining_this_turn(),
        cfg,
    )


def _play_snapshots(spec: dict, seed: int, snap_at: dict) -> dict:
    """Play random legal moves; snapshot after ply i for each i in snap_at.

    ``snap_at`` maps ply-index -> class name. Returns {name: GameState}.
    """
    cfg = _cfg(spec)
    rng = random.Random(seed)
    game = hexo_rs.GameState(cfg)
    out = {}
    for ply in range(max(snap_at) + 1):
        assert not game.is_terminal(), f"game ended before ply {ply}"
        game.apply_move(*rng.choice(game.legal_moves()))
        if ply in snap_at:
            out[snap_at[ply]] = _snapshot(game, cfg)
    return out


def _play_to_terminal(spec: dict, seed: int):
    """Random play until the game ends; return the LAST non-terminal state."""
    cfg = _cfg(spec)
    rng = random.Random(seed)
    game = hexo_rs.GameState(cfg)
    last = None
    while not game.is_terminal():
        last = _snapshot(game, cfg)
        game.apply_move(*rng.choice(game.legal_moves()))
    assert game.is_terminal() and last is not None
    return last


@pytest.fixture(scope="module")
def positions():
    """Multi-graph batches of real positions, keyed by board spec.

    Every batch mixes mid-turn moves_remaining 1 AND 2; the small batch also
    carries a near-terminal position (one ply before game end).
    """
    # Plies alternate mr: after an even ply index mr==1, after odd mr==2
    # (except turn boundaries) — pick indices empirically below and assert.
    small = _play_snapshots(SMALL, seed=11, snap_at={4: "a", 7: "b", 10: "c"})
    small_batch = [small["a"], small["b"], small["c"],
                   _play_to_terminal(SMALL, seed=23)]
    large = _play_snapshots(LARGE, seed=42, snap_at={20: "a", 27: "b"})
    large_batch = [large["a"], large["b"]]

    batches = {"small": (SMALL, small_batch), "large": (LARGE, large_batch)}
    for _name, (_spec, games) in batches.items():
        mrs = {g.moves_remaining_this_turn() for g in games}
        assert mrs == {1, 2}, f"batch must cover mid-turn mr 1 and 2, got {mrs}"
    return batches


# ---------------------------------------------------------------------------
# Wire-body builders
# ---------------------------------------------------------------------------

def _collate_graph_body(games, node_dim: int, **builder_kwargs) -> bytes:
    """MSG_FORWARD BODY collated from per-graph Rust builder outputs.

    Mirrors the Rust client encoder ``SubprocessModel::forward_graphs``
    byte-for-byte (see module docstring); deliberately does NOT go through
    ``axis_states_to_batch`` (the server's states-rebuild path) so the two
    sides of the parity check are independent.
    """
    raws = [hexo_rs.game_to_axis_graph_raw(g, **builder_kwargs) for g in games]
    total_nodes = sum(r["num_nodes"] for r in raws)
    total_edges = sum(len(r["edge_src"]) for r in raws)

    buf = bytearray()
    buf.extend(struct.pack("<III", total_nodes, total_edges, len(raws)))
    buf.extend(struct.pack("<BB", 1, node_dim))
    for r in raws:
        feats = np.asarray(r["features"], dtype=np.float32)
        assert feats.shape[0] == r["num_nodes"] * node_dim
        buf.extend(feats.tobytes())
    offset = 0
    for r in raws:
        buf.extend((np.asarray(r["edge_src"], dtype=np.int64) + offset).tobytes())
        offset += r["num_nodes"]
    offset = 0
    for r in raws:
        buf.extend((np.asarray(r["edge_dst"], dtype=np.int64) + offset).tobytes())
        offset += r["num_nodes"]
    for r in raws:
        buf.extend(np.asarray(r["edge_attr"], dtype=np.float32).tobytes())
    for r in raws:
        buf.extend(np.asarray(r["legal_mask"], dtype=np.uint8).tobytes())
    for r in raws:
        buf.extend(np.asarray(r["stone_mask"], dtype=np.uint8).tobytes())
    for i, r in enumerate(raws):
        buf.extend(np.full(r["num_nodes"], i, dtype=np.int32).tobytes())
    return bytes(buf)


def _pack_hexkey(q: int, r: int) -> int:
    """Canonical HexKey (independent test-side reference of the §1 spec):
    ``(q << 16) | ((r ^ 0x8000) & 0xFFFF)`` as a signed i32 — q sign-extended
    i16 in the high half, ONLY r biased."""
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def _states_body(games, spec: dict, builder_flags: int, node_dim: int) -> bytes:
    """MSG_FORWARD_STATES BODY for the same positions (outer header omitted).

    Per-graph payload: u16 n_p1 | u16 n_p2 | u8 current_player |
    u8 moves_remaining | u16 num_legal | i32 keys[n_p1] (P1 stones) |
    i32 keys[n_p2] (P2 stones), canonical HexKeys, little-endian.
    """
    buf = bytearray()
    buf.extend(struct.pack(
        "<IBBIBB", len(games), spec["win_length"], spec["placement_radius"],
        spec["max_moves"], builder_flags, node_dim,
    ))
    for g in games:
        stones = g.placed_stones()
        p1 = [c for c, p in stones if p == "P1"]
        p2 = [c for c, p in stones if p == "P2"]
        cur = 0 if g.current_player() == "P1" else 1
        buf.extend(struct.pack(
            "<HHBBH", len(p1), len(p2), cur,
            g.moves_remaining_this_turn(), g.legal_move_count(),
        ))
        for q, r in p1:
            buf.extend(struct.pack("<i", _pack_hexkey(q, r)))
        for q, r in p2:
            buf.extend(struct.pack("<i", _pack_hexkey(q, r)))
    return bytes(buf)


def _fnv1a64_ref(data: bytes) -> int:
    """Independent FNV-1a 64 reference (do not import the server's)."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _expected_legal_hash(game) -> int:
    """Client-side hash: (q i32 LE, r i32 LE) per move in legal_moves() order."""
    payload = b"".join(struct.pack("<ii", q, r) for q, r in game.legal_moves())
    return _fnv1a64_ref(payload)


# ---------------------------------------------------------------------------
# Task 2 — tensor parity through the unmodified _prepare_tensors
# ---------------------------------------------------------------------------

TENSOR_NAMES = (
    "x", "edge_index", "legal_mask", "stone_mask", "batch",
    "num_graphs", "edge_attr", "legal_idx", "stone_idx", "stone_batch",
)


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize("flag_name", list(FLAG_MATRIX), ids=list(FLAG_MATRIX))
@pytest.mark.parametrize("board", ["small", "large"])
def test_states_and_graph_mode_prepare_identical_tensors(
    positions, board, flag_name, padded
):
    """Bit-identical _prepare_tensors output, states-mode vs graph-mode.

    TRANSPORT/COLLATION parity (R3T-W1): both sides run the same shared lean
    axis builder, so equal tensors here verify the two wire FRAMINGS collate
    identically (offsets, concatenation, stone_mask scatter, dtype), not that
    the builder's features are correct. Covers the ghost/padding layout too:
    with ``padded=True`` both paths must pick the same bucket and emit
    identical ghost nodes/edges.
    """
    spec, games = positions[board]
    flags, builder_kwargs, node_dim = FLAG_MATRIX[flag_name]

    graph_body = _read_forward_body(
        io.BytesIO(_collate_graph_body(games, node_dim, **builder_kwargs)),
        expected_node_dim=node_dim,
    )
    states_body, _hashes = _read_forward_states_body(
        io.BytesIO(_states_body(games, spec, flags, node_dim)),
        node_dim,
        prune_empty_edges=builder_kwargs.get("prune_empty_edges", False),
        threat_features=builder_kwargs.get("threat_features", False),
        relative_stones=builder_kwargs.get("relative_stones", False),
    )

    # Scalar layout fields first (clearer failure than a tensor mismatch).
    for key in ("total_nodes", "total_edges", "num_graphs", "has_edge_attr",
                "node_dim"):
        assert states_body[key] == graph_body[key], f"body field {key!r}"

    cpu = torch.device("cpu")
    g_tensors, g_real_n = _prepare_tensors(graph_body, cpu, None, padded=padded)
    s_tensors, s_real_n = _prepare_tensors(states_body, cpu, None, padded=padded)

    assert s_real_n == g_real_n == len(games)
    assert len(s_tensors) == len(g_tensors) == len(TENSOR_NAMES)
    for name, sv, gv in zip(TENSOR_NAMES, s_tensors, g_tensors):
        if name == "num_graphs":  # plain int in the tensor tuple
            assert sv == gv, "num_graphs mismatch"
            continue
        assert sv.dtype == gv.dtype, f"{name}: dtype {sv.dtype} != {gv.dtype}"
        assert sv.shape == gv.shape, f"{name}: shape {sv.shape} != {gv.shape}"
        assert torch.equal(sv, gv), f"{name}: tensor values differ"


def test_position_classes(positions):
    """The fixture really covers the required position classes."""
    _spec, small = positions["small"]
    _spec, large = positions["large"]
    # Multi-graph batches on both boards.
    assert len(small) >= 3 and len(large) >= 2
    # Mid-turn coverage asserted inside the fixture; near-terminal state is
    # one ply before game end (built by _play_to_terminal) and non-terminal.
    near_terminal = small[-1]
    assert not near_terminal.is_terminal()
    assert near_terminal.moves_remaining_this_turn() in (1, 2)
    # Large board really is large: placement dilation yields a much bigger
    # legal set than the whole small board can hold.
    assert large[0].legal_move_count() > max(g.legal_move_count() for g in small)


def test_flag_matrix_is_not_vacuous(positions):
    """Sensitivity guard: the builder flags must actually change the build,
    otherwise the parity parametrization proves nothing."""
    spec, games = positions["small"]

    def parse(flag_name):
        flags, kwargs, node_dim = FLAG_MATRIX[flag_name]
        body, _ = _read_forward_states_body(
            io.BytesIO(_states_body(games, spec, flags, node_dim)), node_dim,
            prune_empty_edges=kwargs.get("prune_empty_edges", False),
            threat_features=kwargs.get("threat_features", False),
            relative_stones=kwargs.get("relative_stones", False),
        )
        return body

    all_off = parse("all_off")
    threat = parse("threat_only")
    relative = parse("relative_only")
    all_on = parse("all_on")
    assert threat["node_dim"] == 12 and all_off["node_dim"] == 8
    assert relative["node_dim"] == 7
    assert all_on["node_dim"] == 11
    # prune really prunes edges.
    assert all_on["total_edges"] < all_off["total_edges"]
    # threat features change node payload, not just the width.
    assert bytes(threat["features"]) != bytes(all_off["features"])
    # relative_stone_encoding changes the feature bytes too (TQ2-W4): without
    # this the relative-only parametrization would prove nothing.
    assert bytes(relative["features"]) != bytes(all_off["features"])


# ---------------------------------------------------------------------------
# Task 3 — end-to-end subprocess parity
# ---------------------------------------------------------------------------

def _write_header(buf: bytearray, msg_type: int) -> None:
    buf.extend(struct.pack("<IBB", MAGIC, VERSION, msg_type))


def _read_exact(stream, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")
        data += chunk
    return data


def _read_response_header(stdout, expected_msg_type: int) -> None:
    magic, ver, msg_type = struct.unpack("<IBB", _read_exact(stdout, 6))
    assert magic == MAGIC and ver == VERSION
    assert msg_type == expected_msg_type, f"msg_type 0x{msg_type:02X}"


def _read_forward_response(stdout) -> tuple[bytes, bytes, bytes]:
    """Raw (logits, legal_counts, values) byte payloads of MSG_FORWARD."""
    _read_response_header(stdout, MSG_FORWARD)
    total_legal, num_graphs = struct.unpack("<II", _read_exact(stdout, 8))
    logits = _read_exact(stdout, total_legal * 4)
    counts = _read_exact(stdout, num_graphs * 4)
    values = _read_exact(stdout, num_graphs * 4)
    return logits, counts, values


def _read_states_response(stdout):
    """Parse a MSG_FORWARD_STATES-typed response.

    Returns ``("ok", logits_bytes, counts_bytes, values_bytes, hashes)``,
    ``("error", message)`` or ``("probe_ack",)``.
    """
    _read_response_header(stdout, MSG_FORWARD_STATES)
    status = _read_exact(stdout, 1)[0]
    if status == STATES_STATUS_PROBE_ACK:
        return ("probe_ack",)
    if status == STATES_STATUS_ERROR:
        (msg_len,) = struct.unpack("<I", _read_exact(stdout, 4))
        return ("error", _read_exact(stdout, msg_len).decode("utf-8"))
    assert status == STATES_STATUS_OK, f"unknown status byte {status}"
    total_legal, num_graphs = struct.unpack("<II", _read_exact(stdout, 8))
    logits = _read_exact(stdout, total_legal * 4)
    counts = _read_exact(stdout, num_graphs * 4)
    values = _read_exact(stdout, num_graphs * 4)
    hashes = list(struct.unpack(f"<{num_graphs}Q", _read_exact(stdout, num_graphs * 8)))
    return ("ok", logits, counts, values, hashes)


@pytest.fixture(scope="module")
def states_server():
    """The actual inference server as a subprocess (tiny axis ckpt, CPU).

    The checkpoint embeds ``model_config = {"prune_empty_edges": True}`` so
    the server derives its states-rebuild builder flags from the checkpoint
    (single source of truth per the plan) — clients must send flags 0x01.
    """
    from hexo_a0.scriptable_model import ScriptableHeXONet

    model = ScriptableHeXONet(
        node_features=8, hidden_dim=32, num_layers=2, num_heads=4,
        policy_hidden=16, value_hidden=16, graph_type="axis",
        conv_type="gatv2",
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
        torch.save(
            {"model": model.state_dict(),
             "model_config": {"prune_empty_edges": True}},
            ckpt_path,
        )

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "hexo_a0.inference_server",
            "--checkpoint", ckpt_path,
            "--hidden-dim", "32", "--num-layers", "2", "--num-heads", "4",
            "--policy-hidden", "16", "--value-hidden", "16",
            "--graph-type", "axis", "--conv-type", "gatv2",
            "--device", "cpu", "--no-compile", "--node-dim", "8",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # NON-BLOCKING stderr poll: a plain readline() blocks forever if the
        # child is alive but silent, so the 30s deadline would never fire.
        import os

        fd = proc.stderr.fileno()
        os.set_blocking(fd, False)
        deadline = time.time() + 30
        buf = b""
        while time.time() < deadline:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                buf += chunk
                if b"READY" in buf:
                    break
            elif proc.poll() is not None:
                try:
                    buf += os.read(fd, 65536)
                except (BlockingIOError, OSError):
                    pass
                if b"READY" in buf:
                    break
                raise RuntimeError(
                    f"server died before READY: {buf.decode(errors='replace')}"
                )
            else:
                time.sleep(0.05)
        else:
            raise TimeoutError("inference server didn't send READY within 30s")
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


# The server's checkpoint says prune_empty_edges=True — the client's flags
# come from the same run config, so both wire modes build pruned graphs.
E2E_FLAGS = FLAG_PRUNE
E2E_BUILDER_KWARGS = dict(prune_empty_edges=True)
E2E_NODE_DIM = 8


def _e2e_games():
    small = _play_snapshots(SMALL, seed=7, snap_at={6: "mr2", 9: "mr1"})
    games = [small["mr2"], small["mr1"], _play_to_terminal(SMALL, seed=99)]
    assert {g.moves_remaining_this_turn() for g in games} == {1, 2}
    return games


@pytest.mark.timeout(120)
def test_e2e_states_vs_graph_byte_identical(states_server):
    """Full round trip against the real server process:

    probe ACK -> MSG_FORWARD -> MSG_FORWARD_STATES (same positions,
    byte-identical logits/counts/values + client-side FNV hash check) ->
    in-band ERROR -> server still answers both wire modes.
    """
    proc = states_server
    spec, games = SMALL, _e2e_games()

    # --- 1. startup capability probe (zero graphs, garbage flags/node_dim
    # must be ignored: the probe bypasses all guards) ---
    probe = bytearray()
    _write_header(probe, MSG_FORWARD_STATES)
    probe.extend(struct.pack("<IBBIBB", 0, 0, 0, 0, 0xFF, 99))
    proc.stdin.write(bytes(probe))
    proc.stdin.flush()
    assert _read_states_response(proc.stdout) == ("probe_ack",)

    # --- 2. graph-mode request (built the way the Rust client encodes it) ---
    graph_req = bytearray()
    _write_header(graph_req, MSG_FORWARD)
    graph_req.extend(_collate_graph_body(games, E2E_NODE_DIM, **E2E_BUILDER_KWARGS))
    proc.stdin.write(bytes(graph_req))
    proc.stdin.flush()
    g_logits, g_counts, g_values = _read_forward_response(proc.stdout)

    # --- 3. states-mode request for the SAME positions ---
    states_req = bytearray()
    _write_header(states_req, MSG_FORWARD_STATES)
    states_req.extend(_states_body(games, spec, E2E_FLAGS, E2E_NODE_DIM))
    proc.stdin.write(bytes(states_req))
    proc.stdin.flush()
    resp = _read_states_response(proc.stdout)
    assert resp[0] == "ok", f"states request failed: {resp}"
    _, s_logits, s_counts, s_values, s_hashes = resp

    # Byte-identical model outputs across the two wire modes.
    assert s_logits == g_logits, "logits bytes differ between wire modes"
    assert s_counts == g_counts, "legal_counts bytes differ between wire modes"
    assert s_values == g_values, "values bytes differ between wire modes"

    # Sanity on the shared payload.
    counts = np.frombuffer(g_counts, dtype=np.int32)
    assert counts.tolist() == [g.legal_move_count() for g in games]
    assert np.isfinite(np.frombuffer(g_logits, dtype=np.float32)).all()
    assert np.isfinite(np.frombuffer(g_values, dtype=np.float32)).all()

    # FNV-1a order guard: recomputed client-side from legal_moves() order.
    assert s_hashes == [_expected_legal_hash(g) for g in games]


@pytest.mark.timeout(60)
def test_e2e_error_keeps_server_alive(states_server):
    """An ERROR-triggering states request must answer in-band and leave the
    server serving BOTH wire modes afterwards."""
    proc = states_server
    spec, games = SMALL, _e2e_games()

    # Wire flags 0x00 disagree with the checkpoint-derived prune flag -> ERROR.
    bad_req = bytearray()
    _write_header(bad_req, MSG_FORWARD_STATES)
    bad_req.extend(_states_body(games, spec, 0x00, E2E_NODE_DIM))
    proc.stdin.write(bytes(bad_req))
    proc.stdin.flush()
    resp = _read_states_response(proc.stdout)
    assert resp[0] == "error"
    assert "builder_flags" in resp[1]
    assert proc.poll() is None, "server died on an in-band states error"

    # A valid states request right after the error must be answered...
    ok_req = bytearray()
    _write_header(ok_req, MSG_FORWARD_STATES)
    ok_req.extend(_states_body(games, spec, E2E_FLAGS, E2E_NODE_DIM))
    proc.stdin.write(bytes(ok_req))
    proc.stdin.flush()
    resp = _read_states_response(proc.stdout)
    assert resp[0] == "ok"
    assert resp[4] == [_expected_legal_hash(g) for g in games]

    # ...and so must a graph-mode request (old path untouched by the error).
    graph_req = bytearray()
    _write_header(graph_req, MSG_FORWARD)
    graph_req.extend(_collate_graph_body(games, E2E_NODE_DIM, **E2E_BUILDER_KWARGS))
    proc.stdin.write(bytes(graph_req))
    proc.stdin.flush()
    logits, counts, values = _read_forward_response(proc.stdout)
    assert np.frombuffer(counts, dtype=np.int32).tolist() == [
        g.legal_move_count() for g in games
    ]
