"""Unit tests for the v2 inference wire protocol (no subprocess, no GPU).

Builds forward-message bodies in pure Python bytes and checks that
``_read_forward_body`` / ``_prepare_tensors`` honour the node-feature-dim
field added in protocol v2, and that version mismatches are rejected loudly.

Also covers MSG_FORWARD_STATES (0x03, the A3 state-payload wire): body
parse + server-side graph rebuild must emit the EXACT dict shape
``_read_forward_body`` produces (so the unmodified ``_prepare_tensors``
consumes it), validation failures must be in-band ``StatesRequestError``s
with the body fully consumed (framing intact — the server survives), and
the new states response type must carry per-graph FNV-1a legal-coord
hashes behind a status byte.
"""

import io
import struct

import numpy as np
import pytest
import torch

import hexo_rs
from hexo_a0.graph import axis_states_to_batch
from hexo_a0.inference_server import (
    MAGIC,
    MSG_FORWARD_STATES,
    STATES_STATUS_ERROR,
    STATES_STATUS_OK,
    STATES_STATUS_PROBE_ACK,
    VERSION,
    StatesRequestError,
    _check_protocol_header,
    _prepare_tensors,
    _read_forward_body,
    _read_forward_states_body,
    _unpack_hexkey,
    _write_forward_states_response,
    _write_states_error,
    _write_states_probe_ack,
)

# Byte offset of the node_dim field within the forward-message body header
# (after total_nodes u32 | total_edges u32 | num_graphs u32 | has_edge_attr u8).
NODE_DIM_OFFSET = 13


def _build_forward_body(
    node_dim: int,
    n_nodes: int = 4,
    n_edges: int = 6,
    num_graphs: int = 1,
    edge_attr: bool = False,
) -> tuple[bytes, np.ndarray]:
    """Build a forward-message BODY (the 6-byte outer header is consumed by
    the reader thread before ``_read_forward_body`` runs, so it is omitted).

    Returns ``(body_bytes, features)`` so callers can compare round-tripped
    feature values.
    """
    features = np.random.randn(n_nodes, node_dim).astype(np.float32)
    edge_src = np.arange(n_edges, dtype=np.int64) % n_nodes
    edge_dst = (np.arange(n_edges, dtype=np.int64) + 1) % n_nodes
    legal_mask = np.ones(n_nodes, dtype=np.uint8)
    stone_mask = np.zeros(n_nodes, dtype=np.uint8)
    batch = np.zeros(n_nodes, dtype=np.int32)

    buf = bytearray()
    buf.extend(struct.pack("<III", n_nodes, n_edges, num_graphs))
    buf.extend(struct.pack("<BB", 1 if edge_attr else 0, node_dim))
    buf.extend(features.tobytes())
    buf.extend(edge_src.tobytes())
    buf.extend(edge_dst.tobytes())
    if edge_attr:
        buf.extend(np.random.randn(n_edges, 5).astype(np.float32).tobytes())
    buf.extend(legal_mask.tobytes())
    buf.extend(stone_mask.tobytes())
    buf.extend(batch.tobytes())
    return bytes(buf), features


@pytest.mark.parametrize("node_dim", [8, 12])
@pytest.mark.parametrize("edge_attr", [False, True], ids=["plain", "edge_attr"])
def test_read_forward_body_node_dim(node_dim: int, edge_attr: bool):
    """The features section is sized by the header's node_dim field."""
    body_bytes, features = _build_forward_body(node_dim, edge_attr=edge_attr)
    stream = io.BytesIO(body_bytes)

    body = _read_forward_body(stream, expected_node_dim=node_dim)

    assert body["node_dim"] == node_dim
    assert body["total_nodes"] == 4
    assert body["total_edges"] == 6
    assert body["num_graphs"] == 1
    assert len(body["features"]) == 4 * node_dim * 4
    decoded = np.frombuffer(bytes(body["features"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded.reshape(4, node_dim), features)
    # The whole body must be consumed — a sizing bug leaves trailing bytes
    # (which would corrupt the NEXT message's framing).
    assert stream.read() == b""


def test_read_forward_body_rejects_mismatched_node_dim():
    """A wire dim that disagrees with the server's --node-dim must fail loudly,
    not surface later as an obscure GNN shape error."""
    body_bytes, _ = _build_forward_body(12)
    with pytest.raises(
        ValueError, match=r"wire node_dim 12 != server --node-dim 8"
    ):
        _read_forward_body(io.BytesIO(body_bytes), expected_node_dim=8)


def test_read_forward_body_rejects_zero_node_dim():
    body_bytes, _ = _build_forward_body(8)
    # Patch the node_dim byte to 0 (corrupt/legacy-v1 framing).
    corrupted = bytearray(body_bytes)
    corrupted[NODE_DIM_OFFSET] = 0
    with pytest.raises(ValueError, match="node_dim"):
        _read_forward_body(io.BytesIO(bytes(corrupted)), expected_node_dim=8)


@pytest.mark.parametrize("node_dim", [8, 12])
def test_prepare_tensors_reshapes_to_node_dim(node_dim: int):
    """_prepare_tensors must reshape features to (N, node_dim), unpadded CPU."""
    body_bytes, features = _build_forward_body(node_dim)
    body = _read_forward_body(io.BytesIO(body_bytes), expected_node_dim=node_dim)

    tensors, real_n = _prepare_tensors(body, torch.device("cpu"), None, padded=False)
    feat_tensor = tensors[0]

    assert real_n == 1
    assert feat_tensor.shape == (4, node_dim)
    np.testing.assert_array_equal(feat_tensor.numpy(), features)


def test_check_protocol_header_accepts_current_version():
    _check_protocol_header(MAGIC, VERSION)  # must not raise


def test_check_protocol_header_rejects_wrong_version():
    with pytest.raises(ValueError, match=r"version 1\b"):
        _check_protocol_header(MAGIC, 1)


def test_check_protocol_header_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        _check_protocol_header(0xDEADBEEF, VERSION)


# ---------------------------------------------------------------------------
# MSG_FORWARD_STATES (0x03) — state-payload requests (A3 wire)
# ---------------------------------------------------------------------------

# Small placement radius keeps the rebuilt graphs tiny (CPU unit tests).
WIN_LENGTH = 5
PLACEMENT_RADIUS = 4
MAX_MOVES = 200

# builder_flags bits (must mirror the server's encoding).
FLAG_PRUNE = 0x01
FLAG_THREAT = 0x02
FLAG_RELATIVE = 0x04


def _pack_hexkey(q: int, r: int) -> int:
    """Canonical HexKey (independent test-side reference of the §1 spec):
    ``(q << 16) | ((r ^ 0x8000) & 0xFFFF)`` as a signed i32 — q sign-extended
    i16 in the high half, ONLY r biased."""
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def _mk_config():
    return hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)


def _mk_games():
    """Two differently-sized non-terminal positions (multi-graph batch)."""
    cfg = _mk_config()
    g1 = hexo_rs.GameState.from_state(
        [((1, 0), "P1"), ((0, 1), "P2"), ((2, -1), "P2")], "P1", 2, cfg
    )
    g2 = hexo_rs.GameState.from_state([((1, 0), "P2")], "P2", 1, cfg)
    return [g1, g2]


def _encode_states_body(
    games,
    builder_flags: int = 0,
    node_dim: int = 8,
    *,
    num_graphs_override: int | None = None,
    num_legal_override: int | None = None,
    moves_remaining_override: int | None = None,
    current_player_override: int | None = None,
    extra_stones: dict[int, list] | None = None,
) -> bytes:
    """Encode a MSG_FORWARD_STATES BODY (outer 6-byte header omitted — the
    reader thread consumes it before ``_read_forward_states_body`` runs)."""
    buf = bytearray()
    num_graphs = len(games) if num_graphs_override is None else num_graphs_override
    buf.extend(
        struct.pack(
            "<IBBIBB",
            num_graphs,
            WIN_LENGTH,
            PLACEMENT_RADIUS,
            MAX_MOVES,
            builder_flags,
            node_dim,
        )
    )
    for i, g in enumerate(games):
        stones = sorted(g.placed_stones())
        if extra_stones and i in extra_stones:
            stones = stones + extra_stones[i]
        cur = (
            (0 if g.current_player() == "P1" else 1)
            if current_player_override is None
            else current_player_override
        )
        mr = (
            g.moves_remaining_this_turn()
            if moves_remaining_override is None
            else moves_remaining_override
        )
        nl = g.legal_move_count() if num_legal_override is None else num_legal_override
        p1 = [c for c, p in stones if p == "P1"]
        p2 = [c for c, p in stones if p == "P2"]
        buf.extend(struct.pack("<HHBBH", len(p1), len(p2), cur, mr, nl))
        for q, r in p1:
            buf.extend(struct.pack("<i", _pack_hexkey(q, r)))
        for q, r in p2:
            buf.extend(struct.pack("<i", _pack_hexkey(q, r)))
    return bytes(buf)


def _graph_mode_body_bytes(games, node_dim: int = 8, **builder_kwargs) -> bytes:
    """Build an equivalent MSG_FORWARD (graph-mode) BODY from the
    ``axis_states_to_batch`` buffers for the same positions.

    stone_mask is derived INDEPENDENTLY of the server's stone_idx scatter:
    per graph, the collated node order is stones-sorted, legal-sorted, dummy,
    so the first ``len(placed_stones)`` nodes of each graph are the stones.
    """
    batch, aux = axis_states_to_batch(games, **builder_kwargs)
    total_nodes = batch.x.shape[0]
    total_edges = batch.edge_index.shape[1]

    stone_mask = np.zeros(total_nodes, dtype=np.uint8)
    bvec = batch.batch.numpy()
    for i, g in enumerate(games):
        start = int(np.searchsorted(bvec, i, side="left"))
        stone_mask[start : start + len(g.placed_stones())] = 1

    buf = bytearray()
    buf.extend(struct.pack("<III", total_nodes, total_edges, len(games)))
    buf.extend(struct.pack("<BB", 1, node_dim))
    buf.extend(batch.x.numpy().tobytes())
    buf.extend(batch.edge_index[0].contiguous().numpy().tobytes())
    buf.extend(batch.edge_index[1].contiguous().numpy().tobytes())
    buf.extend(batch.edge_attr.numpy().tobytes())
    buf.extend(batch.legal_mask.to(torch.uint8).numpy().tobytes())
    buf.extend(stone_mask.tobytes())
    buf.extend(batch.batch.to(torch.int32).numpy().tobytes())
    return bytes(buf)


def _reference_fnv1a64(data: bytes) -> int:
    """Independent FNV-1a 64 reference (do not import the server's)."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# Includes negative q, negative r, both-negative, and the four i16 corners.
HEXKEY_CASES = [
    (0, 0), (1, 0), (0, 1), (1, 2),
    (-1, 0), (-5, 7), (-300, 42),          # negative q, positive/zero r
    (0, -1), (3, -11), (299, -300),        # positive/zero q, negative r
    (-1, -1), (-17, -256), (-300, -300),   # both negative
    (-32768, -32768), (-32768, 32767), (32767, -32768), (32767, 32767),
]


def test_hexkey_exact_decode_roundtrip():
    """Pin the exact key layout AND the exact decode.

    key = (q << 16) | ((r ^ 0x8000) & 0xFFFF), signed i32, q sign-extended
    i16 in the high half, ONLY r biased. Decode: q = key >> 16 (arithmetic
    shift), r = i16 reinterpretation of (key & 0xFFFF) ^ 0x8000.
    """
    for q, r in HEXKEY_CASES:
        key = _pack_hexkey(q, r)
        assert -0x80000000 <= key <= 0x7FFFFFFF  # a real i32
        # i32 LE wire round trip (what the parser sees via struct "<i").
        (wire_key,) = struct.unpack("<i", struct.pack("<i", key))
        assert wire_key == key
        # Spelled-out reference decode, independent of the server helper.
        dq = key >> 16  # arithmetic shift (Python ints are signed)
        rr = (key & 0xFFFF) ^ 0x8000
        dr = rr - 0x10000 if rr >= 0x8000 else rr  # i16 reinterpretation
        assert (dq, dr) == (q, r), f"reference decode broke for {(q, r)}"
        # The server's decode helper must agree exactly.
        assert _unpack_hexkey(key) == (q, r)


def test_hexkey_int32_sort_order_is_qr_lexicographic():
    """Signed-i32 key order == lexicographic (q, r) order (the §1 property
    that makes torch.sort/searchsorted agree with the node ordering)."""
    import random

    coords = sorted(
        {(q, r) for q in range(-40, 41, 7) for r in range(-40, 41, 7)}
        | set(HEXKEY_CASES)
    )
    random.Random(1234).shuffle(coords)  # order must come from the keys
    keys = np.array([_pack_hexkey(q, r) for q, r in coords], dtype=np.int32)
    order_by_key = np.argsort(keys, kind="stable").tolist()
    order_by_qr = sorted(range(len(coords)), key=lambda i: coords[i])
    assert order_by_key == order_by_qr
    # Keys are strictly increasing over sorted-(q, r) coords: distinct and
    # signed-comparison-ordered.
    assert (np.diff(keys[order_by_qr]) > 0).all()


def test_read_forward_states_body_roundtrip():
    """Parse + rebuild must fully consume the body and produce a
    _read_forward_body-shaped dict with consistent sizes."""
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games))

    body, hashes = _read_forward_states_body(stream, 8)

    assert stream.read() == b""  # whole body consumed (framing intact)
    assert body["num_graphs"] == 2
    assert body["node_dim"] == 8
    assert body["has_edge_attr"] == 1
    n = body["total_nodes"]
    e = body["total_edges"]
    assert n > 0 and e > 0
    assert len(body["features"]) == n * 8 * 4
    assert len(body["edge_src"]) == e * 8
    assert len(body["edge_dst"]) == e * 8
    assert len(body["edge_attr"]) == e * 5 * 4
    assert len(body["legal_mask"]) == n
    assert len(body["stone_mask"]) == n
    assert len(body["batch"]) == n * 4
    assert len(hashes) == 2


def test_states_body_matches_graph_mode_body():
    """The states parser emits the EXACT dict shape _read_forward_body produces.

    NOTE (R3T-W1): both sides here go through the SAME rebuild
    (axis_states_to_batch), so this is TRANSPORT/COLLATION parity — it pins the
    states-path FRAMING and the stone_mask scatter (which the graph body carries
    directly but the states path reconstructs from stone_idx), NOT the graph
    BUILD. test_wire_states_parity.py cross-checks the states rebuild against the
    Rust client's graph WIRE collation, but that too shares the lean builder by
    design; the builder's feature CORRECTNESS lives in the builder's own
    suites (hexo-rs graph tests + the Python graph tests)."""
    games = _mk_games()
    states_body, _hashes = _read_forward_states_body(
        io.BytesIO(_encode_states_body(games)), 8
    )
    graph_body = _read_forward_body(
        io.BytesIO(_graph_mode_body_bytes(games)), expected_node_dim=8
    )

    assert set(states_body.keys()) == set(graph_body.keys())
    for key in graph_body:
        sv, gv = states_body[key], graph_body[key]
        if isinstance(gv, (bytes, bytearray)):
            assert bytes(sv) == bytes(gv), f"byte mismatch in {key!r}"
        else:
            assert sv == gv, f"value mismatch in {key!r}"


def test_states_body_feeds_unmodified_prepare_tensors():
    games = _mk_games()
    body, _ = _read_forward_states_body(io.BytesIO(_encode_states_body(games)), 8)

    tensors, real_n = _prepare_tensors(body, torch.device("cpu"), None, padded=False)

    assert real_n == 2
    assert tensors[0].shape == (body["total_nodes"], 8)
    # batch vector must be int64 after _prepare_tensors' int32 -> long convert
    assert tensors[4].dtype == torch.long
    assert tensors[4].max().item() == 1


def test_states_multi_graph_batch_dtype():
    """The binding hands back an i64 batch vector; the wire dict must carry
    int32 (the graph-mode wire dtype) — a multi-graph batch catches this."""
    games = _mk_games()
    body, _ = _read_forward_states_body(io.BytesIO(_encode_states_body(games)), 8)

    decoded = np.frombuffer(bytes(body["batch"]), dtype=np.int32)
    assert decoded.shape[0] == body["total_nodes"]
    # Graph indices 0 and 1 both present, monotone non-decreasing.
    assert decoded.min() == 0 and decoded.max() == 1
    assert (np.diff(decoded) >= 0).all()


@pytest.mark.parametrize(
    "flags,kwargs,node_dim",
    [
        (FLAG_THREAT, dict(threat_features=True), 12),
        (FLAG_RELATIVE, dict(relative_stones=True), 7),
        (FLAG_PRUNE, dict(prune_empty_edges=True), 8),
    ],
    ids=["threat", "relative", "prune"],
)
def test_states_body_builder_flag_variants(flags, kwargs, node_dim):
    """Server builder flags drive the rebuild; n_feat/node_dim synthesized.

    Like test_states_body_matches_graph_mode_body, both sides run through
    axis_states_to_batch — this checks FRAMING + stone_mask scatter across the
    flag matrix (transport/collation, R3T-W1), not an independent graph build.
    Builder feature correctness lives in the builder's own suites."""
    games = _mk_games()
    body, _ = _read_forward_states_body(
        io.BytesIO(_encode_states_body(games, builder_flags=flags, node_dim=node_dim)),
        node_dim,
        **kwargs,
    )
    graph_body = _read_forward_body(
        io.BytesIO(_graph_mode_body_bytes(games, node_dim=node_dim, **kwargs)),
        expected_node_dim=node_dim,
    )
    for key in graph_body:
        sv, gv = body[key], graph_body[key]
        if isinstance(gv, (bytes, bytearray)):
            assert bytes(sv) == bytes(gv), f"byte mismatch in {key!r}"
        else:
            assert sv == gv, f"value mismatch in {key!r}"


def test_states_legal_hashes_match_legal_moves_order():
    """Per-graph u64 hash = FNV-1a over (q i32 LE, r i32 LE) per legal move,
    in legal_moves() order — computed here with an independent reference."""
    games = _mk_games()
    _, hashes = _read_forward_states_body(io.BytesIO(_encode_states_body(games)), 8)

    for g, h in zip(games, hashes):
        payload = b"".join(struct.pack("<ii", q, r) for q, r in g.legal_moves())
        assert h == _reference_fnv1a64(payload)


def test_states_flag_mismatch_is_inband_error():
    """Wire builder_flags disagreeing with the server's checkpoint-derived
    flags must raise StatesRequestError AFTER consuming the body."""
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games, builder_flags=FLAG_THREAT))
    with pytest.raises(StatesRequestError, match="builder_flags"):
        _read_forward_states_body(stream, 8)  # server flags: all off
    assert stream.read() == b""  # framing intact — server survives


def test_states_node_dim_mismatch_is_inband_error():
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games, node_dim=12))
    with pytest.raises(StatesRequestError, match="node_dim"):
        _read_forward_states_body(stream, 8)
    assert stream.read() == b""


def test_states_zero_graph_probe_bypasses_guards():
    """A zero-graph request is a capability probe: no rebuild, no guards —
    even nonsense flags/node_dim must be accepted."""
    stream = io.BytesIO(
        _encode_states_body([], builder_flags=0xFF, node_dim=99)
    )
    assert _read_forward_states_body(stream, 8) is None
    assert stream.read() == b""


def test_states_num_legal_mismatch_is_inband_error():
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games, num_legal_override=1))
    with pytest.raises(StatesRequestError, match="num_legal"):
        _read_forward_states_body(stream, 8)
    assert stream.read() == b""


@pytest.mark.parametrize("mr", [0, 3], ids=["terminal_mr0", "mr3"])
def test_states_invalid_moves_remaining_is_inband_error(mr):
    """moves_remaining outside {1,2} (0 = what a terminal state serialises
    to) must be an in-band error, not server death."""
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games, moves_remaining_override=mr))
    with pytest.raises(StatesRequestError, match="moves_remaining"):
        _read_forward_states_body(stream, 8)
    assert stream.read() == b""


@pytest.mark.parametrize("cur", [2, 255], ids=["cur2", "cur255"])
def test_states_invalid_current_player_is_inband_error(cur):
    """current_player byte outside {0, 1} (P1/P2) must be an in-band error, not
    server death — the body is fully consumed so framing survives."""
    games = _mk_games()
    stream = io.BytesIO(_encode_states_body(games, current_player_override=cur))
    with pytest.raises(StatesRequestError, match="current_player"):
        _read_forward_states_body(stream, 8)
    assert stream.read() == b""


def test_states_lying_num_graphs_header_is_framing_error():
    """A header that claims MORE graphs than the body carries is a framing lie:
    the parser reads the real graphs, then hits EOF on the phantom graph's
    header. That is an EOFError — unrecoverable (framing lost), which the reader
    thread turns into a clean server shutdown rather than a desync. (A too-SMALL
    count would silently leave trailing bytes; the too-LARGE case is the one the
    parser can detect, so we assert what the code actually does, framing-safely.)
    """
    games = _mk_games()  # 2 graphs in the body
    stream = io.BytesIO(_encode_states_body(games, num_graphs_override=3))
    with pytest.raises(EOFError):
        _read_forward_states_body(stream, 8)


def test_states_rebuild_valueerror_is_inband_error():
    """from_state rejections (duplicate stones) surface in-band."""
    games = _mk_games()
    dup = games[0].placed_stones()[-1]
    stream = io.BytesIO(_encode_states_body(games, extra_stones={0: [dup]}))
    with pytest.raises(StatesRequestError):
        _read_forward_states_body(stream, 8)
    assert stream.read() == b""


def test_write_forward_states_response_layout():
    """Exact OK-response byte layout: header | status | u32 total_legal |
    u32 num_graphs | f32 logits | i32 counts | f32 values | u64 hashes."""
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    legal_counts = torch.tensor([3, 1], dtype=torch.int64)
    values = torch.tensor([0.5, -0.5], dtype=torch.float32)
    hashes = [0x0123456789ABCDEF, 0xFEDCBA9876543210]
    stream = io.BytesIO()

    _write_forward_states_response(stream, logits, legal_counts, values, hashes)
    raw = stream.getvalue()

    magic, ver, msg_type = struct.unpack_from("<IBB", raw, 0)
    assert (magic, ver, msg_type) == (MAGIC, VERSION, MSG_FORWARD_STATES)
    assert raw[6] == STATES_STATUS_OK
    total_legal, num_graphs = struct.unpack_from("<II", raw, 7)
    assert (total_legal, num_graphs) == (4, 2)
    off = 15
    np.testing.assert_array_almost_equal(
        np.frombuffer(raw, np.float32, 4, off), logits.numpy()
    )
    off += 4 * 4
    np.testing.assert_array_equal(
        np.frombuffer(raw, np.int32, 2, off), [3, 1]
    )
    off += 2 * 4
    np.testing.assert_array_almost_equal(
        np.frombuffer(raw, np.float32, 2, off), values.numpy()
    )
    off += 2 * 4
    assert struct.unpack_from("<2Q", raw, off) == tuple(hashes)
    assert len(raw) == off + 2 * 8  # nothing trailing


def test_write_states_error_layout():
    stream = io.BytesIO()
    _write_states_error(stream, "graph 3: boom")
    raw = stream.getvalue()
    magic, ver, msg_type = struct.unpack_from("<IBB", raw, 0)
    assert (magic, ver, msg_type) == (MAGIC, VERSION, MSG_FORWARD_STATES)
    assert raw[6] == STATES_STATUS_ERROR
    (msg_len,) = struct.unpack_from("<I", raw, 7)
    assert raw[11 : 11 + msg_len].decode("utf-8") == "graph 3: boom"
    assert len(raw) == 11 + msg_len


def test_write_states_probe_ack_layout():
    stream = io.BytesIO()
    _write_states_probe_ack(stream)
    raw = stream.getvalue()
    magic, ver, msg_type = struct.unpack_from("<IBB", raw, 0)
    assert (magic, ver, msg_type) == (MAGIC, VERSION, MSG_FORWARD_STATES)
    assert raw[6] == STATES_STATUS_PROBE_ACK
    assert len(raw) == 7
