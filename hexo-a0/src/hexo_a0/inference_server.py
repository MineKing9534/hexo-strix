"""Inference server for HeXO AlphaZero self-play.

Spawned as a subprocess by the Rust self-play binary. Loads a model
checkpoint, torch.compile's it, and serves inference requests over
stdin/stdout using a binary protocol.

Wire protocol: all little-endian, 6-byte header (u32 magic + u8 version + u8 msg_type).
See task spec for full message definitions.

Protocol v2 added a ``node_dim: u8`` field to the forward-message body header
(after ``has_edge_attr``) so non-8-dim node features (e.g. 12-dim threat
features) survive the wire. The Rust self-play binary spawns this server from
the same checkout, so no rolling version compat is needed — version mismatch
is a hard error.

MSG_FORWARD_STATES (0x03) is the A3 state-payload wire: the client sends
~300 B/graph board-state records instead of collated graph tensors, and the
server rebuilds the identical batch via hexo_rs (GameState.from_state +
axis_states_to_batch). Requests answer with a MSG_FORWARD_STATES-typed
response carrying a status byte and per-graph legal-coord FNV-1a hashes (see
``_read_forward_states_body`` / ``_write_forward_states_response`` for the
exact byte layouts). Validation/rebuild failures reply in-band
(STATES_STATUS_ERROR) — they never kill the server. MSG_FORWARD stays
byte-identical.

With ``--slot-inference`` (plan-A3 task 5), MSG_FORWARD_STATES is served by
the A2 slot backend instead: the wire's canonical int32 HexKeys feed the
batched on-device slot builder directly and a ``slot_model_from_legacy``
conversion of the SAME checkpoint runs ``forward_padded`` (legacy GINE
checkpoints only — anything else fails at startup). The wire format and the
states response layout are unchanged; MSG_FORWARD always stays on the legacy
model. See the ``--slot-inference`` block near ``_load_slot_model``.
"""

import argparse
import os
import struct
import sys
import time as _time
from pathlib import Path

import torch
from torch import Tensor

MAGIC = 0x48583034
VERSION = 2
MSG_FORWARD = 0x01
MSG_RELOAD = 0x02
MSG_FORWARD_STATES = 0x03
MSG_SHUTDOWN = 0xFF

HEADER_SIZE = 6  # u32 + u8 + u8
HEADER_FMT = "<IBB"

# --- MSG_FORWARD_STATES (A3 state-payload wire) -----------------------------
# builder_flags bits (client cross-check against the checkpoint-derived flags)
BUILDER_FLAG_PRUNE = 0x01
BUILDER_FLAG_THREAT = 0x02
BUILDER_FLAG_RELATIVE = 0x04

# Status byte of every MSG_FORWARD_STATES-typed response.
STATES_STATUS_OK = 0
STATES_STATUS_ERROR = 1
STATES_STATUS_PROBE_ACK = 2

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_U64_MASK = 0xFFFFFFFFFFFFFFFF


class StatesRequestError(Exception):
    """In-band MSG_FORWARD_STATES failure.

    Raised only AFTER the request body has been fully consumed, so the
    stream framing is intact: the server replies with a STATES_STATUS_ERROR
    response and keeps serving — states-request errors never kill the server.
    """


def _fnv1a64(data) -> int:
    """FNV-1a 64-bit over a byte sequence (the legal-coord order guard)."""
    h = _FNV64_OFFSET
    for b in data:
        h = ((h ^ b) * _FNV64_PRIME) & _U64_MASK
    return h


def _fnv1a64_qr_hashes(flat_qr, counts) -> "list[int] | None":
    """Per-graph legal-coord FNV-1a hashes via the hexo_rs Rust binding, or
    ``None`` when the (older) extension lacks it so the caller falls back to the
    pure-Python ``_fnv1a64`` loop (the ~6 ms/batch hot spot the binding removes).

    ``flat_qr`` is the interleaved ``[q0, r0, q1, r1, ...]`` int32 coord stream;
    ``counts[i]`` is graph ``i``'s legal-coord count. The binding hashes the
    exact same byte stream (each coord = q i32 LE then r i32 LE) as ``_fnv1a64``
    — the golden-fixture/parity tests pin bit-identity.
    """
    import hexo_rs  # function-level import, per project convention

    fn = getattr(hexo_rs, "fnv1a64_qr_hashes", None)
    if fn is None:
        return None
    return fn(flat_qr, counts)


def _unpack_hexkey(key: int) -> "tuple[int, int]":
    """Decode a canonical HexKey (perf-plan §1) into ``(q, r)``.

    Layout: ``key = (q << 16) | ((r ^ 0x8000) & 0xFFFF)`` as a signed i32 —
    q stored sign-extended i16 in the high half (NOT biased), ONLY r biased,
    so signed-i32 key order == lexicographic (q, r) order.

    Decode: ``q = key >> 16`` (arithmetic shift; Python ints are signed, so
    ``>>`` on the struct-unpacked signed value is exactly that), and
    ``r`` = i16 reinterpretation of ``(key & 0xFFFF) ^ 0x8000``.
    """
    q = key >> 16
    r = (key & 0xFFFF) ^ 0x8000
    if r >= 0x8000:
        r -= 0x10000
    return q, r


def _install_parent_death_watchdog() -> None:
    """Make sure this server dies when the Rust self-play binary dies.

    Without this, abnormal parent termination (SIGKILL, panic, or a parent of
    the parent — like a test harness — getting cancelled) reparents us to PID
    1 and we'd happily keep occupying GPU memory forever.

    Two layers of defence:

    1. **Linux prctl(PR_SET_PDEATHSIG)**: kernel-level signal delivery the
       moment the *real* parent dies. Near-zero latency.
    2. **ppid poll**: portable fallback that catches the race where the parent
       was already dead before we called prctl, and the macOS/non-Linux case
       where prctl isn't available.

    Both paths call ``os._exit(0)`` (not ``sys.exit``) to skip Python
    finalisers — torch/HIP cleanup can hang once the parent's gone.
    """
    import signal
    import threading

    def _hard_exit(*_args):
        os._exit(0)

    # Install the hard-exit handler first so any kernel-delivered SIGTERM
    # (from prctl below, or from someone running `kill <pid>`) does the right
    # thing instead of being intercepted by Python's default raise-SystemExit.
    try:
        signal.signal(signal.SIGTERM, _hard_exit)
    except Exception:
        pass  # signals unavailable on this platform; ppid watchdog still helps.

    # Layer 1: kernel-delivered SIGTERM when our real parent dies (Linux only).
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except Exception:
            pass  # missing libc / not supported; fall through to layer 2.

    # If we were *already* orphaned before prctl had a chance to fire (parent
    # died between fork and exec), bail out immediately.
    if os.getppid() == 1:
        os._exit(0)

    # Layer 2: portable ppid poll. Two-second cadence is plenty — the cost of
    # an extra two seconds of GPU memory occupancy on parent death is small
    # compared to the cost of a forgotten orphan running indefinitely.
    def _ppid_watchdog():
        while True:
            _time.sleep(2.0)
            if os.getppid() == 1:
                os._exit(0)

    threading.Thread(target=_ppid_watchdog, daemon=True).start()


def _check_protocol_header(magic: int, ver: int) -> None:
    """Validate the 6-byte message header's magic + version fields.

    Raises ``ValueError`` with a clear message on mismatch. Version mismatch
    means the Rust binary and this server come from different checkouts —
    a deployment bug, not something to paper over.
    """
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
    if ver != VERSION:
        raise ValueError(
            f"unsupported protocol version {ver} (this server speaks v{VERSION}; "
            "the Rust self_play binary and hexo_a0 must come from the same checkout)"
        )


def _read_exact(stream, n: int) -> bytearray:
    """Read exactly n bytes from stream, raise EOFError on short read."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = stream.readinto(view[pos:])
        if not nbytes:
            raise EOFError(f"Expected {n} bytes, got {pos}")
        pos += nbytes
    return buf


def _write_forward_response(
    stream, logits: Tensor, legal_counts: Tensor, values: Tensor
) -> None:
    """Write a forward response to the binary stream."""
    total_legal = logits.shape[0]
    num_graphs = values.shape[0]

    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_FORWARD))
    buf.extend(struct.pack("<II", total_legal, num_graphs))
    buf.extend(logits.cpu().float().numpy().tobytes())
    buf.extend(legal_counts.cpu().int().numpy().tobytes())
    buf.extend(values.cpu().float().numpy().tobytes())
    stream.write(bytes(buf))
    stream.flush()


def _write_reload_ack(stream, success: bool) -> None:
    """Write a reload ACK to the binary stream."""
    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_RELOAD))
    buf.extend(struct.pack("<B", 1 if success else 0))
    stream.write(bytes(buf))
    stream.flush()


def _load_model(args) -> torch.nn.Module:
    """Build and load a ScriptableHeXONet from checkpoint."""
    from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # The lean/relational schema is authoritative from the checkpoint's embedded
    # model_config (training writes it into model_selfplay.pt), so self-play
    # needs no extra CLI plumbing — the server auto-configures from the ckpt.
    # CLI flags remain as a fallback for standalone/test invocations.
    emb = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if isinstance(emb, dict):
        # prune_empty_edges is not a model-construction knob; it is copied so
        # the MSG_FORWARD_STATES rebuild derives its builder flags from the
        # checkpoint (single source of truth — wire flags are a cross-check).
        for _k in ("axis_relational", "axis_window", "compact_stone_onehot",
                   "node_coords", "moves_scope", "relative_stone_encoding",
                   "threat_features", "value_bins", "value_bin_min",
                   "value_bin_max", "prune_empty_edges"):
            if _k in emb:
                setattr(args, _k, emb[_k])

    # In axis_relational mode the model's input_proj is the LEAN width: the
    # wire graph arrives legacy (args.node_dim, e.g. 11) and the shim reduces it
    # to lean before input_proj. Size the model to the lean width accordingly.
    node_features = args.node_dim
    if getattr(args, "axis_relational", False):
        from types import SimpleNamespace
        from hexo_a0.config import node_feature_dim
        node_features = node_feature_dim(SimpleNamespace(
            relative_stone_encoding=getattr(args, "relative_stone_encoding", False),
            threat_features=getattr(args, "threat_features", False),
            compact_stone_onehot=getattr(args, "compact_stone_onehot", False),
            node_coords=getattr(args, "node_coords", True),
            moves_scope=getattr(args, "moves_scope", "node"),
        ))

    model = ScriptableHeXONet(
        node_features=node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        policy_hidden=args.policy_hidden,
        value_hidden=args.value_hidden,
        graph_type=args.graph_type,
        pre_norm=not args.no_pre_norm,
        conv_type=args.conv_type,
        use_jk=getattr(args, "use_jk", False),
        jk_mode=getattr(args, "jk_mode", "sum"),
        axis_relational=getattr(args, "axis_relational", False),
        axis_window=getattr(args, "axis_window", 8),
        relative_stone_encoding=getattr(args, "relative_stone_encoding", False),
        threat_features=getattr(args, "threat_features", False),
        compact_stone_onehot=getattr(args, "compact_stone_onehot", False),
        node_coords=getattr(args, "node_coords", True),
        moves_scope=getattr(args, "moves_scope", "node"),
        value_bins=getattr(args, "value_bins", 0),
        value_bin_min=getattr(args, "value_bin_min", -1.0),
        value_bin_max=getattr(args, "value_bin_max", 1.0),
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Strip _orig_mod. prefixes that torch.compile adds
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v

    # Try direct load first; fall back to HeXONet key mapping
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        load_from_hexonet(model, cleaned)

    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    # Use bfloat16 on CUDA
    if device.type == "cuda":
        model = model.to(torch.bfloat16)

    if not args.no_compile:
        compile_kwargs = {}
        if getattr(args, "dynamic_compile", False) or os.environ.get("HEXO_DYNAMIC_COMPILE") == "1":
            compile_kwargs["dynamic"] = True
        # The axis-relational encoder now unifies all edges into fixed-shape
        # scatter buckets (no per-axis boolean masking / nonzero split), so it
        # compiles under fullgraph=True like the legacy path — no graph breaks,
        # no ~3x self-play launch overhead. (The heads' index_select path is
        # taken because _prepare_tensors passes legal_idx/stone_idx.)
        model = torch.compile(model, fullgraph=True, **compile_kwargs)

    return model


def _warmup(
    model: torch.nn.Module, device: torch.device, graph_type: str, node_dim: int = 8
) -> None:
    """Run a warmup forward pass to trigger JIT compilation."""
    n, e = 4, 6
    x = torch.randn(n, node_dim, device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device
    )
    legal_mask = torch.ones(n, dtype=torch.bool, device=device)
    stone_mask = torch.ones(n, dtype=torch.bool, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    if device.type == "cuda":
        x = x.to(torch.bfloat16)

    edge_attr = torch.zeros(0, device=device)
    if graph_type == "axis":
        edge_attr = torch.randn(e, 5, device=device)
        if device.type == "cuda":
            edge_attr = edge_attr.to(torch.bfloat16)

    with torch.no_grad():
        model(x, edge_index, legal_mask, stone_mask, batch, 1, edge_attr)


def _read_forward_body(stdin, expected_node_dim: int) -> dict:
    """Read forward request body into memory (header already consumed).

    ``expected_node_dim`` is the model's node-feature width (``--node-dim``).
    A wire/model mismatch fails here with a clear error instead of surfacing
    later as an obscure GNN shape error.
    """
    body_header = _read_exact(stdin, 14)
    total_nodes, total_edges, num_graphs = struct.unpack_from("<III", body_header, 0)
    has_edge_attr = body_header[12]
    node_dim = body_header[13]
    if node_dim != expected_node_dim:
        raise ValueError(
            f"wire node_dim {node_dim} != server --node-dim {expected_node_dim}"
        )

    features_bytes = _read_exact(stdin, total_nodes * node_dim * 4)
    edge_src_bytes = _read_exact(stdin, total_edges * 8)
    edge_dst_bytes = _read_exact(stdin, total_edges * 8)
    edge_attr_bytes = _read_exact(stdin, total_edges * 5 * 4) if has_edge_attr else None
    legal_mask_bytes = _read_exact(stdin, total_nodes)
    stone_mask_bytes = _read_exact(stdin, total_nodes)
    batch_bytes = _read_exact(stdin, total_nodes * 4)

    return {
        "total_nodes": total_nodes, "total_edges": total_edges,
        "num_graphs": num_graphs, "has_edge_attr": has_edge_attr,
        "node_dim": node_dim,
        "features": features_bytes, "edge_src": edge_src_bytes,
        "edge_dst": edge_dst_bytes, "edge_attr": edge_attr_bytes,
        "legal_mask": legal_mask_bytes, "stone_mask": stone_mask_bytes,
        "batch": batch_bytes,
    }


def _states_request_frame(stdin) -> tuple:
    """Read a raw MSG_FORWARD_STATES request BODY (framing only, no
    validation — see ``_read_forward_states_body`` for the byte layout).

    Returns ``(num_graphs, win_length, placement_radius, max_moves,
    builder_flags, node_dim, graphs)`` with ``graphs`` a list of
    ``(p1_keys, p2_keys, current_player, moves_remaining, num_legal)``
    tuples (keys are sign-decoded int32 HexKeys). Raises ``EOFError`` on a
    short read — framing lost, fatal like graph mode. After a successful
    return the body is fully consumed, so every later failure is in-band.
    """
    body_header = _read_exact(stdin, 12)
    (num_graphs, win_length, placement_radius, max_moves,
     builder_flags, node_dim) = struct.unpack("<IBBIBB", body_header)

    graphs = []
    for _ in range(num_graphs):
        gh = _read_exact(stdin, 8)
        n_p1, n_p2, cur, mr, num_legal = struct.unpack("<HHBBH", gh)
        key_bytes = _read_exact(stdin, (n_p1 + n_p2) * 4)
        keys = struct.unpack(f"<{n_p1 + n_p2}i", key_bytes)
        graphs.append((keys[:n_p1], keys[n_p1:], cur, mr, num_legal))

    return (num_graphs, win_length, placement_radius, max_moves,
            builder_flags, node_dim, graphs)


def _read_forward_states_body(
    stdin,
    expected_node_dim: int,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
) -> "tuple[dict, list[int]] | None":
    """Read a MSG_FORWARD_STATES request body and rebuild the graph batch.

    Request BODY layout (all little-endian; the 6-byte outer header is
    consumed by the reader thread before this runs)::

        u32 num_graphs
        u8  win_length
        u8  placement_radius
        u32 max_moves
        u8  builder_flags   (bit0 prune_empty_edges, bit1 threat_features,
                             bit2 relative_stones — cross-check only; the
                             server's checkpoint config is authoritative)
        u8  node_dim        (schema guard: must equal the server's node dim)
        per graph:
            u16 n_p1             (P1 stone count)
            u16 n_p2             (P2 stone count)
            u8  current_player   (0 = P1, 1 = P2)
            u8  moves_remaining  (must be 1 or 2)
            u16 num_legal        (cross-checked against the rebuilt graph)
            i32 keys[n_p1]       (P1 stones, canonical HexKeys)
            i32 keys[n_p2]       (P2 stones, canonical HexKeys)

    Stone coords ride as canonical int32 HexKeys (perf-plan §1):
    ``key = (q << 16) | ((r ^ 0x8000) & 0xFFFF)`` — q sign-extended i16 in
    the high half, ONLY r biased (signed-i32 key order == lexicographic
    (q, r) order); decoded by ``_unpack_hexkey``.

    A zero-graph request is a startup capability probe: it bypasses the
    rebuild and ALL guards (flags/node_dim may be garbage) and returns
    ``None`` — the caller replies with a dedicated STATES_STATUS_PROBE_ACK.

    Otherwise returns ``(body, legal_hashes)`` where ``body`` is EXACTLY the
    dict shape ``_read_forward_body`` produces (so the unmodified
    ``_prepare_tensors`` consumes it) and ``legal_hashes`` is one u64
    FNV-1a hash per graph over the rebuilt graph's legal coords in order —
    hash input is the concatenation of ``(q: i32 LE, r: i32 LE)`` per legal
    move, in ``legal_moves()`` order (== the graph's legal-node order).

    Raises ``EOFError`` on a short read (framing lost — fatal, like graph
    mode) and ``StatesRequestError`` for every validation/rebuild failure
    (body fully consumed — the caller replies in-band and keeps serving).
    """
    (num_graphs, win_length, placement_radius, max_moves,
     builder_flags, node_dim, graphs) = _states_request_frame(stdin)

    # From here on the body is fully consumed — every failure is in-band.
    if num_graphs == 0:
        return None  # capability probe: bypass rebuild + guards

    try:
        return _states_to_forward_body(
            graphs, win_length, placement_radius, max_moves,
            builder_flags, node_dim, expected_node_dim,
            prune_empty_edges=prune_empty_edges,
            threat_features=threat_features,
            relative_stones=relative_stones,
        )
    except StatesRequestError:
        raise
    except Exception as e:
        # Rebuild machinery failure (from_state, batch builder, ...):
        # framing is intact, so keep the server alive and report in-band.
        raise StatesRequestError(f"states rebuild failed: {e}") from e


def _states_to_forward_body(
    graphs: list,
    win_length: int,
    placement_radius: int,
    max_moves: int,
    builder_flags: int,
    node_dim: int,
    expected_node_dim: int,
    *,
    prune_empty_edges: bool,
    threat_features: bool,
    relative_stones: bool,
) -> "tuple[dict, list[int]]":
    """Validate a parsed states request and rebuild the collated axis batch.

    The server's checkpoint-derived builder flags are authoritative; the
    wire ``builder_flags`` are a cross-check. Emits the exact
    ``_read_forward_body`` dict shape: the binding's i64 batch vector is
    converted to the int32 wire dtype, stone_mask is scattered from
    stone_idx, total_nodes/total_edges are derived from the rebuilt batch,
    and has_edge_attr/node_dim are synthesized (axis graphs always carry
    5-dim edge attrs; node_dim = rebuilt n_feat).
    """
    import hexo_rs
    from hexo_a0.graph import axis_states_to_batch

    server_flags = (
        (BUILDER_FLAG_PRUNE if prune_empty_edges else 0)
        | (BUILDER_FLAG_THREAT if threat_features else 0)
        | (BUILDER_FLAG_RELATIVE if relative_stones else 0)
    )
    if builder_flags != server_flags:
        raise StatesRequestError(
            f"wire builder_flags 0x{builder_flags:02X} != server checkpoint "
            f"flags 0x{server_flags:02X} (bit0 prune_empty_edges, "
            f"bit1 threat_features, bit2 relative_stones)"
        )
    if node_dim != expected_node_dim:
        raise StatesRequestError(
            f"wire node_dim {node_dim} != server node_dim {expected_node_dim}"
        )

    config = hexo_rs.GameConfig(win_length, placement_radius, max_moves)
    games = []
    for i, (p1_keys, p2_keys, cur, mr, _num_legal) in enumerate(graphs):
        if mr not in (1, 2):
            raise StatesRequestError(
                f"graph {i}: moves_remaining {mr} not in {{1, 2}} "
                f"(0 means the client sent a terminal state)"
            )
        if cur not in (0, 1):
            raise StatesRequestError(f"graph {i}: current_player byte {cur} not in {{0, 1}}")
        stone_list = [(_unpack_hexkey(k), "P1") for k in p1_keys]
        stone_list += [(_unpack_hexkey(k), "P2") for k in p2_keys]
        try:
            games.append(hexo_rs.GameState.from_state(
                stone_list, "P1" if cur == 0 else "P2", mr, config
            ))
        except ValueError as e:
            raise StatesRequestError(f"graph {i}: from_state rejected state: {e}") from e

    batch, aux = axis_states_to_batch(
        games,
        prune_empty_edges=prune_empty_edges,
        threat_features=threat_features,
        relative_stones=relative_stones,
        device="cpu",
    )

    legal_counts = aux.legal_counts.tolist()
    for i, (_p1, _p2, _cur, _mr, num_legal) in enumerate(graphs):
        if legal_counts[i] != num_legal:
            raise StatesRequestError(
                f"graph {i}: wire num_legal {num_legal} != rebuilt legal "
                f"count {legal_counts[i]} (client/server builder divergence)"
            )

    n_feat = batch.x.shape[1]
    if n_feat != expected_node_dim:
        raise StatesRequestError(
            f"rebuilt n_feat {n_feat} != model input width {expected_node_dim}"
        )

    total_nodes = batch.x.shape[0]
    total_edges = batch.edge_index.shape[1]

    stone_mask = torch.zeros(total_nodes, dtype=torch.uint8)
    stone_mask[aux.stone_idx] = 1

    # bytearray (writable) so torch.frombuffer in _prepare_tensors behaves
    # exactly as with the graph-mode _read_exact buffers.
    body = {
        "total_nodes": total_nodes, "total_edges": total_edges,
        "num_graphs": batch.num_graphs, "has_edge_attr": 1,
        "node_dim": n_feat,
        "features": bytearray(batch.x.numpy().tobytes()),
        "edge_src": bytearray(batch.edge_index[0].contiguous().numpy().tobytes()),
        "edge_dst": bytearray(batch.edge_index[1].contiguous().numpy().tobytes()),
        "edge_attr": bytearray(batch.edge_attr.numpy().tobytes()),
        "legal_mask": bytearray(batch.legal_mask.to(torch.uint8).numpy().tobytes()),
        "stone_mask": bytearray(stone_mask.numpy().tobytes()),
        # dtype trap: the binding's batch vector is i64; the wire dict is i32.
        "batch": bytearray(batch.batch.to(torch.int32).numpy().tobytes()),
    }

    # Per-graph order guard: hash the rebuilt legal coords in graph order
    # (== legal_moves() order); the client compares against its snapshot. The
    # Rust binding does the byte-identical hashing (removes the ~6 ms/batch
    # pure-Python loop); fall back to it only when the extension lacks the fn.
    legal_coords = aux.coords[aux.legal_idx].numpy().astype("<i4", copy=False)
    hashes = _fnv1a64_qr_hashes(legal_coords.reshape(-1).tolist(), legal_counts)
    if hashes is None:
        hashes = []
        offset = 0
        for count in legal_counts:
            hashes.append(_fnv1a64(legal_coords[offset:offset + count].tobytes()))
            offset += count

    return body, hashes


# --- --slot-inference: A2 slot backend for MSG_FORWARD_STATES ---------------
#
# With --slot-inference, states requests bypass the legacy graph rebuild
# entirely: the wire's canonical int32 HexKeys are fed DIRECTLY to the batched
# on-device slot builder (slot_graph.build_slot_batch_from_keys — no per-stone
# unpack/repack), the converted SlotHeXONet runs forward_padded, and the [B, N]
# padded logits are gathered back to the per-graph legal order the states
# response promises. MSG_FORWARD (graph mode) always stays on the legacy model.


def _args_prune_default(args) -> bool:
    """Single source of truth for the ``prune_empty_edges`` default across the
    slot path (``_load_slot_model`` / ``_slot_builder_config`` /
    ``_validate_states_slot``).

    Default is ``False`` — it matches the graph builder's own default (an
    absent flag means "don't prune"). In production the checkpoint's embedded
    model_config sets ``prune_empty_edges`` explicitly at load time, so this
    default only bites standalone/test invocations; keeping ONE helper stops
    the three call sites from silently disagreeing (they previously split
    True/False, which could flip the builder-flag cross-check under a
    schema-less checkpoint)."""
    return bool(getattr(args, "prune_empty_edges", False))


def _slot_builder_config(args, win_length: int, placement_radius: int):
    """SlotBuilderConfig from the server's checkpoint-derived schema flags
    (single source of truth — wire flags are only a cross-check) plus the
    request's game geometry."""
    from hexo_a0.slot_graph import SlotBuilderConfig

    return SlotBuilderConfig(
        win_length=win_length,
        placement_radius=placement_radius,
        prune_empty_edges=_args_prune_default(args),
        threat_features=bool(getattr(args, "threat_features", False)),
        relative_stones=bool(getattr(args, "relative_stone_encoding", False)),
        node_coords=bool(getattr(args, "node_coords", True)),
        moves_scope=str(getattr(args, "moves_scope", "node")),
        compact_stone_onehot=bool(getattr(args, "compact_stone_onehot", False)),
    )


def _states_slot_node_counts(graphs: list, placement_radius: int) -> "list[int]":
    """EXACT per-graph slot node count (stones + legal), computed cheaply on
    CPU ints — no torch allocation.

    The slot builder's legal region is the union of hex-disks(placement_radius)
    around the stones, MINUS the stones (see ``slot_graph.build_slot_batch``);
    node count = stones + |union \\ stones|. Since the disk includes its centre,
    every stone is itself in the union, so this is exactly ``|union of disks|``.
    Deduplicating the union (a Python set of packed keys) captures the disk
    OVERLAP between nearby stones — the reason the old zero-overlap bound
    (``stones × disk_cells``) over-reported node counts 50-100x and made the
    default budget reject ordinary production batches.

    Keys pack as ``q * STRIDE + r`` with ``STRIDE`` larger than any in-range
    ``|r|`` (i16 coords + radius ≤ 32767 + 64 < 2^19), so the disk deltas add
    directly onto each stone's base key — the same additive
    ``(dq << ?, dr)`` arithmetic as ``slot_graph._disk_deltas``, but on Python
    ints. O(stones × disk_cells) small-int ops per graph.
    """
    r = placement_radius
    STRIDE = 1 << 20  # > any in-range |r| (see docstring)
    # Disk deltas: all (dq, dr) with hex-distance ≤ r (matches hex_offsets).
    disk = [
        dq * STRIDE + dr
        for dq in range(-r, r + 1)
        for dr in range(-r, r + 1)
        if max(abs(dq), abs(dr), abs(dq + dr)) <= r
    ]
    counts = []
    for p1, p2, _cur, _mr, _num_legal in graphs:
        union = set()
        for k in list(p1) + list(p2):
            q, rr = _unpack_hexkey(k)
            base = q * STRIDE + rr
            for d in disk:
                union.add(base + d)
        counts.append(len(union))
    return counts


def _validate_states_slot(frame: tuple, args, *, bytes_per_elem: int = 4) -> tuple:
    """Validate a parsed states request for the slot backend (reader thread).

    Mirrors the legacy path's guards (builder-flag cross-check, node_dim,
    per-graph moves_remaining/current_player) plus two slot-specific ones:

    - win_length must match ``--win-length`` (the slot model's edge tables are
      built for a fixed slot count ``6 * (win_length - 1)``);
    - the A2 memory sharp edge: the message-passing activations are DENSE
      ``[B, N_max, S, H]`` tensors (bf16 on CUDA, f32 on CPU — the loaded slot
      model's parameter dtype, passed as ``bytes_per_elem``), so a big batch of
      big graphs can OOM instead of just running slow. The estimate below is
      computed from the request's geometry (an EXACT node count, before any
      tensor is allocated) and over-budget requests get an in-band ERROR,
      never an OOM.

    ``bytes_per_elem`` is the slot model's activation element size (2 for
    bf16 on CUDA, 4 for f32 on CPU); the caller reads it from the live model.

    Returns ``(win_length, placement_radius, graphs)`` for the main loop.
    """
    (num_graphs, win_length, placement_radius, _max_moves,
     builder_flags, node_dim, graphs) = frame

    server_flags = (
        (BUILDER_FLAG_PRUNE if _args_prune_default(args) else 0)
        | (BUILDER_FLAG_THREAT if getattr(args, "threat_features", False) else 0)
        | (BUILDER_FLAG_RELATIVE if getattr(args, "relative_stone_encoding", False) else 0)
    )
    if builder_flags != server_flags:
        raise StatesRequestError(
            f"wire builder_flags 0x{builder_flags:02X} != server checkpoint "
            f"flags 0x{server_flags:02X} (bit0 prune_empty_edges, "
            f"bit1 threat_features, bit2 relative_stones)"
        )
    if node_dim != args.node_dim:
        raise StatesRequestError(
            f"wire node_dim {node_dim} != server node_dim {args.node_dim}"
        )
    if win_length != args.win_length:
        raise StatesRequestError(
            f"wire win_length {win_length} != server --win-length "
            f"{args.win_length} (the slot model's edge tables are built for a "
            f"fixed win_length)"
        )
    # Range-check placement_radius BEFORE it drives the disk-cell bound below
    # (and the slot builder's disk meshgrid alloc). Matches hexo_rs.GameConfig's
    # own 1..=64 bound — an out-of-band radius is an in-band ERROR, never an OOM.
    if not (1 <= placement_radius <= 64):
        raise StatesRequestError(
            f"placement_radius {placement_radius} out of range 1..=64 "
            f"(matches hexo_rs.GameConfig)"
        )

    for i, (p1, p2, cur, mr, _num_legal) in enumerate(graphs):
        if mr not in (1, 2):
            raise StatesRequestError(
                f"graph {i}: moves_remaining {mr} not in {{1, 2}} "
                f"(0 means the client sent a terminal state)"
            )
        if cur not in (0, 1):
            raise StatesRequestError(f"graph {i}: current_player byte {cur} not in {{0, 1}}")
        if len(p1) + len(p2) == 0:
            raise StatesRequestError(f"graph {i}: no stones")

    # A2 activation-memory guard. The estimate must NOT trust the wire's
    # num_legal (unverified — an under-report would slip past the guard and
    # then OOM in the on-device build). Instead compute the EXACT per-graph node
    # count server-side from the geometry: the slot builder's legal region is
    # the union of hex-disks(radius) around the stones minus the stones, so the
    # node count is |union of stone disks| — computed cheaply via a CPU set of
    # packed keys (_states_slot_node_counts, O(stones·disk_cells) int ops, no
    # torch alloc). This captures inter-stone disk OVERLAP, which the old
    # zero-overlap bound (stones·disk_cells) ignored — over-reporting 50-100x
    # and rejecting ordinary production batches under the default budget.
    #
    # PY2-W1: estimate on the PADDED shape the forward actually allocates.
    # _handle_states_slot pads B and N before forward_padded (B to a power of
    # two, N to a multiple of 128), and that padding can enlarge the dense
    # [B, N, S, H] tensor; estimating on the raw counts under-reports and could
    # still OOM. Round both dims through the SAME shared helpers the allocator
    # uses (_slot_bucket_b / _slot_bucket_n) so guard and allocator cannot
    # diverge.
    #
    # bytes_per_elem is the slot model's activation dtype size (bf16=2 on CUDA,
    # f32=4 on CPU). The ×3 covers the handful of dense [B, N, S, H]-shaped
    # intermediates forward_padded holds live at once (message-passing input +
    # output + reductions); it also dominates the smaller int64 candidate/
    # partner index tensors ([B, N, S] × 8 B/elem), which are a factor ~H below
    # a single H-wide activation and so are amply absorbed by the ×3 margin.
    num_slots = 6 * (args.win_length - 1)
    n_max_est = max(_states_slot_node_counts(graphs, placement_radius))
    b_bucket = _slot_bucket_b(num_graphs)
    n_bucket = _slot_bucket_n(n_max_est)
    est_bytes = b_bucket * n_bucket * num_slots * args.hidden_dim * bytes_per_elem * 3
    budget_bytes = args.slot_activation_budget_mb * 1024 * 1024
    if est_bytes > budget_bytes:
        raise StatesRequestError(
            f"slot activation budget exceeded: estimated "
            f"{est_bytes / 2**20:.0f} MiB (padded {b_bucket} graphs x {n_bucket} "
            f"max nodes x {num_slots} slots x {args.hidden_dim} hidden x "
            f"{bytes_per_elem} B x 3; from {num_graphs} graphs x {n_max_est} "
            f"exact nodes) > --slot-activation-budget-mb "
            f"{args.slot_activation_budget_mb:g} — send smaller batches"
        )

    return (win_length, placement_radius, graphs)


def _load_slot_model(args, device: torch.device) -> torch.nn.Module:
    """Build the A2 slot model from the loaded LEGACY GINE checkpoint.

    Raises ``ValueError`` with a clear message for every unsupported
    architecture (axis_relational, gatv2, lstm-JK, moves_scope != 'node',
    non-axis graphs) or a checkpoint whose state dict is not HeXONet-format —
    the caller turns that into a STARTUP exit, never a mid-request failure.

    Must run AFTER ``_load_model`` so the checkpoint's embedded model_config
    has already been merged into ``args`` (same single source of truth as the
    legacy states rebuild).
    """
    from hexo_a0.config import ModelConfig, node_feature_dim
    from hexo_a0.model import HeXONet
    from hexo_a0.model_slots import slot_model_from_legacy

    # Explicit A2 coverage-boundary checks first, for clear startup messages.
    if getattr(args, "axis_relational", False):
        raise ValueError(
            "--slot-inference covers legacy GINE checkpoints only (the A2 "
            "boundary); this checkpoint is axis_relational"
        )
    if args.conv_type != "gine":
        raise ValueError(
            "--slot-inference covers conv_type='gine' checkpoints only, "
            f"got {args.conv_type!r}"
        )
    if args.graph_type != "axis":
        raise ValueError(
            f"--slot-inference requires graph_type='axis', got {args.graph_type!r}"
        )
    if str(getattr(args, "moves_scope", "node")) != "node":
        raise ValueError(
            "--slot-inference supports moves_scope='node' only, got "
            f"{getattr(args, 'moves_scope', 'node')!r}"
        )
    if getattr(args, "use_jk", False) and getattr(args, "jk_mode", "sum") == "lstm":
        raise ValueError("--slot-inference does not support jk_mode='lstm'")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    emb = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    emb = emb if isinstance(emb, dict) else {}

    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        conv_type=args.conv_type,
        pre_norm=not args.no_pre_norm,
        dropout=0.0,
        # Not merged into args by _load_model (the scriptable path ignores it),
        # so read it straight from the embedded config.
        use_layer_scale=bool(emb.get("use_layer_scale", False)),
        use_jk=bool(getattr(args, "use_jk", False)),
        jk_mode=str(getattr(args, "jk_mode", "sum")),
        policy_hidden=args.policy_hidden,
        value_hidden=args.value_hidden,
        value_bins=int(getattr(args, "value_bins", 0) or 0),
        value_bin_min=float(getattr(args, "value_bin_min", -1.0)),
        value_bin_max=float(getattr(args, "value_bin_max", 1.0)),
        graph_type=args.graph_type,
        prune_empty_edges=_args_prune_default(args),
        threat_features=bool(getattr(args, "threat_features", False)),
        relative_stone_encoding=bool(getattr(args, "relative_stone_encoding", False)),
        compact_stone_onehot=bool(getattr(args, "compact_stone_onehot", False)),
        node_coords=bool(getattr(args, "node_coords", True)),
        moves_scope=str(getattr(args, "moves_scope", "node")),
    )
    if node_feature_dim(config) != args.node_dim:
        raise ValueError(
            f"checkpoint schema implies node width {node_feature_dim(config)} "
            f"but --node-dim is {args.node_dim}"
        )

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        # Train-only extras (never used at inference; absent from HeXONet
        # built with default q_head=False / value_horizons=[]).
        if new_key.startswith(("q_head.", "horizon_value_heads.")):
            continue
        cleaned[new_key] = v

    legacy = HeXONet(config)
    try:
        legacy.load_state_dict(cleaned, strict=True)
    except RuntimeError as e:
        raise ValueError(
            "--slot-inference needs a legacy HeXONet-format GINE checkpoint; "
            f"the state dict did not load into HeXONet({config.conv_type}/"
            f"{config.graph_type}): {e}"
        ) from e
    legacy.eval()

    slot = slot_model_from_legacy(legacy, config, args.win_length)
    slot.eval()
    slot = slot.to(device)
    if device.type == "cuda":
        slot = slot.to(torch.bfloat16)

    # Compiled batched build core (Task 2): the on-device slot build runs
    # through this when compile is enabled, and eager (None) otherwise. It rides
    # on the model instance so reload rebinds it and the handler/warmup read it
    # off the live slot model — no module-level global state.
    slot._slot_build_core = None
    if not args.no_compile:
        compile_kwargs = {}
        if getattr(args, "dynamic_compile", False) or os.environ.get("HEXO_DYNAMIC_COMPILE") == "1":
            compile_kwargs["dynamic"] = True
        # mult128-on-N bucketing yields more distinct static [B, N] shapes than
        # pow2 did, so raise the dynamo recompile cap (default 8) to avoid
        # falling back to eager once the shape set grows (the bench needed this).
        torch._dynamo.config.recompile_limit = 256
        # Same compile policy as the legacy model (eager slot LOSES — the A2
        # bench showed compile is the point); forward_padded is the static-
        # shape entry point the server calls.
        slot.forward_padded = torch.compile(
            slot.forward_padded, fullgraph=True, **compile_kwargs
        )
        # Compile the batched slot BUILD core too. torch.cumprod is gone (Task 1)
        # and mode="default" + dynamic=False at mult-128 N buckets gives ~1.34ms
        # at b=8 vs ~6.3ms eager on ROCm, bit-identical (scripts/
        # bench_build_compile_e2e.py). Graph breaks at unique/nonzero/.item are
        # allowed (NOT fullgraph); the mult-128 N bucketing keeps distinct build
        # shapes under the raised recompile cap.
        from hexo_a0.slot_graph import _build_slot_batch_core
        slot._slot_build_core = torch.compile(
            _build_slot_batch_core, mode="default", dynamic=False
        )
    return slot


def _warmup_slot(slot_model: torch.nn.Module, args, device: torch.device) -> None:
    """One tiny states->slot forward so schema/shape errors (and the first
    torch.compile trace) surface before READY, not on the first request."""
    from hexo_a0.slot_graph import build_slot_batch_from_keys, pack

    origin_key = int(pack(torch.tensor([0]), torch.tensor([0]))[0])
    config = _slot_builder_config(args, args.win_length, placement_radius=1)
    # Build through the (optionally compiled) build core at its mult-128 N
    # bucket, exactly as _handle_states_slot does, so BOTH the build-core compile
    # AND forward_padded are traced before READY (not on the first request).
    core = getattr(slot_model, "_slot_build_core", None)
    bucket_n = _slot_pad_n([([origin_key], [], 0, 2, 0)], placement_radius=1)
    batch = build_slot_batch_from_keys(
        [([origin_key], [], 0, 2)], config, device=device,
        pad_to=bucket_n, core_fn=core,
    )
    # Trace the SAME padded (B, N) bucket class production hits (mirror
    # _handle_states_slot exactly, via the shared _slot_bucket_shape helper).
    # Without this the compile trace is on the unpadded shape and the first real
    # request pays a recompile.
    batch = _pad_slot_batch(batch, *_slot_bucket_shape(batch))
    if device.type == "cuda":
        batch.x = batch.x.to(torch.bfloat16)
        batch.dummy_x = batch.dummy_x.to(torch.bfloat16)
    with torch.no_grad():
        slot_model.forward_padded(batch)


def _slot_legal_hashes(aux) -> "list[int]":
    """Per-graph FNV-1a legal-coord hashes from the slot batch's OWN legal
    ordering (``SlotBatchAux``) — same hash-input layout as the legacy states
    path: ``(q: i32 LE, r: i32 LE)`` per legal move, in legal-column order."""
    from hexo_a0.slot_graph import unpack

    # Unpack keys -> (q, r) vectorized (on-device if CUDA) BEFORE the .cpu()
    # sync, then flatten to the interleaved i32 stream the hasher consumes.
    lq, lr = unpack(aux.legal_keys)
    coords = torch.stack([lq, lr], dim=1).to(torch.int32).cpu().numpy().astype("<i4", copy=False)
    counts = aux.legal_counts.tolist()

    hashes = _fnv1a64_qr_hashes(coords.reshape(-1).tolist(), counts)
    if hashes is not None:
        return hashes
    hashes = []
    offset = 0
    for count in counts:
        hashes.append(_fnv1a64(coords[offset:offset + count].tobytes()))
        offset += count
    return hashes


def _round_pow2(n: int, floor: int = 1) -> int:
    """Smallest power of two >= max(n, floor). The slot forward runs under
    torch.compile(dynamic=False), so every distinct [B, N] shape triggers a
    fresh trace; rounding to a coarse grid collapses nearby request sizes onto
    a handful of shapes (few recompiles) at a bounded pad cost."""
    n = max(int(n), floor, 1)
    return 1 << (n - 1).bit_length()


# --- Slot [B, N] bucketing (single source of truth for allocator + guard) ----
# B and N use DIFFERENT grids. B keeps powers of two (few distinct batch sizes).
# N (max node count) rounds up to multiples of 128: pow2 pathologically rounds
# the production N≈2053 up to 4096, whereas mult-128 lands on 2176. The
# slot-padding ablation bench (scripts/bench_slot_padding_ablation.py) measured
# mult128-on-N fastest — it beats pow2 and beats mult256 by ~5%. Both the
# allocator (_slot_bucket_shape) and the activation-budget guard
# (_validate_states_slot) MUST route through these two helpers so their padded
# shapes cannot diverge (review finding PY2-W1).


def _slot_bucket_b(b: int) -> int:
    """Bucket the batch (B) dim: smallest power of two >= B."""
    return _round_pow2(b)


def _slot_bucket_n(n: int) -> int:
    """Bucket the max-node (N) dim: smallest multiple of 128 >= max(n, 128)."""
    n = max(int(n), 128)
    return -(-n // 128) * 128


def _slot_pad_n(graphs: list, placement_radius: int) -> int:
    """The mult-128 N bucket to build a states-slot request at, from the EXACT
    per-graph node counts (== the builder's ``n_max``; see
    ``_states_slot_node_counts``). Passed as ``pad_to`` so the compiled build
    core runs at a static bucketed N — otherwise every distinct raw ``n_max``
    would trace a fresh graph. Shared by ``_handle_states_slot`` and
    ``_warmup_slot`` so both hit the same bucket class."""
    return _slot_bucket_n(max(_states_slot_node_counts(graphs, placement_radius)))


def _pad_slot_batch(batch, b_to: int, n_to: int):
    """Pad a :class:`SlotBatch`'s graph (B) and node (N) dims up to bucket sizes
    so the compiled ``forward_padded`` sees a small fixed set of shapes.

    Padding graphs/nodes carry ``node_mask``/``filled``/masks all False, so they
    contribute no legal columns: ``logits[legal_mask]`` excludes them and the
    real-graph legal ordering (hence the FNV order guard and legal_counts, taken
    from the UNPADDED ``SlotBatchAux``) is unchanged. The caller slices the
    ghost graphs off ``values``."""
    from hexo_a0.slot_graph import SlotBatch

    b, n = batch.node_mask.shape
    if b_to == b and n_to == n:
        return batch
    if b_to < b or n_to < n:
        raise RuntimeError(
            f"_pad_slot_batch: target ({b_to}, {n_to}) smaller than batch ({b}, {n})"
        )
    s = batch.partner.shape[2]
    f = batch.x.shape[2]
    dev = batch.x.device

    def _mk(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=dev)

    x = _mk((b_to, n_to, f), batch.x.dtype); x[:b, :n] = batch.x
    dummy_x = _mk((b_to, f), batch.dummy_x.dtype); dummy_x[:b] = batch.dummy_x
    partner = _mk((b_to, n_to, s), batch.partner.dtype); partner[:b, :n] = batch.partner
    filled = _mk((b_to, n_to, s), batch.filled.dtype); filled[:b, :n] = batch.filled
    src_player = _mk((b_to, n_to), batch.src_player.dtype); src_player[:b, :n] = batch.src_player
    node_mask = _mk((b_to, n_to), batch.node_mask.dtype); node_mask[:b, :n] = batch.node_mask
    stone_mask = _mk((b_to, n_to), batch.stone_mask.dtype); stone_mask[:b, :n] = batch.stone_mask
    legal_mask = _mk((b_to, n_to), batch.legal_mask.dtype); legal_mask[:b, :n] = batch.legal_mask
    return SlotBatch(
        x=x, dummy_x=dummy_x, partner=partner, filled=filled,
        src_player=src_player, node_mask=node_mask,
        stone_mask=stone_mask, legal_mask=legal_mask,
    )


def _slot_bucket_shape(batch) -> "tuple[int, int]":
    """The pow2 ``(B, N)`` bucket a slot batch is padded to before
    ``forward_padded``. Single source of truth for BOTH the request handler
    (``_handle_states_slot``) and warmup (``_warmup_slot``) so the compile trace
    warmup produces matches the shape class production requests hit — otherwise
    warmup traces the UNPADDED shape and the first real request recompiles."""
    return _slot_bucket_b(batch.num_graphs), _slot_bucket_n(batch.node_mask.shape[1])


def _write_forward_states_response(
    stream, logits: Tensor, legal_counts: Tensor, values: Tensor,
    legal_hashes: "list[int]",
) -> None:
    """Write a MSG_FORWARD_STATES OK response.

    Layout (little-endian)::

        u32 magic | u8 version | u8 msg_type = MSG_FORWARD_STATES (0x03)
        u8  status = STATES_STATUS_OK (0)
        u32 total_legal
        u32 num_graphs
        f32[total_legal]  logits        (same layout as _write_forward_response)
        i32[num_graphs]   legal_counts  (same layout as _write_forward_response)
        f32[num_graphs]   values        (same layout as _write_forward_response)
        u64[num_graphs]   legal-coord FNV-1a hashes (order guard; see
                          _read_forward_states_body for the hash input layout)
    """
    total_legal = logits.shape[0]
    num_graphs = values.shape[0]
    if len(legal_hashes) != num_graphs:
        raise RuntimeError(
            f"_write_forward_states_response: legal_hashes length "
            f"{len(legal_hashes)} != num_graphs {num_graphs} (per-graph order "
            f"guard would be misaligned)"
        )

    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_FORWARD_STATES))
    buf.append(STATES_STATUS_OK)
    buf.extend(struct.pack("<II", total_legal, num_graphs))
    buf.extend(logits.cpu().float().numpy().tobytes())
    buf.extend(legal_counts.cpu().int().numpy().tobytes())
    buf.extend(values.cpu().float().numpy().tobytes())
    buf.extend(struct.pack(f"<{num_graphs}Q", *legal_hashes))
    stream.write(bytes(buf))
    stream.flush()


def _write_states_error(stream, message: str) -> None:
    """Write a MSG_FORWARD_STATES in-band ERROR response.

    Layout (little-endian)::

        u32 magic | u8 version | u8 msg_type = MSG_FORWARD_STATES (0x03)
        u8  status = STATES_STATUS_ERROR (1)
        u32 msg_len
        u8[msg_len] utf-8 error message
    """
    msg = message.encode("utf-8")
    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_FORWARD_STATES))
    buf.append(STATES_STATUS_ERROR)
    buf.extend(struct.pack("<I", len(msg)))
    buf.extend(msg)
    stream.write(bytes(buf))
    stream.flush()


def _write_states_probe_ack(stream) -> None:
    """Write the dedicated capability-probe ACK for a zero-graph request.

    Layout (little-endian)::

        u32 magic | u8 version | u8 msg_type = MSG_FORWARD_STATES (0x03)
        u8  status = STATES_STATUS_PROBE_ACK (2)
    """
    buf = bytearray()
    buf.extend(struct.pack(HEADER_FMT, MAGIC, VERSION, MSG_FORWARD_STATES))
    buf.append(STATES_STATUS_PROBE_ACK)
    stream.write(bytes(buf))
    stream.flush()


# Static buckets ported verbatim from hexo-rs/hexo-mcts/src/inference.rs as a
# fallback when --static-buckets is given. The adaptive bucketizer (default)
# learns its own from the live size distribution.
NODE_BUCKETS = (4096, 16384, 32768, 49152, 65536, 131072, 196608)
EDGE_BUCKETS = (98304, 393216, 786432, 1572864, 3145728, 4718592)
EDGE_BUCKETS_PRUNED = (16384, 65536, 131072, 196608, 262144, 393216, 524288, 1048576)
_BUCKET_OVERFLOWS = 0


def pick_bucket(n: int, buckets: tuple) -> int:
    """Return the smallest bucket >= n; on overflow, pad to a multiple of the largest.

    Mirrors ``pick_bucket`` in ``hexo-rs/hexo-mcts/src/inference.rs``. Sustained
    overflows mean the bucket set is undersized — first five overflows always
    warn, then every 100th.
    """
    global _BUCKET_OVERFLOWS
    for b in buckets:
        if n <= b:
            return b
    largest = buckets[-1]
    padded = -(-n // largest) * largest  # ceil-div * largest
    _BUCKET_OVERFLOWS += 1
    c = _BUCKET_OVERFLOWS
    if c <= 5 or c % 100 == 0:
        sys.stderr.write(
            f"[pick_bucket] WARNING: size {n} exceeds top bucket {largest}, "
            f"padding to {padded} (overflow count: {c}). Consider enlarging buckets.\n"
        )
        sys.stderr.flush()
    return padded


class _AdaptiveBuckets:
    """Observe-then-lock bucketizer with auto re-lock on workload drift.

    Behaviour:

    - ``observe(n)`` is called on EVERY batch (even after lock) and keeps the
      last ``ring_size`` observations in a deque.
    - First lock fires once ``observe_batches`` samples have accumulated:
      buckets are chosen at uniform quantiles of the observed distribution,
      padded by ``headroom_pct``, rounded up to ``quantum``.
    - ``pick(n)`` returns the smallest fitting bucket. Overflows (size above
      the top bucket) pad to ``ceil(n / largest) * largest``.
    - If the post-lock overflow rate exceeds ``relock_threshold`` over at
      least ``observe_batches`` batches, the bucketizer re-picks buckets from
      the current ring buffer. This handles drift (e.g. early-game-only
      observation → games-now-end-game distribution) without manual tuning.
      Limited to ``max_relocks`` total to bound torch.compile re-trace cost.
    """

    def __init__(
        self,
        max_buckets: int,
        headroom_pct: float,
        quantum: int,
        observe_batches: int,
        name: str,
        ring_size: int = 5000,
        relock_threshold: float = 0.10,
        max_relocks: int = 4,
    ) -> None:
        import collections
        self.max_buckets = max_buckets
        self.headroom_pct = headroom_pct
        self.quantum = quantum
        self.observe_batches = max(1, observe_batches)
        self.name = name
        self.observed: "collections.deque[int]" = collections.deque(
            maxlen=max(ring_size, observe_batches),
        )
        self.buckets: list[int] = []
        self.is_ready = False
        self.relock_threshold = relock_threshold
        self.max_relocks = max_relocks
        self._relocks_done = 0
        self._batches_since_lock = 0
        self._overflows_since_lock = 0

    def observe(self, n: int) -> bool:
        """Record one observation; return ``is_ready`` afterwards."""
        self.observed.append(n)
        if not self.is_ready and len(self.observed) >= self.observe_batches:
            self._lock(is_relock=False)
        return self.is_ready

    def _quantum_round_up(self, n: int) -> int:
        return -(-n // self.quantum) * self.quantum

    def _bucket_for(self, n: int) -> int:
        """Bucket lookup with overflow-pad fallback (no side effects)."""
        for b in self.buckets:
            if n <= b:
                return b
        return -(-n // self.buckets[-1]) * self.buckets[-1]

    def _lock(self, *, is_relock: bool) -> None:
        sizes = sorted(self.observed)
        n_obs = len(sizes)
        chosen: set[int] = set()
        # Uniform quantile bins — each bucket k covers an equal share of the
        # observed distribution; rounded up to quantum.
        for k in range(1, self.max_buckets + 1):
            idx = min(n_obs - 1, max(0, int(k * n_obs / self.max_buckets) - 1))
            target = int(sizes[idx] * (1.0 + self.headroom_pct))
            chosen.add(self._quantum_round_up(target))
        # Guarantee a bucket >= the largest observed size.
        chosen.add(self._quantum_round_up(int(sizes[-1] * (1.0 + self.headroom_pct))))
        self.buckets = sorted(chosen)
        self.is_ready = True
        if is_relock:
            self._relocks_done += 1
        self._batches_since_lock = 0
        self._overflows_since_lock = 0
        # Overhead diagnostic against the observed distribution.
        total_real = sum(sizes)
        total_padded = sum(self._bucket_for(s) for s in sizes)
        overhead_pct = (total_padded - total_real) / total_real * 100.0 if total_real else 0.0
        verb = "RE-LOCKED" if is_relock else "LOCKED"
        suffix = f" (relock #{self._relocks_done}/{self.max_relocks})" if is_relock else ""
        sys.stderr.write(
            f"[adaptive_buckets:{self.name}] {verb} from {n_obs} observations: "
            f"buckets={self.buckets} mean_overhead={overhead_pct:.1f}%{suffix}\n"
        )
        sys.stderr.flush()

    def pick(self, n: int) -> int:
        assert self.is_ready, "pick() called before _lock()"
        self._batches_since_lock += 1
        import bisect
        idx = bisect.bisect_left(self.buckets, n)
        if idx < len(self.buckets):
            return self.buckets[idx]
        largest = self.buckets[-1]
        padded = -(-n // largest) * largest
        self._overflows_since_lock += 1
        if (
            self._relocks_done < self.max_relocks
            and self._batches_since_lock >= self.observe_batches
            and self._overflows_since_lock / self._batches_since_lock >= self.relock_threshold
        ):
            self._lock(is_relock=True)
        return padded


def _prepare_tensors(
    body: dict,
    device: torch.device,
    transfer_stream: "torch.cuda.Stream | None",
    padded: bool = False,
    node_buckets: "_AdaptiveBuckets | None" = None,
    edge_buckets: "_AdaptiveBuckets | None" = None,
) -> tuple[tuple, int]:
    """Deserialize a forward request body into GPU tensors.

    When *transfer_stream* is not None, all H2D copies are issued on that
    stream so they can overlap with a forward pass running on the default
    stream.  The caller must synchronize *transfer_stream* before using the
    returned tensors for compute.

    When ``padded`` is True, a "ghost" graph (all-zero features, false masks,
    self-loops only) is appended at batch index ``real_num_graphs`` so that
    ``total_nodes`` and ``total_edges`` round up to one of a small fixed set of
    bucket sizes — stabilising the CUDA / HIP caching allocator and letting
    ``torch.compile`` reuse a single specialised kernel across batches.

    Returns ``(tensor_tuple, real_num_graphs)``. Callers must slice
    ``values`` and ``legal_counts`` outputs to ``real_num_graphs`` before
    responding; ``all_logits`` is already correctly sized because the ghost
    contributes 0 legal positions.
    """
    real_nodes = body["total_nodes"]
    real_edges = body["total_edges"]
    real_num_graphs = body["num_graphs"]
    node_dim = body["node_dim"]

    if padded:
        if node_buckets is not None:
            target_nodes = node_buckets.pick(real_nodes + 1)
            target_edges = edge_buckets.pick(real_edges + 1)
        else:
            target_nodes = pick_bucket(real_nodes + 1, NODE_BUCKETS)
            # Same heuristic as Rust: pruned axis graphs have edge/node ratio
            # ~3.5-8x, unpruned ~20-28x; switch bucket set on observed ratio.
            edge_ratio = (real_edges // real_nodes) if real_nodes > 0 else 0
            edge_bucket_set = EDGE_BUCKETS_PRUNED if edge_ratio <= 10 else EDGE_BUCKETS
            target_edges = pick_bucket(real_edges + 1, edge_bucket_set)
    else:
        target_nodes = real_nodes
        target_edges = real_edges

    ctx = torch.cuda.stream(transfer_stream) if transfer_stream is not None else _nullctx()

    with ctx:
        # Note: we avoid pin_memory() here. Pinned allocations grow
        # monotonically (never returned to OS) and with variable-sized
        # batches the pinned pool slowly balloons. The double-buffered
        # transfer stream already hides most of the H2D latency.
        real_features = torch.frombuffer(body["features"], dtype=torch.float32).reshape(real_nodes, node_dim)
        real_edge_src = torch.frombuffer(body["edge_src"], dtype=torch.int64)
        real_edge_dst = torch.frombuffer(body["edge_dst"], dtype=torch.int64)
        real_legal = torch.frombuffer(body["legal_mask"], dtype=torch.uint8).to(dtype=torch.bool)
        real_stone = torch.frombuffer(body["stone_mask"], dtype=torch.uint8).to(dtype=torch.bool)
        real_batch = torch.frombuffer(body["batch"], dtype=torch.int32).to(dtype=torch.long)
        if body["has_edge_attr"] and body["edge_attr"] is not None:
            real_edge_attr = torch.frombuffer(body["edge_attr"], dtype=torch.float32).reshape(real_edges, 5)
        else:
            real_edge_attr = None

        if padded:
            ghost_nodes = target_nodes - real_nodes
            ghost_edges = target_edges - real_edges
            assert ghost_nodes >= 1 and ghost_edges >= 1

            features_cpu = torch.zeros((target_nodes, node_dim), dtype=torch.float32)
            features_cpu[:real_nodes] = real_features
            features = features_cpu.to(device)

            edge_src_cpu = torch.empty(target_edges, dtype=torch.int64)
            edge_dst_cpu = torch.empty(target_edges, dtype=torch.int64)
            edge_src_cpu[:real_edges] = real_edge_src
            edge_dst_cpu[:real_edges] = real_edge_dst
            # Spread ghost edges as self-loops across the ghost-node range so no
            # single ghost node accumulates a degenerate in-degree.
            ghost_self_loops = (
                torch.arange(ghost_edges, dtype=torch.int64) % ghost_nodes + real_nodes
            )
            edge_src_cpu[real_edges:] = ghost_self_loops
            edge_dst_cpu[real_edges:] = ghost_self_loops
            edge_index = torch.stack([edge_src_cpu, edge_dst_cpu], dim=0).to(device)

            legal_cpu = torch.zeros(target_nodes, dtype=torch.bool)
            legal_cpu[:real_nodes] = real_legal
            legal_mask = legal_cpu.to(device)

            stone_cpu = torch.zeros(target_nodes, dtype=torch.bool)
            stone_cpu[:real_nodes] = real_stone
            stone_mask = stone_cpu.to(device)

            batch_cpu = torch.empty(target_nodes, dtype=torch.long)
            batch_cpu[:real_nodes] = real_batch
            batch_cpu[real_nodes:] = real_num_graphs  # ghost batch index
            batch_vec = batch_cpu.to(device)

            if real_edge_attr is not None:
                ea_cpu = torch.zeros((target_edges, 5), dtype=torch.float32)
                ea_cpu[:real_edges] = real_edge_attr
                edge_attr = ea_cpu.to(device)
            else:
                edge_attr = torch.zeros(0, device=device)

            num_graphs = real_num_graphs + 1
        else:
            features = real_features.to(device)
            edge_index = torch.stack([real_edge_src, real_edge_dst], dim=0).to(device)
            legal_mask = real_legal.to(device)
            stone_mask = real_stone.to(device)
            batch_vec = real_batch.to(device)
            edge_attr = (
                real_edge_attr.to(device) if real_edge_attr is not None
                else torch.zeros(0, device=device)
            )
            num_graphs = real_num_graphs

        if device.type == "cuda":
            features = features.to(torch.bfloat16)
            if edge_attr.numel() > 0:
                edge_attr = edge_attr.to(torch.bfloat16)

        # Compute index tensors over the FULL (possibly padded) masks. Ghost
        # nodes are False in legal_mask/stone_mask, so they are naturally
        # excluded — same effect as Rust's "iterate real_nodes only".
        legal_idx = torch.where(legal_mask)[0]
        stone_idx = torch.where(stone_mask)[0]
        stone_batch = batch_vec[stone_idx]

    tensors = (features, edge_index, legal_mask, stone_mask, batch_vec,
               num_graphs, edge_attr, legal_idx, stone_idx, stone_batch)
    return tensors, real_num_graphs


class _nullctx:
    """No-op context manager for CPU path."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _forward_and_dispatch(
    tensors: tuple, model: torch.nn.Module,
    transfer_stream: "torch.cuda.Stream | None",
    writer_queue: "queue.Queue",
    real_num_graphs: int,
) -> tuple[float, float, float]:
    """Wait for H2D transfer, launch forward, hand off response to writer.

    Records a CUDA event right after the model launch and enqueues
    ``("forward", event, logits, legal_counts, values)`` for the writer
    thread, which event-waits and then serialises + writes the response.

    Returns ``(stream_sync_ms, launch_ms, dispatch_ms)``. Note that
    ``launch_ms`` measures the model's *Python-side* call time only —
    actual GPU compute happens asynchronously and is paid for by the
    writer thread (or by the next iteration's stream_sync).
    """
    on_cuda = transfer_stream is not None

    t_sync_start = _time.perf_counter()
    if on_cuda:
        torch.cuda.current_stream().wait_stream(transfer_stream)
    t_sync_end = _time.perf_counter()

    with torch.no_grad():
        all_logits, legal_counts, values = model(*tensors)

    # If the inputs were ghost-padded, the last bin of legal_counts/values
    # belongs to the ghost — drop it before responding. ``all_logits`` is
    # already correctly sized because the ghost contributes 0 legal positions.
    if legal_counts.shape[0] != real_num_graphs:
        legal_counts = legal_counts[:real_num_graphs]
        values = values[:real_num_graphs]

    event = torch.cuda.Event() if on_cuda else None
    if event is not None:
        event.record()
    t_launch_end = _time.perf_counter()

    writer_queue.put(("forward", event, all_logits, legal_counts, values))
    t_dispatch_end = _time.perf_counter()

    return (
        (t_sync_end - t_sync_start) * 1000.0,
        (t_launch_end - t_sync_end) * 1000.0,
        (t_dispatch_end - t_launch_end) * 1000.0,
    )


def _forward_and_dispatch_states(
    tensors: tuple, model: torch.nn.Module,
    transfer_stream: "torch.cuda.Stream | None",
    writer_queue: "queue.Queue",
    real_num_graphs: int,
    legal_hashes: "list[int]",
) -> tuple[float, float, float]:
    """States-mode mirror of ``_forward_and_dispatch`` (which stays untouched
    for the live graph path): identical forward/ghost-slice logic, but the
    writer item carries the per-graph legal-coord hashes and is written as a
    MSG_FORWARD_STATES-typed response by the writer thread.
    """
    on_cuda = transfer_stream is not None

    t_sync_start = _time.perf_counter()
    if on_cuda:
        torch.cuda.current_stream().wait_stream(transfer_stream)
    t_sync_end = _time.perf_counter()

    with torch.no_grad():
        all_logits, legal_counts, values = model(*tensors)

    if legal_counts.shape[0] != real_num_graphs:
        legal_counts = legal_counts[:real_num_graphs]
        values = values[:real_num_graphs]

    event = torch.cuda.Event() if on_cuda else None
    if event is not None:
        event.record()
    t_launch_end = _time.perf_counter()

    writer_queue.put(
        ("forward_states", event, all_logits, legal_counts, values, legal_hashes)
    )
    t_dispatch_end = _time.perf_counter()

    return (
        (t_sync_end - t_sync_start) * 1000.0,
        (t_launch_end - t_sync_end) * 1000.0,
        (t_dispatch_end - t_launch_end) * 1000.0,
    )


def main() -> None:
    # Configure the CUDA/HIP caching allocator BEFORE the device context is
    # initialised. This inference subprocess shares the GPU with the training
    # process; the allocator config (gc threshold + split-size, optional
    # expandable_segments) is the real safety net against cross-process OOM and
    # replaces the old per-forward empty_cache(). Must run before any
    # torch.cuda.* / .to(cuda) below — the env var is read once at first device
    # touch.
    from hexo_a0.gpu_memory import configure_cuda_alloc
    configure_cuda_alloc()

    # Isolate our torch.compile/inductor cache from the trainer's. We inherit
    # the trainer's env (TORCHINDUCTOR_CACHE_DIR=.../train) when spawned by the
    # Rust self-play binary; a shared kernel cache between the two compiling
    # processes on the one APU is the leading suspect for the 2026-06-02 compile
    # NaN. FORCE our own dir (override the inherited trainer value);
    # HEXO_INDUCTOR_CACHE_DIR_INFER overrides if explicitly set.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ.get(
        "HEXO_INDUCTOR_CACHE_DIR_INFER", "/tmp/torchinductor_hexo/infer"
    )

    # Bind us to the parent's lifetime before doing anything heavyweight.
    # A torch.compile pass can take minutes; if the parent dies during that
    # window we want to die too, not finish compiling against a dead parent
    # and then start serving requests no one will read.
    _install_parent_death_watchdog()

    parser = argparse.ArgumentParser(description="HeXO inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--policy-hidden", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--graph-type", default="hex", choices=["hex", "axis"])
    parser.add_argument(
        "--node-dim", type=int, default=8,
        help="Node-feature dimension of the model/checkpoint: 8 absolute, "
             "7 relative stone encoding, +4 with threat features (12/11). "
             "Drives model construction and warmup input width; "
             "must match the node_dim the Rust side sends in v2 forward "
             "messages.",
    )
    parser.add_argument("--conv-type", default="gatv2", choices=["gatv2", "gine"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pre-norm", action="store_true")
    parser.add_argument(
        "--use-jk", action="store_true",
        help="Enable Jumping Knowledge aggregation in the inference model. "
             "Must match the training config used to produce the checkpoint.",
    )
    parser.add_argument(
        "--jk-mode", default="sum", choices=["sum", "cat", "max"],
        help="JK aggregation mode (only honored when --use-jk is set). "
             "'lstm' is not supported in ScriptableHeXONet — use --jk-mode "
             "sum/cat/max for self-play.",
    )
    # --- D6-invariant lean schema (must match the training config) ---
    parser.add_argument(
        "--axis-relational", action="store_true",
        help="Use the D6-invariant axis-relational encoder (edge-type partition "
             "+ tied-weight AxisRelationalConv). The server converts the legacy "
             "wire graph to lean inputs internally; must match the checkpoint.",
    )
    parser.add_argument(
        "--axis-window", type=int, default=8,
        help="Max unsigned hop for the axis-relational distance embedding "
             "(only with --axis-relational; must be >= win_length-1).",
    )
    parser.add_argument(
        "--relative-stone-encoding", action="store_true",
        help="Relative (own/opp) stone encoding — needed for the lean-column map.",
    )
    parser.add_argument(
        "--threat-features", action="store_true",
        help="Threat node features present — needed for the lean-column map.",
    )
    parser.add_argument(
        "--compact-stone-onehot", action="store_true",
        help="Lean schema: the redundant 'empty' stone one-hot dim is dropped.",
    )
    parser.add_argument(
        "--no-node-coords", dest="node_coords", action="store_false", default=True,
        help="Lean schema: norm-q/r node coordinates are dropped.",
    )
    parser.add_argument(
        "--moves-scope", default="node", choices=["node", "graph"],
        help="moves-remaining scope in the lean node schema.",
    )
    # --- Distributional (binned) value head (must match the checkpoint) ---
    parser.add_argument(
        "--value-bins", type=int, default=0,
        help="Number of value bins for the distributional (C51-style) value "
             "head. 0 = scalar tanh head (legacy). Normally auto-configured "
             "from the checkpoint's embedded model_config.",
    )
    parser.add_argument(
        "--value-bin-min", type=float, default=-1.0,
        help="Lower edge of the value-bin center grid (only with --value-bins).",
    )
    parser.add_argument(
        "--value-bin-max", type=float, default=1.0,
        help="Upper edge of the value-bin center grid (only with --value-bins).",
    )
    parser.add_argument(
        "--dynamic-compile", action="store_true",
        help="Pass dynamic=True to torch.compile so Dynamo traces with "
             "symbolic shapes from the start (no per-shape specialisation). "
             "Trades per-call kernel speed for zero recompile cost on shape "
             "changes — usually a wash or regression for stable workloads.",
    )
    parser.add_argument(
        "--padded-inference", action="store_true",
        help="Pad inputs to bucket shapes (mirrors Rust forward_graphs_padded) "
             "so torch.compile sees a small fixed set of shapes.",
    )
    parser.add_argument(
        "--static-buckets", action="store_true",
        help="Use the static NODE_BUCKETS/EDGE_BUCKETS from Rust instead of "
             "the adaptive bucketizer (which learns the bucket set from live "
             "size distribution). Only relevant with --padded-inference.",
    )
    parser.add_argument(
        "--max-buckets", type=int, default=6,
        help="Adaptive bucketizer: cap on distinct bucket sizes per dimension "
             "(node, edge). When exceeded, closest-adjacent pair is merged.",
    )
    parser.add_argument(
        "--bucket-headroom-pct", type=float, default=0.10,
        help="Adaptive bucketizer: padding headroom above each chosen bucket "
             "size (default 10%%).",
    )
    parser.add_argument(
        "--bucket-observe-batches", type=int, default=1000,
        help="Adaptive bucketizer: minimum number of batches to observe "
             "before locking bucket sizes (default 1000 ~= 15s of self-play; "
             "needs to be long enough to cover end-of-game positions, not just "
             "early-game ones). Bucketizer auto re-locks if the post-lock "
             "overflow rate exceeds --bucket-relock-threshold, so under-sizing "
             "is self-correcting. Use 1 in tests.",
    )
    # --- --slot-inference: A2 slot backend for MSG_FORWARD_STATES ---
    parser.add_argument(
        "--slot-inference", action="store_true",
        help="Serve MSG_FORWARD_STATES via the A2 slot model (batched "
             "on-device slot build from the wire's int32 HexKeys + "
             "SlotHeXONet.forward_padded) instead of the legacy graph "
             "rebuild. MSG_FORWARD (graph mode) always stays on the legacy "
             "model. Legacy GINE checkpoints only — unsupported architectures "
             "(axis_relational, gatv2, lstm-JK, moves_scope='graph') fail at "
             "startup. Requires --win-length.",
    )
    parser.add_argument(
        "--win-length", type=int, default=None,
        help="Game win_length (only with --slot-inference): fixes the slot "
             "model's edge-table size 6*(win_length-1); cross-checked against "
             "every states request.",
    )
    parser.add_argument(
        "--slot-activation-budget-mb", type=float, default=4096.0,
        help="A2 memory guard (only with --slot-inference): cap in MiB on the "
             "estimated dense [B, N, slots, hidden] message-passing activation "
             "of one states request (bf16 on CUDA / f32 on CPU, padded to the "
             "pow2 [B, N] the forward allocates, from an exact server-computed "
             "node count — never any allocation). Over-budget requests get an "
             "in-band ERROR instead of an OOM.",
    )
    parser.add_argument(
        "--bucket-relock-threshold", type=float, default=0.10,
        help="Adaptive bucketizer: fraction of post-lock batches that must "
             "overflow the top bucket before triggering a re-lock from the "
             "ring buffer (default 10%%). Capped at 4 re-locks per run.",
    )
    args = parser.parse_args()

    if args.slot_inference and (args.win_length is None or args.win_length < 2):
        parser.error("--slot-inference requires --win-length >= 2")

    device = torch.device(args.device)

    # Load model
    model = _load_model(args)

    # Warmup
    _warmup(model, device, args.graph_type, args.node_dim)

    # --slot-inference: also build the A2 slot model from the same checkpoint.
    # Unsupported architectures are a clear STARTUP failure (exit code 2),
    # never a mid-request one. (_load_model above already merged the
    # checkpoint's embedded model_config into args.)
    slot_model = None
    if args.slot_inference:
        try:
            slot_model = _load_slot_model(args, device)
            _warmup_slot(slot_model, args, device)
        except Exception as e:
            # ANY startup failure (unsupported arch ValueError, a bf16/dtype
            # RuntimeError in the warmup forward, a torch.compile trace error,
            # ...) must produce the clean FATAL line + exit(2), never a raw
            # traceback that the supervisor can't distinguish from a crash.
            sys.stderr.write(f"FATAL: --slot-inference startup check failed: {e}\n")
            sys.stderr.flush()
            sys.exit(2)

    # Redirect text-mode stdout to stderr so that any stray print(),
    # library warning, or unhandled traceback goes to the log pipe
    # instead of corrupting the binary protocol on stdout.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    # Signal ready
    sys.stderr.write("READY\n")
    sys.stderr.flush()

    # Use a background thread to pre-read the next request while the GPU
    # is processing the current one. This overlaps pipe I/O with GPU compute.
    import queue
    import threading

    request_queue: queue.Queue = queue.Queue(maxsize=1)
    # Writer queue: ("forward", event, logits, legal_counts, values) or
    # ("reload_ack", success_bool) or None (shutdown). Bounded so a slow
    # writer applies backpressure rather than letting GPU outputs pile up.
    # Sized to allow the main dispatch thread to launch several batches ahead
    # of the writer's event-sync — at 4 the main thread blocks every time the
    # writer falls one batch behind. 16 gives ~300ms of in-flight forwards.
    writer_queue: queue.Queue = queue.Queue(maxsize=16)
    writer_perf = {"count": 0, "total_ms": 0.0, "wait_ms": 0.0}
    writer_perf_lock = threading.Lock()

    def _writer_thread() -> None:
        """Serialise model outputs and write them to stdout in order."""
        while True:
            item = writer_queue.get()
            if item is None:
                return
            kind = item[0]
            t0 = _time.perf_counter()
            if kind == "forward":
                _, event, logits, legal_counts, values = item
                if event is not None:
                    event.synchronize()
                t_synced = _time.perf_counter()
                _write_forward_response(stdout, logits, legal_counts, values)
                t_done = _time.perf_counter()
                with writer_perf_lock:
                    writer_perf["count"] += 1
                    writer_perf["wait_ms"] += (t_synced - t0) * 1000.0
                    writer_perf["total_ms"] += (t_done - t0) * 1000.0
            elif kind == "reload_ack":
                _write_reload_ack(stdout, success=item[1])
            elif kind == "forward_states":
                _, event, logits, legal_counts, values, hashes = item
                if event is not None:
                    event.synchronize()
                t_synced = _time.perf_counter()
                _write_forward_states_response(
                    stdout, logits, legal_counts, values, hashes
                )
                t_done = _time.perf_counter()
                with writer_perf_lock:
                    writer_perf["count"] += 1
                    writer_perf["wait_ms"] += (t_synced - t0) * 1000.0
                    writer_perf["total_ms"] += (t_done - t0) * 1000.0
            elif kind == "states_error":
                _write_states_error(stdout, item[1])
            elif kind == "states_probe":
                _write_states_probe_ack(stdout)
            else:
                sys.stderr.write(f"Unknown writer queue item: {kind!r}\n")

    def _reader_thread():
        """Read requests from stdin and enqueue them."""
        while True:
            try:
                header = _read_exact(stdin, HEADER_SIZE)
            except EOFError:
                request_queue.put(None)
                return
            magic, ver, msg_type = struct.unpack(HEADER_FMT, header)
            try:
                _check_protocol_header(magic, ver)
            except ValueError as e:
                sys.stderr.write(f"protocol error: {e}\n")
                sys.stderr.flush()
                request_queue.put(None)
                return
            if msg_type == MSG_SHUTDOWN:
                request_queue.put(None)
                return
            # Pre-read the full request body into memory
            if msg_type == MSG_FORWARD:
                try:
                    body = _read_forward_body(stdin, args.node_dim)
                except ValueError as e:
                    # Unrecoverable mid-stream (framing is lost); die loudly.
                    sys.stderr.write(f"protocol error: {e}\n")
                    sys.stderr.flush()
                    request_queue.put(None)
                    return
                request_queue.put(("forward", body))
            elif msg_type == MSG_FORWARD_STATES and args.slot_inference:
                # Slot backend: framing + validation only here (cheap); the
                # on-device slot build + forward run in the main loop.
                try:
                    frame = _states_request_frame(stdin)
                except EOFError:
                    request_queue.put(None)
                    return
                if frame[0] == 0:  # zero-graph capability probe
                    request_queue.put(("states_probe",))
                else:
                    # Activation dtype size (bf16=2 on CUDA, f32=4 on CPU) read
                    # straight from the live slot model's parameters so the
                    # budget estimate matches the tensors forward_padded builds.
                    bpe = next(slot_model.parameters()).element_size()
                    try:
                        payload = _validate_states_slot(
                            frame, args, bytes_per_elem=bpe
                        )
                    except StatesRequestError as e:
                        sys.stderr.write(f"states request error: {e}\n")
                        sys.stderr.flush()
                        request_queue.put(("states_error", str(e)))
                    else:
                        # The client is strictly synchronous, so the on-device
                        # build+forward run inline in the main loop
                        # (_handle_states_slot); the reader only frames+validates.
                        request_queue.put(("forward_states_slot", payload))
            elif msg_type == MSG_FORWARD_STATES:
                # Rebuild runs HERE on the reader thread so it overlaps GPU
                # compute exactly like graph-mode body reads + H2D staging.
                try:
                    parsed = _read_forward_states_body(
                        stdin, args.node_dim,
                        prune_empty_edges=bool(getattr(args, "prune_empty_edges", False)),
                        threat_features=bool(getattr(args, "threat_features", False)),
                        relative_stones=bool(getattr(args, "relative_stone_encoding", False)),
                    )
                except StatesRequestError as e:
                    # Body fully consumed — framing intact. Reply in-band and
                    # keep serving; the CLIENT hard-fails on the error.
                    sys.stderr.write(f"states request error: {e}\n")
                    sys.stderr.flush()
                    request_queue.put(("states_error", str(e)))
                except EOFError:
                    request_queue.put(None)
                    return
                else:
                    if parsed is None:
                        request_queue.put(("states_probe",))
                    else:
                        states_body, states_hashes = parsed
                        # Private key: _prepare_tensors ignores it; the main
                        # loop uses it to pick the states-typed response.
                        states_body["_states_hashes"] = states_hashes
                        request_queue.put(("forward_states", states_body))
            elif msg_type == MSG_RELOAD:
                path_len = struct.unpack("<I", _read_exact(stdin, 4))[0]
                path_bytes = _read_exact(stdin, path_len)
                request_queue.put(("reload", path_bytes.decode("utf-8")))
            else:
                request_queue.put(None)
                return

    reader = threading.Thread(target=_reader_thread, daemon=True)
    reader.start()
    writer = threading.Thread(target=_writer_thread, daemon=True)
    writer.start()

    # Main loop — double-buffered: prepare batch N+1's tensors on a transfer
    # stream while batch N's forward pass runs on the default (compute) stream.
    #
    # Perf logging is a 100-batch rolling window: every accumulator below is
    # reset after each `[perf]` line, so the printed averages reflect only
    # the most recent window (not torch.compile cold start + every batch
    # ever). `_perf_count` is cumulative — it drives the `batches=N` line so
    # users can still see total progress.
    _perf_count = 0
    _perf_window_total_ms = 0.0
    _perf_window_gpu_ms = 0.0
    _perf_window_sync_ms = 0.0
    _perf_window_fwd_ms = 0.0
    _perf_window_write_ms = 0.0
    _perf_window_graphs = 0
    _perf_window_nodes = 0

    transfer_stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    # Pressure-triggered empty_cache(): variable-shape inference makes the HIP
    # caching allocator reserve a block per distinct (node,edge) size and never
    # give it back (gc_threshold is inert on unified memory), so the reserve
    # creeps until the kernel OOM-kills self-play after hours. BUT clearing on a
    # batch cadence is a ~3x throughput regression on this unified-memory APU:
    # empty_cache() returns GTT to the driver and forces EVERY process (this
    # worker + the trainer) to re-fault pages back in. So instead of clearing on
    # a fixed schedule, clear ONLY when the allocator reserve is genuinely high
    # (a real leak), checked cheaply (memory_reserved(), no sync) every N batches.
    # In steady state (reserve ~1-2 GB, tens of GB free) it NEVER fires => zero
    # churn; it only kicks in if the reserve balloons past the threshold.
    #   HEXO_CACHE_CLEAR_RESERVED_MB : clear when reserved exceeds this (default
    #     8192; <=0 disables clearing entirely)
    #   HEXO_CACHE_CLEAR_CHECK_EVERY : how often to test the threshold (default 128)
    _clear_threshold_mb = float(os.environ.get("HEXO_CACHE_CLEAR_RESERVED_MB", "8192"))
    _clear_check_every = int(os.environ.get("HEXO_CACHE_CLEAR_CHECK_EVERY", "128"))
    _clear_check = 0
    _clear_on_cuda = device.type == "cuda" and _clear_threshold_mb > 0

    # Adaptive bucketizer (default when --padded-inference is set).
    if args.padded_inference and not args.static_buckets:
        node_bucketizer = _AdaptiveBuckets(
            max_buckets=args.max_buckets,
            headroom_pct=args.bucket_headroom_pct,
            quantum=4096,
            observe_batches=args.bucket_observe_batches,
            name="nodes",
            relock_threshold=args.bucket_relock_threshold,
        )
        edge_bucketizer = _AdaptiveBuckets(
            max_buckets=args.max_buckets,
            headroom_pct=args.bucket_headroom_pct,
            quantum=16384,
            observe_batches=args.bucket_observe_batches,
            name="edges",
            relock_threshold=args.bucket_relock_threshold,
        )
    else:
        node_bucketizer = None
        edge_bucketizer = None

    # Attributes _load_model mutates in place on `args` (checkpoint path + the
    # schema flags it merges from the new checkpoint's embedded model_config).
    # A NACKed reload must leave `args` consistent with the STILL-LIVE models,
    # so we snapshot and restore them if the rebuild fails partway.
    _RELOAD_MUTATED_ATTRS = (
        "checkpoint", "axis_relational", "axis_window", "compact_stone_onehot",
        "node_coords", "moves_scope", "relative_stone_encoding",
        "threat_features", "value_bins", "value_bin_min", "value_bin_max",
        "prune_empty_edges",
    )

    def _handle_reload(path: str) -> None:
        nonlocal model, slot_model
        _sentinel = object()
        saved = {k: getattr(args, k, _sentinel) for k in _RELOAD_MUTATED_ATTRS}
        try:
            args.checkpoint = path
            new_model = _load_model(args)
            _warmup(new_model, device, args.graph_type, args.node_dim)
            new_slot = None
            if args.slot_inference:
                new_slot = _load_slot_model(args, device)
                _warmup_slot(new_slot, args, device)
            model = new_model
            slot_model = new_slot if args.slot_inference else slot_model
            print(f"Model reloaded from {path}", file=sys.stderr, flush=True)
            writer_queue.put(("reload_ack", True))
        except Exception as e:
            # Roll back every args mutation so the flags the live models rely on
            # (builder-flag cross-check, node dim, slot schema) stay consistent.
            for k, v in saved.items():
                if v is _sentinel:
                    if hasattr(args, k):
                        delattr(args, k)
                else:
                    setattr(args, k, v)
            print(f"Reload failed: {e}", file=sys.stderr, flush=True)
            writer_queue.put(("reload_ack", False))

    def _handle_states_slot(payload: tuple) -> None:
        """Slot-backend states request (--slot-inference): keys -> on-device
        slot batch -> forward_padded -> per-graph legal-order response.

        Every failure past the reader's validation is still in-band: the
        server replies STATES_STATUS_ERROR and keeps serving. The [B, N]
        padded logits are flattened with ``logits[legal_mask]`` — row-major,
        so per graph in the batch's legal-column order, which is ascending
        HexKey == (q, r)-lexicographic == ``legal_moves()`` order; the FNV
        hashes are recomputed from that same ordering (SlotBatchAux) so the
        client-side order guard stays honest.
        """
        from hexo_a0.slot_graph import build_slot_batch_from_keys

        win_length, placement_radius, graphs = payload
        try:
            builder_config = _slot_builder_config(args, win_length, placement_radius)
            states = [(p1, p2, cur, mr) for (p1, p2, cur, mr, _nl) in graphs]
            # Build at a mult-128 N bucket through the (optionally compiled)
            # build core: pad_to fixes the core's N so compile(dynamic=False)
            # sees a small static shape set. bucket_n comes from the EXACT node
            # count (== the builder's n_max), so it is always >= n_max. aux
            # (legal ordering/counts) is independent of pad_to, so the response
            # contract is unchanged.
            bucket_n = _slot_pad_n(graphs, placement_radius)
            core = getattr(slot_model, "_slot_build_core", None)
            batch, aux = build_slot_batch_from_keys(
                states, builder_config, device=device,
                pad_to=bucket_n, return_aux=True, core_fn=core,
            )
            rebuilt_counts = aux.legal_counts.cpu().tolist()
            for i, (_p1, _p2, _c, _m, num_legal) in enumerate(graphs):
                if rebuilt_counts[i] != num_legal:
                    raise StatesRequestError(
                        f"graph {i}: wire num_legal {num_legal} != rebuilt "
                        f"legal count {rebuilt_counts[i]} (client/server "
                        f"builder divergence)"
                    )
            # Pad B to its bucket so compile(dynamic=False) sees a small fixed
            # set of forward_padded shapes (N is already at its mult-128 bucket
            # from the build's pad_to, so _slot_bucket_shape is idempotent on N).
            # aux (legal ordering, counts) comes from the pre-B-pad build, so the
            # response contract is unaffected; ghost graphs/nodes are inert
            # (masks False) and their values are sliced off below.
            real_b = batch.num_graphs
            bucket_b, bucket_n = _slot_bucket_shape(batch)
            batch = _pad_slot_batch(batch, bucket_b, bucket_n)
            if device.type == "cuda":
                batch.x = batch.x.to(torch.bfloat16)
                batch.dummy_x = batch.dummy_x.to(torch.bfloat16)

            with torch.no_grad():
                logits, values = slot_model.forward_padded(batch)
            flat_logits = logits[batch.legal_mask]  # per-graph legal order
            values = values[:real_b]  # drop ghost-graph values
            legal_counts = aux.legal_counts.to(torch.int32)
            hashes = _slot_legal_hashes(aux)

            event = torch.cuda.Event() if device.type == "cuda" else None
            if event is not None:
                event.record()
            writer_queue.put(
                ("forward_states", event, flat_logits, legal_counts, values, hashes)
            )
        except StatesRequestError as e:
            sys.stderr.write(f"states request error: {e}\n")
            sys.stderr.flush()
            writer_queue.put(("states_error", str(e)))
        except Exception as e:
            # Builder/forward machinery failure: framing is intact (the body
            # was consumed by the reader), so reply in-band and keep serving.
            sys.stderr.write(f"states request error: slot forward failed: {e}\n")
            sys.stderr.flush()
            writer_queue.put(("states_error", f"slot forward failed: {e}"))

    def _next_forward_body():
        """Block until we get a forward request, handling reloads inline."""
        while True:
            item = request_queue.get()
            if item is None:
                return None
            if item[0] == "reload":
                _handle_reload(item[1])
                continue
            if item[0] == "states_error":
                # In-band error for a MSG_FORWARD_STATES request; routed
                # through the request queue so response order is preserved.
                writer_queue.put(("states_error", item[1]))
                continue
            if item[0] == "states_probe":
                writer_queue.put(("states_probe",))
                continue
            if item[0] == "forward_states_slot":
                # Slot backend (--slot-inference): build + forward run inline
                # here, in request order, outside the legacy double-buffered
                # prepare path (_prepare_tensors never sees these). The client is
                # strictly synchronous, so there is nothing to overlap the build
                # against — the reader frames+validates, the main loop builds and
                # forwards.
                _handle_states_slot(item[1])
                continue
            return item[1]  # forward body (graph-mode or rebuilt states-mode)

    # State for double-buffering: when not None, tensors for the next
    # batch have already been prepared on the transfer stream.
    prefetched_body = None
    prefetched_tensors = None
    prefetched_real_n = 0

    _perf_window_prepare_ms = 0.0
    PERF_WINDOW = 100

    while True:
        t0 = _time.perf_counter()

        if prefetched_tensors is not None:
            # Tensors were pre-prepared during the previous forward pass
            body = prefetched_body
            tensors = prefetched_tensors
            real_n = prefetched_real_n
            prefetched_body = None
            prefetched_tensors = None
            t_prepared = t0  # prepare cost was hidden in the previous iteration
        else:
            body = _next_forward_body()
            if body is None:
                break
            if node_bucketizer is not None:
                node_bucketizer.observe(body["total_nodes"])
                edge_bucketizer.observe(body["total_edges"])
            effective_padded = (
                args.padded_inference
                and (node_bucketizer is None or node_bucketizer.is_ready)
            )
            tensors, real_n = _prepare_tensors(
                body, device, transfer_stream, padded=effective_padded,
                node_buckets=node_bucketizer, edge_buckets=edge_bucketizer,
            )
            t_prepared = _time.perf_counter()

        # Launch forward on the default stream and hand the outputs off
        # to the writer thread (which event-waits + writes to stdout).
        # States-mode bodies carry the per-graph legal-coord hashes and get
        # the MSG_FORWARD_STATES-typed response; graph mode is unchanged.
        states_hashes = body.get("_states_hashes")
        if states_hashes is None:
            sync_ms, fwd_ms, write_ms = _forward_and_dispatch(
                tensors, model, transfer_stream, writer_queue, real_n,
            )
        else:
            sync_ms, fwd_ms, write_ms = _forward_and_dispatch_states(
                tensors, model, transfer_stream, writer_queue, real_n,
                states_hashes,
            )

        # Drop our local refs; the writer thread holds its own refs until
        # it's done with these tensors.
        del tensors
        t_forward = _time.perf_counter()

        # While the response is being written, pre-fetch the next batch
        # and start H2D transfer on the transfer stream.
        next_body = _next_forward_body()
        if next_body is not None:
            if node_bucketizer is not None:
                node_bucketizer.observe(next_body["total_nodes"])
                edge_bucketizer.observe(next_body["total_edges"])
            next_effective_padded = (
                args.padded_inference
                and (node_bucketizer is None or node_bucketizer.is_ready)
            )
            prefetched_tensors, prefetched_real_n = _prepare_tensors(
                next_body, device, transfer_stream, padded=next_effective_padded,
                node_buckets=node_bucketizer, edge_buckets=edge_bucketizer,
            )
            prefetched_body = next_body

        t_done = _time.perf_counter()

        # Perf tracking (100-batch rolling window; reset after each print).
        #   prepare:  tensor deserialization + H2D (0 when prefetched)
        #   forward:  model forward + response write
        #   prefetch: preparing next batch (hidden overlap)
        _perf_count += 1
        _perf_window_prepare_ms += (t_prepared - t0) * 1000
        _perf_window_gpu_ms += (t_forward - t_prepared) * 1000
        _perf_window_sync_ms += sync_ms
        _perf_window_fwd_ms += fwd_ms
        _perf_window_write_ms += write_ms  # now ~dispatch time, write happens on writer thread
        _perf_window_total_ms += (t_forward - t0) * 1000  # prepare + dispatch (excludes prefetch)
        _perf_window_graphs += body["num_graphs"]
        _perf_window_nodes += body["total_nodes"]
        if _perf_count % PERF_WINDOW == 0:
            n = PERF_WINDOW
            # Drain + reset writer thread's counters under the lock so the
            # window stat there matches the main-loop window.
            with writer_perf_lock:
                w_count = writer_perf["count"]
                w_total = writer_perf["total_ms"]
                w_wait = writer_perf["wait_ms"]
                writer_perf["count"] = 0
                writer_perf["total_ms"] = 0.0
                writer_perf["wait_ms"] = 0.0
            w_total_avg = w_total / w_count if w_count else 0.0
            w_wait_avg = w_wait / w_count if w_count else 0.0
            w_io_avg = w_total_avg - w_wait_avg
            bucket_suffix = ""
            if node_bucketizer is not None:
                bucket_suffix = (
                    f" nodes={node_bucketizer.buckets}"
                    f" edges={edge_bucketizer.buckets}"
                )
            print(
                f"[perf] batches={_perf_count} "
                f"last_total={_perf_window_total_ms/n:.1f}ms "
                f"last_prepare={_perf_window_prepare_ms/n:.1f}ms "
                f"last_dispatch={_perf_window_gpu_ms/n:.1f}ms "
                f"(sync={_perf_window_sync_ms/n:.2f} launch={_perf_window_fwd_ms/n:.2f} put={_perf_window_write_ms/n:.2f}) "
                f"writer={w_total_avg:.2f}ms (event_wait={w_wait_avg:.2f} io={w_io_avg:.2f}) "
                f"last_graphs={_perf_window_graphs/n:.1f} "
                f"last_nodes={_perf_window_nodes/n:.0f}"
                f"{bucket_suffix}",
                file=sys.stderr, flush=True,
            )
            # Reset the rolling window so the next line is independent.
            _perf_window_total_ms = 0.0
            _perf_window_prepare_ms = 0.0
            _perf_window_gpu_ms = 0.0
            _perf_window_sync_ms = 0.0
            _perf_window_fwd_ms = 0.0
            _perf_window_write_ms = 0.0
            _perf_window_graphs = 0
            _perf_window_nodes = 0

        # Return the allocator's unused reserve to the system ONLY under real
        # memory pressure (see above) — never on a fixed cadence, which churns
        # GTT and slows every process on this unified-memory APU.
        if _clear_on_cuda:
            _clear_check += 1
            if _clear_check >= _clear_check_every:
                _clear_check = 0
                if torch.cuda.memory_reserved() / 1e6 > _clear_threshold_mb:
                    torch.cuda.empty_cache()

        if next_body is None:
            break

    # Drain and stop the writer cleanly so any in-flight responses get out.
    writer_queue.put(None)
    writer.join(timeout=5.0)


if __name__ == "__main__":
    main()
