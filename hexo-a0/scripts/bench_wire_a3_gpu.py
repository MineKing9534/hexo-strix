"""A3 plan-task-7 GPU bench: the PLAN'S CLAIM end-to-end on the real server.

Compares the two inference-server wire paths at PRODUCTION shapes on GPU,
each as a full write-request -> read-response pipe round-trip against the
REAL ``hexo_a0.inference_server`` subprocess (torch.compile ON, --device cuda):

  A) LEGACY graph-wire path: MSG_FORWARD body (per-graph axis builder output
     collated the way the Rust client encodes it) -> server _read_forward_body
     -> _prepare_tensors -> compiled legacy ScriptableHeXONet.

  B) SLOT states path (--slot-inference): MSG_FORWARD_STATES body (canonical
     int32 HexKeys) -> on-device batched slot build -> compiled
     SlotHeXONet.forward_padded.

The two servers are spawned SEQUENTIALLY (one GPU) with random weights at
production serving dims (hidden 128 / 4 layers / 8 heads / policy_hidden 128 /
value_hidden 32 / jk cat / axis / gine / prune+threat+relative node_dim 11 /
win_length 6). Random weights are fine for THROUGHPUT: the kernels executed and
their shapes are identical to trained weights; only the numbers differ.

Batch shapes: an edge-budgeted production batch (accumulate real radius-8 games
until ~45k graph edges, the Rust client's ~4-graph shape) plus fixed batch
sizes 1, 8, 16 — the SAME positions fed to both modes.

Protocol per (server, batch-shape): request bytes prebuilt OUTSIDE timing;
>=5 warmup round-trips (lets torch.compile settle on the shape — discarded);
>=30 timed round-trips; report median + p10/p90 wall ms/batch and graphs/sec.
Before timing server B, one batch is verified: the slot path's response FNV-1a
legal hashes must match a client-side recompute over legal_moves() order.

Usage:
    uv run --no-sync python hexo-a0/scripts/bench_wire_a3_gpu.py \
        [--reps 30] [--warmup 5] [--device cuda] [--ready-timeout 600]
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
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
)

# --- Production serving config ----------------------------------------------
WIN_LENGTH = 6
PLACEMENT_RADIUS = 8
MAX_MOVES = 300
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 8
POLICY_HIDDEN = 128
VALUE_HIDDEN = 32
CONV_TYPE = "gine"
GRAPH_TYPE = "axis"
JK_MODE = "cat"
NODE_DIM = 11  # relative (7) + threat (4)
BUILDER_KWARGS = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)
FLAG_PRUNE, FLAG_THREAT, FLAG_RELATIVE = 0x01, 0x02, 0x04
BUILDER_FLAGS = FLAG_PRUNE | FLAG_THREAT | FLAG_RELATIVE
EDGE_BUDGET = 45000  # Rust client's max_batch_edges

# Mid-game depths (plies == stone placements), ~20-40 stones, covers mr 1 & 2.
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)


# --- Positions: real random-legal radius-8 games ----------------------------

def _cfg() -> "hexo_rs.GameConfig":
    return hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)


def _snapshot(game, cfg):
    return hexo_rs.GameState.from_state(
        game.placed_stones(), game.current_player(),
        game.moves_remaining_this_turn(), cfg,
    )


def _play_to_depth(seed: int, depth: int):
    cfg = _cfg()
    rng = random.Random(seed)
    game = hexo_rs.GameState(cfg)
    last = None
    for _ in range(depth):
        if game.is_terminal():
            break
        game.apply_move(*rng.choice(game.legal_moves()))
        if not game.is_terminal():
            last = _snapshot(game, cfg)
    assert last is not None, f"seed {seed}: no non-terminal state reached"
    return last


def build_pool(n: int) -> list:
    return [_play_to_depth(seed=3000 + i, depth=DEPTHS[i % len(DEPTHS)]) for i in range(n)]


# --- Wire-body builders (new HexKey format, mirrors the parity tests) --------

def _pack_hexkey(q: int, r: int) -> int:
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def collate_graph_body(games) -> bytes:
    """MSG_FORWARD BODY, byte-for-byte as SubprocessModel::forward_graphs."""
    raws = [hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS) for g in games]
    total_nodes = sum(r["num_nodes"] for r in raws)
    total_edges = sum(len(r["edge_src"]) for r in raws)
    buf = bytearray()
    buf.extend(struct.pack("<III", total_nodes, total_edges, len(raws)))
    buf.extend(struct.pack("<BB", 1, NODE_DIM))
    for r in raws:
        buf.extend(np.asarray(r["features"], dtype=np.float32).tobytes())
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
    return bytes(buf), total_nodes, total_edges


def states_body(games) -> bytes:
    """MSG_FORWARD_STATES BODY: canonical int32 HexKeys (§1 wire revision)."""
    buf = bytearray()
    buf.extend(struct.pack(
        "<IBBIBB", len(games), WIN_LENGTH, PLACEMENT_RADIUS,
        MAX_MOVES, BUILDER_FLAGS, NODE_DIM,
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


# --- Client-side FNV legal-order hash (independent of the server's) ----------

def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _expected_legal_hash(game) -> int:
    payload = b"".join(struct.pack("<ii", q, r) for q, r in game.legal_moves())
    return _fnv1a64(payload)


# --- Framing helpers ---------------------------------------------------------

def _framed(msg_type: int, body: bytes) -> bytes:
    return struct.pack("<IBB", MAGIC, VERSION, msg_type) + body


def _read_exact(stream, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError(f"Expected {n} bytes, got {len(data)}")
        data += chunk
    return data


def _read_forward_response(stdout):
    magic, ver, mt = struct.unpack("<IBB", _read_exact(stdout, 6))
    assert (magic, ver) == (MAGIC, VERSION) and mt == MSG_FORWARD, (magic, ver, mt)
    total_legal, num_graphs = struct.unpack("<II", _read_exact(stdout, 8))
    logits = _read_exact(stdout, total_legal * 4)
    counts = _read_exact(stdout, num_graphs * 4)
    values = _read_exact(stdout, num_graphs * 4)
    return logits, counts, values


def _read_states_response(stdout):
    magic, ver, mt = struct.unpack("<IBB", _read_exact(stdout, 6))
    assert (magic, ver) == (MAGIC, VERSION) and mt == MSG_FORWARD_STATES, (magic, ver, mt)
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


# --- Checkpoint: legacy HeXONet at production dims (random weights) -----------
# Follows tests/test_slot_inference_wire.py::_make_hexonet_ckpt exactly:
# embed model_config with the checkpoint-authoritative builder flags; the server
# derives its states builder flags from it (single source of truth).

def make_ckpt() -> str:
    from hexo_a0.config import ModelConfig
    from hexo_a0.model import HeXONet

    cfg = ModelConfig(
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        conv_type=CONV_TYPE, policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN,
        graph_type=GRAPH_TYPE, use_jk=True, jk_mode=JK_MODE,
        prune_empty_edges=True, threat_features=True, relative_stone_encoding=True,
    )
    torch.manual_seed(1234)
    model = HeXONet(cfg)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {"model": model.state_dict(),
             "model_config": {
                 "prune_empty_edges": True,
                 "threat_features": True,
                 "relative_stone_encoding": True,
             }},
            f.name,
        )
        return f.name


# --- Server spawn / ready wait (pattern from the wire tests) ------------------

def spawn_server(ckpt_path: str, device: str, *extra: str) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "hexo_a0.inference_server",
        "--checkpoint", ckpt_path,
        "--hidden-dim", str(HIDDEN_DIM), "--num-layers", str(NUM_LAYERS),
        "--num-heads", str(NUM_HEADS), "--policy-hidden", str(POLICY_HIDDEN),
        "--value-hidden", str(VALUE_HIDDEN), "--graph-type", GRAPH_TYPE,
        "--conv-type", CONV_TYPE, "--device", device, "--node-dim", str(NODE_DIM),
        "--use-jk", "--jk-mode", JK_MODE,
        "--threat-features", "--relative-stone-encoding",
        *extra,
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=os.environ.copy(),
    )


def wait_ready(proc, timeout: float) -> str:
    """Non-blocking stderr poll until READY (compile warmup can be minutes)."""
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
                return buf.decode(errors="replace")
        elif proc.poll() is not None:
            try:
                buf += os.read(fd, 65536)
            except (BlockingIOError, OSError):
                pass
            if b"READY" in buf:
                return buf.decode(errors="replace")
            raise RuntimeError(
                f"server died before READY (exit {proc.returncode}):\n"
                f"{buf.decode(errors='replace')}"
            )
        else:
            time.sleep(0.05)
    raise TimeoutError(
        f"server didn't send READY in {timeout}s:\n{buf.decode(errors='replace')}"
    )


def drain_stderr(proc) -> None:
    """Keep the stderr pipe empty so the server never blocks on a full pipe."""
    def _run():
        fd = proc.stderr.fileno()
        os.set_blocking(fd, True)
        while True:
            try:
                if not os.read(fd, 65536):
                    return
            except OSError:
                return
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def shutdown(proc) -> None:
    if proc.poll() is None:
        try:
            proc.stdin.write(struct.pack("<IBB", MAGIC, VERSION, MSG_SHUTDOWN))
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


# --- Timing ------------------------------------------------------------------

def roundtrip(proc, request: bytes, read_response) -> object:
    proc.stdin.write(request)
    proc.stdin.flush()
    return read_response(proc.stdout)


def time_config(proc, request: bytes, read_response, warmup: int, reps: int):
    for _ in range(warmup):
        roundtrip(proc, request, read_response)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        roundtrip(proc, request, read_response)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median": statistics.median(times),
        "p10": times[max(0, int(0.10 * len(times)) - 0)],
        "p90": times[min(len(times) - 1, int(0.90 * len(times)))],
        "min": times[0],
    }


# --- Batch selection ---------------------------------------------------------

def edge_budgeted_batch(pool):
    """Accumulate real radius-8 games until the actual graph edge count would
    exceed EDGE_BUDGET (the Rust client's ~45k-edge / ~4-graph production
    shape)."""
    batch, total_edges = [], 0
    for g in pool:
        raw = hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS)
        e = len(raw["edge_src"])
        if batch and total_edges + e > EDGE_BUDGET:
            break
        batch.append(g)
        total_edges += e
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    is_cuda = dev.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    hip = getattr(torch.version, "hip", None)

    print("# A3 plan-task-7 GPU bench — states+slot forward_padded vs legacy graph wire")
    print(f"# torch={torch.__version__} device={args.device} gpu={gpu_name!r} "
          f"hip={hip} dtype={'bf16 (CUDA activations)' if is_cuda else 'f32'}")
    print(f"# compile: torch.compile(fullgraph=True, dynamic=False) ON both models")
    print(f"# model: legacy GINE, hidden={HIDDEN_DIM} layers={NUM_LAYERS} "
          f"heads={NUM_HEADS} policy_h={POLICY_HIDDEN} value_h={VALUE_HIDDEN} "
          f"jk={JK_MODE} axis prune+threat+relative node_dim={NODE_DIM} "
          f"win_length={WIN_LENGTH}")
    print(f"# WEIGHTS ARE RANDOM (throughput only — identical kernels/shapes to "
          f"trained weights; only the numbers differ)")
    print(f"# reps={args.reps} (>=30 timed), warmup={args.warmup} (>=5, discarded), "
          f"edge_budget={EDGE_BUDGET}")

    # --- positions + batch shapes (built once, shared by both modes) ---------
    t0 = time.perf_counter()
    pool = build_pool(48)
    eb = edge_budgeted_batch(pool)
    batches = {
        f"edge-budget(n={len(eb)})": eb,
        "b=1": pool[:1],
        "b=8": pool[:8],
        "b=16": pool[:16],
    }
    stones = [len(g.placed_stones()) for g in pool]
    print(f"# pool: {len(pool)} games in {time.perf_counter()-t0:.1f}s, "
          f"stones/game min={min(stones)} max={max(stones)}; "
          f"mr set={sorted({g.moves_remaining_this_turn() for g in pool})}")

    # Prebuild request bytes OUTSIDE timing.
    prebuilt = {}  # name -> (graph_req, states_req, num_graphs, nodes, edges)
    for name, games in batches.items():
        gbody, nodes, edges = collate_graph_body(games)
        sbody = states_body(games)
        prebuilt[name] = (
            _framed(MSG_FORWARD, gbody),
            _framed(MSG_FORWARD_STATES, sbody),
            len(games), nodes, edges,
        )
        print(f"#   {name}: graphs={len(games)} nodes={nodes} edges={edges} "
              f"graph_req={len(gbody)+6}B states_req={len(sbody)+6}B")

    ckpt = make_ckpt()
    results = {}  # name -> {"legacy": stats, "slot": stats}
    slot_error = None
    try:
        # ================= Server A: legacy graph wire, compile ON ===========
        print("\n# --- Server A: legacy MSG_FORWARD (compile ON) ---", flush=True)
        procA = spawn_server(ckpt, args.device)
        try:
            rdy = wait_ready(procA, args.ready_timeout)
            print("#   READY", flush=True)
            drain_stderr(procA)
            for name in batches:
                greq = prebuilt[name][0]
                stats = time_config(procA, greq, _read_forward_response,
                                    args.warmup, args.reps)
                results.setdefault(name, {})["legacy"] = stats
                print(f"#   {name}: legacy median={stats['median']:.2f}ms "
                      f"p10={stats['p10']:.2f} p90={stats['p90']:.2f}", flush=True)
        finally:
            shutdown(procA)

        # ================= Server B: slot states wire, compile ON ============
        print("\n# --- Server B: --slot-inference MSG_FORWARD_STATES (compile ON) ---",
              flush=True)
        procB = spawn_server(ckpt, args.device, "--slot-inference",
                             "--win-length", str(WIN_LENGTH))
        try:
            rdy = wait_ready(procB, args.ready_timeout)
            print("#   READY", flush=True)
            drain_stderr(procB)

            # Sanity: FNV legal-order hashes must match client recompute on one
            # batch BEFORE timing (correctness guard for the slot path).
            probe_name = "b=8"
            sreq = prebuilt[probe_name][1]
            resp = roundtrip(procB, sreq, _read_states_response)
            if resp[0] != "ok":
                slot_error = f"slot path returned {resp} for {probe_name}"
                raise RuntimeError(slot_error)
            _, _s_logits, s_counts, _s_values, s_hashes = resp
            games = batches[probe_name]
            exp_counts = [g.legal_move_count() for g in games]
            got_counts = np.frombuffer(s_counts, dtype=np.int32).tolist()
            exp_hashes = [_expected_legal_hash(g) for g in games]
            assert got_counts == exp_counts, (got_counts, exp_counts)
            assert s_hashes == exp_hashes, "FNV legal-order hash mismatch (slot path)"
            print(f"#   sanity OK: {probe_name} counts + FNV legal-order hashes "
                  f"match client recompute", flush=True)

            for name in batches:
                sreq = prebuilt[name][1]
                # confirm this shape answers OK before timing
                resp = roundtrip(procB, sreq, _read_states_response)
                if resp[0] != "ok":
                    print(f"#   {name}: slot ERROR -> {resp}", flush=True)
                    results.setdefault(name, {})["slot"] = {"error": str(resp)}
                    continue
                stats = time_config(procB, sreq, _read_states_response,
                                    args.warmup, args.reps)
                results.setdefault(name, {})["slot"] = stats
                print(f"#   {name}: slot median={stats['median']:.2f}ms "
                      f"p10={stats['p10']:.2f} p90={stats['p90']:.2f}", flush=True)
        finally:
            shutdown(procB)
    finally:
        Path(ckpt).unlink(missing_ok=True)

    # ================= Results table =========================================
    print("\n" + "=" * 100)
    print("## Results — wall round-trip ms/batch (median [p10..p90]) and graphs/sec\n")
    hdr = (f"| {'batch shape':<18} | {'graphs':>6} | {'nodes':>6} | {'edges':>6} | "
           f"{'legacy ms':>22} | {'slot ms':>22} | {'ratio':>6} | "
           f"{'legacy g/s':>10} | {'slot g/s':>9} | {'graph req':>10} | {'states req':>10} |")
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for name in batches:
        ng, nodes, edges = prebuilt[name][2], prebuilt[name][3], prebuilt[name][4]
        greq_b, sreq_b = len(prebuilt[name][0]), len(prebuilt[name][1])
        r = results.get(name, {})
        leg = r.get("legacy")
        slot = r.get("slot")

        def fmt(s):
            if not s or "error" in (s or {}):
                return "ERROR"
            return f"{s['median']:.2f} [{s['p10']:.2f}..{s['p90']:.2f}]"
        leg_s = fmt(leg)
        slot_s = fmt(slot)
        if leg and slot and "error" not in slot:
            ratio = f"{slot['median']/leg['median']:.2f}x"
            leg_gs = f"{ng/(leg['median']/1e3):.0f}"
            slot_gs = f"{ng/(slot['median']/1e3):.0f}"
        else:
            ratio, leg_gs, slot_gs = "-", (f"{ng/(leg['median']/1e3):.0f}" if leg else "-"), "-"
        print(f"| {name:<18} | {ng:>6} | {nodes:>6} | {edges:>6} | {leg_s:>22} | "
              f"{slot_s:>22} | {ratio:>6} | {leg_gs:>10} | {slot_gs:>9} | "
              f"{greq_b:>9}B | {sreq_b:>9}B |")

    print("\n# ratio = slot median / legacy median  (>1.0 => slot SLOWER)")
    if slot_error:
        print(f"# SLOT PATH ERROR: {slot_error}")
    print("# NOTE: single shared APU GPU (unified memory). A `tournament/"
          "deliberate.py` process may contend — p90 captures contention spikes.")


if __name__ == "__main__":
    main()
