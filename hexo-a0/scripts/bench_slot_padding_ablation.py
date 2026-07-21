"""Slot padding ablation: how much of the compiled slot forward's loss vs the
legacy forward is the pow2 padding, and how much is fundamental (the dense
[B, N, S] slot algebra itself)?

Context: ``bench_slot_decomposition.py`` found the compiled slot ``forward_padded``
loses to the compiled legacy ``ScriptableHeXONet`` forward at the SAME production
shapes, even uncontended and forward-only. The server pads B and N to powers of
two (``_round_pow2``, floor=8 on N) so the ``torch.compile(dynamic=False)`` slot
forward sees a handful of shapes. Pow2 padding on N inflates the slot node count
well past the batch's real max-N; this script isolates how much of the slot
forward's cost is that padding overhang vs the dense slot algebra that survives
even at the batch's EXACT max-N.

Padding regimes, all measuring the compiled bf16 slot ``forward_padded``
ms/batch at b = 1, 8, 16, 32, 64, 128, on the SAME real random-legal radius-8
positions:

  1. pow2       — server behavior: _round_pow2 both dims, floor=8 on N. Control;
                  reproduces the decomposition bench's slot forward numbers.
  2. mult256    — B padded to pow2; N padded up to the next multiple of 256.
                  NOTE: B is now pow2-padded (was B-unpadded in the earlier
                  variant). The server needs a bounded set of B variants, so the
                  bounded-B forms all pad B to pow2 and vary only in N granularity.
  3. mult128    — B padded to pow2; N padded up to the next multiple of 128.
                  Same as mult256 but a finer N bucket — decides server bucket
                  granularity (does the finer N bucket measurably help?).
  4. exact      — pad_to=None: real B, batch's real max-N. One compile per exact
                  shape (recompiles; trace time reported separately, EXCLUDED
                  from the timed median). This is the padding-free floor.
  5. legacy     — compiled ScriptableHeXONet forward on the same batches
                  (reference control from bench_slot_decomposition.py).

Each regime uses its OWN torch.compile callable (dynamic=False) so guard caches
never interfere and compile counts are attributable. Trace time = the single
first invocation on each fresh shape (synced), measured and reported apart from
the warmup/reps median.

Verdict logic: if the exact regime (3) forward still loses to legacy at b=8/16,
the slot forward's deficit is FUNDAMENTAL (dense slot algebra), not padding.

Random weights (throughput only — identical kernels/shapes to trained weights).

Usage:
    uv run --no-sync python hexo-a0/scripts/bench_slot_padding_ablation.py \
        [--device cuda] [--reps 50] [--warmup 20]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch

# The four slot regimes feed different (B,N) shapes to the SAME forward_padded
# code object; dynamo's recompile limit is per-code-object, so 4 regimes x 6
# batch sizes = up to 24 distinct static traces (fewer after pow2 dedup) far
# exceeds the default 8. Raise it generously (each distinct shape is an intended,
# separate static compile in this bench).
torch._dynamo.config.recompile_limit = 256
if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
    torch._dynamo.config.accumulated_recompile_limit = 1024

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_a0.model_slots import slot_model_from_legacy
from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet
from hexo_a0.slot_graph import SlotBuilderConfig, build_slot_batch_from_keys
from hexo_a0.inference_server import (
    _pad_slot_batch,
    _prepare_tensors,
    _round_pow2,
)

# --- Production serving config (identical to bench_slot_decomposition.py) -----
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
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)
BATCH_SIZES = (1, 8, 16, 32, 64, 128)
# N-bucket granularities for the bounded-B (pow2-B) regimes. mult128 is the
# finer bucket under test; mult256 the coarser reference.
N_MULTIPLES = {"mult256": 256, "mult128": 128}
REGIMES = ("pow2", "mult256", "mult128", "exact")


# --- Positions: real random-legal radius-8 games (mirror decomposition bench) -

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


# --- Slot keys input: (p1_keys, p2_keys, current_player, moves_remaining) -----

def _pack_hexkey(q: int, r: int) -> int:
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def games_to_keys(games) -> list:
    out = []
    for g in games:
        stones = g.placed_stones()
        p1 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P1"]
        p2 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P2"]
        cur = 0 if g.current_player() == "P1" else 1
        out.append((p1, p2, cur, g.moves_remaining_this_turn()))
    return out


# --- Legacy graph body dict (exact _read_forward_body shape) ------------------

def _buf(arr) -> bytearray:
    return bytearray(np.ascontiguousarray(arr).tobytes())


def build_forward_body(games) -> dict:
    raws = [hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS) for g in games]
    total_nodes = sum(r["num_nodes"] for r in raws)
    total_edges = sum(len(r["edge_src"]) for r in raws)

    features = np.concatenate([np.asarray(r["features"], dtype=np.float32).reshape(-1) for r in raws])
    esrc, edst, off = [], [], 0
    for r in raws:
        esrc.append(np.asarray(r["edge_src"], dtype=np.int64) + off)
        edst.append(np.asarray(r["edge_dst"], dtype=np.int64) + off)
        off += r["num_nodes"]
    edge_src = np.concatenate(esrc)
    edge_dst = np.concatenate(edst)
    edge_attr = np.concatenate([np.asarray(r["edge_attr"], dtype=np.float32).reshape(-1) for r in raws])
    legal_mask = np.concatenate([np.asarray(r["legal_mask"], dtype=np.uint8) for r in raws])
    stone_mask = np.concatenate([np.asarray(r["stone_mask"], dtype=np.uint8) for r in raws])
    batch = np.concatenate([np.full(r["num_nodes"], i, dtype=np.int32) for i, r in enumerate(raws)])

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "num_graphs": len(raws),
        "node_dim": NODE_DIM,
        "features": _buf(features),
        "edge_src": _buf(edge_src),
        "edge_dst": _buf(edge_dst),
        "edge_attr": _buf(edge_attr),
        "has_edge_attr": True,
        "legal_mask": _buf(legal_mask),
        "stone_mask": _buf(stone_mask),
        "batch": _buf(batch),
    }


# --- Model construction -------------------------------------------------------

def build_models(device):
    cfg = ModelConfig(
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        conv_type=CONV_TYPE, policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN,
        graph_type=GRAPH_TYPE, use_jk=True, jk_mode=JK_MODE,
        prune_empty_edges=True, threat_features=True, relative_stone_encoding=True,
    )
    torch.manual_seed(1234)
    legacy = HeXONet(cfg).to(device).eval()

    slot = slot_model_from_legacy(legacy, cfg, WIN_LENGTH).to(device).eval()

    scriptable = ScriptableHeXONet(
        node_features=NODE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN,
        graph_type=GRAPH_TYPE, conv_type=CONV_TYPE, use_jk=True, jk_mode=JK_MODE,
        relative_stone_encoding=True, threat_features=True,
    ).eval()
    load_from_hexonet(scriptable, legacy.state_dict())
    scriptable = scriptable.to(device)

    if device.type == "cuda":
        scriptable = scriptable.to(torch.bfloat16)
        slot = slot.to(torch.bfloat16)

    scriptable_c = torch.compile(scriptable, fullgraph=True, dynamic=False)
    # one independent compiled callable per slot regime (own guard cache)
    slot_c = {
        r: torch.compile(slot.forward_padded, fullgraph=True, dynamic=False)
        for r in REGIMES
    }
    return scriptable_c, slot_c


# --- Padding regimes ----------------------------------------------------------

def _next_mult(n: int, m: int) -> int:
    return ((int(n) + m - 1) // m) * m


def regime_shape(regime: str, real_b: int, real_n: int) -> "tuple[int, int]":
    if regime == "pow2":
        return _round_pow2(real_b), _round_pow2(real_n, floor=8)
    if regime in N_MULTIPLES:
        # bounded-B server variant: B pow2-padded, N to the next bucket multiple
        return _round_pow2(real_b), _next_mult(real_n, N_MULTIPLES[regime])
    if regime == "exact":
        return real_b, real_n
    raise ValueError(regime)


def make_padded(keys, builder_cfg, device, regime, is_cuda):
    """Fresh unpadded build, pad to the regime's (B,N), cast x/dummy_x to bf16.

    Rebuilt per regime so casting never mutates a shared unpadded batch."""
    unpadded, _aux = build_slot_batch_from_keys(
        keys, builder_cfg, device=device, return_aux=True)
    real_b = int(unpadded.num_graphs)
    real_n = int(unpadded.node_mask.shape[1])
    b_to, n_to = regime_shape(regime, real_b, real_n)
    padded = _pad_slot_batch(unpadded, b_to, n_to)
    if is_cuda:
        padded.x = padded.x.to(torch.bfloat16)
        padded.dummy_x = padded.dummy_x.to(torch.bfloat16)
    return padded, (real_b, real_n), (b_to, n_to)


# --- Timing -------------------------------------------------------------------

def timed(fn, device, warmup, reps):
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def trace_once(fn, device):
    """Single first invocation on a fresh shape — includes the compile trace.
    Returned separately so it is EXCLUDED from the timed median."""
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def act_bytes(b_to: int, n_to: int, slots: int, hidden: int, bytes_per: int = 2) -> int:
    """Peak dense slot-message activation size [B, N, S, H] in bytes (bf16=2).

    This is the ``msgs``/``agg`` tensor gathered per slot in the representation
    body — the single largest transient the pow2/mult padding inflates."""
    return int(b_to) * int(n_to) * int(slots) * int(hidden) * bytes_per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    hip = getattr(torch.version, "hip", None)

    print("# slot padding ABLATION — pow2 vs mult256 vs mult128 vs exact vs legacy, forward-only")
    print(f"# torch={torch.__version__} device={args.device} gpu={gpu!r} hip={hip} "
          f"dtype={'bf16' if is_cuda else 'f32'}")
    print(f"# model: GINE hidden={HIDDEN_DIM} layers={NUM_LAYERS} heads={NUM_HEADS} "
          f"jk={JK_MODE} axis prune+threat+relative node_dim={NODE_DIM} wl={WIN_LENGTH}")
    print(f"# compile: fullgraph=True dynamic=False (per-regime callable; legacy scriptable)")
    print(f"# reps={args.reps} warmup={args.warmup} — RANDOM weights (throughput only)")
    print(f"# batches: {BATCH_SIZES}")
    print(f"# N regimes: pow2(B pow2, N pow2 floor=8) | mult256/mult128 (B pow2, "
          f"N->next mult) | exact (real B, real max-N)")

    scriptable_c, slot_c = build_models(device)

    pool = build_pool(max(BATCH_SIZES))
    builder_cfg = SlotBuilderConfig(
        win_length=WIN_LENGTH, placement_radius=PLACEMENT_RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )

    fwd_ms = {r: {} for r in REGIMES}
    fwd_ms["legacy"] = {}
    trace_ms = {r: {} for r in REGIMES}
    real_shape = {}     # b -> (real_b, real_n)
    pad_shape = {r: {} for r in REGIMES}   # regime -> b -> (b_to, n_to)
    legacy_nodes = {}   # b -> total real nodes
    compile_shapes = {r: set() for r in REGIMES}
    slot_s = {}         # b -> S (num_slots; constant, tracked for the byte est.)
    est_bytes = {r: {} for r in REGIMES}   # regime -> b -> peak act bytes
    oom_events = []     # (regime, b, (b_to,n_to), est_bytes)

    for b in BATCH_SIZES:
        games = pool[:b]
        keys = games_to_keys(games)

        body = build_forward_body(games)
        legacy_nodes[b] = body["total_nodes"]
        legacy_tensors, _real_n = _prepare_tensors(body, device, None, padded=False)

        with torch.no_grad():
            # legacy control (sparse graph body — guard it too, though unlikely OOM)
            try:
                fwd_ms["legacy"][b] = timed(
                    lambda: scriptable_c(*legacy_tensors), device, args.warmup, args.reps)
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                fwd_ms["legacy"][b] = None
                if is_cuda:
                    torch.cuda.empty_cache()
                oom_events.append(("legacy", b, real_shape.get(b), None))
                print(f"#  !! OOM legacy b={b} — skipped, continuing")

            for regime in REGIMES:
                padded, real_bn, pad_bn = make_padded(
                    keys, builder_cfg, device, regime, is_cuda)
                real_shape[b] = real_bn
                pad_shape[regime][b] = pad_bn
                compile_shapes[regime].add(pad_bn)
                s = int(padded.filled.shape[2])
                slot_s[b] = s
                eb = act_bytes(pad_bn[0], pad_bn[1], s, HIDDEN_DIM,
                               2 if is_cuda else 4)
                est_bytes[regime][b] = eb
                # print the peak dense [B,N,S,H] activation estimate BEFORE running
                print(f"#  est {regime:<8} b={b:<3} pad(B,N)={pad_bn} S={s} "
                      f"H={HIDDEN_DIM} -> peak [B,N,S,H] act ~= "
                      f"{eb/1e9:.3f} GB ({eb:,} bytes)")
                fn = slot_c[regime]
                try:
                    # trace (compile) time = first call on this fresh shape, excluded
                    trace_ms[regime][b] = trace_once(lambda: fn(padded), device)
                    fwd_ms[regime][b] = timed(
                        lambda: fn(padded), device, args.warmup, args.reps)
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    fwd_ms[regime][b] = None
                    trace_ms[regime].setdefault(b, None)
                    oom_events.append((regime, b, pad_bn, eb))
                    print(f"#  !! OOM {regime} b={b} pad(B,N)={pad_bn} "
                          f"est~{eb/1e9:.2f}GB — skipped, continuing")
                    del padded
                    if is_cuda:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

    # ---------------- results ----------------
    def slot_count(bn):
        return bn[0] * bn[1]

    def fmt_ms(v):
        return f"{v:>7.3f}" if v is not None else f"{'OOM':>7}"

    all_regimes = ("legacy",) + REGIMES

    print("\n" + "=" * 78)
    print("## Forward-only ms/batch + ratio vs legacy\n")
    hdr = f"| {'regime':<10} |" + "".join(
        f" b={b} ms | b={b} xLeg |" for b in BATCH_SIZES)
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    print(f"| {'legacy':<10} |" + "".join(
        f" {fmt_ms(fwd_ms['legacy'][b])} | {'1.00' if fwd_ms['legacy'][b] is not None else 'n/a':>8} |"
        for b in BATCH_SIZES))
    for regime in REGIMES:
        line = f"| {regime:<10} |"
        for b in BATCH_SIZES:
            ms = fwd_ms[regime][b]
            leg = fwd_ms["legacy"][b]
            if ms is None:
                line += f" {'OOM':>7} | {'--':>8} |"
            elif leg:
                line += f" {ms:>7.3f} | {ms/leg:>8.2f} |"
            else:
                line += f" {ms:>7.3f} | {'n/a':>8} |"
        print(line)

    print("\n## Throughput graphs/sec (real batch b / ms) — legacy vs best slot regime\n")
    print(f"| {'metric':<18} |" + "".join(f" b={b:<5} |" for b in BATCH_SIZES))
    print("|" + "-" * (20 + 9 * len(BATCH_SIZES)) + "|")
    leg_line = f"| {'legacy gr/s':<18} |"
    best_line = f"| {'best-slot gr/s':<18} |"
    which_line = f"| {'best-slot regime':<18} |"
    for b in BATCH_SIZES:
        leg = fwd_ms["legacy"][b]
        leg_line += f" {b/(leg/1e3):>6.0f} |" if leg else f" {'--':>6} |"
        cand = [(fwd_ms[r][b], r) for r in REGIMES if fwd_ms[r][b] is not None]
        if cand:
            bms, br = min(cand)
            best_line += f" {b/(bms/1e3):>6.0f} |"
            which_line += f" {br:>6} |"
        else:
            best_line += f" {'--':>6} |"
            which_line += f" {'--':>6} |"
    print(leg_line)
    print(best_line)
    print(which_line)

    print("\n## Node-slot count (B*N) vs legacy real node count\n")
    hdr2 = f"| {'regime':<10} |" + "".join(
        f" b={b} slots | b={b} vsReal |" for b in BATCH_SIZES)
    print(hdr2)
    print("|" + "-" * (len(hdr2) - 2) + "|")
    print(f"| {'legacy':<10} |" + "".join(
        f" {legacy_nodes[b]:>9} | {'1.00x':>10} |" for b in BATCH_SIZES))
    for regime in REGIMES:
        line = f"| {regime:<10} |"
        for b in BATCH_SIZES:
            sc = slot_count(pad_shape[regime][b])
            line += f" {sc:>9} | {sc/legacy_nodes[b]:>9.2f}x |"
        print(line)

    print("\n## Padded (B,N) shapes and real (B,N) / real node count\n")
    for b in BATCH_SIZES:
        rb, rn = real_shape[b]
        print(f"#  b={b:<3} real(B,N)=({rb},{rn}) real_nodes={legacy_nodes[b]} S={slot_s[b]}  "
              f"pow2->{pad_shape['pow2'][b]}  mult256->{pad_shape['mult256'][b]}  "
              f"mult128->{pad_shape['mult128'][b]}  exact->{pad_shape['exact'][b]}")

    print("\n## Peak dense [B,N,S,H] activation estimate per shape (bf16)\n")
    for b in BATCH_SIZES:
        parts = " ".join(
            f"{r}={est_bytes[r][b]/1e9:.2f}GB" for r in REGIMES)
        print(f"#  b={b:<3} {parts}")

    print("\n## Compile counts (distinct (B,N) shapes traced) + trace ms (excluded from timing)\n")
    for regime in REGIMES:
        traces = " ".join(
            f"b={b}:{trace_ms[regime].get(b):.0f}ms" if trace_ms[regime].get(b) is not None
            else f"b={b}:--" for b in BATCH_SIZES)
        print(f"#  {regime:<8} compiles={len(compile_shapes[regime])} "
              f"shapes={sorted(compile_shapes[regime])}  trace: {traces}")

    print("\n## Verdict — is the slot forward deficit padding or fundamental?\n")
    for b in BATCH_SIZES:
        ex = fwd_ms["exact"][b]
        leg = fwd_ms["legacy"][b]
        p2 = fwd_ms["pow2"][b]
        if ex is None or leg is None:
            print(f"#  b={b:<3}: exact={fmt_ms(ex).strip()} legacy={fmt_ms(leg).strip()} "
                  f"pow2={fmt_ms(p2).strip()}  -> incomplete (OOM)")
            continue
        pad_overhang = (p2 - ex) if p2 is not None else float("nan")
        verdict = "LOSES to legacy -> FUNDAMENTAL" if ex > leg else "beats legacy -> padding-bound"
        print(f"#  b={b:<3}: exact={ex:.3f} legacy={leg:.3f} ({ex/leg:.2f}x)  "
              f"pow2={fmt_ms(p2).strip()} pad_overhang(pow2-exact)={pad_overhang:+.3f}ms  -> exact {verdict}")

    # ---------------- explicit answers requested ----------------
    print("\n" + "=" * 78)
    print("## (a) Does the slot forward's advantage grow / hold / shrink at b=32/64/128?\n")
    print("#  slot ratio vs legacy (best slot regime; <1 = slot faster) across batches:")
    trend = []
    for b in BATCH_SIZES:
        leg = fwd_ms["legacy"][b]
        cand = [(fwd_ms[r][b], r) for r in REGIMES if fwd_ms[r][b] is not None]
        if leg and cand:
            bms, br = min(cand)
            trend.append((b, bms / leg, br))
            print(f"#    b={b:<3} best={br:<8} {bms:.3f}ms  ratio_vs_legacy={bms/leg:.2f}x")
        else:
            print(f"#    b={b:<3} incomplete (OOM)")
    print("#  exact-regime ratio vs legacy across batches (padding-free floor):")
    for b in BATCH_SIZES:
        ex, leg = fwd_ms["exact"][b], fwd_ms["legacy"][b]
        if ex and leg:
            print(f"#    b={b:<3} exact/legacy={ex/leg:.2f}x")
    if len(trend) >= 2:
        big = [t for t in trend if t[0] in (32, 64, 128)]
        if big:
            r0 = trend[0][1]
            print(f"#  Read: ratio at b=1 is {r0:.2f}x; "
                  + ", ".join(f"b={b}:{r:.2f}x" for b, r, _ in big)
                  + " (rising ratio = slot advantage SHRINKS, falling = GROWS).")

    print("\n## (b) Is mult128 measurably better than mult256 at these shapes?\n")
    print(f"| {'batch':<6} | {'mult256 ms':<11} | {'mult128 ms':<11} | {'delta(256-128) ms':<18} | {'delta %':<8} |")
    print("|" + "-" * 66 + "|")
    for b in BATCH_SIZES:
        m2, m1 = fwd_ms["mult256"][b], fwd_ms["mult128"][b]
        if m2 is None or m1 is None:
            print(f"| {b:<6} | {fmt_ms(m2).strip():<11} | {fmt_ms(m1).strip():<11} | {'--':<18} | {'--':<8} |")
            continue
        d = m2 - m1
        pct = 100.0 * d / m2 if m2 else float("nan")
        print(f"| {b:<6} | {m2:<11.3f} | {m1:<11.3f} | {d:<+18.3f} | {pct:<+8.1f} |")
    print("#  (positive delta = mult128 faster; also compare their padded-N below)")
    for b in BATCH_SIZES:
        n256 = pad_shape["mult256"][b][1]
        n128 = pad_shape["mult128"][b][1]
        real_n = real_shape[b][1]
        print(f"#    b={b:<3} real_N={real_n} padN mult256={n256} mult128={n128} "
              f"(slot waste 256={n256-real_n} 128={n128-real_n})")

    print("\n## (c) OOM / recompile-limit events\n")
    if oom_events:
        for regime, b, shape, eb in oom_events:
            ebs = f"~{eb/1e9:.2f}GB" if eb else "n/a"
            print(f"#  OOM: regime={regime} b={b} shape={shape} est_act={ebs}")
    else:
        print("#  No OOM events — all shapes ran to completion.")
    print(f"#  recompile_limit={torch._dynamo.config.recompile_limit} "
          f"accumulated={getattr(torch._dynamo.config, 'accumulated_recompile_limit', 'n/a')}")
    total_traces = sum(len(compile_shapes[r]) for r in REGIMES)
    print(f"#  distinct static traces across all slot regimes = {total_traces} "
          f"(per-regime: " + " ".join(f"{r}:{len(compile_shapes[r])}" for r in REGIMES) + ")")


if __name__ == "__main__":
    main()
