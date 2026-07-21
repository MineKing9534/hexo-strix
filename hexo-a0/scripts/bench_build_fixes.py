"""Q2/Q3/Q5: measure full-build impact of (a) replacing cumprod with an
unrolled scan, (b) torch.compile on the whole core, (c) sub-profile threat.

Monkeypatches slot_graph in-process only (no file edits).

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_fixes.py
"""
from __future__ import annotations

import random
import statistics
import time

import torch

import hexo_rs
from hexo_a0 import slot_graph as sg
from hexo_a0.slot_graph import SlotBuilderConfig, build_slot_batch_from_keys

WIN, RAD, MM = 6, 8, 300
CONFIG = SlotBuilderConfig(WIN, RAD, True, True, True)
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)


def play(seed, depth):
    cfg = hexo_rs.GameConfig(WIN, RAD, MM); rng = random.Random(seed)
    g = hexo_rs.GameState(cfg); last = None
    for _ in range(depth):
        if g.is_terminal(): break
        g.apply_move(*rng.choice(g.legal_moves()))
        if not g.is_terminal():
            last = g.from_state(g.placed_stones(), g.current_player(),
                                g.moves_remaining_this_turn(), cfg)
    return last


def pk(q, r):
    k = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return k - 0x100000000 if k >= 0x80000000 else k


def to_state(g):
    p1, p2 = [], []
    for (q, r), pl in g.placed_stones():
        (p1 if pl == "P1" else p2).append(pk(q, r))
    return (p1, p2, 0 if g.current_player() == "P1" else 1, g.moves_remaining_this_turn())


def median_ms(fn, dev, reps=50, warmup=12):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    dev = torch.device("cuda")
    pool = [play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]
    batches = {b: [to_state(pool[i % len(pool)]) for i in range(b)] for b in (1, 8, 16)}

    print("## Full-build medians: baseline vs cumprod-unroll patch")
    print(f"{'b':>4} {'baseline':>10} {'unroll':>10} {'speedup':>8}")

    # ---- capture original cumprod usage by patching torch.cumprod inside the
    # module's reach computation. Easiest: monkeypatch torch.cumprod globally
    # to an unrolled version for the last-dim, small-W case.
    orig_cumprod = torch.cumprod

    def fast_cumprod(t, dim):
        if dim in (-1, t.dim() - 1) and t.shape[-1] <= 8:
            out = torch.empty_like(t)
            acc = t[..., 0]
            out[..., 0] = acc
            for d in range(1, t.shape[-1]):
                acc = acc * t[..., d]
                out[..., d] = acc
            return out
        return orig_cumprod(t, dim)

    base = {}
    for b, games in batches.items():
        base[b] = median_ms(lambda g=games: build_slot_batch_from_keys(
            g, CONFIG, device=dev, return_aux=True), dev)

    torch.cumprod = fast_cumprod
    patched = {}
    for b, games in batches.items():
        patched[b] = median_ms(lambda g=games: build_slot_batch_from_keys(
            g, CONFIG, device=dev, return_aux=True), dev)
    torch.cumprod = orig_cumprod

    for b in (1, 8, 16):
        print(f"{b:>4} {base[b]:>10.3f} {patched[b]:>10.3f} {base[b]/patched[b]:>7.2f}x")

    # ---- sub-profile threat features at b=8 ----
    print("\n## threat feature sub-profile (b=8)")
    games = batches[8]
    # reproduce inputs + intermediate qs/rs/sorted_keys/kinds_sorted
    from hexo_a0.slot_graph import _threat_features_batched
    keys_l, counts, tm = [], [], []
    for (p1, p2, cur, mr) in games:
        tm.append(cur); keys_l += p1; keys_l += p2; counts += [len(p1), len(p2)]
    # Just call the real builder once to get shapes, then time threat alone by
    # re-running the batched threat on reconstructed args via a quick build.
    batch, aux = build_slot_batch_from_keys(games, CONFIG, device=dev, return_aux=True)
    # We cannot easily extract qs/rs; instead time threat via patching a flag.
    # Approx: time full build with threat_features on vs off.
    cfg_no_threat = SlotBuilderConfig(WIN, RAD, True, False, True)
    t_on = median_ms(lambda: build_slot_batch_from_keys(games, CONFIG, device=dev, return_aux=True), dev)
    t_off = median_ms(lambda: build_slot_batch_from_keys(games, cfg_no_threat, device=dev, return_aux=True), dev)
    print(f"  build with threat ON : {t_on:.3f} ms")
    print(f"  build with threat OFF: {t_off:.3f} ms")
    print(f"  threat delta         : {t_on - t_off:.3f} ms")
    # and with unroll patch, threat delta:
    torch.cumprod = fast_cumprod
    t_on2 = median_ms(lambda: build_slot_batch_from_keys(games, CONFIG, device=dev, return_aux=True), dev)
    t_off2 = median_ms(lambda: build_slot_batch_from_keys(games, cfg_no_threat, device=dev, return_aux=True), dev)
    torch.cumprod = orig_cumprod
    print(f"  [unroll] threat ON:{t_on2:.3f}  OFF:{t_off2:.3f}  delta:{t_on2-t_off2:.3f} ms")


if __name__ == "__main__":
    main()
