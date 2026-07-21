"""Q2/Q5: end-to-end build time with torch.compile applied to the core.

Two approaches:
 (A) compile _build_slot_batch_core whole, mode=default (graph breaks allowed
     at unique/nonzero/.item), fixed pad_to bucket.
 (B) monkeypatch the two hot regions (reach-scan + threat) with compiled
     fixed-shape helpers, leaving the dynamic prep eager.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_compile_e2e.py
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
# pad buckets: mult-of-128 ceilings that cover the observed n_max (2053/2128)
PAD = {1: 1280, 8: 2176, 16: 2176}


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


def median_ms(fn, reps=50, warmup=15):
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

    # baseline
    print("## baseline (no compile, dynamic shapes)")
    base = {}
    for b, games in batches.items():
        base[b] = median_ms(lambda g=games: build_slot_batch_from_keys(g, CONFIG, device=dev, return_aux=True))
        print(f"  b={b:>2}: {base[b]:.3f} ms")

    # baseline with fixed pad_to (bucketed)
    print("## baseline + fixed pad_to bucket")
    for b, games in batches.items():
        t = median_ms(lambda g=games, p=PAD[b]: build_slot_batch_from_keys(g, CONFIG, device=dev, return_aux=True, pad_to=p))
        print(f"  b={b:>2} pad={PAD[b]}: {t:.3f} ms")

    # ---- Approach A: compile whole core, mode=default, fixed pad_to ----
    print("\n## Approach A: torch.compile(core, mode=default) + fixed pad_to")
    orig_core = sg._build_slot_batch_core
    try:
        torch._dynamo.reset()
        compiled_core = torch.compile(orig_core, dynamic=False)
        sg._build_slot_batch_core = compiled_core
        for b, games in batches.items():
            try:
                t = median_ms(lambda g=games, p=PAD[b]: build_slot_batch_from_keys(
                    g, CONFIG, device=dev, return_aux=True, pad_to=p), reps=40, warmup=20)
                print(f"  b={b:>2} pad={PAD[b]}: {t:.3f} ms  (speedup {base[b]/t:.2f}x)")
            except Exception as e:
                print(f"  b={b:>2}: FAILED {type(e).__name__}: {str(e)[:100]}")
    finally:
        sg._build_slot_batch_core = orig_core
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
