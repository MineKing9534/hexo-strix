"""Q5: scale baseline vs compiled(core)+pad_to across b up to 256.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_large.py
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
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36, 18, 22, 26, 30, 33, 38)
BATCHES = (1, 8, 16, 32, 64, 128, 256)


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


def median_ms(fn, reps=30, warmup=12):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def ceil128(n):
    return ((n + 127) // 128) * 128


def main():
    dev = torch.device("cuda")
    pool = [play(4000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(300)]
    batches = {b: [to_state(pool[i % len(pool)]) for i in range(b)] for b in BATCHES}

    # observe n_max per batch and choose pad bucket
    pad = {}
    print(f"{'b':>4} {'n_max':>7} {'pad':>6}")
    for b, games in batches.items():
        batch, _ = build_slot_batch_from_keys(games, CONFIG, device=dev, return_aux=True)
        nm = batch.x.shape[1]
        pad[b] = ceil128(nm + 64)  # headroom
        print(f"{b:>4} {nm:>7} {pad[b]:>6}")

    print("\n## baseline (no compile)")
    base = {}
    for b, games in batches.items():
        try:
            base[b] = median_ms(lambda g=games: build_slot_batch_from_keys(g, CONFIG, device=dev, return_aux=True))
            gps = b / (base[b] / 1e3)
            print(f"  b={b:>3}: {base[b]:8.3f} ms   ({gps:8.0f} graphs/s)")
        except RuntimeError as e:
            base[b] = None
            print(f"  b={b:>3}: OOM/ERR {str(e)[:60]}")

    print("\n## compiled(core, default) + fixed pad_to bucket")
    orig = sg._build_slot_batch_core
    torch._dynamo.reset()
    sg._build_slot_batch_core = torch.compile(orig, dynamic=False)
    try:
        for b, games in batches.items():
            try:
                t = median_ms(lambda g=games, p=pad[b]: build_slot_batch_from_keys(
                    g, CONFIG, device=dev, return_aux=True, pad_to=p), reps=25, warmup=18)
                gps = b / (t / 1e3)
                sp = f"{base[b]/t:.2f}x" if base.get(b) else "n/a"
                print(f"  b={b:>3} pad={pad[b]:>4}: {t:8.3f} ms   ({gps:8.0f} graphs/s)  {sp}")
            except RuntimeError as e:
                print(f"  b={b:>3}: OOM/ERR {str(e)[:70]}")
                torch._dynamo.reset()
                sg._build_slot_batch_core = torch.compile(orig, dynamic=False)
    finally:
        sg._build_slot_batch_core = orig
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
