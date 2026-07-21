"""Q1/Q3: decompose the build into (python prep / H2D input / core compute),
and count host<->device syncs, to locate the ~3.5ms hipMemcpyWithStream cost.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_decompose.py
"""
from __future__ import annotations

import random
import statistics
import time

import torch

import hexo_rs
from hexo_a0 import slot_graph as sg
from hexo_a0.slot_graph import SlotBuilderConfig, build_slot_batch_from_keys

WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES = 6, 8, 300
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)
CONFIG = SlotBuilderConfig(WIN_LENGTH, PLACEMENT_RADIUS, True, True, True)


def _play(seed, depth):
    cfg = hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)
    rng = random.Random(seed)
    g = hexo_rs.GameState(cfg)
    last = None
    for _ in range(depth):
        if g.is_terminal():
            break
        g.apply_move(*rng.choice(g.legal_moves()))
        if not g.is_terminal():
            last = g.from_state(g.placed_stones(), g.current_player(),
                                g.moves_remaining_this_turn(), cfg)
    return last


def _pk(q, r):
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def to_state(g):
    p1, p2 = [], []
    for (q, r), pl in g.placed_stones():
        (p1 if pl == "P1" else p2).append(_pk(q, r))
    return (p1, p2, 0 if g.current_player() == "P1" else 1,
            g.moves_remaining_this_turn())


def median_ms(fn, reps=50, warmup=10, dev=None):
    for _ in range(warmup):
        fn()
    if dev and dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if dev and dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


# ---- a copy of the from_keys prep, split so we can time each stage ----
def prep_lists(games):
    keys_l, counts, to_move_l, pf_l, mf_l = [], [], [], [], []
    for (p1, p2, cur, mr) in games:
        to_move_l.append(cur)
        pf_l.append(1.0 if cur == 0 else -1.0)
        mf_l.append(mr / 2.0)
        keys_l.extend(p1)
        keys_l.extend(p2)
        counts.extend((len(p1), len(p2)))
    return keys_l, counts, to_move_l, pf_l, mf_l


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# device={dev}")
    pool = [_play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]

    for b in (1, 8, 16):
        games = [to_state(pool[i % len(pool)]) for i in range(b)]
        print(f"\n=== b={b} ===")

        # (0) full end-to-end
        full = median_ms(lambda: build_slot_batch_from_keys(
            games, CONFIG, device=dev, return_aux=True), dev=dev)
        print(f"full build_slot_batch_from_keys : {full:.3f} ms")

        # (1) just python list prep (CPU only)
        pl = median_ms(lambda: prep_lists(games))
        print(f"  python list prep              : {pl:.3f} ms")

        # (2) H2D of the input tensors (list -> cpu tensor -> device)
        keys_l, counts, tm, pf, mf = prep_lists(games)

        def h2d():
            torch.tensor(keys_l, dtype=torch.int32).to(dev)
            torch.tensor(counts, dtype=torch.int64, device=dev)
            torch.tensor(tm, dtype=torch.int64, device=dev)
            torch.tensor(pf, dtype=torch.float32, device=dev)
            torch.tensor(mf, dtype=torch.float32, device=dev)
        print(f"  H2D input tensor creation     : {median_ms(h2d, dev=dev):.3f} ms")

        # (3) core only, with device tensors pre-built (measures compute+syncs
        #     inside _build_slot_batch_core, no python parse / H2D of lists)
        skey = torch.tensor(keys_l, dtype=torch.int32).to(dev)
        cnt = torch.tensor(counts, dtype=torch.int64, device=dev)
        sgame = torch.repeat_interleave(
            torch.arange(b, dtype=torch.int64, device=dev), cnt[0::2] + cnt[1::2])
        skind = torch.repeat_interleave(
            torch.tensor([0, 1], dtype=torch.int64, device=dev).repeat(b), cnt)
        tmk = torch.tensor(tm, dtype=torch.int64, device=dev)
        pft = torch.tensor(pf, dtype=torch.float32, device=dev)
        mft = torch.tensor(mf, dtype=torch.float32, device=dev)

        def core():
            sg._build_slot_batch_core(sgame, skey, skind, tmk, pft, mft,
                                      b, CONFIG, dev, None, True)
        print(f"  _build_slot_batch_core        : {median_ms(core, dev=dev):.3f} ms")

        # (4) core with a FIXED pad_to (removes n_max data-dep for shape, but
        #     .max() sync still present). Static-shape lower bound-ish.
        def core_pad():
            sg._build_slot_batch_core(sgame, skey, skind, tmk, pft, mft,
                                      b, CONFIG, dev, 2176, True)
        print(f"  core (pad_to=2176)            : {median_ms(core_pad, dev=dev):.3f} ms")


if __name__ == "__main__":
    main()
