"""Q1/Q2 investigation: profile + time the on-device slot-batch build.

Measures build_slot_batch_from_keys at production shapes (win_length=6,
radius=8, threat+relative+prune) on GPU, counts kernel launches via
torch.profiler, and reports a top-N op table + baseline wall ms at b=1/8/16.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_profile.py
"""
from __future__ import annotations

import random
import statistics
import sys
import time

import torch

import hexo_rs
from hexo_a0.slot_graph import (
    SlotBuilderConfig,
    build_slot_batch_from_keys,
    pack,
)

WIN_LENGTH = 6
PLACEMENT_RADIUS = 8
MAX_MOVES = 300
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)
CONFIG = SlotBuilderConfig(
    win_length=WIN_LENGTH,
    placement_radius=PLACEMENT_RADIUS,
    prune_empty_edges=True,
    threat_features=True,
    relative_stones=True,
)


def _cfg():
    return hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)


def _play_to_depth(seed, depth):
    cfg = _cfg()
    rng = random.Random(seed)
    game = hexo_rs.GameState(cfg)
    last = None
    for _ in range(depth):
        if game.is_terminal():
            break
        game.apply_move(*rng.choice(game.legal_moves()))
        if not game.is_terminal():
            last = game.from_state(
                game.placed_stones(), game.current_player(),
                game.moves_remaining_this_turn(), cfg,
            )
    assert last is not None
    return last


def _pack_key(q, r):
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def game_to_keys_state(g):
    p1, p2 = [], []
    for (q, r), player in g.placed_stones():
        (p1 if player == "P1" else p2).append(_pack_key(q, r))
    cur = 0 if g.current_player() == "P1" else 1
    mr = g.moves_remaining_this_turn()
    return (p1, p2, cur, mr)


def build_pool(n):
    return [_play_to_depth(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(n)]


def make_states(pool, b):
    return [game_to_keys_state(pool[i % len(pool)]) for i in range(b)]


def time_build(states, device, reps=50, warmup=10, pad_to=None):
    for _ in range(warmup):
        build_slot_batch_from_keys(states, CONFIG, device=device, return_aux=True, pad_to=pad_to)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        build_slot_batch_from_keys(states, CONFIG, device=device, return_aux=True, pad_to=pad_to)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "median": statistics.median(ts),
        "p10": ts[int(0.10 * len(ts))],
        "p90": ts[min(len(ts) - 1, int(0.90 * len(ts)))],
        "min": ts[0],
    }


def profile_build(states, device, active=20):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(10):
        build_slot_batch_from_keys(states, CONFIG, device=device, return_aux=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(active):
            build_slot_batch_from_keys(states, CONFIG, device=device, return_aux=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return prof, active


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# device={dev} torch={torch.__version__} hip={getattr(torch.version,'hip',None)}")
    print(f"# gpu={torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}")
    print(f"# config: wl={WIN_LENGTH} radius={PLACEMENT_RADIUS} threat+relative+prune")

    pool = build_pool(20)
    stone_counts = [len(g.placed_stones()) for g in pool]
    print(f"# pool stones: min={min(stone_counts)} max={max(stone_counts)} "
          f"mean={sum(stone_counts)/len(stone_counts):.1f}")

    # ---- baseline timing ----
    print("\n## Baseline build wall ms (dynamic shapes, no pad_to)")
    print(f"{'b':>4} {'median':>8} {'p10':>8} {'p90':>8} {'min':>8}")
    for b in (1, 8, 16):
        states = make_states(pool, b)
        # report N (padded node count) for context
        batch, _ = build_slot_batch_from_keys(states, CONFIG, device=dev, return_aux=True)
        n = batch.x.shape[1]
        st = time_build(states, dev)
        print(f"{b:>4} {st['median']:>8.3f} {st['p10']:>8.3f} {st['p90']:>8.3f} "
              f"{st['min']:>8.3f}   (N={n} S={batch.filled.shape[-1]})")

    # ---- kernel-launch profile at b=8 ----
    print("\n## torch.profiler op table at b=8 (20 build iters)")
    states = make_states(pool, 8)
    prof, iters = profile_build(states, dev)
    ka = prof.key_averages()

    # total launches: count of CUDA kernel/runtime events
    ev = prof.events()
    n_cuda_launch = sum(1 for e in ev if getattr(e, "device_type", None) is not None
                        and str(getattr(e, "device_type", "")).endswith("CUDA"))
    print(f"# profiler captured {len(ev)} events over {iters} iters "
          f"(~{len(ev)//iters}/iter)")

    # Sort by self CUDA time if available else self CPU time
    def sortkey(k):
        cu = getattr(k, "self_device_time_total", 0) or getattr(k, "self_cuda_time_total", 0)
        return cu if cu else k.self_cpu_time_total
    rows = sorted(ka, key=sortkey, reverse=True)
    print(f"\n{'op':<40} {'#calls':>7} {'#/iter':>7} {'selfCUDAus':>11} {'selfCPUus':>10}")
    total_calls = 0
    for k in rows[:25]:
        cu = getattr(k, "self_device_time_total", 0) or getattr(k, "self_cuda_time_total", 0)
        total_calls += k.count
        print(f"{k.key[:40]:<40} {k.count:>7} {k.count/iters:>7.1f} "
              f"{cu/1:>11.1f} {k.self_cpu_time_total:>10.1f}")
    # total device launches
    total_device_calls = sum(k.count for k in ka
                             if (getattr(k, "self_device_time_total", 0) or
                                 getattr(k, "self_cuda_time_total", 0)))
    print(f"\n# total op-level calls (all): {sum(k.count for k in ka)} "
          f"(~{sum(k.count for k in ka)//iters}/iter)")
    print(f"# ops with device time (kernels): {total_device_calls} "
          f"(~{total_device_calls//iters}/iter)")

    # aggregate device vs cpu time
    tot_cuda = sum((getattr(k,'self_device_time_total',0) or getattr(k,'self_cuda_time_total',0)) for k in ka)
    tot_cpu = sum(k.self_cpu_time_total for k in ka)
    print(f"# aggregate self CUDA us/iter: {tot_cuda/iters:.1f}  "
          f"self CPU us/iter: {tot_cpu/iters:.1f}")


if __name__ == "__main__":
    main()
