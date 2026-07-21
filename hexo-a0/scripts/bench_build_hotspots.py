"""Q3: sub-instrument edge_walk_reach + threat, and test cheap rewrites
(int32 partner, contiguous reshape, dtype). Also test 'no-sync' variant to
prove syncs are/aren't the bottleneck.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_hotspots.py
"""
from __future__ import annotations

import random
import time

import torch

import hexo_rs
from hexo_a0.slot_graph import (
    KIND_EMPTY, KIND_P1, _KEY_SENTINEL, SlotBuilderConfig,
    _disk_deltas, _wrap_i32, axis_deltas, unpack,
)

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


def build_inputs(pool, b, dev):
    games = [to_state(pool[i % len(pool)]) for i in range(b)]
    keys_l, counts = [], []
    for (p1, p2, cur, mr) in games:
        keys_l += p1; keys_l += p2; counts += [len(p1), len(p2)]
    skey = torch.tensor(keys_l, dtype=torch.int32).to(dev)
    cnt = torch.tensor(counts, dtype=torch.int64, device=dev)
    sgame = torch.repeat_interleave(torch.arange(b, dtype=torch.int64, device=dev),
                                    cnt[0::2] + cnt[1::2])
    skind = torch.repeat_interleave(
        torch.tensor([0, 1], dtype=torch.int64, device=dev).repeat(b), cnt)
    return sgame, skey, skind


def prep_upto_edges(sgame, skey, skind, b, dev):
    """Reproduce the build up to the point edge slots start; return the
    tensors edge_walk needs (keys_id, kinds_id, node_mask, sorted_keys,
    sort_idx, n_max)."""
    wl = WIN; window = wl - 1; radius = RAD
    scomb = (sgame << 32) | (skey.to(torch.int64) + 0x8000_0000)
    order = torch.argsort(scomb)
    scomb_s = scomb[order]; skey_s = skey[order]; skind_s = skind[order]; sgame_s = sgame[order]
    disk = _disk_deltas(radius, dev)
    cand_keys = _wrap_i32(skey_s.to(torch.int64)[:, None] + disk.to(torch.int64)[None, :])
    cand_comb = (sgame_s[:, None] << 32) | (cand_keys.to(torch.int64) + 0x8000_0000)
    ucomb = torch.unique(cand_comb.reshape(-1))
    n_stones_total = skey.shape[0]
    pos = torch.searchsorted(scomb_s, ucomb)
    occupied = (pos < n_stones_total) & (scomb_s[pos.clamp(max=n_stones_total - 1)] == ucomb)
    lcomb = ucomb[~occupied]
    lgame = lcomb >> 32
    lkey = ((lcomb & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)
    ns = torch.bincount(sgame_s, minlength=b); nl = torch.bincount(lgame, minlength=b)
    n_real = ns + nl; n_max = int(n_real.max())
    zero = torch.zeros(1, dtype=torch.int64, device=dev)
    scum = torch.cat([zero, ns.cumsum(0)]); lcum = torch.cat([zero, nl.cumsum(0)])
    spos = torch.arange(n_stones_total, device=dev) - scum[sgame_s]
    lpos = ns[lgame] + torch.arange(lgame.shape[0], device=dev) - lcum[lgame]
    keys_id = torch.full((b, n_max), _KEY_SENTINEL, dtype=torch.int32, device=dev)
    kinds_id = torch.full((b, n_max), KIND_EMPTY, dtype=torch.int8, device=dev)
    keys_id[sgame_s, spos] = skey_s
    keys_id[lgame, lpos] = lkey
    kinds_id[sgame_s, spos] = skind_s.to(torch.int8)
    ar = torch.arange(n_max, device=dev)
    node_mask = ar[None, :] < n_real[:, None]
    sorted_keys, sort_idx = torch.sort(keys_id, dim=1)
    return keys_id, kinds_id, node_mask, sorted_keys, sort_idx, n_max, window


def timeit(fn, dev, reps=60, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    dev = torch.device("cuda")
    pool = [play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]
    for b in (8, 16):
        sgame, skey, skind = build_inputs(pool, b, dev)
        keys_id, kinds_id, node_mask, sorted_keys, sort_idx, n_max, window = \
            prep_upto_edges(sgame, skey, skind, b, dev)
        print(f"\n=== b={b} n_max={n_max} S={3*2*window} elems={b*n_max*3*2*window} ===")

        deltas = axis_deltas(WIN, dev)

        def edge_cand():
            cand = _wrap_i32(keys_id.to(torch.int64)[:, :, None, None, None]
                             + deltas.to(torch.int64)[None, None])
            cand2 = cand.reshape(b, -1)
            pos = torch.searchsorted(sorted_keys, cand2)
            pos_c = pos.clamp(max=n_max - 1)
            present = (pos < n_max) & (torch.gather(sorted_keys, 1, pos_c) == cand2)
            present = present.reshape(b, n_max, 3, 2, window) & node_mask[:, :, None, None, None]
            partner = torch.where(present,
                                  torch.gather(sort_idx, 1, pos_c).reshape(b, n_max, 3, 2, window),
                                  torch.zeros((), dtype=torch.int64, device=dev))
            return present, partner
        print(f"  edge_cand+searchsorted     : {timeit(edge_cand, dev):.3f} ms")

        present, partner = edge_cand()

        # --- current edge_walk_reach ---
        def walk_current():
            tk = torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
            wk = kinds_id[:, :, None, None, None]
            stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
            cont = (present & ~stop).to(torch.uint8)
            ones = torch.ones_like(cont[..., :1])
            carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
            reach = present & carry.bool()
            return reach
        print(f"  walk_reach (current int64) : {timeit(walk_current, dev):.3f} ms")

        # sub-parts
        def sub_gather():
            return torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
        print(f"    - gather tk              : {timeit(sub_gather, dev):.3f} ms")
        tk = sub_gather()
        wk = kinds_id[:, :, None, None, None]
        def sub_stop():
            return present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
        print(f"    - stop (where)           : {timeit(sub_stop, dev):.3f} ms")
        stop = sub_stop()
        def sub_cumprod():
            cont = (present & ~stop).to(torch.uint8)
            ones = torch.ones_like(cont[..., :1])
            return torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
        print(f"    - cumprod+cat            : {timeit(sub_cumprod, dev):.3f} ms")

        # --- int32 partner variant ---
        partner32 = partner.to(torch.int32)
        def walk_int32():
            tk = torch.gather(kinds_id, 1, partner32.reshape(b, -1).to(torch.int64)).reshape(partner.shape)
            wk = kinds_id[:, :, None, None, None]
            stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
            cont = (present & ~stop).to(torch.uint8)
            ones = torch.ones_like(cont[..., :1])
            carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
            return present & carry.bool()
        print(f"  walk_reach (int32 partner) : {timeit(walk_int32, dev):.3f} ms")

        # --- contiguous partner (avoid reshape copies) ---
        partner_c = partner.contiguous()
        def walk_contig():
            pflat = partner_c.reshape(b, -1)
            tk = torch.gather(kinds_id, 1, pflat).reshape(partner.shape)
            wk = kinds_id[:, :, None, None, None]
            stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
            cont = (present & ~stop).to(torch.uint8)
            ones = torch.ones_like(cont[..., :1])
            carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
            return present & carry.bool()
        print(f"  walk_reach (contig)        : {timeit(walk_contig, dev):.3f} ms")

        # --- manual cumprod over W=5 (unrolled, no cat) ---
        def walk_unroll():
            tk = torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
            wk = kinds_id[:, :, None, None, None]
            stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
            cont = present & ~stop
            # reach[d] = cont[0..d-1] all true; unroll W
            reach = torch.empty_like(cont)
            acc = torch.ones_like(cont[..., 0])
            for d in range(cont.shape[-1]):
                reach[..., d] = present[..., d] & acc
                acc = acc & cont[..., d]
            return reach
        print(f"  walk_reach (unrolled W)    : {timeit(walk_unroll, dev):.3f} ms")


if __name__ == "__main__":
    main()
