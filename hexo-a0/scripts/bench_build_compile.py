"""Q2/Q3: (a) sub-profile threat internals; (b) test torch.compile on the two
hot regions (walk_reach with cumprod, threat) at FIXED shapes to see if the
inductor/triton backend fuses them and fixes the cumprod pathology.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_compile.py
"""
from __future__ import annotations

import random
import time

import torch

import hexo_rs
from hexo_a0.slot_graph import (
    KIND_EMPTY, KIND_P1, _KEY_SENTINEL, SlotBuilderConfig, WIN_AXES,
    _disk_deltas, _wrap_i32, axis_deltas, pack, unpack,
)

WIN, RAD, MM = 6, 8, 300
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


def timeit(fn, reps=60, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    dev = torch.device("cuda")
    b, n_max = 8, 2053
    window = WIN - 1
    # synthetic but representative tensors at fixed shapes
    torch.manual_seed(0)
    kinds_id = torch.randint(0, 3, (b, n_max), dtype=torch.int8, device=dev)
    present = torch.rand(b, n_max, 3, 2, window, device=dev) > 0.4
    partner = torch.randint(0, n_max, (b, n_max, 3, 2, window), dtype=torch.int64, device=dev)

    def walk_current(present, partner, kinds_id):
        tk = torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
        wk = kinds_id[:, :, None, None, None]
        stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
        cont = (present & ~stop).to(torch.uint8)
        ones = torch.ones_like(cont[..., :1])
        carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
        return present & carry.bool()

    print(f"# b={b} n_max={n_max}")
    print(f"walk_reach eager (cumprod) : {timeit(lambda: walk_current(present, partner, kinds_id)):.3f} ms")

    for mode, kw in [("default", {}), ("max-autotune", {"mode": "max-autotune"})]:
        try:
            cw = torch.compile(walk_current, dynamic=False, **kw)
            t = timeit(lambda: cw(present, partner, kinds_id))
            print(f"walk_reach compiled {mode:<12}: {t:.3f} ms")
        except Exception as e:
            print(f"walk_reach compiled {mode}: FAILED {type(e).__name__}: {str(e)[:80]}")

    # ---- threat: build real qs/rs/sorted_keys/kinds_sorted at fixed shapes ----
    pool = [play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]
    games = [to_state(pool[i % len(pool)]) for i in range(b)]
    # minimal reproduction of the identity/sorted tables
    keys_l, counts, tm = [], [], []
    for (p1, p2, cur, mr) in games:
        tm.append(cur); keys_l += p1; keys_l += p2; counts += [len(p1), len(p2)]
    skey = torch.tensor(keys_l, dtype=torch.int32, device=dev)
    cnt = torch.tensor(counts, dtype=torch.int64, device=dev)
    sgame = torch.repeat_interleave(torch.arange(b, dtype=torch.int64, device=dev), cnt[0::2] + cnt[1::2])
    skind = torch.repeat_interleave(torch.tensor([0, 1], dtype=torch.int64, device=dev).repeat(b), cnt)
    scomb = (sgame << 32) | (skey.to(torch.int64) + 0x8000_0000)
    order = torch.argsort(scomb); scomb_s = scomb[order]; skey_s = skey[order]
    skind_s = skind[order]; sgame_s = sgame[order]
    disk = _disk_deltas(RAD, dev)
    cand_keys = _wrap_i32(skey_s.to(torch.int64)[:, None] + disk.to(torch.int64)[None, :])
    cand_comb = (sgame_s[:, None] << 32) | (cand_keys.to(torch.int64) + 0x8000_0000)
    ucomb = torch.unique(cand_comb.reshape(-1))
    nst = skey.shape[0]
    pos = torch.searchsorted(scomb_s, ucomb)
    occ = (pos < nst) & (scomb_s[pos.clamp(max=nst - 1)] == ucomb)
    lcomb = ucomb[~occ]; lgame = lcomb >> 32
    lkey = ((lcomb & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)
    ns = torch.bincount(sgame_s, minlength=b); nl = torch.bincount(lgame, minlength=b)
    n_real = ns + nl; nm = int(n_real.max())
    zero = torch.zeros(1, dtype=torch.int64, device=dev)
    scum = torch.cat([zero, ns.cumsum(0)]); lcum = torch.cat([zero, nl.cumsum(0)])
    spos = torch.arange(nst, device=dev) - scum[sgame_s]
    lpos = ns[lgame] + torch.arange(lgame.shape[0], device=dev) - lcum[lgame]
    keys_id = torch.full((b, nm), _KEY_SENTINEL, dtype=torch.int32, device=dev)
    kinds_id2 = torch.full((b, nm), KIND_EMPTY, dtype=torch.int8, device=dev)
    keys_id[sgame_s, spos] = skey_s; keys_id[lgame, lpos] = lkey
    kinds_id2[sgame_s, spos] = skind_s.to(torch.int8)
    node_mask = torch.arange(nm, device=dev)[None, :] < n_real[:, None]
    qs, rs = unpack(keys_id)
    qs = torch.where(node_mask, qs, torch.zeros_like(qs))
    rs = torch.where(node_mask, rs, torch.zeros_like(rs))
    sorted_keys, sort_idx = torch.sort(keys_id, dim=1)
    kinds_sorted = torch.gather(kinds_id2, 1, sort_idx)
    to_move_kind = torch.tensor(tm, dtype=torch.int64, device=dev)

    def threat(qs, rs, sorted_keys, kinds_sorted, to_move_kind):
        wl = WIN; bb, nn = qs.shape
        opp_kind = 1 - to_move_kind
        own_max = torch.zeros((bb, nn), device=dev); opp_max = torch.zeros((bb, nn), device=dev)
        own_axes = torch.zeros((bb, nn), device=dev); opp_axes = torch.zeros((bb, nn), device=dev)
        ks = torch.arange(-(wl - 1), wl, device=dev)
        kinds64 = kinds_sorted.to(torch.int64)
        for dq, dr in WIN_AXES:
            cq = qs[:, :, None] + ks[None, None, :] * dq
            cr = rs[:, :, None] + ks[None, None, :] * dr
            cand = pack(cq, cr); k = cand.shape[-1]
            pos = torch.searchsorted(sorted_keys, cand.reshape(bb, -1)).clamp(max=nn - 1)
            present = torch.gather(sorted_keys, 1, pos).reshape(bb, nn, k) == cand
            ck = torch.gather(kinds64, 1, pos).reshape(bb, nn, k)
            cell_kind = torch.where(present, ck, torch.full_like(ck, KIND_EMPTY))
            is_own = cell_kind == to_move_kind[:, None, None]
            is_opp = cell_kind == opp_kind[:, None, None]
            axis_own = torch.zeros((bb, nn), device=dev); axis_opp = torch.zeros((bb, nn), device=dev)
            for start in range(wl):
                win_own = is_own[:, :, start:start + wl].sum(dim=-1).float()
                win_opp = is_opp[:, :, start:start + wl].sum(dim=-1).float()
                axis_own = torch.where(win_opp == 0, torch.maximum(axis_own, win_own), axis_own)
                axis_opp = torch.where(win_own == 0, torch.maximum(axis_opp, win_opp), axis_opp)
            own_max = torch.maximum(own_max, axis_own); opp_max = torch.maximum(opp_max, axis_opp)
            own_axes = own_axes + (axis_own >= wl - 2).float(); opp_axes = opp_axes + (axis_opp >= wl - 2).float()
        return torch.stack([own_max / wl, opp_max / wl, own_axes / 3.0, opp_axes / 3.0], dim=2)

    print(f"\nthreat eager             : {timeit(lambda: threat(qs, rs, sorted_keys, kinds_sorted, to_move_kind)):.3f} ms")
    for mode, kw in [("default", {}), ("max-autotune", {"mode": "max-autotune"})]:
        try:
            ct = torch.compile(threat, dynamic=False, **kw)
            t = timeit(lambda: ct(qs, rs, sorted_keys, kinds_sorted, to_move_kind))
            print(f"threat compiled {mode:<12} : {t:.3f} ms")
        except Exception as e:
            print(f"threat compiled {mode}: FAILED {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
