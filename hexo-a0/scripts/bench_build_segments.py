"""Q1/Q3: instrumented copy of _build_slot_batch_core with per-segment
sync'd timers, to attribute the ~6ms (b=8) to code regions. Segments mirror
the real function; only timing calls are added.

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_segments.py
"""
from __future__ import annotations

import random
import statistics
import time

import torch

import hexo_rs
from hexo_a0 import slot_graph as sg
from hexo_a0.slot_graph import (
    KIND_EMPTY, KIND_P1, KIND_P2, _KEY_SENTINEL, SlotBuilderConfig,
    _disk_deltas, _node_layout, _threat_features_batched, _wrap_i32,
    axis_deltas, unpack,
)

WIN, RAD, MM = 6, 8, 300
CONFIG = SlotBuilderConfig(WIN, RAD, True, True, True)
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)


def play(seed, depth):
    cfg = hexo_rs.GameConfig(WIN, RAD, MM)
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


def pk(q, r):
    k = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return k - 0x100000000 if k >= 0x80000000 else k


def to_state(g):
    p1, p2 = [], []
    for (q, r), pl in g.placed_stones():
        (p1 if pl == "P1" else p2).append(pk(q, r))
    return (p1, p2, 0 if g.current_player() == "P1" else 1,
            g.moves_remaining_this_turn())


class Timer:
    def __init__(self, dev):
        self.dev = dev
        self.acc = {}
        self.t0 = None
        self.label = None

    def tick(self, label):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self.label is not None:
            self.acc[self.label] = self.acc.get(self.label, 0.0) + (now - self.t0)
        self.label = label
        self.t0 = now

    def stop(self):
        self.tick(None)


def core_instrumented(sgame, skey, skind, to_move_kind, player_feat, moves_feat,
                      b, config, device, T):
    wl = config.win_length
    window = wl - 1
    radius = config.placement_radius
    n_stones_total = skey.shape[0]

    T.tick("sort_stones")
    scomb = (sgame << 32) | (skey.to(torch.int64) + 0x8000_0000)
    order = torch.argsort(scomb)
    scomb_s = scomb[order]
    skey_s = skey[order]
    skind_s = skind[order]
    sgame_s = sgame[order]
    sq_s, sr_s = unpack(skey_s)

    T.tick("dup_check[SYNC]")
    if n_stones_total > 1 and bool((scomb_s[1:] == scomb_s[:-1]).any()):
        raise ValueError("dup")

    T.tick("legal_disk_expand")
    disk = _disk_deltas(radius, device)
    cand_keys = _wrap_i32(skey_s.to(torch.int64)[:, None] + disk.to(torch.int64)[None, :])
    cand_comb = (sgame_s[:, None] << 32) | (cand_keys.to(torch.int64) + 0x8000_0000)

    T.tick("unique[SYNC]")
    ucomb = torch.unique(cand_comb.reshape(-1))

    T.tick("legal_searchsorted+boolidx")
    pos = torch.searchsorted(scomb_s, ucomb)
    occupied = (pos < n_stones_total) & (
        scomb_s[pos.clamp(max=n_stones_total - 1)] == ucomb)
    lcomb = ucomb[~occupied]  # boolean index
    lgame = lcomb >> 32
    lkey = ((lcomb & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)

    T.tick("counts+nmax[SYNC]")
    ns = torch.bincount(sgame_s, minlength=b)
    nl = torch.bincount(lgame, minlength=b)
    n_real = ns + nl
    n_max = int(n_real.max())
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    scum = torch.cat([zero, ns.cumsum(0)])
    lcum = torch.cat([zero, nl.cumsum(0)])
    spos = torch.arange(n_stones_total, device=device) - scum[sgame_s]
    lpos = ns[lgame] + torch.arange(lgame.shape[0], device=device) - lcum[lgame]

    T.tick("scatter_id_tables")
    keys_id = torch.full((b, n_max), _KEY_SENTINEL, dtype=torch.int32, device=device)
    kinds_id = torch.full((b, n_max), KIND_EMPTY, dtype=torch.int8, device=device)
    keys_id[sgame_s, spos] = skey_s
    keys_id[lgame, lpos] = lkey
    kinds_id[sgame_s, spos] = skind_s.to(torch.int8)
    ar = torch.arange(n_max, device=device)
    node_mask = ar[None, :] < n_real[:, None]
    stone_mask = ar[None, :] < ns[:, None]
    legal_mask = node_mask & ~stone_mask
    qs, rs = unpack(keys_id)
    qs = torch.where(node_mask, qs, torch.zeros_like(qs))
    rs = torch.where(node_mask, rs, torch.zeros_like(rs))

    T.tick("sort_lookup")
    sorted_keys, sort_idx = torch.sort(keys_id, dim=1)
    kinds_sorted = torch.gather(kinds_id, 1, sort_idx)

    T.tick("edge_cand+searchsorted")
    deltas = axis_deltas(wl, device)
    cand = _wrap_i32(keys_id.to(torch.int64)[:, :, None, None, None]
                     + deltas.to(torch.int64)[None, None])
    s_flat = 3 * 2 * window
    cand2 = cand.reshape(b, -1)
    pos = torch.searchsorted(sorted_keys, cand2)
    pos_c = pos.clamp(max=n_max - 1)
    present = (pos < n_max) & (torch.gather(sorted_keys, 1, pos_c) == cand2)
    present = present.reshape(b, n_max, 3, 2, window) & node_mask[:, :, None, None, None]
    partner = torch.where(present,
                          torch.gather(sort_idx, 1, pos_c).reshape(b, n_max, 3, 2, window),
                          torch.zeros((), dtype=torch.int64, device=device))

    T.tick("edge_walk_reach")
    tk = torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
    wk = kinds_id[:, :, None, None, None]
    stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
    cont = (present & ~stop).to(torch.uint8)
    ones = torch.ones_like(cont[..., :1])
    carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
    reach = present & carry.bool()

    T.tick("edge_mirror_union")
    reach_flip = reach.flip(dims=[3]).reshape(b, n_max, s_flat)
    mirror = torch.gather(reach_flip, 1, partner.reshape(b, n_max, s_flat))
    filled = reach | (present & mirror.reshape(partner.shape))
    if config.prune_empty_edges:
        filled = filled & ~((wk == KIND_EMPTY) & (tk == KIND_EMPTY))

    T.tick("feat_onehot[SYNC boolidx]")
    layout = _node_layout(config.relative_stones, config.node_coords,
                          config.moves_scope, config.compact_stone_onehot)
    base_dim = layout["base_dim"]
    fdim = base_dim + (4 if config.threat_features else 0)
    x = torch.zeros((b, n_max, fdim), dtype=torch.float32, device=device)
    fzero = torch.zeros((), dtype=torch.float32, device=device)
    own_is_p1 = player_feat > 0.0
    if config.relative_stones:
        to_own = (skind_s == KIND_P1) == own_is_p1[sgame_s]
    else:
        to_own = skind_s == KIND_P1
    x[sgame_s[to_own], spos[to_own], layout["own"]] = 1.0
    x[sgame_s[~to_own], spos[~to_own], layout["opp"]] = 1.0
    if layout["empty"] is not None:
        x[lgame, lpos, layout["empty"]] = 1.0
    if layout["to_move"] is not None:
        x[:, :, layout["to_move"]] = torch.where(node_mask, player_feat[:, None], fzero)
    if layout["moves"] is not None:
        x[:, :, layout["moves"]] = torch.where(node_mask, moves_feat[:, None], fzero)

    T.tick("feat_centroid[SYNC smax]")
    if config.node_coords:
        sumq = torch.zeros(b, dtype=torch.float64, device=device)
        sumr = torch.zeros(b, dtype=torch.float64, device=device)
        sumq.index_add_(0, sgame_s, sq_s.double())
        sumr.index_add_(0, sgame_s, sr_s.double())
        cq = sumq / ns.double()
        cr = sumr / ns.double()
        dev_ = torch.maximum((sq_s.double() - cq[sgame_s]).abs(),
                             (sr_s.double() - cr[sgame_s]).abs())
        spread = torch.zeros(b, dtype=torch.float64, device=device)
        spread.scatter_reduce_(0, sgame_s, dev_, reduce="amax", include_self=False)
        spread = spread.clamp(min=1.0)
        norm_q = ((qs.double() - cq[:, None]) / spread[:, None]).float()
        norm_r = ((rs.double() - cr[:, None]) / spread[:, None]).float()
        x[:, :, layout["norm_q"]] = torch.where(node_mask, norm_q, fzero)
        x[:, :, layout["norm_r"]] = torch.where(node_mask, norm_r, fzero)

    T.tick("feat_invdist[SYNC smax]")
    s_max = int(ns.max())
    stq = torch.zeros((b, s_max), dtype=torch.float32, device=device)
    str_ = torch.zeros((b, s_max), dtype=torch.float32, device=device)
    stq[sgame_s, spos] = sq_s.float()
    str_[sgame_s, spos] = sr_s.float()
    stone_valid = torch.arange(s_max, device=device)[None, :] < ns[:, None]
    dq = qs.float()[:, :, None] - stq[:, None, :]
    dr = rs.float()[:, :, None] - str_[:, None, :]
    hexd = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
    hexd = torch.where(stone_valid[:, None, :], hexd, torch.full_like(hexd, torch.inf))
    min_d = hexd.min(dim=2).values.clamp(min=1.0)
    x[:, :, layout["inv_dist"]] = torch.where(legal_mask, 1.0 / min_d, fzero)

    T.tick("feat_threat")
    if config.threat_features:
        tf = _threat_features_batched(qs, rs, sorted_keys, kinds_sorted, to_move_kind, wl)
        x[:, :, base_dim:base_dim + 4] = torch.where(node_mask[:, :, None], tf.float(), fzero)

    T.tick("finalize")
    src_player = (kinds_id == KIND_P1).float() - (kinds_id == KIND_P2).float()
    T.stop()
    return x


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# device={dev}")
    pool = [play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]

    for b in (1, 8, 16):
        games = [to_state(pool[i % len(pool)]) for i in range(b)]
        keys_l, counts, tm, pf, mf = [], [], [], [], []
        for (p1, p2, cur, mr) in games:
            tm.append(cur); pf.append(1.0 if cur == 0 else -1.0); mf.append(mr / 2.0)
            keys_l += p1; keys_l += p2; counts += [len(p1), len(p2)]
        skey = torch.tensor(keys_l, dtype=torch.int32).to(dev)
        cnt = torch.tensor(counts, dtype=torch.int64, device=dev)
        sgame = torch.repeat_interleave(torch.arange(b, dtype=torch.int64, device=dev),
                                        cnt[0::2] + cnt[1::2])
        skind = torch.repeat_interleave(
            torch.tensor([0, 1], dtype=torch.int64, device=dev).repeat(b), cnt)
        tmk = torch.tensor(tm, dtype=torch.int64, device=dev)
        pft = torch.tensor(pf, dtype=torch.float32, device=dev)
        mft = torch.tensor(mf, dtype=torch.float32, device=dev)

        # warmup + accumulate over reps
        T = Timer(dev)
        REPS = 40
        for _ in range(10):
            core_instrumented(sgame, skey, skind, tmk, pft, mft, b, CONFIG, dev, Timer(dev))
        for _ in range(REPS):
            core_instrumented(sgame, skey, skind, tmk, pft, mft, b, CONFIG, dev, T)
        total = sum(T.acc.values())
        print(f"\n=== b={b}  total={total/REPS*1e3:.3f} ms/build ===")
        for label, t in sorted(T.acc.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<28} {t/REPS*1e3:>7.3f} ms  ({100*t/total:>4.1f}%)")


if __name__ == "__main__":
    main()
