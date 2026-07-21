"""Q3/Q5: confirm cumprod-unroll and torch.compile(core) produce BIT-IDENTICAL
SlotBatch outputs vs eager baseline (correctness gate for the proposals).

Usage: uv run --no-sync python hexo-a0/scripts/bench_build_correctness.py
"""
from __future__ import annotations

import random

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


def cmp(ref, other, n, tag):
    ok = True
    for f in ("x", "dummy_x", "partner", "filled", "src_player", "node_mask",
              "stone_mask", "legal_mask"):
        ta = getattr(ref, f)
        tb = getattr(other, f)
        if f != "dummy_x":
            tb = tb[:, :n]  # slice off padding rows
        if ta.dtype.is_floating_point:
            eq = torch.equal(ta, tb)
        else:
            eq = torch.equal(ta, tb)
        if not eq:
            ok = False
            d = (ta.float() - tb.float()).abs().max().item()
            print(f"    {f}: DIFF maxabs={d}")
    print(f"  [{tag}] {'BIT-IDENTICAL' if ok else 'MISMATCH'}")


def main():
    dev = torch.device("cuda")
    pool = [play(3000 + i, DEPTHS[i % len(DEPTHS)]) for i in range(20)]

    orig_cumprod = torch.cumprod
    def fast_cumprod(t, dim):
        if dim in (-1, t.dim() - 1) and t.shape[-1] <= 8:
            out = torch.empty_like(t); acc = t[..., 0]; out[..., 0] = acc
            for d in range(1, t.shape[-1]):
                acc = acc * t[..., d]; out[..., d] = acc
            return out
        return orig_cumprod(t, dim)

    for b in (1, 8, 16):
        games = [to_state(pool[i % len(pool)]) for i in range(b)]
        print(f"\n=== b={b} ===")
        ref = build_slot_batch_from_keys(games, CONFIG, device=dev)
        n = ref.x.shape[1]

        torch.cumprod = fast_cumprod
        u = build_slot_batch_from_keys(games, CONFIG, device=dev)
        torch.cumprod = orig_cumprod
        cmp(ref, u, n, "cumprod-unroll (no pad)")

    # compiled core, separate loop (compile once, reuse)
    print("\n=== compiled core (pad_to=2432) ===")
    orig = sg._build_slot_batch_core
    torch._dynamo.reset()
    sg._build_slot_batch_core = torch.compile(orig, dynamic=False)
    try:
        for b in (8, 16):
            games = [to_state(pool[i % len(pool)]) for i in range(b)]
            ref = build_slot_batch_from_keys(games, CONFIG, device=dev)  # eager (patched back below? no)
            # ref uses compiled now; recompute eager ref by temporarily restoring
            sg._build_slot_batch_core = orig
            ref = build_slot_batch_from_keys(games, CONFIG, device=dev)
            sg._build_slot_batch_core = torch.compile(orig, dynamic=False)
            n = ref.x.shape[1]
            c = build_slot_batch_from_keys(games, CONFIG, device=dev, pad_to=2432)
            cmp(ref, c, n, f"compiled b={b}")
    finally:
        sg._build_slot_batch_core = orig
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
