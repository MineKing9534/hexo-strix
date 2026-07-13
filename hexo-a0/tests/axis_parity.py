"""Parity test: vectorized axis-window edge building on packed u32 keys
vs a faithful port of hexo-strix build_axis_graph (axis_graph.rs) walk semantics.

Reference semantics (from axis_graph.rs):
  - For each real node i, 3 axes x 2 signs, walk d = 1..=window.
  - Missing cell (off-graph) -> ray stops, no edge.
  - Edge emitted BEFORE the stop check (so edges to the stopping node exist).
  - prune_empty_edges: skip emission for empty->empty pairs, but keep walking.
  - Stop rules: stone walker stops at opponent stone; empty walker stops at ANY stone.
  - Both directions (i->j, j->i) emitted at discovery; dedup by (src, dst, axis).
  => final directed edge set = union over both endpoints' walks.
"""
import numpy as np
import time

AXES = [(1, 0), (0, 1), (1, -1)]
WINDOW = 5  # win_length - 1

# ---------------- packed u32 keys ----------------
def pack(qs, rs):
    qs = np.asarray(qs); rs = np.asarray(rs)
    return (((qs.astype(np.int64) + 0x8000).astype(np.uint32)) << np.uint32(16)) | \
            ((rs.astype(np.int64) + 0x8000).astype(np.uint32))

def make_deltas():
    """deltas[a, s, d-1] = packed offset for s*d steps along axis a (mod 2^32)."""
    out = np.zeros((3, 2, WINDOW), dtype=np.uint32)
    for a, (dq, dr) in enumerate(AXES):
        for si, s in enumerate((1, -1)):
            for d in range(1, WINDOW + 1):
                out[a, si, d - 1] = np.uint32(((s * d * dq) << 16) + (s * d * dr) & 0xFFFFFFFF)
    return out

# kind codes: 0 = P1 stone, 1 = P2 stone, 2 = empty
def stop_rule(walker_kind, target_kind):
    if walker_kind == 2:
        return target_kind != 2            # empty stops at any stone
    return target_kind == (1 - walker_kind)  # stone stops at opponent stone

# ---------------- reference: faithful port of the Rust loop ----------------
def edges_reference(coords, kinds, prune):
    idx_of = {c: i for i, c in enumerate(coords)}
    emitted, seen = set(), set()
    for i, (iq, ir) in enumerate(coords):
        ik = kinds[i]
        for a, (dq, dr) in enumerate(AXES):
            for s in (1, -1):
                for d in range(1, WINDOW + 1):
                    t = (iq + s * d * dq, ir + s * d * dr)
                    j = idx_of.get(t)
                    if j is None:
                        break
                    jk = kinds[j]
                    if not (prune and ik == 2 and jk == 2):
                        for (src, dst, sd) in ((i, j, s * d), (j, i, -s * d)):
                            k = (src, dst, a)
                            if k not in seen:
                                seen.add(k)
                                emitted.add((coords[src], coords[dst], a, sd))
                    if stop_rule(ik, jk):
                        break
    return emitted

# ---------------- vectorized: searchsorted + cumprod + mirror gather ----------------
def edges_vectorized(coords, kinds, prune):
    qs = np.array([c[0] for c in coords]); rs = np.array([c[1] for c in coords])
    keys = pack(qs, rs)
    order = np.argsort(keys)                    # u64/u32 sort == (q, r) lexicographic
    keys_s = keys[order]
    kinds_s = np.asarray(kinds)[order]
    N = len(keys_s)

    deltas = make_deltas()                                     # (3, 2, W)
    cand = keys_s[:, None, None, None] + deltas[None]          # (N, 3, 2, W), wraps mod 2^32
    pos = np.searchsorted(keys_s, cand)
    present = (pos < N) & (keys_s[np.minimum(pos, N - 1)] == cand)
    partner = np.where(present, pos, 0)                        # sorted-index of target
    tk = kinds_s[partner]                                      # target kinds (garbage where !present)

    wk = kinds_s[:, None, None, None]                          # walker kind, broadcast
    stop = present & np.where(wk == 2, tk != 2, tk == (1 - wk))

    # alive[d]: ray from this node reaches step d
    cont = present & ~stop                                     # may continue PAST step d
    carry = np.cumprod(np.concatenate(
        [np.ones_like(cont[..., :1]), cont[..., :-1]], axis=-1).astype(np.uint8), axis=-1)
    reach = present & carry.astype(bool)                       # (N, 3, 2, W)

    # union of both endpoints' walks: slot (i, a, s, d) also fills if the
    # partner's mirrored slot (j, a, -s, d) reached i.
    mirror = reach[partner, np.arange(3)[None, :, None, None], 1 - np.arange(2)[None, None, :, None],
                   np.arange(WINDOW)[None, None, None, :]]
    filled = reach | (present & mirror)

    if prune:
        filled &= ~((wk == 2) & (tk == 2))

    # emit per-destination in-edges: src = partner at slot (a, s, d) sits at +s*d
    # from dst, so signed_dist (offset of dst from src) = -s*d
    ii, aa, ss, dd = np.nonzero(filled)
    src = partner[ii, aa, ss, dd]
    sd = -np.where(ss == 0, 1, -1) * (dd + 1)
    cs = [tuple(x) for x in np.stack([qs[order], rs[order]], 1)]
    return {(cs[s_], cs[d_], int(a_), int(sd_)) for s_, d_, a_, sd_ in zip(src, ii, aa, sd)}

# ---------------- test harness ----------------
def random_position(n_stones, seed, legal_radius):
    rng = np.random.default_rng(seed)
    stones = {}
    p = 0
    while len(stones) < n_stones:
        q, r = int(rng.integers(-12, 13)), int(rng.integers(-12, 13))
        if (q, r) not in stones:
            stones[(q, r)] = p; p = 1 - p
    offs = [(dq, dr) for dq in range(-legal_radius, legal_radius + 1)
                     for dr in range(-legal_radius, legal_radius + 1)
            if max(abs(dq), abs(dr), abs(dq + dr)) <= legal_radius]
    legal = sorted({(q + dq, r + dr) for (q, r) in stones for (dq, dr) in offs} - set(stones))
    coords = sorted(stones) + legal              # stones sorted, then empties sorted
    kinds = [stones[c] for c in sorted(stones)] + [2] * len(legal)
    return coords, kinds

total_edges = 0
for seed in range(6):
    for prune in (False, True):
        radius = 8 if seed == 0 else 3
        coords, kinds = random_position(120, seed, radius)
        ref = edges_reference(coords, kinds, prune)
        vec = edges_vectorized(coords, kinds, prune)
        assert vec == ref, f"MISMATCH seed={seed} prune={prune}: " \
            f"only_ref={list(ref - vec)[:3]} only_vec={list(vec - ref)[:3]}"
        total_edges += len(ref)
        print(f"seed={seed} prune={int(prune)} r={radius}: "
              f"{len(coords):>5} nodes, {len(ref):>6} directed edges — EXACT MATCH")

print(f"\nparity: {total_edges} edges compared across 12 configs, zero mismatches")

# rough single-thread CPU timing at full size (radius 8, 120 stones)
coords, kinds = random_position(120, 0, 8)
t = time.perf_counter(); [edges_reference(coords, kinds, True) for _ in range(3)]
t_ref = (time.perf_counter() - t) / 3
t = time.perf_counter(); [edges_vectorized(coords, kinds, True) for _ in range(20)]
t_vec = (time.perf_counter() - t) / 20
print(f"CPU timing ({len(coords)} nodes): python-loop reference {t_ref*1e3:.1f} ms, "
      f"vectorized numpy {t_vec*1e3:.1f} ms ({t_ref/t_vec:.0f}x)")
print("(reference is Python, your real builder is Rust — the point is the "
      "vectorized form ports 1:1 to torch and runs on GPU)")
