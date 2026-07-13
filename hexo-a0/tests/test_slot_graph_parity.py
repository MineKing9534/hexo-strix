"""Parity: the torch slot-table builder (slot_graph.py, Workstream A1) vs the
actual Rust builder ``hexo_rs.game_to_axis_graph_raw`` — edge sets AND node
features — plus the canonical HexKey round-trip suite (§1 of the perf plan).

Runs CPU-only (the builder is device-agnostic; parity is a numeric property).
"""

from __future__ import annotations

import random

import pytest
import torch

hexo_rs = pytest.importorskip("hexo_rs")

from hexo_a0 import slot_graph as sg  # noqa: E402


# --------------------------------------------------------------------------
# §1 canonical HexKey round-trip / step / sort-order tests
# --------------------------------------------------------------------------
def test_hexkey_roundtrip_grid():
    qs = torch.arange(-300, 301, dtype=torch.int64)
    rs = torch.arange(-300, 301, dtype=torch.int64)
    Q, R = torch.meshgrid(qs, rs, indexing="ij")
    q, r = sg.unpack(sg.pack(Q.reshape(-1), R.reshape(-1)))
    assert torch.equal(q, Q.reshape(-1))
    assert torch.equal(r, R.reshape(-1))


def test_hexkey_extreme_corners():
    q = torch.tensor([-32000, 32000, -32000, 32000], dtype=torch.int64)
    r = torch.tensor([-32000, -32000, 32000, 32000], dtype=torch.int64)
    uq, ur = sg.unpack(sg.pack(q, r))
    assert torch.equal(uq, q) and torch.equal(ur, r)


def test_hexkey_single_step_deltas():
    # All 6 single-step neighbours via a +1 step along each of the 3 axes/2 signs.
    qs = torch.arange(-64, 65, dtype=torch.int64)
    rs = torch.arange(-64, 65, dtype=torch.int64)
    Q, R = torch.meshgrid(qs, rs, indexing="ij")
    q = Q.reshape(-1)
    r = R.reshape(-1)
    base = sg.pack(q, r)
    deltas = sg.axis_deltas(win_length=2)  # window 1 -> single step
    for a, (dq, dr) in enumerate(sg.WIN_AXES):
        for si, s in enumerate((1, -1)):
            stepped = ((base.to(torch.int64) + deltas[a, si, 0].to(torch.int64)) & 0xFFFFFFFF)
            stepped = (stepped - 0x100000000 * (stepped >= 0x80000000)).to(torch.int32)
            uq, ur = sg.unpack(stepped)
            assert torch.equal(uq, q + s * dq), f"axis {a} sign {s} q mismatch"
            assert torch.equal(ur, r + s * dr), f"axis {a} sign {s} r mismatch"


def test_hexkey_multistep_deltas():
    q = torch.arange(-40, 41, dtype=torch.int64)
    r = torch.arange(-40, 41, dtype=torch.int64)
    base = sg.pack(q, r)
    deltas = sg.axis_deltas(win_length=6)  # window 5
    for a, (dq, dr) in enumerate(sg.WIN_AXES):
        for si, s in enumerate((1, -1)):
            for d in range(1, 6):
                stepped = (base.to(torch.int64) + deltas[a, si, d - 1].to(torch.int64)) & 0xFFFFFFFF
                stepped = (stepped - 0x100000000 * (stepped >= 0x80000000)).to(torch.int32)
                uq, ur = sg.unpack(stepped)
                assert torch.equal(uq, q + s * d * dq)
                assert torch.equal(ur, r + s * d * dr)


def test_hexkey_sort_order_matches_tuple_order():
    rng = random.Random(0)
    pts = [(rng.randint(-500, 500), rng.randint(-500, 500)) for _ in range(2000)]
    pts = list(dict.fromkeys(pts))  # dedup
    q = torch.tensor([p[0] for p in pts], dtype=torch.int64)
    r = torch.tensor([p[1] for p in pts], dtype=torch.int64)
    keys = sg.pack(q, r)
    by_key = torch.argsort(keys).tolist()
    by_tuple = sorted(range(len(pts)), key=lambda i: pts[i])
    assert by_key == by_tuple


# --------------------------------------------------------------------------
# Parity fixtures: build assorted GameStates (incl. mid-turn)
# --------------------------------------------------------------------------
def _random_game(seed: int, n_moves: int):
    cfg = hexo_rs.GameConfig(win_length=6, placement_radius=6, max_moves=300)
    g = hexo_rs.GameState(cfg)
    rng = random.Random(seed)
    for _ in range(n_moves):
        if g.is_terminal():
            break
        moves = g.legal_moves()
        if not moves:
            break
        q, r = moves[rng.randrange(len(moves))]
        try:
            g.apply_move(q, r)
        except Exception:
            break
    return g


def _rust_axis_edge_set(raw):
    """Decode the Rust legacy edge lists into the same
    {(src_coord, dst_coord, axis, signed_dist)} set the slot builder produces
    (dummy/global edges excluded — they are a separate broadcast relation)."""
    n = raw["num_nodes"]
    dummy = n - 1
    coords = raw["coords"]
    coord = lambda i: (coords[2 * i], coords[2 * i + 1])
    src, dst, attr = raw["edge_src"], raw["edge_dst"], raw["edge_attr"]
    edges = set()
    for e in range(len(src)):
        s, d = src[e], dst[e]
        if s == dummy or d == dummy:
            continue
        a0, a1, a2 = attr[e * 5], attr[e * 5 + 1], attr[e * 5 + 2]
        axis = 0 if a0 > 0.5 else (1 if a1 > 0.5 else 2)
        signed = int(round(attr[e * 5 + 3]))
        edges.add((coord(s), coord(d), axis, signed))
    return edges


FLAG_COMBOS = [
    dict(prune_empty_edges=p, threat_features=t, relative_stones=r)
    for p in (False, True)
    for t in (False, True)
    for r in (False, True)
]

# ≥12 positions: 6 seeds × 2 depths (one mid-turn-heavy, one full-turn-heavy).
POSITIONS = [(seed, nmoves) for seed in range(6) for nmoves in (7, 12)]


@pytest.mark.parametrize("seed,nmoves", POSITIONS)
@pytest.mark.parametrize(
    "flags", FLAG_COMBOS, ids=[f"p{int(f['prune_empty_edges'])}t{int(f['threat_features'])}r{int(f['relative_stones'])}" for f in FLAG_COMBOS]
)
def test_slot_graph_matches_rust(seed, nmoves, flags):
    g = _random_game(seed, nmoves)
    if g.is_terminal():
        pytest.skip("terminal position — builder is defined for non-terminal states")

    raw = hexo_rs.game_to_axis_graph_raw(
        g, flags["prune_empty_edges"], flags["threat_features"], flags["relative_stones"]
    )
    out = sg.build_slot_graph(
        g.placed_stones(), g.legal_moves(), g.current_player(),
        g.moves_remaining_this_turn(), g.config().win_length, **flags,
    )

    # 1. Node count + masks + coords.
    assert out["num_nodes"] == raw["num_nodes"]
    assert out["stone_mask"].tolist() == raw["stone_mask"]
    assert out["legal_mask"].tolist() == raw["legal_mask"]
    rust_coords = torch.tensor(raw["coords"], dtype=torch.int32).reshape(-1, 2)
    assert torch.equal(out["coords"], rust_coords), "coord/node ordering mismatch"

    # 2. Directed axis edge-set equality (the union-of-walks subtlety lives here).
    assert out["edge_set"] == _rust_axis_edge_set(raw), (
        f"edge-set mismatch seed={seed} nmoves={nmoves} flags={flags}"
    )

    # 3. Node features to float32 tolerance.
    n = raw["num_nodes"]
    rust_feats = torch.tensor(raw["features"], dtype=torch.float32).reshape(n, -1)
    assert out["features"].shape == rust_feats.shape, (
        f"fdim mismatch: slot {tuple(out['features'].shape)} vs rust {tuple(rust_feats.shape)}"
    )
    torch.testing.assert_close(out["features"], rust_feats, atol=1e-5, rtol=1e-4)
