"""Fixed-slot axis-window graph builder in pure torch (Workstream A1).

Reproduces ``hexo_rs.game_to_axis_graph_raw`` (axis_graph.rs) entirely in torch
int32/float32 tensors, using the canonical packed HexKey (§1 of the perf plan).
The axis-window graph has a fixed max in-degree of 30 (3 axes × 2 signs ×
window ``win_length - 1``); the slot index ``(axis, sign, dist)`` fully
determines the edge's axis one-hot + signed distance, so edges never need to be
materialised as a ragged ``edge_index`` — they live in dense ``[N, 3, 2, W]``
``partner`` (source node index) and ``filled`` (mask) tensors.

The edge construction is a 1:1 torch port of the numpy reference in
``tests/axis_parity.py``, which is verified EXACTLY equal to the Rust walk
semantics across 157k directed edges / 12 configs (including the
union-of-both-endpoints'-walks term). ``tests/test_slot_graph_parity.py``
re-verifies this module against the actual Rust builder.

No ``torch_geometric`` dependency (by design — see the plan). CPU + CUDA.
"""

from __future__ import annotations

import torch

# The 3 win axes, matching hexo_engine::types::WIN_AXES.
WIN_AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))

# kind codes, matching axis_parity.py: 0 = P1 stone, 1 = P2 stone, 2 = empty.
KIND_P1, KIND_P2, KIND_EMPTY = 0, 1, 2

_R_BIAS = 0x8000  # low-16 r-field bias (only r is biased — see §1)


# --------------------------------------------------------------------------
# Canonical packed HexKey  (§1: i32, layout (q << 16) | ((r ^ 0x8000) & 0xFFFF))
# --------------------------------------------------------------------------
def pack(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Pack integer (q, r) coordinate tensors into canonical int32 HexKeys.

    q occupies the high 16 bits as two's complement (NOT biased); r occupies the
    low 16 bits bias-flipped by ``^ 0x8000``. Because only r is biased, signed
    i32 ordering equals lexicographic (q, r) ordering, so ``torch.sort`` /
    ``torch.searchsorted`` agree with the documented node coordinate order.
    """
    q = q.to(torch.int64)
    r = r.to(torch.int64)
    key = (q << 16) | ((r ^ _R_BIAS) & 0xFFFF)
    # Fold into the signed int32 range (wrapping two's complement).
    key = ((key + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000
    return key.to(torch.int32)


def unpack(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack`. Returns (q, r) as int64 tensors."""
    ki = k.to(torch.int64)
    q = ki >> 16  # arithmetic shift sign-extends the two's-complement q field
    t = (ki & 0xFFFF) ^ _R_BIAS
    r = torch.where(t >= _R_BIAS, t - 0x1_0000, t)
    return q, r


def axis_deltas(win_length: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Packed multi-step offsets ``[3, 2, W]`` (W = win_length - 1).

    ``deltas[a, s, d-1]`` is the HexKey delta for a step of ``sign * d`` along
    axis ``a`` — a single ``wrapping_add`` walks one step. Because r stays in the
    biased low-16 range for all in-range coordinates, borrows/carries across the
    field boundary resolve correctly (§1).
    """
    window = win_length - 1
    out = torch.zeros((3, 2, window), dtype=torch.int64, device=device)
    for a, (dq, dr) in enumerate(WIN_AXES):
        for si, s in enumerate((1, -1)):
            for d in range(1, window + 1):
                # delta(dq,dr) = (dq << 16) + dr, wrapping — matches §1.
                out[a, si, d - 1] = ((s * d * dq) << 16) + (s * d * dr)
    out = ((out + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000
    return out.to(torch.int32)


# --------------------------------------------------------------------------
# Edge slots: partner [N,3,2,W] (source node index) + filled [N,3,2,W] (mask)
# --------------------------------------------------------------------------
def build_edge_slots(
    keys: torch.Tensor,
    kinds: torch.Tensor,
    win_length: int,
    prune_empty_edges: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(partner, filled)`` slot tensors for nodes in identity order.

    ``keys`` [N] int32 canonical HexKeys, ``kinds`` [N] int8 (0/1/2). Node
    identity order is caller-defined (stones-then-legal); this routine sorts a
    private copy for the ``searchsorted`` lookup and remaps results back to
    identity indices, so the returned ``partner`` values are identity indices.

    Returns ``partner`` [N,3,2,W] int64 and ``filled`` [N,3,2,W] bool. Slot
    ``(i, a, s, d)`` describes the incoming edge to node ``i`` from
    ``partner[i,a,s,d]`` along axis ``a``; ``filled`` says the edge exists.
    """
    device = keys.device
    n = keys.shape[0]
    window = win_length - 1
    deltas = axis_deltas(win_length, device)  # [3,2,W] int32

    # Globally sort keys for lookup; keep a map sorted-pos -> identity index.
    sort_idx = torch.argsort(keys)  # int32 order == (q, r) lexicographic
    sorted_keys = keys[sort_idx]

    # Candidate neighbour keys for every (node, axis, sign, dist) slot.
    # int32 add wraps (two's complement) — matches Rust stepping.
    cand = (keys[:, None, None, None].to(torch.int64) + deltas[None].to(torch.int64))
    cand = (((cand + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)

    pos = torch.searchsorted(sorted_keys, cand)
    pos_c = pos.clamp(max=n - 1)
    present = (pos < n) & (sorted_keys[pos_c] == cand)
    partner = torch.where(present, sort_idx[pos_c].to(torch.int64), torch.zeros_like(pos, dtype=torch.int64))

    tk = kinds[partner]                       # target kind (garbage where !present)
    wk = kinds[:, None, None, None]           # walker kind, broadcast

    # Walk-stop rule: empty walker stops at any stone; stone walker stops at the
    # opponent stone. (kinds: 0=P1, 1=P2, 2=empty.)
    stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))

    # reach[...,d]: the ray from this node actually reaches step d (all earlier
    # steps present-and-not-stopping).
    cont = (present & ~stop).to(torch.uint8)
    ones = torch.ones_like(cont[..., :1])
    carry = torch.cumprod(torch.cat([ones, cont[..., :-1]], dim=-1), dim=-1)
    reach = present & carry.bool()

    # Union of both endpoints' walks (REQUIRED — parity fails without it): slot
    # (i,a,s,d) also fills if the partner's mirrored slot (j,a,-s,d) reached i.
    # reach.flip(dim=2) maps sign index s -> 1-s, so indexing it at s gives the
    # opposite-sign slot of the partner (matches axis_parity.py's mirror gather).
    mirror = reach.flip(dims=[2])[
        partner,
        torch.arange(3, device=device)[None, :, None, None],
        torch.arange(2, device=device)[None, None, :, None],
        torch.arange(window, device=device)[None, None, None, :],
    ]
    filled = reach | (present & mirror)

    if prune_empty_edges:
        filled = filled & ~((wk == KIND_EMPTY) & (tk == KIND_EMPTY))

    return partner, filled


def slots_to_edge_set(
    coords: torch.Tensor,
    partner: torch.Tensor,
    filled: torch.Tensor,
) -> set[tuple[tuple[int, int], tuple[int, int], int, int]]:
    """Decode ``(partner, filled)`` into a directed edge set
    ``{(src_coord, dst_coord, axis, signed_dist)}`` matching the Rust axis
    edge_attr convention (signed_dist = step from src to dst)."""
    ii, aa, ss, dd = torch.nonzero(filled, as_tuple=True)
    src = partner[ii, aa, ss, dd]
    s_sign = torch.where(ss == 0, 1, -1)
    signed_dist = (-s_sign * (dd + 1)).tolist()
    src_c = coords[src].tolist()
    dst_c = coords[ii].tolist()
    aa_l = aa.tolist()
    return {
        (tuple(sc), tuple(dc), int(a), int(sd))
        for sc, dc, a, sd in zip(src_c, dst_c, aa_l, signed_dist)
    }


# --------------------------------------------------------------------------
# Node features (matches axis_graph.rs build_axis_graph + fill_threat_features)
# --------------------------------------------------------------------------
def _node_layout(relative_stones: bool, node_coords: bool, moves_scope: str, compact: bool) -> dict:
    """Column indices for the base node features (mirrors NodeLayout::new)."""
    idx = 0

    def take():
        nonlocal idx
        c = idx
        idx += 1
        return c

    own = take()
    opp = take()
    empty = None if compact else take()
    to_move = None if relative_stones else take()
    moves = take() if moves_scope == "node" else None
    if node_coords:
        norm_q, norm_r = take(), take()
    else:
        norm_q = norm_r = None
    inv_dist = take()
    return dict(
        own=own, opp=opp, empty=empty, to_move=to_move, moves=moves,
        norm_q=norm_q, norm_r=norm_r, inv_dist=inv_dist, base_dim=idx,
    )


def _threat_features(
    keys: torch.Tensor,
    kinds: torch.Tensor,
    coords: torch.Tensor,
    to_move_kind: int,
    win_length: int,
) -> torch.Tensor:
    """Vectorised port of hexo_engine::threat::node_threat_features for every
    real node. Returns ``[N, 4]`` = [own_max/wl, opp_max/wl, own_axes/3,
    opp_axes/3], where "own" is the side to move.
    """
    device = keys.device
    n = keys.shape[0]
    wl = win_length
    opp_kind = 1 - to_move_kind

    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]

    own_max = torch.zeros(n, device=device)
    opp_max = torch.zeros(n, device=device)
    own_axes = torch.zeros(n, device=device)
    opp_axes = torch.zeros(n, device=device)

    ks = torch.arange(-(wl - 1), wl, device=device)  # 2*wl-1 line offsets
    for dq, dr in WIN_AXES:
        # Gather the kind of each of the 2*wl-1 cells centred on each node.
        cq = coords[:, 0:1] + ks[None, :] * dq
        cr = coords[:, 1:2] + ks[None, :] * dr
        cand = pack(cq, cr)
        pos = torch.searchsorted(sorted_keys, cand).clamp(max=n - 1)
        present = sorted_keys[pos] == cand
        cell_kind = torch.where(present, kinds[sort_idx[pos]], torch.full_like(cand, KIND_EMPTY, dtype=kinds.dtype))
        is_own = (cell_kind == to_move_kind)
        is_opp = (cell_kind == opp_kind)

        axis_own = torch.zeros(n, device=device)
        axis_opp = torch.zeros(n, device=device)
        for start in range(wl):
            win_own = is_own[:, start:start + wl].sum(dim=1).float()
            win_opp = is_opp[:, start:start + wl].sum(dim=1).float()
            clean_for_own = win_opp == 0
            clean_for_opp = win_own == 0
            axis_own = torch.where(clean_for_own, torch.maximum(axis_own, win_own), axis_own)
            axis_opp = torch.where(clean_for_opp, torch.maximum(axis_opp, win_opp), axis_opp)

        own_max = torch.maximum(own_max, axis_own)
        opp_max = torch.maximum(opp_max, axis_opp)
        own_axes = own_axes + (axis_own >= wl - 2).float()
        opp_axes = opp_axes + (axis_opp >= wl - 2).float()

    return torch.stack([own_max / wl, opp_max / wl, own_axes / 3.0, opp_axes / 3.0], dim=1)


def build_node_features(
    coords: torch.Tensor,
    kinds: torch.Tensor,
    keys: torch.Tensor,
    n_stones: int,
    player_feat: float,
    moves_feat: float,
    win_length: int,
    to_move_kind: int | None,
    *,
    threat_features: bool,
    relative_stones: bool,
    node_coords: bool = True,
    moves_scope: str = "node",
    compact_stone_onehot: bool = False,
) -> torch.Tensor:
    """Build the ``[N+1, fdim]`` node feature matrix (incl. the dummy node),
    matching build_axis_graph. ``coords``/``kinds``/``keys`` cover the N real
    nodes (stones-then-legal); the dummy row is appended here."""
    device = coords.device
    n_real = coords.shape[0]
    layout = _node_layout(relative_stones, node_coords, moves_scope, compact_stone_onehot)
    base_dim = layout["base_dim"]
    fdim = base_dim + (4 if threat_features else 0)

    feats = torch.zeros((n_real + 1, fdim), dtype=torch.float32, device=device)
    own_is_p1 = player_feat > 0.0

    # Stone / legal split.
    is_stone = torch.arange(n_real, device=device) < n_stones
    stone_kind = kinds[is_stone]

    # --- stone one-hot (own/opp) ---
    if relative_stones:
        # own if (player==P1) == own_is_p1 else opp
        player_is_p1 = stone_kind == KIND_P1
        to_own = player_is_p1 == own_is_p1
    else:
        to_own = stone_kind == KIND_P1  # absolute: P1 -> own, P2 -> opp
    stone_idx = torch.nonzero(is_stone, as_tuple=True)[0]
    feats[stone_idx[to_own], layout["own"]] = 1.0
    feats[stone_idx[~to_own], layout["opp"]] = 1.0

    # --- empty one-hot for legal nodes ---
    legal_idx = torch.nonzero(~is_stone, as_tuple=True)[0]
    if layout["empty"] is not None:
        feats[legal_idx, layout["empty"]] = 1.0

    # --- to_move (absolute only) + moves, on ALL real nodes ---
    if layout["to_move"] is not None:
        feats[:n_real, layout["to_move"]] = player_feat
    if layout["moves"] is not None:
        feats[:n_real, layout["moves"]] = moves_feat

    # --- centroid / spread over stones; norm_q/norm_r on all real nodes ---
    if node_coords:
        if n_stones > 0:
            sc = coords[:n_stones].float()
            cq = sc[:, 0].mean()
            cr = sc[:, 1].mean()
            dev = torch.maximum((sc[:, 0] - cq).abs(), (sc[:, 1] - cr).abs())
            spread = torch.clamp(dev.max(), min=1.0)
        else:
            cq = cr = torch.tensor(0.0, device=device)
            spread = torch.tensor(1.0, device=device)
        feats[:n_real, layout["norm_q"]] = (coords[:, 0].float() - cq) / spread
        feats[:n_real, layout["norm_r"]] = (coords[:, 1].float() - cr) / spread

    # --- inv_dist on legal nodes: 1 / max(min hex-dist to any stone, 1) ---
    if n_stones > 0 and legal_idx.numel() > 0:
        legal_c = coords[legal_idx].float()          # [L, 2]
        stone_c = coords[:n_stones].float()          # [S, 2]
        dq = legal_c[:, None, 0] - stone_c[None, :, 0]
        dr = legal_c[:, None, 1] - stone_c[None, :, 1]
        hexd = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
        min_d = hexd.min(dim=1).values.clamp(min=1.0)
        feats[legal_idx, layout["inv_dist"]] = 1.0 / min_d
    # stones keep inv_dist = 0.

    # --- dummy node: moves_feat (+ to_move in absolute mode); threat stays 0 ---
    if layout["to_move"] is not None:
        feats[n_real, layout["to_move"]] = player_feat
    if layout["moves"] is not None:
        feats[n_real, layout["moves"]] = moves_feat

    # --- threat dims (real nodes only) ---
    if threat_features:
        assert to_move_kind is not None, "threat features require a non-terminal to_move"
        tf = _threat_features(keys, kinds, coords, to_move_kind, win_length)
        feats[:n_real, base_dim:base_dim + 4] = tf

    return feats


# --------------------------------------------------------------------------
# Top-level: build from ingredients extracted from a GameState
# --------------------------------------------------------------------------
def _kind_of(player_str: str) -> int:
    return KIND_P1 if player_str == "P1" else KIND_P2


def build_slot_graph(
    stones: list[tuple[tuple[int, int], str]],
    legal: list[tuple[int, int]],
    current_player: str | None,
    moves_remaining: int,
    win_length: int,
    *,
    prune_empty_edges: bool = False,
    threat_features: bool = False,
    relative_stones: bool = False,
    node_coords: bool = True,
    moves_scope: str = "node",
    compact_stone_onehot: bool = False,
    device: torch.device | str = "cpu",
) -> dict:
    """Build the slot-table axis-window graph from the same ingredients the Rust
    builder reads from a GameState.

    ``stones`` = ``game.placed_stones()`` (list of ((q,r), 'P1'|'P2')),
    ``legal`` = ``game.legal_moves()``, ``current_player`` =
    ``game.current_player()`` ('P1'|'P2'|None), etc.

    Returns a dict with: ``features`` [N+1, fdim], ``partner`` / ``filled``
    [N,3,2,W], ``coords`` [N+1, 2], ``stone_mask`` / ``legal_mask`` [N+1],
    ``num_nodes``, and ``edge_set`` (directed axis edges as coord tuples).
    """
    device = torch.device(device)

    # Node identity order: stones sorted by (q, r), then legal sorted by (q, r).
    stones_sorted = sorted(stones, key=lambda s: s[0])
    legal_sorted = sorted(legal)
    n_stones = len(stones_sorted)

    stone_qr = [c for c, _ in stones_sorted]
    all_qr = stone_qr + list(legal_sorted)
    n_real = len(all_qr)

    coords_real = torch.tensor(all_qr, dtype=torch.int32, device=device).reshape(-1, 2)
    kinds = torch.tensor(
        [_kind_of(p) for _, p in stones_sorted] + [KIND_EMPTY] * len(legal_sorted),
        dtype=torch.int8, device=device,
    )
    keys = pack(coords_real[:, 0], coords_real[:, 1])

    partner, filled = build_edge_slots(keys, kinds, win_length, prune_empty_edges)

    player_feat = 1.0 if current_player == "P1" else -1.0
    moves_feat = moves_remaining / 2.0
    to_move_kind = None if current_player is None else _kind_of(current_player)

    feats = build_node_features(
        coords_real, kinds, keys, n_stones, player_feat, moves_feat, win_length,
        to_move_kind,
        threat_features=threat_features, relative_stones=relative_stones,
        node_coords=node_coords, moves_scope=moves_scope,
        compact_stone_onehot=compact_stone_onehot,
    )

    # coords incl. dummy at (0, 0) (Rust pushes (0,0) for the dummy).
    coords = torch.cat([coords_real, torch.zeros((1, 2), dtype=torch.int32, device=device)], dim=0)
    stone_mask = torch.zeros(n_real + 1, dtype=torch.bool, device=device)
    stone_mask[:n_stones] = True
    legal_mask = torch.zeros(n_real + 1, dtype=torch.bool, device=device)
    legal_mask[n_stones:n_real] = True

    edge_set = slots_to_edge_set(coords_real, partner, filled)

    return dict(
        features=feats,
        partner=partner,
        filled=filled,
        coords=coords,
        stone_mask=stone_mask,
        legal_mask=legal_mask,
        num_nodes=n_real + 1,
        edge_set=edge_set,
    )
