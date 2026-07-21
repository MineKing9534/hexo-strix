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

from dataclasses import dataclass, fields

import torch
from torch import Tensor

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
    # steps present-and-not-stopping). Unrolled cumulative-AND over the size-W
    # walk dim, bit-identical to the ``cumprod`` scan it replaces
    # (reach[d] = present[d] & cont[0] & ... & cont[d-1]) but ~28x faster on the
    # scan segment — ``torch.cumprod`` over this tiny dim is pathological on ROCm
    # (see scripts/bench_build_hotspots.py).
    cont = present & ~stop
    reaches = []
    acc = torch.ones_like(cont[..., 0])
    for d in range(window):
        reaches.append(present[..., d] & acc)
        acc = acc & cont[..., d]
    reach = torch.stack(reaches, dim=-1)

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
    # Computed in f64 like the Rust builder (which only casts the final
    # normalised value to f32) so non-representable centroids (e.g. 1/3)
    # don't diverge in the last ulps.
    if node_coords:
        if n_stones > 0:
            sc = coords[:n_stones].double()
            cq = sc[:, 0].mean()
            cr = sc[:, 1].mean()
            dev = torch.maximum((sc[:, 0] - cq).abs(), (sc[:, 1] - cr).abs())
            spread = torch.clamp(dev.max(), min=1.0)
        else:
            cq = cr = torch.tensor(0.0, dtype=torch.float64, device=device)
            spread = torch.tensor(1.0, dtype=torch.float64, device=device)
        feats[:n_real, layout["norm_q"]] = ((coords[:, 0].double() - cq) / spread).float()
        feats[:n_real, layout["norm_r"]] = ((coords[:, 1].double() - cr) / spread).float()

    # --- inv_dist on legal nodes: 1 / max(min hex-dist to any stone, 1) ---
    if n_stones > 0 and legal_idx.numel() > 0:
        legal_c = coords[legal_idx].float()          # [L, 2]
        stone_c = coords[:n_stones].float()          # [S, 2]
        dq = legal_c[:, None, 0] - stone_c[None, :, 0]
        dr = legal_c[:, None, 1] - stone_c[None, :, 1]
        hexd = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
        min_d = hexd.min(dim=1).values.clamp(min=1.0)
        feats[legal_idx, layout["inv_dist"]] = 1.0 / min_d
    elif legal_idx.numel() > 0:
        # No stones: Rust's min-dist fold is `.min().unwrap_or(1)` → every
        # legal node gets inv_dist = 1.0 (unreachable from a real GameState —
        # the engine seeds (0,0) — but the port must match exactly).
        feats[legal_idx, layout["inv_dist"]] = 1.0
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
    with_edge_set: bool = False,
) -> dict:
    """Build the slot-table axis-window graph from the same ingredients the Rust
    builder reads from a GameState.

    ``stones`` = ``game.placed_stones()`` (list of ((q,r), 'P1'|'P2')),
    ``legal`` = ``game.legal_moves()``, ``current_player`` =
    ``game.current_player()`` ('P1'|'P2'|None), etc.

    Returns a dict with: ``features`` [N+1, fdim], ``partner`` / ``filled``
    [N,3,2,W], ``coords`` [N+1, 2], ``stone_mask`` / ``legal_mask`` [N+1],
    ``kinds`` [N] int8 (real nodes; 0=P1, 1=P2, 2=empty), ``num_nodes``, and —
    only when ``with_edge_set`` (a parity-test decode that costs a host sync +
    Python set build) — ``edge_set`` (directed axis edges as coord tuples).
    """
    device = torch.device(device)

    # Terminal states are outside the contract (no side to move — features
    # like to_move/threat are undefined); match game_to_axis_graph's raise.
    if current_player is None:
        raise ValueError("build_slot_graph: game is terminal (no current player)")

    # Node identity order: stones sorted by (q, r), then legal sorted by (q, r).
    stones_sorted = sorted(stones, key=lambda s: s[0])
    legal_sorted = sorted(legal)
    n_stones = len(stones_sorted)

    stone_qr = [c for c, _ in stones_sorted]
    all_qr = stone_qr + list(legal_sorted)
    n_real = len(all_qr)

    coords_real = torch.tensor(all_qr, dtype=torch.int32, device=device).reshape(-1, 2)
    if n_real and int(coords_real.abs().max()) > 32000:
        # §1 of the plan: coords leaving [-32000, 32000] are an assertion
        # failure (HexKey packs q,r into 16-bit fields), not a real case.
        raise ValueError("build_slot_graph: coordinate outside the packable ±32000 range")
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

    out = dict(
        features=feats,
        partner=partner,
        filled=filled,
        coords=coords,
        stone_mask=stone_mask,
        legal_mask=legal_mask,
        kinds=kinds,
        num_nodes=n_real + 1,
    )
    if with_edge_set:
        out["edge_set"] = slots_to_edge_set(coords_real, partner, filled)
    return out


# --------------------------------------------------------------------------
# Padded batch container + per-graph collate (used by model_slots.SlotHeXONet;
# lives here so the batched builder below can emit it without a PyG import)
# --------------------------------------------------------------------------
@dataclass
class SlotBatch:
    """Padded static-shape batch of slot graphs.

    ``N`` is the padded real-node count (excludes the dummy node, which is
    carried separately as a per-graph row), ``S = 3 * 2 * W`` the flattened
    slot count. Padded rows have ``node_mask=False``, ``filled=False``
    everywhere, zero features, and are excluded from the heads via the masks.
    """

    x: Tensor            # [B, N, F] float32 — real-node features
    dummy_x: Tensor      # [B, F]    float32 — dummy-node features
    partner: Tensor      # [B, N, S] int64   — source node index per slot
    filled: Tensor       # [B, N, S] bool    — slot carries an edge
    src_player: Tensor   # [B, N]    float32 — +1 P1 stone / -1 P2 stone / 0 empty
    node_mask: Tensor    # [B, N]    bool    — real (non-padding) node
    stone_mask: Tensor   # [B, N]    bool
    legal_mask: Tensor   # [B, N]    bool

    def to(self, device) -> "SlotBatch":
        return SlotBatch(**{
            f.name: getattr(self, f.name).to(device) for f in fields(self)
        })

    @property
    def num_graphs(self) -> int:
        return self.x.shape[0]


def collate_slot_graphs(graphs: list[dict], pad_to: int | None = None) -> SlotBatch:
    """Collate ``build_slot_graph`` outputs into a padded :class:`SlotBatch`.

    ``pad_to`` fixes the padded node count (for shape bucketing /
    ``torch.compile`` static shapes); default is the batch max. All graphs
    must share the same slot count (same ``win_length``) and feature width.
    """
    if not graphs:
        raise ValueError("collate_slot_graphs: empty graph list")
    device = graphs[0]["features"].device
    sizes = [g["features"].shape[0] - 1 for g in graphs]  # real nodes (dummy is last row)
    n_max = max(sizes)
    if pad_to is not None:
        if pad_to < n_max:
            raise ValueError(
                f"pad_to={pad_to} smaller than largest graph ({n_max} real nodes)"
            )
        n_max = pad_to
    b = len(graphs)
    fdim = graphs[0]["features"].shape[1]
    s = graphs[0]["filled"].shape[1] * graphs[0]["filled"].shape[2] * graphs[0]["filled"].shape[3]

    x = torch.zeros((b, n_max, fdim), dtype=torch.float32, device=device)
    dummy_x = torch.zeros((b, fdim), dtype=torch.float32, device=device)
    partner = torch.zeros((b, n_max, s), dtype=torch.int64, device=device)
    filled = torch.zeros((b, n_max, s), dtype=torch.bool, device=device)
    src_player = torch.zeros((b, n_max), dtype=torch.float32, device=device)
    node_mask = torch.zeros((b, n_max), dtype=torch.bool, device=device)
    stone_mask = torch.zeros((b, n_max), dtype=torch.bool, device=device)
    legal_mask = torch.zeros((b, n_max), dtype=torch.bool, device=device)

    for i, g in enumerate(graphs):
        n = sizes[i]
        if g["features"].shape[1] != fdim or g["filled"].reshape(g["filled"].shape[0], -1).shape[1] != s:
            raise ValueError("collate_slot_graphs: inconsistent feature/slot shapes across graphs")
        x[i, :n] = g["features"][:-1]
        dummy_x[i] = g["features"][-1]
        partner[i, :n] = g["partner"].reshape(n, -1)
        filled[i, :n] = g["filled"].reshape(n, -1)
        kinds = g["kinds"]
        src_player[i, :n] = (kinds == KIND_P1).float() - (kinds == KIND_P2).float()
        node_mask[i, :n] = True
        stone_mask[i, :n] = g["stone_mask"][:-1]
        legal_mask[i, :n] = g["legal_mask"][:-1]

    return SlotBatch(
        x=x, dummy_x=dummy_x, partner=partner, filled=filled,
        src_player=src_player, node_mask=node_mask,
        stone_mask=stone_mask, legal_mask=legal_mask,
    )


# --------------------------------------------------------------------------
# Batched builder: wire states -> SlotBatch, end-to-end torch (plan-A3 task 4)
# --------------------------------------------------------------------------
# Sentinel padding key for row-sorted lookup tables. Strictly greater than any
# packable key: coords are asserted within ±32000 and walks step ≤ win_length-1
# further, so real keys stay ≤ pack(~32005, ·) < 0x7D07_0000 < 0x7FFF_FFFF, and
# no candidate key can ever EQUAL the sentinel.
_KEY_SENTINEL = 0x7FFF_FFFF


@dataclass(frozen=True)
class SlotBuilderConfig:
    """Game + builder parameters for :func:`build_slot_batch` (the fields a
    MSG_FORWARD_STATES request carries besides the per-graph states)."""

    win_length: int
    placement_radius: int
    prune_empty_edges: bool = False
    threat_features: bool = False
    relative_stones: bool = False
    node_coords: bool = True
    moves_scope: str = "node"
    compact_stone_onehot: bool = False


def _wrap_i32(x: torch.Tensor) -> torch.Tensor:
    """Fold an int64 tensor into wrapping-two's-complement int32."""
    return (((x + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)


def _norm_player(p) -> int:
    """Normalise a player designator ('P1'/'P2' or wire 0/1) to a kind code."""
    if isinstance(p, str):
        if p == "P1":
            return KIND_P1
        if p == "P2":
            return KIND_P2
    elif isinstance(p, (int, bool)) and p in (0, 1):
        return int(p)
    raise ValueError(f"build_slot_batch: invalid player {p!r}")


def _disk_deltas(radius: int, device: torch.device) -> torch.Tensor:
    """Packed HexKey deltas for all offsets with hex-distance ≤ radius
    (including (0, 0)), matching hexo_engine::hex::hex_offsets."""
    rng = torch.arange(-radius, radius + 1, dtype=torch.int64, device=device)
    dq, dr = torch.meshgrid(rng, rng, indexing="ij")
    dq = dq.reshape(-1)
    dr = dr.reshape(-1)
    keep = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs()) <= radius
    return _wrap_i32((dq[keep] << 16) + dr[keep])


def _threat_features_batched(
    qs: torch.Tensor,
    rs: torch.Tensor,
    sorted_keys: torch.Tensor,
    kinds_sorted: torch.Tensor,
    to_move_kind: torch.Tensor,
    win_length: int,
) -> torch.Tensor:
    """Batched port of :func:`_threat_features`: ``qs``/``rs`` [B, N] int64
    identity-order coordinates (pad rows arbitrary — mask the output),
    ``sorted_keys`` [B, N] row-sorted int32 keys (sentinel-padded),
    ``kinds_sorted`` [B, N] kinds in that order, ``to_move_kind`` [B] int64.
    Returns [B, N, 4] (same normalisation as the single-graph version)."""
    device = qs.device
    b, n = qs.shape
    wl = win_length
    opp_kind = 1 - to_move_kind

    own_max = torch.zeros((b, n), device=device)
    opp_max = torch.zeros((b, n), device=device)
    own_axes = torch.zeros((b, n), device=device)
    opp_axes = torch.zeros((b, n), device=device)

    ks = torch.arange(-(wl - 1), wl, device=device)  # 2*wl-1 line offsets
    kinds64 = kinds_sorted.to(torch.int64)
    for dq, dr in WIN_AXES:
        cq = qs[:, :, None] + ks[None, None, :] * dq
        cr = rs[:, :, None] + ks[None, None, :] * dr
        cand = pack(cq, cr)  # [B, N, K]
        k = cand.shape[-1]
        pos = torch.searchsorted(sorted_keys, cand.reshape(b, -1)).clamp(max=n - 1)
        present = torch.gather(sorted_keys, 1, pos).reshape(b, n, k) == cand
        ck = torch.gather(kinds64, 1, pos).reshape(b, n, k)
        cell_kind = torch.where(present, ck, torch.full_like(ck, KIND_EMPTY))
        is_own = cell_kind == to_move_kind[:, None, None]
        is_opp = cell_kind == opp_kind[:, None, None]

        axis_own = torch.zeros((b, n), device=device)
        axis_opp = torch.zeros((b, n), device=device)
        for start in range(wl):
            win_own = is_own[:, :, start:start + wl].sum(dim=-1).float()
            win_opp = is_opp[:, :, start:start + wl].sum(dim=-1).float()
            clean_for_own = win_opp == 0
            clean_for_opp = win_own == 0
            axis_own = torch.where(clean_for_own, torch.maximum(axis_own, win_own), axis_own)
            axis_opp = torch.where(clean_for_opp, torch.maximum(axis_opp, win_opp), axis_opp)

        own_max = torch.maximum(own_max, axis_own)
        opp_max = torch.maximum(opp_max, axis_opp)
        own_axes = own_axes + (axis_own >= wl - 2).float()
        opp_axes = opp_axes + (axis_opp >= wl - 2).float()

    return torch.stack(
        [own_max / wl, opp_max / wl, own_axes / 3.0, opp_axes / 3.0], dim=2
    )


def _parse_states(states: list) -> tuple[list, list[int], list[float], list[float]]:
    """Validate + normalise wire states. Returns (stone_rows, to_move_kinds,
    player_feats, moves_feats); stone_rows = flat [(game, q, r, kind)]."""
    stone_rows: list[tuple[int, int, int, int]] = []
    to_move: list[int] = []
    player_feats: list[float] = []
    moves_feats: list[float] = []
    for gi, (stones, current_player, moves_remaining) in enumerate(states):
        if current_player is None:
            raise ValueError(
                f"build_slot_batch: state {gi} is terminal (no current player)"
            )
        if moves_remaining not in (1, 2):
            raise ValueError(
                f"build_slot_batch: state {gi} has moves_remaining="
                f"{moves_remaining!r}, expected 1 or 2"
            )
        if not stones:
            raise ValueError(
                f"build_slot_batch: state {gi} has no stones — the wire always "
                "carries the engine's (0,0) seed stone, and the legal region "
                "is undefined without stones"
            )
        kind = _norm_player(current_player)
        to_move.append(kind)
        player_feats.append(1.0 if kind == KIND_P1 else -1.0)
        moves_feats.append(moves_remaining / 2.0)
        for stone in stones:
            if len(stone) == 3:
                q, r, p = stone
            else:  # placed_stones()-style ((q, r), player)
                (q, r), p = stone
            stone_rows.append((gi, int(q), int(r), _norm_player(p)))
    return stone_rows, to_move, player_feats, moves_feats


@dataclass
class SlotBatchAux:
    """Sidecar of the batched builders (``return_aux=True``): the batch's own
    legal-node ordering, for response mapping and legal-order guards.

    ``legal_keys`` [L] int32 canonical HexKeys of every legal node, grouped by
    graph in batch order and ascending (== (q, r)-lexicographic) within each
    graph — EXACTLY the column order the padded batch assigns legal nodes, and
    therefore the order ``logits[legal_mask]`` flattens to.
    ``legal_counts`` [B] int64 legal-node count per graph.
    """

    legal_keys: Tensor
    legal_counts: Tensor


def build_slot_batch(
    states: list,
    config: SlotBuilderConfig,
    device: torch.device | str = "cpu",
    pad_to: int | None = None,
    return_aux: bool = False,
    core_fn=None,
) -> "SlotBatch | tuple[SlotBatch, SlotBatchAux]":
    """Build a padded :class:`SlotBatch` directly from a batch of board states
    — exactly the fields the MSG_FORWARD_STATES wire carries — with torch ops
    end-to-end (CPU or CUDA), replacing per-graph ``build_slot_graph`` +
    :func:`collate_slot_graphs`.

    ``states`` is a list of ``(stones, current_player, moves_remaining)``
    where ``stones`` is ``[(q, r, player)]`` (or placed_stones()-style
    ``[((q, r), player)]``) and players are ``'P1'``/``'P2'`` or wire ``0``/``1``.
    The legal region is derived here with the Rust engine's semantics: every
    empty cell within hex-distance ≤ ``config.placement_radius`` of any stone.

    Per-game node lookup runs as ONE batched ``torch.searchsorted`` over
    sentinel-padded row-sorted canonical int32 HexKeys (plan §1); node identity
    order is stones-then-legal, each (q, r)-sorted, via an index remap; the
    union-of-both-endpoints'-walks mirror term is included (plan §7, sharp
    edge 2).

    With ``return_aux=True`` also returns a :class:`SlotBatchAux` describing
    the batch's legal-node ordering.

    ``core_fn`` overrides the shared ``_build_slot_batch_core`` implementation
    (default) — the server passes a ``torch.compile``d core here so the
    on-device build runs compiled while tests/other callers keep the eager
    default.
    """
    device = torch.device(device)
    b = len(states)
    if b == 0:
        raise ValueError("build_slot_batch: empty state list")

    stone_rows, to_move_l, player_feats_l, moves_feats_l = _parse_states(states)

    st = torch.tensor(stone_rows, dtype=torch.int64).to(device)  # [Ts, 4]
    sgame, sq, sr, skind = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    if int(torch.maximum(sq.abs().max(), sr.abs().max())) > 32000:
        # §1 of the plan: coords leaving [-32000, 32000] are an assertion
        # failure (HexKey packs q,r into 16-bit fields) — checked BEFORE pack,
        # which would silently wrap them.
        raise ValueError(
            "build_slot_batch: coordinate outside the packable ±32000 range"
        )
    core = core_fn if core_fn is not None else _build_slot_batch_core
    return core(
        sgame, pack(sq, sr), skind,
        torch.tensor(to_move_l, dtype=torch.int64, device=device),
        torch.tensor(player_feats_l, dtype=torch.float32, device=device),
        torch.tensor(moves_feats_l, dtype=torch.float32, device=device),
        b, config, device, pad_to, return_aux,
    )


def build_slot_batch_from_keys(
    games: list,
    config: SlotBuilderConfig,
    device: torch.device | str = "cpu",
    pad_to: int | None = None,
    return_aux: bool = False,
    core_fn=None,
) -> "SlotBatch | tuple[SlotBatch, SlotBatchAux]":
    """Keys-input fast path of :func:`build_slot_batch` for the
    MSG_FORWARD_STATES wire: stones arrive as canonical int32 HexKeys (§1) and
    are fed to the batched builder DIRECTLY — no per-stone unpack/repack; (q, r)
    are recovered once, vectorised, inside the shared core (they are needed for
    the coordinate-derived node features anyway).

    ``games`` is a list of ``(p1_keys, p2_keys, current_player,
    moves_remaining)`` where ``p1_keys``/``p2_keys`` are sequences of canonical
    int32 HexKeys (P1 / P2 stones) and ``current_player`` is ``'P1'``/``'P2'``
    or wire ``0``/``1``. Semantics (validation, ordering, outputs) are
    identical to :func:`build_slot_batch` on the equivalent unpacked states.
    """
    device = torch.device(device)
    b = len(games)
    if b == 0:
        raise ValueError("build_slot_batch: empty state list")

    keys_l: list[int] = []
    counts: list[int] = []  # interleaved [n_p1_0, n_p2_0, n_p1_1, ...]
    to_move_l: list[int] = []
    player_feats_l: list[float] = []
    moves_feats_l: list[float] = []
    for gi, (p1_keys, p2_keys, current_player, moves_remaining) in enumerate(games):
        if current_player is None:
            raise ValueError(
                f"build_slot_batch: state {gi} is terminal (no current player)"
            )
        if moves_remaining not in (1, 2):
            raise ValueError(
                f"build_slot_batch: state {gi} has moves_remaining="
                f"{moves_remaining!r}, expected 1 or 2"
            )
        if len(p1_keys) + len(p2_keys) == 0:
            raise ValueError(
                f"build_slot_batch: state {gi} has no stones — the wire always "
                "carries the engine's (0,0) seed stone, and the legal region "
                "is undefined without stones"
            )
        kind = _norm_player(current_player)
        to_move_l.append(kind)
        player_feats_l.append(1.0 if kind == KIND_P1 else -1.0)
        moves_feats_l.append(moves_remaining / 2.0)
        keys_l.extend(p1_keys)
        keys_l.extend(p2_keys)
        counts.extend((len(p1_keys), len(p2_keys)))

    skey = torch.tensor(keys_l, dtype=torch.int32).to(device)
    cnt = torch.tensor(counts, dtype=torch.int64, device=device)
    sgame = torch.repeat_interleave(
        torch.arange(b, dtype=torch.int64, device=device), cnt[0::2] + cnt[1::2]
    )
    skind = torch.repeat_interleave(
        torch.tensor([KIND_P1, KIND_P2], dtype=torch.int64, device=device).repeat(b),
        cnt,
    )
    q, r = unpack(skey)
    if int(torch.maximum(q.abs().max(), r.abs().max())) > 32000:
        # Any i32 key decodes to i16 coords, but the walk-safety margin of the
        # sentinel (§1) requires |q|,|r| ≤ 32000 — same contract as the
        # states-input path.
        raise ValueError(
            "build_slot_batch: coordinate outside the packable ±32000 range"
        )
    core = core_fn if core_fn is not None else _build_slot_batch_core
    return core(
        sgame, skey, skind,
        torch.tensor(to_move_l, dtype=torch.int64, device=device),
        torch.tensor(player_feats_l, dtype=torch.float32, device=device),
        torch.tensor(moves_feats_l, dtype=torch.float32, device=device),
        b, config, device, pad_to, return_aux,
    )


def _build_slot_batch_core(
    sgame: Tensor,
    skey: Tensor,
    skind: Tensor,
    to_move_kind: Tensor,
    player_feat: Tensor,
    moves_feat: Tensor,
    b: int,
    config: SlotBuilderConfig,
    device: torch.device,
    pad_to: int | None,
    return_aux: bool,
) -> "SlotBatch | tuple[SlotBatch, SlotBatchAux]":
    """Shared core of :func:`build_slot_batch` / :func:`build_slot_batch_from_keys`.

    ``sgame``/``skind`` int64 [Ts], ``skey`` int32 [Ts] canonical HexKeys
    (unsorted flat stones), ``to_move_kind`` int64 [B], ``player_feat``/
    ``moves_feat`` float32 [B]. Coordinates are recovered from the keys after
    the sort (pack/unpack is an exact roundtrip for in-range coords).
    """
    wl = config.win_length
    window = wl - 1
    radius = config.placement_radius
    if wl < 2:
        raise ValueError(f"build_slot_batch: win_length={wl} must be ≥ 2")
    if radius < 1:
        raise ValueError(f"build_slot_batch: placement_radius={radius} must be ≥ 1")
    n_stones_total = skey.shape[0]

    # ---- sort stones by (game, key): int32 key order == (q, r) order (§1),
    # the +2^31 bias maps it onto unsigned order so it packs under the game id.
    scomb = (sgame << 32) | (skey.to(torch.int64) + 0x8000_0000)
    order = torch.argsort(scomb)
    scomb_s = scomb[order]
    skey_s = skey[order]
    skind_s = skind[order]
    sgame_s = sgame[order]
    sq_s, sr_s = unpack(skey_s)  # exact pack-inverse for in-range coords
    if n_stones_total > 1 and bool((scomb_s[1:] == scomb_s[:-1]).any()):
        raise ValueError("build_slot_batch: duplicate stone coordinate in a state")

    # ---- legal region: union of hex-disks(radius) around stones, minus stones
    # (hexo_engine::legal_moves semantics), deduped per game via torch.unique.
    disk = _disk_deltas(radius, device)  # [K] packed deltas
    cand_keys = _wrap_i32(skey_s.to(torch.int64)[:, None] + disk.to(torch.int64)[None, :])
    cand_comb = (sgame_s[:, None] << 32) | (cand_keys.to(torch.int64) + 0x8000_0000)
    ucomb = torch.unique(cand_comb.reshape(-1))  # sorted ascending
    pos = torch.searchsorted(scomb_s, ucomb)
    occupied = (pos < n_stones_total) & (
        scomb_s[pos.clamp(max=n_stones_total - 1)] == ucomb
    )
    lcomb = ucomb[~occupied]
    lgame = lcomb >> 32
    lkey = ((lcomb & 0xFFFF_FFFF) - 0x8000_0000).to(torch.int32)

    # ---- per-game counts and identity positions (stones-first node order)
    ns = torch.bincount(sgame_s, minlength=b)  # [B]
    nl = torch.bincount(lgame, minlength=b)
    n_real = ns + nl
    n_max = int(n_real.max())
    if pad_to is not None:
        if pad_to < n_max:
            raise ValueError(
                f"pad_to={pad_to} smaller than largest graph ({n_max} real nodes)"
            )
        n_max = pad_to
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    scum = torch.cat([zero, ns.cumsum(0)])
    lcum = torch.cat([zero, nl.cumsum(0)])
    # Flat arrays are sorted by (game, key) with games contiguous, so the
    # within-game rank is arange minus the game's start offset.
    spos = torch.arange(n_stones_total, device=device) - scum[sgame_s]
    lpos = ns[lgame] + torch.arange(lgame.shape[0], device=device) - lcum[lgame]

    # ---- padded identity-order tables (keys / kinds / coords / masks)
    keys_id = torch.full((b, n_max), _KEY_SENTINEL, dtype=torch.int32, device=device)
    kinds_id = torch.full((b, n_max), KIND_EMPTY, dtype=torch.int8, device=device)
    keys_id[sgame_s, spos] = skey_s
    keys_id[lgame, lpos] = lkey
    kinds_id[sgame_s, spos] = skind_s.to(torch.int8)

    ar = torch.arange(n_max, device=device)
    node_mask = ar[None, :] < n_real[:, None]
    stone_mask = ar[None, :] < ns[:, None]
    legal_mask = node_mask & ~stone_mask

    qs, rs = unpack(keys_id)  # int64 [B, N]; pad rows carry sentinel coords →
    qs = torch.where(node_mask, qs, torch.zeros_like(qs))  # zero them (inert)
    rs = torch.where(node_mask, rs, torch.zeros_like(rs))

    # ---- one global row-sorted lookup view (sentinels sort last)
    sorted_keys, sort_idx = torch.sort(keys_id, dim=1)
    kinds_sorted = torch.gather(kinds_id, 1, sort_idx)

    # ---- edge slots: batched port of build_edge_slots
    deltas = axis_deltas(wl, device)  # [3, 2, W] int32
    cand = _wrap_i32(
        keys_id.to(torch.int64)[:, :, None, None, None]
        + deltas.to(torch.int64)[None, None]
    )  # [B, N, 3, 2, W]
    s_flat = 3 * 2 * window
    cand2 = cand.reshape(b, -1)
    pos = torch.searchsorted(sorted_keys, cand2)
    pos_c = pos.clamp(max=n_max - 1)
    present = (pos < n_max) & (torch.gather(sorted_keys, 1, pos_c) == cand2)
    present = present.reshape(b, n_max, 3, 2, window) & node_mask[:, :, None, None, None]
    partner = torch.where(
        present,
        torch.gather(sort_idx, 1, pos_c).reshape(b, n_max, 3, 2, window),
        torch.zeros((), dtype=torch.int64, device=device),
    )

    tk = torch.gather(kinds_id, 1, partner.reshape(b, -1)).reshape(partner.shape)
    wk = kinds_id[:, :, None, None, None]
    stop = present & torch.where(wk == KIND_EMPTY, tk != KIND_EMPTY, tk == (1 - wk))
    # Unrolled cumulative-AND over the size-W walk dim — bit-identical to the
    # ``cumprod`` scan (reach[d] = present[d] & cont[0] & ... & cont[d-1]) but
    # ~28x faster on the scan segment and compile-friendly (``torch.cumprod``
    # over this tiny dim is pathological on ROCm — scripts/bench_build_hotspots.py).
    cont = present & ~stop
    reaches = []
    acc = torch.ones_like(cont[..., 0])
    for d in range(window):
        reaches.append(present[..., d] & acc)
        acc = acc & cont[..., d]
    reach = torch.stack(reaches, dim=-1)

    # Union of both endpoints' walks (REQUIRED — plan §7 sharp edge 2): slot
    # (i, a, s, d) also fills if the partner's mirrored slot (j, a, -s, d)
    # reached i. flip on the sign dim + a same-shape gather along the node dim.
    reach_flip = reach.flip(dims=[3]).reshape(b, n_max, s_flat)
    mirror = torch.gather(reach_flip, 1, partner.reshape(b, n_max, s_flat))
    filled = reach | (present & mirror.reshape(partner.shape))
    if config.prune_empty_edges:
        filled = filled & ~((wk == KIND_EMPTY) & (tk == KIND_EMPTY))

    # ---- node features: batched port of build_node_features
    layout = _node_layout(
        config.relative_stones, config.node_coords, config.moves_scope,
        config.compact_stone_onehot,
    )
    base_dim = layout["base_dim"]
    fdim = base_dim + (4 if config.threat_features else 0)
    x = torch.zeros((b, n_max, fdim), dtype=torch.float32, device=device)
    fzero = torch.zeros((), dtype=torch.float32, device=device)

    # stone one-hot (own/opp)
    own_is_p1 = player_feat > 0.0  # [B]
    if config.relative_stones:
        to_own = (skind_s == KIND_P1) == own_is_p1[sgame_s]
    else:
        to_own = skind_s == KIND_P1
    x[sgame_s[to_own], spos[to_own], layout["own"]] = 1.0
    x[sgame_s[~to_own], spos[~to_own], layout["opp"]] = 1.0

    # empty one-hot for legal nodes
    if layout["empty"] is not None:
        x[lgame, lpos, layout["empty"]] = 1.0

    # to_move (absolute only) + moves, on all real nodes
    if layout["to_move"] is not None:
        x[:, :, layout["to_move"]] = torch.where(node_mask, player_feat[:, None], fzero)
    if layout["moves"] is not None:
        x[:, :, layout["moves"]] = torch.where(node_mask, moves_feat[:, None], fzero)

    # centroid / spread over stones (f64, like the Rust builder) → norm_q/norm_r
    if config.node_coords:
        sumq = torch.zeros(b, dtype=torch.float64, device=device)
        sumr = torch.zeros(b, dtype=torch.float64, device=device)
        sumq.index_add_(0, sgame_s, sq_s.double())
        sumr.index_add_(0, sgame_s, sr_s.double())
        cq = sumq / ns.double()
        cr = sumr / ns.double()
        dev = torch.maximum(
            (sq_s.double() - cq[sgame_s]).abs(), (sr_s.double() - cr[sgame_s]).abs()
        )
        spread = torch.zeros(b, dtype=torch.float64, device=device)
        spread.scatter_reduce_(0, sgame_s, dev, reduce="amax", include_self=False)
        spread = spread.clamp(min=1.0)
        norm_q = ((qs.double() - cq[:, None]) / spread[:, None]).float()
        norm_r = ((rs.double() - cr[:, None]) / spread[:, None]).float()
        x[:, :, layout["norm_q"]] = torch.where(node_mask, norm_q, fzero)
        x[:, :, layout["norm_r"]] = torch.where(node_mask, norm_r, fzero)

    # inv_dist on legal nodes: 1 / max(min hex-dist to any stone, 1)
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

    # threat dims (real nodes only)
    if config.threat_features:
        tf = _threat_features_batched(qs, rs, sorted_keys, kinds_sorted, to_move_kind, wl)
        x[:, :, base_dim:base_dim + 4] = torch.where(
            node_mask[:, :, None], tf.float(), fzero
        )

    # dummy node: moves (+ to_move in absolute mode); everything else stays 0
    dummy_x = torch.zeros((b, fdim), dtype=torch.float32, device=device)
    if layout["to_move"] is not None:
        dummy_x[:, layout["to_move"]] = player_feat
    if layout["moves"] is not None:
        dummy_x[:, layout["moves"]] = moves_feat

    src_player = (kinds_id == KIND_P1).float() - (kinds_id == KIND_P2).float()

    batch = SlotBatch(
        x=x,
        dummy_x=dummy_x,
        partner=partner.reshape(b, n_max, s_flat),
        filled=filled.reshape(b, n_max, s_flat),
        src_player=src_player,
        node_mask=node_mask,
        stone_mask=stone_mask,
        legal_mask=legal_mask,
    )
    if not return_aux:
        return batch
    # lkey/lgame come out of torch.unique sorted by (game, key) — exactly the
    # identity order legal nodes were scattered at (lpos), so this IS the
    # batch's legal-column order.
    return batch, SlotBatchAux(legal_keys=lkey, legal_counts=nl)
