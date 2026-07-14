"""Slot-based static-shape model variant (Workstream A2).

Consumes the fixed-slot axis-window graphs from :mod:`hexo_a0.slot_graph`
(dense ``partner``/``filled`` ``[N, 3, 2, W]`` tensors, padded to a static
``[B, N_max, ...]`` batch shape) instead of ragged PyG ``edge_index`` +
``(E, 5) edge_attr``. Message passing per layer is a single gather + masked
sum over the 30 slots — no scatter, no per-edge linear.

Exact functional continuity with the legacy GINE path: the legacy per-layer
edge embedding is the affine chain ``conv.lin(edge_proj(attr))`` where
``attr = [axis_onehot(3), signed_dist, src_player]``. The slot index
``(axis, sign, dist)`` fully determines the first 4 components and
``src_player`` is a static property of the gathered source node, so the chain
decomposes EXACTLY into

    edge_emb = slot_table[axis, sign, dist] + src_player * src_vec

with the all-zero-attr dummy (global) edges collapsing to a constant
``dummy_emb`` per layer. :func:`slot_model_from_legacy` computes these tables
from a trained legacy checkpoint's ``edge_proj`` + per-conv ``lin`` weights;
the converted model reproduces the legacy forward to float tolerance
(``tests/test_slot_model.py``).

Scope (A2): ``conv_type='gine'`` + ``graph_type='axis'`` legacy schema only —
the production configuration. GATv2 attention over slots, the lean
``axis_relational`` schema, and jk_mode='lstm' are not ported. Train-only
extras (q_head, value_horizons) are deferred to the training integration.

No ``torch_geometric`` dependency (by design — the point is static shapes and
CUDA-graph friendliness via ``torch.compile(mode="reduce-overhead")``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
from torch import Tensor

from hexo_a0.config import node_feature_dim
from hexo_a0.loss import decode_binned_value, value_bin_centers
from hexo_a0.model import PolicyHead, ValueHead

# kind codes (slot_graph convention): 0 = P1 stone, 1 = P2 stone, 2 = empty.
_KIND_P1, _KIND_P2 = 0, 1


# --------------------------------------------------------------------------
# Padded batch container + collate
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
        src_player[i, :n] = (kinds == _KIND_P1).float() - (kinds == _KIND_P2).float()
        node_mask[i, :n] = True
        stone_mask[i, :n] = g["stone_mask"][:-1]
        legal_mask[i, :n] = g["legal_mask"][:-1]

    return SlotBatch(
        x=x, dummy_x=dummy_x, partner=partner, filled=filled,
        src_player=src_player, node_mask=node_mask,
        stone_mask=stone_mask, legal_mask=legal_mask,
    )


# --------------------------------------------------------------------------
# Representation network
# --------------------------------------------------------------------------
class SlotRepresentationNetwork(nn.Module):
    """GINE-style message passing over fixed slots, static shapes throughout.

    Mirrors ``RepresentationNetwork`` (conv_type='gine', graph_type='axis',
    legacy schema) layer for layer: pre-/post-norm residual blocks, optional
    LayerScale, optional JK ('sum'/'cat'/'max'). Per layer the parameters are

    - ``slot_tables[l]``: [S, H] — edge embedding per (axis, sign, dist) slot
      (legacy: ``conv.lin(edge_proj(attr))`` at ``src_player=0``),
    - ``src_vecs[l]``:   [H] — the src_player direction
      (legacy: ``conv.lin.weight @ edge_proj.weight[:, 4]``),
    - ``dummy_embs[l]``: [H] — the all-zero-attr dummy-edge embedding
      (legacy: ``conv.lin(edge_proj(0))``),
    - ``mlps[l]``: the GINE update MLP (legacy ``conv.nn``).

    The dummy (global) node is carried as a separate ``[B, H]`` state: it
    receives the masked sum of ``relu(x_i + dummy_emb)`` over real nodes and
    broadcasts ``relu(x_dummy + dummy_emb)`` back to every real node —
    exactly the legacy bidirectional dummy edges.
    """

    def __init__(self, config, win_length: int) -> None:
        super().__init__()
        if getattr(config, "graph_type", "axis") != "axis":
            raise ValueError("slot model requires graph_type='axis'")
        if getattr(config, "axis_relational", False):
            raise ValueError("slot model covers the legacy schema, not axis_relational")
        conv_type = getattr(config, "conv_type", "gatv2")
        if conv_type != "gine":
            raise ValueError(
                f"slot model supports conv_type='gine' only, got {conv_type!r}"
            )
        if str(getattr(config, "moves_scope", "node")) != "node":
            # 'graph' drops the per-node moves column and expects the MODEL to
            # inject the scalar — the slot model has no such injection, so it
            # would silently lose the moves-remaining signal.
            raise ValueError("slot model supports moves_scope='node' only")

        h = config.hidden_dim
        layers = config.num_layers
        self.hidden_dim = h
        self.window = win_length - 1
        self.num_slots = 6 * self.window

        self.input_proj = nn.Linear(node_feature_dim(config), h)

        self.slot_tables = nn.Parameter(torch.empty(layers, self.num_slots, h))
        self.src_vecs = nn.Parameter(torch.empty(layers, h))
        self.dummy_embs = nn.Parameter(torch.empty(layers, h))
        nn.init.normal_(self.slot_tables, std=h ** -0.5)
        nn.init.normal_(self.src_vecs, std=h ** -0.5)
        nn.init.normal_(self.dummy_embs, std=h ** -0.5)
        # GINE eps (non-trainable in the legacy config: GINEConv default).
        self.register_buffer("eps", torch.zeros(layers))

        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))

        self.use_layer_scale = bool(getattr(config, "use_layer_scale", False))
        if self.use_layer_scale:
            self.layer_scales = nn.ParameterList(
                nn.Parameter(torch.ones(h)) for _ in range(layers)
            )

        self.use_jk = bool(getattr(config, "use_jk", False))
        self.jk_mode = str(getattr(config, "jk_mode", "sum"))
        if self.use_jk:
            if self.jk_mode not in ("sum", "cat", "max"):
                raise ValueError(
                    f"slot model supports jk_mode 'sum'/'cat'/'max', got {self.jk_mode!r}"
                )
            if self.jk_mode == "sum":
                self.jk_weights = nn.Parameter(torch.zeros(layers))

        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.final_norm = nn.LayerNorm(h)

        self.output_dim = (
            layers * h if (self.use_jk and self.jk_mode == "cat") else h
        )
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

    def forward(self, batch: SlotBatch) -> Tensor:
        """Node embeddings ``[B, N, output_dim]`` (real nodes; padded rows are
        garbage and must be masked by the caller — the heads do)."""
        b, n, _ = batch.x.shape
        h = self.hidden_dim
        s = self.num_slots
        if batch.filled.shape[-1] != s:
            raise ValueError(
                f"batch has {batch.filled.shape[-1]} slots but the model expects "
                f"{s} — graph builder and model disagree on win_length"
            )

        x = self.input_proj(batch.x)          # [B, N, H]
        xd = self.input_proj(batch.dummy_x)   # [B, H]

        nm = batch.node_mask.unsqueeze(-1).to(x.dtype)      # [B, N, 1]
        fl = batch.filled.unsqueeze(-1).to(x.dtype)         # [B, N, S, 1]
        partner_flat = batch.partner.reshape(b, n * s)      # [B, N*S]
        # src_player of each slot's SOURCE node (a static node property).
        sp = torch.gather(batch.src_player, 1, partner_flat)
        sp = sp.reshape(b, n, s, 1)

        hs: list[Tensor] = []
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norms)):
            residual, residual_d = x, xd
            if self.pre_norm:
                xn, xdn = norm(x), norm(xd)
            else:
                xn, xdn = x, xd

            # Gather each slot's source embedding: [B, N, S, H].
            msgs = torch.gather(
                xn, 1, partner_flat.unsqueeze(-1).expand(b, n * s, h)
            ).reshape(b, n, s, h)
            edge = self.slot_tables[i].view(1, 1, s, h) + sp * self.src_vecs[i].view(1, 1, 1, h)
            agg = (torch.relu(msgs + edge) * fl).sum(dim=2)  # [B, N, H]

            # Dummy relation (legacy bidirectional zero-attr edges).
            g = self.dummy_embs[i]
            agg = agg + torch.relu(xdn + g).unsqueeze(1) * nm
            agg_d = (torch.relu(xn + g) * nm).sum(dim=1)     # [B, H]

            eps1 = 1.0 + self.eps[i]
            out = mlp(eps1 * xn + agg)
            out_d = mlp(eps1 * xdn + agg_d)
            if self.use_layer_scale:
                out = self.layer_scales[i] * out
                out_d = self.layer_scales[i] * out_d

            x = out + residual
            xd = out_d + residual_d
            if not self.pre_norm:
                x, xd = norm(x), norm(xd)
            x, xd = torch.relu(x), torch.relu(xd)
            x, xd = self.dropout(x), self.dropout(xd)
            if self.use_jk:
                hs.append(x)

        if self.use_jk:
            if self.jk_mode == "sum":
                stacked = torch.stack(hs, dim=0)                     # [L, B, N, H]
                w = self.jk_weights.softmax(dim=0).view(-1, 1, 1, 1)
                x = (w * stacked).sum(dim=0)
                if self.pre_norm:
                    x = self.final_norm(x)
            elif self.jk_mode == "cat":
                if self.pre_norm:
                    hs = [self.final_norm(hh) for hh in hs]
                x = torch.cat(hs, dim=-1)                            # [B, N, L*H]
            else:  # max
                x = torch.stack(hs, dim=0).max(dim=0).values
                if self.pre_norm:
                    x = self.final_norm(x)
        elif self.pre_norm:
            x = self.final_norm(x)

        return x


class SlotHeXONet(nn.Module):
    """Slot-model counterpart of ``HeXONet``: representation + policy/value.

    Head modules are the same classes as the legacy model (their MLPs are
    applied to padded ``[B, N, D]`` tensors here), so legacy head weights load
    directly.
    """

    def __init__(self, config, win_length: int) -> None:
        super().__init__()
        self.representation = SlotRepresentationNetwork(config, win_length)
        head_in = self.representation.output_dim
        self.value_bins = int(getattr(config, "value_bins", 0) or 0)
        self.policy_head = PolicyHead(head_in, config.policy_hidden)
        self.value_head = ValueHead(head_in, config.value_hidden, value_bins=self.value_bins)
        if self.value_bins > 0:
            self.register_buffer(
                "value_bin_centers",
                value_bin_centers(
                    self.value_bins,
                    float(getattr(config, "value_bin_min", -1.0)),
                    float(getattr(config, "value_bin_max", 1.0)),
                ),
                persistent=False,
            )

    def forward_padded(self, batch: SlotBatch) -> tuple[Tensor, Tensor]:
        """Static-shape forward: ``(logits [B, N], values [B])``.

        ``logits`` is ``-inf`` outside ``legal_mask`` (padding included), so a
        softmax over dim 1 is the move distribution directly.
        """
        emb = self.representation(batch)                       # [B, N, D]
        logits = self.policy_head.mlp(emb).squeeze(-1)         # [B, N]
        logits = logits.masked_fill(~batch.legal_mask, float("-inf"))

        sm = batch.stone_mask.unsqueeze(-1).to(emb.dtype)
        pooled = (emb * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1.0)
        if self.value_bins > 0:
            values = decode_binned_value(
                self.value_head.mlp(pooled), self.value_bin_centers
            )
        else:
            values = self.value_head.mlp(pooled).squeeze(-1)
        return logits, values

    def forward_batch(self, batch: SlotBatch) -> tuple[list[Tensor], Tensor]:
        """Legacy-compatible API: per-graph legal logits (node order) + values."""
        logits, values = self.forward_padded(batch)
        policy_list = [
            logits[i][batch.legal_mask[i]] for i in range(batch.num_graphs)
        ]
        return policy_list, values


# --------------------------------------------------------------------------
# Conversion from a legacy HeXONet (exact functional continuity)
# --------------------------------------------------------------------------
def _slot_base_attrs(win_length: int) -> Tensor:
    """The legacy 5-dim edge attr for each slot at ``src_player = 0``,
    ordered to match the ``[3, 2, W] -> S`` flattening. Slot ``(a, si, d)``
    is the incoming edge FROM the node at offset ``sign * d`` along axis
    ``a``, so its src->dst signed distance is ``-sign * d``."""
    window = win_length - 1
    attrs = torch.zeros(3, 2, window, 5)
    for a in range(3):
        for si, sign in enumerate((1, -1)):
            for d in range(1, window + 1):
                attrs[a, si, d - 1, a] = 1.0
                attrs[a, si, d - 1, 3] = float(-sign * d)
    return attrs.reshape(-1, 5)


def slot_model_from_legacy(model, config, win_length: int) -> SlotHeXONet:
    """Build a :class:`SlotHeXONet` reproducing a legacy ``HeXONet`` exactly.

    Requires the legacy model to be the production shape: ``graph_type='axis'``
    + ``conv_type='gine'`` (GINEConv with the shared ``edge_proj``). Train-only
    extras (q_head, value_horizons) are not carried over.
    """
    rep = model.representation
    if not hasattr(rep, "edge_proj"):
        raise ValueError(
            "slot_model_from_legacy needs a legacy axis-graph model with "
            "edge_proj (graph_type='axis', conv_type='gine')"
        )
    # `config` must describe the checkpoint's actual architecture — a mismatch
    # would otherwise silently produce a diverging model (surplus random-init
    # layers, dropped final_norm, ...). Fail loudly instead.
    if config.num_layers != len(rep.convs):
        raise ValueError(
            f"config.num_layers={config.num_layers} != checkpoint layers {len(rep.convs)}"
        )
    if config.hidden_dim != rep.hidden_dim:
        raise ValueError(
            f"config.hidden_dim={config.hidden_dim} != checkpoint hidden {rep.hidden_dim}"
        )
    if bool(config.pre_norm) != bool(rep.pre_norm):
        raise ValueError(
            f"config.pre_norm={config.pre_norm} != checkpoint pre_norm {rep.pre_norm}"
        )
    if bool(getattr(config, "use_layer_scale", False)) != rep.use_layer_scale:
        raise ValueError(
            f"config.use_layer_scale mismatch: config "
            f"{getattr(config, 'use_layer_scale', False)} vs checkpoint {rep.use_layer_scale}"
        )
    cfg_jk = (bool(getattr(config, "use_jk", False)), str(getattr(config, "jk_mode", "sum")))
    model_jk = (rep.use_jk, rep.jk_mode)
    if cfg_jk[0] != model_jk[0] or (cfg_jk[0] and cfg_jk[1] != model_jk[1]):
        raise ValueError(f"config jk settings {cfg_jk} != checkpoint {model_jk}")

    device = rep.input_proj.weight.device
    slot = SlotHeXONet(config, win_length).to(device)
    srep = slot.representation

    attrs = _slot_base_attrs(win_length).to(device)
    with torch.no_grad():
        srep.input_proj.weight.copy_(rep.input_proj.weight)
        srep.input_proj.bias.copy_(rep.input_proj.bias)

        zero_attr = torch.zeros(1, 5, device=device)
        for i, conv in enumerate(rep.convs):
            lin = getattr(conv, "lin", None)
            if lin is None:
                raise ValueError(
                    "slot_model_from_legacy supports conv_type='gine' convs "
                    f"with an edge lin; layer {i} is {type(conv).__name__}"
                )
            srep.slot_tables[i] = lin(rep.edge_proj(attrs))
            srep.src_vecs[i] = lin.weight @ rep.edge_proj.weight[:, 4]
            srep.dummy_embs[i] = lin(rep.edge_proj(zero_attr)).squeeze(0)
            srep.eps[i] = float(conv.eps)
            srep.mlps[i].load_state_dict(conv.nn.state_dict())
            srep.norms[i].load_state_dict(rep.norms[i].state_dict())

        if srep.use_layer_scale:
            for i in range(len(rep.layer_scales)):
                srep.layer_scales[i].copy_(rep.layer_scales[i])
        if srep.use_jk and srep.jk_mode == "sum":
            srep.jk_weights.copy_(rep.jk_weights)
        if srep.pre_norm:
            srep.final_norm.load_state_dict(rep.final_norm.state_dict())

    slot.policy_head.load_state_dict(model.policy_head.state_dict())
    slot.value_head.load_state_dict(model.value_head.state_dict())
    return slot
