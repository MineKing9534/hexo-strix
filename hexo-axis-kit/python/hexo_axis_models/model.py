from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fused_line import axis_line_gather
from .ops import (
    AXIS_RAY_PAIRS,
    NUM_RAYS,
    PACKED_RAY_RADIUS,
    RAY_DIRS,
    ChannelLayerNorm,
    PointwiseMLP,
    SpatialMLPHead,
    gather_dense_actions,
    masked_mean,
    roll_source,
    unpack_ray_bits,
)


class AxisModelOutput(NamedTuple):
    policy_logits: Tensor  # [B,H,W], illegal cells set to dtype minimum
    q_values: Tensor       # [B,H,W], illegal cells set to zero
    value: Tensor          # [B]
    value_logits: Tensor   # [B,bins], or [B,0] for scalar value


@dataclass(frozen=True)
class AxisGineConfig:
    """Checkpoint-portable dense compilation of Strix's axis-relational GNN."""

    plane_count: int = 8
    scalar_count: int = 5
    input_dim: int = 8  # exact lean Strix node schema assembled internally
    hidden_dim: int = 128
    num_layers: int = 4
    line_radius: int = 5
    distance_bins: int = 8
    policy_hidden: int = 128
    q_hidden: int = 64
    value_hidden: int = 32
    value_bins: int = 65
    value_bin_min: float = -1.0
    value_bin_max: float = 1.0
    value_horizons: tuple[int, ...] = (4, 12, 32)
    jk_mode: str = "cat"  # "none", "sum", or "cat"
    use_layer_scale: bool = False
    use_extra_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.plane_count < 8:
            raise ValueError("plane_count must be at least 8")
        if self.scalar_count < 1:
            raise ValueError("scalar_count must be at least 1")
        if self.input_dim != 8:
            raise ValueError("the checkpoint-compatible lean schema has input_dim=8")
        if not 1 <= self.line_radius <= PACKED_RAY_RADIUS:
            raise ValueError(f"line_radius must be in 1..={PACKED_RAY_RADIUS}")
        if self.distance_bins < self.line_radius:
            raise ValueError("distance_bins must be at least line_radius")
        if self.jk_mode not in {"none", "sum", "cat"}:
            raise ValueError("jk_mode must be one of: none, sum, cat")
        if self.value_bins not in {0} and self.value_bins < 2:
            raise ValueError("value_bins must be 0 or >=2")


@dataclass(frozen=True)
class PersistentRayConfig(AxisGineConfig):
    """Axis-GINE backbone plus a narrow persistent six-ray side stream."""

    ray_channels: int = 12
    ray_update_hidden: int = 48
    ray_branch_scale: float = 1.0
    exact_graft_init: bool = True
    ray_after_layers: tuple[int, ...] = field(default_factory=tuple)

    def active_ray_layers(self) -> tuple[int, ...]:
        if self.ray_after_layers:
            return self.ray_after_layers
        return tuple(range(self.num_layers))


class ValueHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bins: int,
        vmin: float,
        vmax: float,
    ) -> None:
        super().__init__()
        self.bins = int(bins)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, bins if bins > 0 else 1)
        centers = torch.linspace(vmin, vmax, bins) if bins > 0 else torch.zeros(0)
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.fc2(F.relu(self.fc1(pooled)))
        if self.bins > 0:
            value = (torch.softmax(logits.float(), dim=-1) * self.centers.float()).sum(dim=-1)
            return value.to(logits.dtype), logits
        value = torch.tanh(logits.squeeze(-1))
        empty = logits.new_empty((logits.shape[0], 0))
        return value, empty


class DenseAxisGineBlock(nn.Module):
    """Dense destination-gather equivalent of one AxisRelationalConv block.

    The scalar raster trunk corresponds to real graph nodes. ``global_state``
    corresponds to the graph's dummy node. Exact Rust ray masks replace
    ``edge_index`` and preserve blocker / pruning semantics.
    """

    def __init__(
        self,
        channels: int,
        radius: int,
        distance_bins: int,
        use_layer_scale: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.radius = int(radius)
        self.norm = ChannelLayerNorm(channels)

        self.distance_embedding = nn.Embedding(distance_bins, channels)
        self.axis_edge_proj = nn.Linear(channels, channels)
        self.axis_eps = nn.Parameter(torch.zeros(1))
        self.axis_mlp = PointwiseMLP(channels, channels, channels)

        self.global_edge_embed = nn.Parameter(torch.randn(channels) * 0.1)
        self.global_edge_proj = nn.Linear(channels, channels)
        self.global_eps = nn.Parameter(torch.zeros(1))
        self.global_mlp = PointwiseMLP(channels, channels, channels)

        self.node_update = PointwiseMLP(2 * channels, channels, channels)
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(channels))
        else:
            self.register_buffer(
                "layer_scale",
                torch.ones(channels),
                persistent=False,
            )

    def forward(
        self,
        h: Tensor,
        global_state: Tensor,
        active_mask: Tensor,
        ray_mask: Tensor,
        active_flat_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if ray_mask.ndim != 5 or ray_mask.shape[1] != NUM_RAYS:
            raise ValueError("ray_mask must be [B,6,R,H,W]")
        if ray_mask.shape[2] < self.radius:
            raise ValueError(
                f"ray_mask radius {ray_mask.shape[2]} "
                f"< model radius {self.radius}"
            )

        residual_h = h
        residual_g = global_state
        active = active_mask.to(dtype=h.dtype)
        x = self.norm(h) * active
        gx = self.norm.forward_vector(global_state)

        distance_table = self.axis_edge_proj(
            self.distance_embedding.weight
        )
        b, c, height, width = x.shape
        stacked = axis_line_gather(
            x,
            ray_mask,
            distance_table,
            self.axis_eps,
            self.radius,
        )
        if active_flat_indices is None:
            axis_real = self.axis_mlp(stacked).reshape(
                b, 3, c, height, width
            ).sum(dim=1)
        else:
            # Inactive cells are discarded by this block. Apply the expensive
            # three-axis MLP only to live nodes, then scatter its relation sum
            # back into the dense trunk. All parameters and active outputs are
            # identical to the padded 1x1-convolution formulation.
            cells = height * width
            axis_points = (
                stacked.reshape(b, 3, c, cells)
                .permute(1, 0, 3, 2)
                .reshape(3, b * cells, c)
            )
            active_points = axis_points.index_select(
                1, active_flat_indices
            )
            axis_active = self.axis_mlp.forward_vector(
                active_points
            ).sum(dim=0)
            active_batch = torch.div(
                active_flat_indices, cells, rounding_mode="floor"
            )
            x_points = x.permute(0, 2, 3, 1).reshape(b * cells, c)
            x_active = x_points.index_select(0, active_flat_indices)

        # The dummy node has no axis edges, but GINE still applies its self term
        # once per relation. All three outputs are identical because weights tie.
        axis_global = 3.0 * self.axis_mlp.forward_vector(
            (1.0 + self.axis_eps) * gx
        )

        global_edge = self.global_edge_proj(self.global_edge_embed)
        global_to_real = F.relu(gx + global_edge)
        if active_flat_indices is None:
            global_real_input = (
                (1.0 + self.global_eps) * x
                + global_to_real.unsqueeze(-1).unsqueeze(-1)
            )
            global_real = self.global_mlp(global_real_input)
        else:
            global_real_active = self.global_mlp.forward_vector(
                (1.0 + self.global_eps) * x_active
                + global_to_real.index_select(0, active_batch)
            )

        real_to_global = F.relu(x + global_edge.view(1, -1, 1, 1)) * active
        real_to_global = real_to_global.sum(dim=(-2, -1))
        global_input = (1.0 + self.global_eps) * gx + real_to_global
        global_out = self.global_mlp.forward_vector(global_input)

        conv_g = self.node_update.forward_vector(
            torch.cat([gx, axis_global + global_out], dim=1)
        )
        if active_flat_indices is None:
            conv_h = self.node_update(
                torch.cat([x, axis_real + global_real], dim=1)
            )
            scale = self.layer_scale.view(1, -1, 1, 1)
            h = F.relu(residual_h + scale * conv_h) * active
        else:
            conv_active = self.node_update.forward_vector(
                torch.cat(
                    [x_active, axis_active + global_real_active],
                    dim=-1,
                )
            )
            residual_points = residual_h.permute(
                0, 2, 3, 1
            ).reshape(b * cells, c)
            residual_active = residual_points.index_select(
                0, active_flat_indices
            )
            h_active = F.relu(
                residual_active
                + self.layer_scale.view(1, -1) * conv_active
            )
            h_points = h_active.new_zeros((b * cells, c))
            h_points.index_copy_(
                0, active_flat_indices, h_active
            )
            h = h_points.reshape(
                b, height, width, c
            ).permute(0, 3, 1, 2)
        global_state = F.relu(
            residual_g + self.layer_scale.view(1, -1) * conv_g
        )
        return h, global_state


class AxisGineCompatNet(nn.Module):
    """Recommended first model: dense, blocker-aware, D6-equivariant Axis-GINE.

    Its node schema and module algebra intentionally mirror the current lean
    axis-relational Strix model, allowing a checkpoint conversion rather than a
    cold restart. Floating-point reduction order differs, so parity should be
    close rather than bit-identical.
    """

    def __init__(self, config: AxisGineConfig | None = None) -> None:
        super().__init__()
        self.config = config or AxisGineConfig()
        cfg = self.config
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DenseAxisGineBlock(
                    cfg.hidden_dim,
                    cfg.line_radius,
                    cfg.distance_bins,
                    cfg.use_layer_scale,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = ChannelLayerNorm(cfg.hidden_dim)
        if cfg.jk_mode == "sum":
            self.jk_weights = nn.Parameter(torch.zeros(cfg.num_layers))

        if cfg.use_extra_conditioning and cfg.scalar_count > 1:
            self.conditioner = nn.Sequential(
                nn.Linear(cfg.scalar_count - 1, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.conditioner = None

        head_dim = cfg.hidden_dim * cfg.num_layers if cfg.jk_mode == "cat" else cfg.hidden_dim
        self.head_dim = head_dim
        self.policy_head = SpatialMLPHead(head_dim, cfg.policy_hidden)
        self.q_head = SpatialMLPHead(head_dim, cfg.q_hidden, tanh=True)
        self.value_head = ValueHead(
            head_dim,
            cfg.value_hidden,
            cfg.value_bins,
            cfg.value_bin_min,
            cfg.value_bin_max,
        )
        self.horizon_value_heads = nn.ModuleList(
            [
                ValueHead(
                    head_dim,
                    cfg.value_hidden,
                    cfg.value_bins,
                    cfg.value_bin_min,
                    cfg.value_bin_max,
                )
                for _ in cfg.value_horizons
            ]
        )

    @staticmethod
    def _strix_lean_features(planes: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assemble current Strix's 8-D lean relative node schema.

        Rust planes are:
        own, opp, legal, inv_dist, own_line, opp_line, own_axes, opp_axes.
        Current lean GNN input is:
        own, opp, moves_remaining, inv_dist, and four threat features.
        """
        own = planes[:, 0:1]
        opp = planes[:, 1:2]
        legal = planes[:, 2:3] > 0.5
        moves = scalars[:, 0:1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, planes.shape[-2], planes.shape[-1])
        node_features = torch.cat([own, opp, moves, planes[:, 3:8]], dim=1)
        active = ((own + opp) > 0.5) | legal
        stones = (own + opp) > 0.5
        return node_features, active, legal, stones

    def _initial_state(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if planes.ndim != 4:
            raise ValueError(f"planes must be [B,P,H,W], got {tuple(planes.shape)}")
        if scalars.ndim != 2:
            raise ValueError(f"scalars must be [B,S], got {tuple(scalars.shape)}")
        if planes.shape[1] < 8:
            raise ValueError("planes must contain at least the eight standard channels")
        if scalars.shape[1] < 1:
            raise ValueError("scalars must contain moves_remaining")

        features, derived_active, legal, stones = self._strix_lean_features(planes, scalars)
        if active_mask is None:
            active = derived_active
        else:
            if active_mask.ndim == 3:
                active_mask = active_mask.unsqueeze(1)
            active = active_mask.to(torch.bool)
        active_f = active.to(dtype=planes.dtype)

        h = F.linear(
            features.permute(0, 2, 3, 1),
            self.input_proj.weight,
            self.input_proj.bias,
        ).permute(0, 3, 1, 2).contiguous()
        h = h * active_f

        global_features = planes.new_zeros((planes.shape[0], self.config.input_dim))
        global_features[:, 2] = scalars[:, 0]
        global_state = self.input_proj(global_features)

        if self.conditioner is not None:
            condition = self.conditioner(scalars[:, 1 : self.config.scalar_count])
            h = h + condition.unsqueeze(-1).unsqueeze(-1) * active_f
            global_state = global_state + condition
        return h, global_state, active, legal, stones

    def _combine_jk(self, states: list[Tensor]) -> Tensor:
        if self.config.jk_mode == "cat":
            return torch.cat([self.final_norm(h) for h in states], dim=1)
        if self.config.jk_mode == "sum":
            weights = torch.softmax(self.jk_weights, dim=0).view(-1, 1, 1, 1, 1)
            mixed = (weights * torch.stack(states, dim=0)).sum(dim=0)
            return self.final_norm(mixed)
        return self.final_norm(states[-1])

    def forward_features(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
        ray_mask: Tensor,
        active_flat_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h, global_state, active, legal, stones = self._initial_state(planes, scalars, active_mask)
        states: list[Tensor] = []
        for block in self.blocks:
            h, global_state = block(
                h,
                global_state,
                active,
                ray_mask,
                active_flat_indices,
            )
            states.append(h)
        representation = self._combine_jk(states)
        return representation, active, legal, stones

    def _apply_heads(self, representation: Tensor, legal: Tensor, stones: Tensor) -> AxisModelOutput:
        policy = self.policy_head(representation)
        q_values = self.q_head(representation)
        legal_2d = legal.squeeze(1)
        policy = policy.masked_fill(~legal_2d, torch.finfo(policy.dtype).min)
        q_values = q_values.masked_fill(~legal_2d, 0.0)

        pooled = masked_mean(representation, stones.to(representation.dtype), dims=(-2, -1))
        value, value_logits = self.value_head(pooled)
        return AxisModelOutput(policy, q_values, value, value_logits)

    def forward(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
        ray_mask: Tensor,
    ) -> AxisModelOutput:
        representation, _active, legal, stones = self.forward_features(
            planes, scalars, active_mask, ray_mask
        )
        return self._apply_heads(representation, legal, stones)

    def forward_packed(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
        ray_bits: Tensor,
    ) -> AxisModelOutput:
        return self(planes, scalars, active_mask, unpack_ray_bits(ray_bits, self.config.line_radius))

    def forward_with_aux(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
        ray_mask: Tensor,
    ) -> tuple[AxisModelOutput, Tensor]:
        representation, _active, legal, stones = self.forward_features(
            planes, scalars, active_mask, ray_mask
        )
        output = self._apply_heads(representation, legal, stones)
        pooled = masked_mean(representation, stones.to(representation.dtype), dims=(-2, -1))
        if self.horizon_value_heads:
            horizon_logits = torch.stack([head(pooled)[1] for head in self.horizon_value_heads], dim=1)
        else:
            horizon_logits = pooled.new_empty((pooled.shape[0], 0, 0))
        return output, horizon_logits

    @staticmethod
    def gather_legal(dense: Tensor, legal_flat_indices: Tensor) -> Tensor:
        return gather_dense_actions(dense, legal_flat_indices)


class RayRingMixer(nn.Module):
    """D6-equivariant pointwise mixing around the six-direction orientation ring."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.same = nn.Conv2d(channels, channels, 1, bias=False)
        self.adjacent = nn.Conv2d(channels, channels, 1, bias=False)
        self.next_nearest = nn.Conv2d(channels, channels, 1, bias=False)
        self.opposite = nn.Conv2d(channels, channels, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(channels))

    @staticmethod
    def _apply_conv(module: nn.Conv2d, rays: Tensor) -> Tensor:
        b, directions, channels, height, width = rays.shape
        y = module(rays.reshape(b * directions, channels, height, width))
        return y.reshape(b, directions, channels, height, width)

    def forward(self, rays: Tensor) -> Tensor:
        same = self._apply_conv(self.same, rays)
        adjacent = self._apply_conv(
            self.adjacent,
            torch.roll(rays, 1, dims=1) + torch.roll(rays, -1, dims=1),
        )
        next_nearest = self._apply_conv(
            self.next_nearest,
            torch.roll(rays, 2, dims=1) + torch.roll(rays, -2, dims=1),
        )
        opposite = self._apply_conv(self.opposite, torch.roll(rays, 3, dims=1))
        return same + adjacent + next_nearest + opposite + self.bias.view(1, 1, -1, 1, 1)


class PersistentRayMixer(nn.Module):
    """Narrow persistent directed-ray stream with invariant fork folding."""

    def __init__(
        self,
        trunk_channels: int,
        ray_channels: int,
        update_hidden: int,
        radius: int,
        branch_scale: float,
        exact_graft_init: bool,
    ) -> None:
        super().__init__()
        self.trunk_channels = int(trunk_channels)
        self.ray_channels = int(ray_channels)
        self.radius = int(radius)
        self.branch_scale = float(branch_scale)

        self.trunk_norm = ChannelLayerNorm(trunk_channels)
        self.source_proj = nn.Conv2d(trunk_channels, ray_channels, 1)
        self.trunk_proj = nn.Conv2d(trunk_channels, ray_channels, 1)
        self.distance_embedding = nn.Embedding(radius, ray_channels)
        self.edge_proj = nn.Linear(ray_channels, ray_channels)
        self.ring_mixer = RayRingMixer(ray_channels)

        update_in = 4 * ray_channels
        self.update_mlp = PointwiseMLP(update_in, update_hidden, ray_channels)
        self.gate = nn.Conv2d(update_in, ray_channels, 1)

        fold_in = 9 * ray_channels
        fold_hidden = max(trunk_channels, 2 * ray_channels)
        self.fold = PointwiseMLP(fold_in, fold_hidden, trunk_channels)
        if exact_graft_init:
            nn.init.zeros_(self.fold.fc2.weight)
            nn.init.zeros_(self.fold.fc2.bias)

    def _directional_messages(self, h: Tensor, ray_mask: Tensor) -> Tensor:
        source_base = self.source_proj(self.trunk_norm(h))
        distance_table = self.edge_proj(self.distance_embedding.weight)
        directions: list[Tensor] = []
        for ray, (dq, dr) in enumerate(RAY_DIRS):
            aggregate = torch.zeros_like(source_base)
            for distance in range(1, self.radius + 1):
                source = roll_source(source_base, dq * distance, dr * distance)
                edge = distance_table[distance - 1].view(1, -1, 1, 1)
                message = F.silu(source + edge)
                mask = ray_mask[:, ray, distance - 1].unsqueeze(1).to(source.dtype)
                aggregate = aggregate + message * mask
            directions.append(aggregate)
        return torch.stack(directions, dim=1)

    @staticmethod
    def _flatten_rays(x: Tensor) -> tuple[Tensor, tuple[int, int, int, int, int]]:
        shape = x.shape
        b, directions, channels, height, width = shape
        return x.reshape(b * directions, channels, height, width), shape

    @staticmethod
    def _unflatten_rays(x: Tensor, shape: tuple[int, int, int, int, int]) -> Tensor:
        b, directions, channels, height, width = shape
        return x.reshape(b, directions, channels, height, width)

    def _fold_invariant(self, rays: Tensor) -> Tensor:
        axes: list[Tensor] = []
        for a, b in AXIS_RAY_PAIRS:
            ra, rb = rays[:, a], rays[:, b]
            axes.append(torch.cat([ra + rb, ra * rb, (ra - rb).abs()], dim=1))
        axis = torch.stack(axes, dim=1)  # [B,3,3R,H,W]
        axis_sum = axis.sum(dim=1)
        axis_sq_sum = (axis * axis).sum(dim=1)
        pair_product = 0.5 * (axis_sum * axis_sum - axis_sq_sum)
        axis_max = axis.max(dim=1).values
        return self.fold(torch.cat([axis_sum, pair_product, axis_max], dim=1))

    def forward(
        self,
        h: Tensor,
        ray_state: Tensor | None,
        active_mask: Tensor,
        ray_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, _, height, width = h.shape
        if ray_state is None:
            ray_state = h.new_zeros((b, NUM_RAYS, self.ray_channels, height, width))

        directional = self._directional_messages(h, ray_mask)
        ring = self.ring_mixer(ray_state)
        trunk = self.trunk_proj(h).unsqueeze(1).expand(-1, NUM_RAYS, -1, -1, -1)
        update_input = torch.cat([ray_state, directional, trunk, ring], dim=2)
        flat, shape = self._flatten_rays(update_input)
        delta = torch.tanh(self.update_mlp(flat))
        gate = torch.sigmoid(self.gate(flat))
        delta = self._unflatten_rays(delta, (shape[0], shape[1], self.ray_channels, shape[3], shape[4]))
        gate = self._unflatten_rays(gate, (shape[0], shape[1], self.ray_channels, shape[3], shape[4]))
        ray_state = ray_state + gate * delta
        ray_state = ray_state * active_mask.to(h.dtype).unsqueeze(1)

        folded = self._fold_invariant(ray_state)
        h = F.relu(h + self.branch_scale * folded) * active_mask.to(h.dtype)
        return h, ray_state


class PersistentRayAxisNet(AxisGineCompatNet):
    """Recommended strength model: Axis-GINE plus persistent six-ray memory."""

    config: PersistentRayConfig

    def __init__(self, config: PersistentRayConfig | None = None) -> None:
        cfg = config or PersistentRayConfig()
        super().__init__(cfg)
        self.config = cfg
        active_layers = set(cfg.active_ray_layers())
        self.ray_mixers = nn.ModuleList(
            [
                PersistentRayMixer(
                    cfg.hidden_dim,
                    cfg.ray_channels,
                    cfg.ray_update_hidden,
                    cfg.line_radius,
                    cfg.ray_branch_scale,
                    cfg.exact_graft_init,
                )
                if layer in active_layers
                else nn.Identity()
                for layer in range(cfg.num_layers)
            ]
        )

    def forward_features(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor | None,
        ray_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h, global_state, active, legal, stones = self._initial_state(planes, scalars, active_mask)
        states: list[Tensor] = []
        ray_state: Tensor | None = None
        for block, mixer in zip(self.blocks, self.ray_mixers):
            h, global_state = block(h, global_state, active, ray_mask)
            if isinstance(mixer, PersistentRayMixer):
                h, ray_state = mixer(h, ray_state, active, ray_mask)
            states.append(h)
        representation = self._combine_jk(states)
        return representation, active, legal, stones


def axis_gine_compat_4x128() -> AxisGineCompatNet:
    """Preset matching the current 4-layer, 128-wide lean Strix family."""
    return AxisGineCompatNet(AxisGineConfig())


def persistent_ray_4x128() -> PersistentRayAxisNet:
    """Function-preserving six-ray graft on the 4x128 compatibility model."""
    return PersistentRayAxisNet(PersistentRayConfig())
