"""KLENT policy/Q network built on the existing HeXO representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hexo_a0.config import ModelConfig
from hexo_a0.model import PolicyHead, QHead, RepresentationNetwork
from hexo_axis_models import (
    AxisGineCompatNet,
    AxisGineConfig,
    PersistentRayAxisNet,
    PersistentRayConfig,
)
from hexo_axis_models.model import PersistentRayMixer
from hexo_axis_models.checkpoint import (
    ConversionReport,
    convert_strix_axis_state_dict,
    extract_state_dict,
)
from hexo_klent.hex_axial_cnn import (
    HexAxialCNNBackbone,
    HexD6DilatedCNNBackbone,
    HexDilatedCNNBackbone,
)


@dataclass(frozen=True)
class BatchOutput:
    """Flat legal-action outputs and their per-position lengths."""

    policy_logits: Tensor
    q_values: Tensor
    legal_counts: Tensor
    critic_logits: Tensor | None = None
    q_mass: Tensor | None = None


def categorical_q(critic_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Compose three outcome logits into scalar Q and committed mass."""

    if critic_logits.ndim < 1 or critic_logits.shape[-1] != 3:
        raise ValueError("categorical critic logits must end in dimension 3")
    # Spell out the fixed-width reduction. On ROCm, Inductor's generic
    # split-reduction softmax can produce non-finite values for this dynamic
    # legal-action dimension under bf16 autocast. Float32 shifted exponentials
    # are both stable and compile into a simple three-lane pointwise kernel.
    work = critic_logits.float()
    positive_logit, negative_logit, zero_logit = work.unbind(dim=-1)
    maximum = torch.maximum(
        torch.maximum(positive_logit, negative_logit),
        zero_logit,
    )
    positive_weight = (positive_logit - maximum).exp()
    negative_weight = (negative_logit - maximum).exp()
    zero_weight = (zero_logit - maximum).exp()
    inverse_sum = (
        positive_weight + negative_weight + zero_weight
    ).reciprocal()
    positive = positive_weight * inverse_sum
    negative = negative_weight * inverse_sum
    return positive - negative, positive + negative


def acting_q_values(output: BatchOutput, *, mass_floor: float) -> Tensor:
    """Return raw scalar Q or Mantis-style per-position normalized Q."""

    if output.q_mass is None:
        return output.q_values
    if not 0.0 < mass_floor <= 1.0:
        raise ValueError("mass_floor must be in (0, 1]")
    counts = [int(count) for count in output.legal_counts.detach().cpu()]
    q_chunks = output.q_values.split(counts)
    mass_chunks = output.q_mass.split(counts)
    normalized = [
        q / mass.max().clamp_min(mass_floor)
        for q, mass in zip(q_chunks, mass_chunks, strict=True)
    ]
    return torch.cat(normalized)


class CategoricalQHead(nn.Module):
    """Per-action three-outcome critic with the scalar Q head's hidden shape."""

    def __init__(self, hidden_dim: int, q_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, q_hidden),
            nn.ReLU(),
            nn.Linear(q_hidden, 3),
        )


def improved_policy(
    policy_logits: Tensor,
    q_values: Tensor,
    *,
    alpha: float,
    beta: float,
) -> Tensor:
    """KLENT closed-form policy improvement over one position's legal moves."""

    if policy_logits.shape != q_values.shape:
        raise ValueError("policy_logits and q_values must have the same shape")
    denominator = alpha + beta
    if denominator <= 0:
        raise ValueError("alpha + beta must be positive")
    return torch.softmax((beta * policy_logits + q_values) / denominator, dim=0)


class KlentNet(nn.Module):
    """Shared graph representation with policy-logit and per-action Q heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(
            config, graph_type=config.graph_type
        )
        head_dim = self.representation.output_dim
        self.policy_head = PolicyHead(head_dim, config.policy_hidden)
        self.critic_type = str(getattr(config, "critic", "scalar"))
        self.q_head = (
            CategoricalQHead(head_dim, config.q_hidden)
            if self.critic_type == "categorical"
            else QHead(head_dim, config.q_hidden)
        )
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start KLENT at a uniform policy with zero action values."""

        policy_out = self.policy_head.mlp[-1]
        q_out = self.q_head.mlp[2]
        if not isinstance(policy_out, nn.Linear) or not isinstance(q_out, nn.Linear):
            raise TypeError("unexpected policy/Q head layout")
        nn.init.zeros_(policy_out.weight)
        nn.init.zeros_(policy_out.bias)
        nn.init.zeros_(q_out.weight)
        nn.init.zeros_(q_out.bias)

    def _forward_batch_core(
        self,
        batch,
        *,
        legal_idx: Tensor | None = None,
    ) -> BatchOutput:
        """Evaluate flat legal-action outputs, optionally reusing legal indices."""

        if self.representation.axis_relational:
            embeddings = self.representation(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_type=getattr(batch, "edge_type", None),
                edge_dist=getattr(batch, "edge_dist", None),
                global_edge_index=getattr(batch, "global_edge_index", None),
            )
        else:
            embeddings = self.representation(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
            )

        if legal_idx is None:
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        legal_embeddings = torch.index_select(embeddings, 0, legal_idx)

        # Policy and Q consume the same wide JK-cat rows. Fold their first
        # projections into one larger GEMM; this preserves every parameter and
        # state-dict key while avoiding a second read of legal_embeddings.
        policy_first = self.policy_head.mlp[0]
        policy_second = self.policy_head.mlp[2]
        q_first = self.q_head.mlp[0]
        q_second = self.q_head.mlp[2]
        if not all(
            isinstance(layer, nn.Linear)
            for layer in (
                policy_first,
                policy_second,
                q_first,
                q_second,
            )
        ):
            raise TypeError("unexpected policy/Q head layout")
        joint_hidden = F.linear(
            legal_embeddings,
            torch.cat((policy_first.weight, q_first.weight), dim=0),
            torch.cat((policy_first.bias, q_first.bias), dim=0),
        )
        policy_hidden, q_hidden = joint_hidden.split(
            (policy_first.out_features, q_first.out_features),
            dim=1,
        )
        policy_logits = F.linear(
            F.relu(policy_hidden),
            policy_second.weight,
            policy_second.bias,
        ).squeeze(-1)
        critic_output = F.linear(
            F.relu(q_hidden),
            q_second.weight,
            q_second.bias,
        )
        if self.critic_type == "categorical":
            critic_logits = critic_output
            q_values, q_mass = categorical_q(critic_logits)
        else:
            critic_logits = None
            q_values = torch.tanh(critic_output.squeeze(-1))
            q_mass = None

        legal_counts = getattr(batch, "legal_counts", None)
        if legal_counts is None:
            legal_counts = torch.zeros(
                batch.num_graphs,
                dtype=torch.long,
                device=embeddings.device,
            )
            legal_counts.scatter_add_(
                0, batch.batch, batch.legal_mask.to(dtype=torch.long)
            )
        return BatchOutput(
            policy_logits,
            q_values,
            legal_counts,
            critic_logits=critic_logits,
            q_mass=q_mass,
        )

    def _forward_fit_core(
        self,
        batch,
        *,
        chosen: Tensor,
        legal_idx: Tensor | None = None,
    ) -> BatchOutput:
        """Evaluate the full policy but Q only for each played action."""

        if self.representation.axis_relational:
            embeddings = self.representation(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                edge_type=getattr(batch, "edge_type", None),
                edge_dist=getattr(batch, "edge_dist", None),
                global_edge_index=getattr(batch, "global_edge_index", None),
            )
        else:
            embeddings = self.representation(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
            )

        if legal_idx is None:
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        legal_embeddings = torch.index_select(embeddings, 0, legal_idx)
        chosen_embeddings = torch.index_select(
            legal_embeddings,
            0,
            chosen,
        )
        policy_logits = self.policy_head.mlp(legal_embeddings).squeeze(-1)
        critic_output = self.q_head.mlp(chosen_embeddings)
        if self.critic_type == "categorical":
            critic_logits = critic_output
            q_values, q_mass = categorical_q(critic_logits)
        else:
            critic_logits = None
            q_values = critic_output.squeeze(-1)
            q_mass = None

        legal_counts = getattr(batch, "legal_counts", None)
        if legal_counts is None:
            legal_counts = torch.zeros(
                batch.num_graphs,
                dtype=torch.long,
                device=embeddings.device,
            )
            legal_counts.scatter_add_(
                0, batch.batch, batch.legal_mask.to(dtype=torch.long)
            )
        return BatchOutput(
            policy_logits,
            q_values,
            legal_counts,
            critic_logits=critic_logits,
            q_mass=q_mass,
        )

    def forward_batch(self, batch) -> BatchOutput:
        """Evaluate all legal actions in a PyG batch."""

        # Native KLENT batches resolve this on CPU before device transfer.  The
        # fallback preserves compatibility with ad-hoc PyG batches while
        # keeping the data-dependent nonzero outside the compiled GNN core.
        legal_idx = getattr(batch, "legal_idx", None)
        if legal_idx is None:
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        return self._forward_batch_core(batch, legal_idx=legal_idx)

    def forward_fit(self, batch, chosen: Tensor) -> BatchOutput:
        """Evaluate training outputs without materializing unused action Qs."""

        legal_idx = getattr(batch, "legal_idx", None)
        if legal_idx is None:
            legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        return self._forward_fit_core(
            batch,
            chosen=chosen,
            legal_idx=legal_idx,
        )


def is_dense_axis_config(config: object) -> bool:
    """Whether a KLENT model uses the dense raster execution path."""

    return getattr(config, "architecture", "graph") in {
        "dense_axis",
        "persistent_ray_axis",
        "hex_axial_cnn",
        "hex_dilated_cnn",
        "hex_d6_dilated_cnn",
    }


def is_persistent_ray_config(config: object) -> bool:
    """Whether a KLENT model adds persistent directed-ray state."""

    return (
        getattr(config, "architecture", "graph")
        == "persistent_ray_axis"
    )


def _dense_axis_config(config: ModelConfig) -> AxisGineConfig:
    jk_mode = (
        str(getattr(config, "jk_mode", "sum"))
        if bool(getattr(config, "use_jk", False))
        else "none"
    )
    return AxisGineConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        line_radius=int(getattr(config, "dense_ray_radius", 5)),
        distance_bins=int(getattr(config, "axis_window", 8)),
        policy_hidden=config.policy_hidden,
        q_hidden=config.q_hidden,
        value_hidden=1,
        value_bins=0,
        value_horizons=(),
        jk_mode=jk_mode,
        use_layer_scale=bool(getattr(config, "use_layer_scale", False)),
    )


def _persistent_ray_config(config: ModelConfig) -> PersistentRayConfig:
    base = _dense_axis_config(config)
    return PersistentRayConfig(
        **vars(base),
        ray_channels=int(getattr(config, "ray_channels", 12)),
        ray_update_hidden=int(
            getattr(config, "ray_update_hidden", 48)
        ),
        ray_branch_scale=float(
            getattr(config, "ray_branch_scale", 1.0)
        ),
        exact_graft_init=bool(
            getattr(config, "exact_graft_init", True)
        ),
        ray_after_layers=tuple(
            int(layer)
            for layer in getattr(config, "ray_after_layers", ())
        ),
    )


class _RasterKlentOutputMixin:
    """Legal-only KLENT heads shared by both dense raster backbones."""

    def _configure_klent_outputs(self, config: ModelConfig) -> None:
        # KLENT derives V from its improved policy and action Q values. The
        # production value and horizon heads therefore do not belong to this
        # network or its optimizer.
        del self.value_head
        del self.horizon_value_heads
        self.critic_type = str(getattr(config, "critic", "scalar"))
        if self.critic_type == "categorical":
            old_output = self.q_head.fc2
            self.q_head.fc2 = nn.Conv2d(
                old_output.in_channels,
                3,
                kernel_size=1,
            )
            self.q_head.tanh = False
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start from uniform policy and zero action values."""

        nn.init.zeros_(self.policy_head.fc2.weight)
        nn.init.zeros_(self.policy_head.fc2.bias)
        nn.init.zeros_(self.q_head.fc2.weight)
        nn.init.zeros_(self.q_head.fc2.bias)

    @staticmethod
    def _legal_head(head: nn.Module, embeddings: Tensor) -> Tensor:
        first = head.fc1
        second = head.fc2
        hidden = F.linear(
            embeddings,
            first.weight[:, :, 0, 0],
            first.bias,
        )
        output = F.linear(
            F.relu(hidden),
            second.weight[:, :, 0, 0],
            second.bias,
        )
        return (
            torch.tanh(output.squeeze(-1))
            if bool(getattr(head, "tanh", False))
            else output.squeeze(-1) if output.shape[-1] == 1 else output
        )

    def _forward_batch_core(
        self,
        batch,
        *,
        legal_idx: Tensor | None = None,
    ) -> BatchOutput:
        del legal_idx
        representation_active = self.forward_active_features(
            batch.planes,
            batch.scalars,
            batch.ray_bits,
            batch.active_flat_indices,
            batch.active_flat_lookup,
        )
        legal_active_indices = batch.active_flat_lookup.index_select(
            0,
            batch.legal_flat_indices,
        ).to(torch.long)
        legal_embeddings = representation_active.index_select(
            0,
            legal_active_indices,
        )
        policy_logits = self._legal_head(
            self.policy_head,
            legal_embeddings,
        )
        critic_output = self._legal_head(
            self.q_head,
            legal_embeddings,
        )
        if self.critic_type == "categorical":
            critic_logits = critic_output
            q_values, q_mass = categorical_q(critic_logits)
        else:
            critic_logits = None
            q_values = critic_output
            q_mass = None
        legal_counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
        return BatchOutput(
            policy_logits,
            q_values,
            legal_counts,
            critic_logits=critic_logits,
            q_mass=q_mass,
        )

    def forward_batch(self, batch) -> BatchOutput:
        return self._forward_batch_core(batch)


class DenseAxisKlentNet(_RasterKlentOutputMixin, AxisGineCompatNet):
    """KLENT policy/Q network using the compatibility dense axis trunk."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(_dense_axis_config(config))
        self._configure_klent_outputs(config)


class PersistentRayKlentNet(
    _RasterKlentOutputMixin,
    PersistentRayAxisNet,
):
    """KLENT dense trunk augmented with persistent six-ray latent state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(_persistent_ray_config(config))
        self._configure_klent_outputs(config)


class HexCNNKlentNet(nn.Module):
    """Shared legal policy/Q outputs for native hex-raster CNN trunks."""

    def __init__(self, config: ModelConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.critic_type = str(getattr(config, "critic", "scalar"))
        self.backbone = backbone
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.policy_hidden),
            nn.SiLU(),
            nn.Linear(config.policy_hidden, 1),
        )
        self.q_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.q_hidden),
            nn.SiLU(),
            nn.Linear(
                config.q_hidden,
                3 if self.critic_type == "categorical" else 1,
            ),
        )
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        for head in (self.policy_head, self.q_head):
            output = head[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("unexpected hex CNN output-head layout")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _forward_batch_core(
        self,
        batch,
        *,
        legal_idx: Tensor | None = None,
    ) -> BatchOutput:
        del legal_idx
        features = self.backbone(
            batch.planes,
            batch.scalars,
            batch.active_mask,
        )
        flattened = features.permute(0, 2, 3, 1).reshape(
            -1,
            features.shape[1],
        )
        legal = flattened.index_select(
            0,
            batch.legal_flat_indices.to(torch.long),
        )
        policy_logits = self.policy_head(legal).squeeze(-1)
        critic_output = self.q_head(legal)
        if self.critic_type == "categorical":
            critic_logits = critic_output
            q_values, q_mass = categorical_q(critic_logits)
        else:
            critic_logits = None
            q_values = torch.tanh(critic_output.squeeze(-1))
            q_mass = None
        legal_counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
        return BatchOutput(
            policy_logits,
            q_values,
            legal_counts,
            critic_logits=critic_logits,
            q_mass=q_mass,
        )

    def forward_batch(self, batch) -> BatchOutput:
        return self._forward_batch_core(batch)


class HexAxialCNNKlentNet(HexCNNKlentNet):
    """KLENT policy/Q network using a hex CNN plus axial attention."""

    def __init__(self, config: ModelConfig) -> None:
        backbone = HexAxialCNNBackbone(
            channels=config.hidden_dim,
            blocks=config.num_layers,
            heads=config.num_heads,
            attention_radius=int(
                getattr(config, "axial_attention_radius", 8)
            ),
            attention_layers=tuple(
                int(layer)
                for layer in getattr(config, "axial_attention_layers", ())
            ),
            expansion=int(getattr(config, "cnn_expansion", 2)),
            dropout=config.dropout,
        )
        super().__init__(config, backbone)


class HexDilatedCNNKlentNet(HexCNNKlentNet):
    """KLENT network using gated multiscale hex-axis convolutions."""

    def __init__(self, config: ModelConfig) -> None:
        backbone = HexDilatedCNNBackbone(
            channels=config.hidden_dim,
            dilations=tuple(
                int(value)
                for value in getattr(config, "cnn_dilations", ())
            ),
            expansion=int(getattr(config, "cnn_expansion", 2)),
            dropout=config.dropout,
        )
        super().__init__(config, backbone)


class HexD6DilatedCNNKlentNet(HexCNNKlentNet):
    """KLENT CNN with exact D6-equivariant multiscale spatial mixing."""

    def __init__(self, config: ModelConfig) -> None:
        backbone = HexD6DilatedCNNBackbone(
            channels=config.hidden_dim,
            dilations=tuple(
                int(value)
                for value in getattr(config, "cnn_dilations", ())
            ),
            expansion=int(getattr(config, "cnn_expansion", 2)),
            dropout=config.dropout,
        )
        super().__init__(config, backbone)


@dataclass(frozen=True)
class D6CNNConversionReport:
    """Summary of projecting an orientation-specific CNN onto D6 orbits."""

    copied_tensors: int
    projected_blocks: int
    source_parameters: int
    target_parameters: int


@dataclass(frozen=True)
class D6CNNDepthGraftReport:
    """Summary of losslessly extending an exact-D6 CNN with identity blocks."""

    copied_tensors: int
    source_blocks: int
    target_blocks: int
    source_parameters: int
    target_parameters: int


def convert_hex_dilated_to_d6(
    source: HexDilatedCNNKlentNet,
    target: HexD6DilatedCNNKlentNet,
) -> D6CNNConversionReport:
    """Project a learned dilated CNN into the exactly equivariant model.

    Non-spatial tensors are copied exactly.  For every axial mixer, the three
    axis centres and biases are averaged, as are the six reflected endpoints.
    The old pointwise gate has no non-trivial scalar-field D6 projection: its
    row-mean component adds the same logit to every axis and therefore cancels
    in the softmax. The new gate over shared line responses starts at zero.
    """

    source_blocks = source.backbone.blocks
    target_blocks = target.backbone.blocks
    if len(source_blocks) != len(target_blocks):
        raise ValueError("source and target CNNs must have the same block count")

    source_state = source.state_dict()
    converted = target.state_dict()
    copied = 0
    for name, value in source_state.items():
        target_value = converted.get(name)
        if target_value is not None and target_value.shape == value.shape:
            target_value.copy_(value)
            copied += 1

    endpoint_indices = (
        ((1, 0), (1, 2)),
        ((0, 1), (2, 1)),
        ((0, 2), (2, 0)),
    )
    centre = (1, 1)
    with torch.no_grad():
        for source_block, target_block in zip(
            source_blocks,
            target_blocks,
            strict=True,
        ):
            old = source_block.axis_conv
            new = target_block.axis_conv
            weights = old.axis_conv.weight.reshape(
                old.channels,
                3,
                3,
                3,
            )
            centre_weights = torch.stack(
                [weights[:, axis, centre[0], centre[1]] for axis in range(3)],
                dim=1,
            )
            endpoint_weights = torch.stack(
                [
                    weights[:, axis, row, column]
                    for axis, endpoints in enumerate(endpoint_indices)
                    for row, column in endpoints
                ],
                dim=1,
            )
            new.main_center.copy_(centre_weights.mean(dim=1))
            new.main_neighbor.copy_(endpoint_weights.mean(dim=1))
            new.main_bias.copy_(
                old.axis_conv.bias.reshape(old.channels, 3).mean(dim=1)
            )
            new.axis_gate.zero_()

    target.load_state_dict(converted, strict=True)
    return D6CNNConversionReport(
        copied_tensors=copied,
        projected_blocks=len(source_blocks),
        source_parameters=sum(parameter.numel() for parameter in source.parameters()),
        target_parameters=sum(parameter.numel() for parameter in target.parameters()),
    )


def graft_hex_d6_depth(
    source: HexD6DilatedCNNKlentNet,
    target: HexD6DilatedCNNKlentNet,
) -> D6CNNDepthGraftReport:
    """Copy an exact-D6 CNN into a deeper, initially equivalent network.

    The target's dilation schedule must extend the source schedule. All source
    tensors, including the learned heads and output normalization, are copied
    exactly. Added residual blocks retain their random internal initialization
    but start with a zero layer scale, so they are exact identities until
    optimization begins to open them.
    """

    source_blocks = source.backbone.blocks
    target_blocks = target.backbone.blocks
    if len(target_blocks) <= len(source_blocks):
        raise ValueError("target D6 CNN must have more blocks than the source")
    source_dilations = tuple(block.axis_conv.dilation for block in source_blocks)
    target_dilations = tuple(block.axis_conv.dilation for block in target_blocks)
    if target_dilations[: len(source_dilations)] != source_dilations:
        raise ValueError(
            "target D6 CNN dilation schedule must extend the source schedule"
        )

    source_state = source.state_dict()
    grafted = target.state_dict()
    missing_or_mismatched = [
        name
        for name, value in source_state.items()
        if name not in grafted or grafted[name].shape != value.shape
    ]
    if missing_or_mismatched:
        raise ValueError(
            "source and target D6 CNN tensors are not depth-compatible: "
            + ", ".join(missing_or_mismatched)
        )

    with torch.no_grad():
        for name, value in source_state.items():
            grafted[name].copy_(value)
        for block_index in range(len(source_blocks), len(target_blocks)):
            grafted[f"backbone.blocks.{block_index}.layer_scale"].zero_()

    target.load_state_dict(grafted, strict=True)
    return D6CNNDepthGraftReport(
        copied_tensors=len(source_state),
        source_blocks=len(source_blocks),
        target_blocks=len(target_blocks),
        source_parameters=sum(parameter.numel() for parameter in source.parameters()),
        target_parameters=sum(parameter.numel() for parameter in target.parameters()),
    )


RasterKlentNet = (
    DenseAxisKlentNet
    | PersistentRayKlentNet
    | HexAxialCNNKlentNet
    | HexDilatedCNNKlentNet
    | HexD6DilatedCNNKlentNet
)


def make_klent_net(
    config: ModelConfig,
) -> KlentNet | RasterKlentNet:
    """Construct the configured KLENT execution backend."""

    if is_persistent_ray_config(config):
        return PersistentRayKlentNet(config)
    if getattr(config, "architecture", "graph") == "hex_axial_cnn":
        return HexAxialCNNKlentNet(config)
    if getattr(config, "architecture", "graph") == "hex_dilated_cnn":
        return HexDilatedCNNKlentNet(config)
    if getattr(config, "architecture", "graph") == "hex_d6_dilated_cnn":
        return HexD6DilatedCNNKlentNet(config)
    if is_dense_axis_config(config):
        return DenseAxisKlentNet(config)
    return KlentNet(config)


def compile_klent_forward(
    model: KlentNet | RasterKlentNet,
    *,
    fit_max_autotune: bool = False,
    fit_compile_seed_nodes: int = 0,
) -> None:
    """Compile the fit/inference core using backend-appropriate shape policy."""

    import torch._inductor.config as inductor_config

    # Inductor otherwise creates one compiler worker per CPU. Four preserves
    # useful parallelism on this host without multiplying first-use memory by
    # 32 for either graph or raster models.
    inductor_config.compile_threads = min(
        int(inductor_config.compile_threads),
        4,
    )

    if isinstance(
        model,
        (HexDilatedCNNKlentNet, HexD6DilatedCNNKlentNet),
    ):
        # The trunk is composed only of convolutions, pointwise operations,
        # and legal gathers. Inductor can retain dynamic raster and population
        # dimensions in one reusable graph, avoiding a cold compile for every
        # finite crop bucket.
        model._forward_batch_core = torch.compile(
            model._forward_batch_core,
            dynamic=True,
        )
        return

    if isinstance(model, HexAxialCNNKlentNet):
        # Keep raster H/W static so Inductor sees one of the finite crop
        # buckets. Mark only population dimensions dynamic; fully static
        # compilation would recompile for every final partial microbatch.
        eager_core = model._forward_batch_core
        compiled_core = torch.compile(eager_core)

        def spatially_specialized_core(
            batch,
            *,
            legal_idx: Tensor | None = None,
        ):
            # Raster legal indices are already embedded in the batch. Accept
            # the graph-compatible MCTS argument without specializing on it.
            del legal_idx
            for tensor in (
                batch.planes,
                batch.scalars,
                batch.active_mask,
                batch.legal_offsets,
                batch.legal_flat_indices,
            ):
                # A final crop bucket can contain exactly one state, making
                # legal_offsets length two provably static. AUTO-style marking
                # retains dynamic populations when possible without turning
                # that valid specialization into a constraint violation.
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
            return compiled_core(batch)

        model._forward_batch_core = spatially_specialized_core
        return

    if isinstance(
        model,
        (DenseAxisKlentNet, PersistentRayKlentNet),
    ):
        # Inductor defaults to one compiler process per CPU on this machine
        # (32 here). Each dense AOTAutograd block compiler can consume hundreds
        # of MiB, so first-fit compilation otherwise rivals the model itself
        # for system memory. Respect a stricter caller setting, but bound the
        # dense default to four concurrent compiler workers.
        import torch._dynamo.config as dynamo_config
        # Dense batches intentionally use a finite vocabulary of spatial crop
        # buckets. Training and inference mode each specialize those static
        # H/W dimensions, so PyTorch's default limit of eight variants is too
        # small: periodic MCTS can legitimately introduce the ninth bucket
        # after several successful fit iterations. Keep the limit bounded to
        # avoid recreating the original compiler-memory spike.
        dynamo_config.recompile_limit = max(
            int(dynamo_config.recompile_limit),
            32,
        )
        # Compile one relational block at a time. A monolithic four-block
        # forward creates a very large AOTAutograd graph; on gfx1151 its
        # backward both takes many minutes to compile and can trigger an
        # invalid fused Triton reduction. Block boundaries keep compilation
        # tractable while retaining the expensive line gather/MLPs inside
        # Inductor.
        #
        # Raster height/width come from a small fixed bucket vocabulary and
        # remain static. Only the batch dimension varies between actor ticks
        # and fit microbatches.
        for block in model.blocks:
            eager_block = block.forward_compact
            # The block is graph-clean (one graph, zero breaks), so fullgraph
            # does not improve fusion. Leaving it disabled lets Dynamo fall
            # back to eager if a truly unusual crop exhausts the bounded
            # specialization cache instead of turning that event into a fatal
            # FailOnRecompileLimitHit exception.
            compiled_block = torch.compile(eager_block)

            def spatially_specialized_block(
                h_active,
                global_state,
                ray_bits,
                active_flat_indices,
                active_flat_lookup,
                *,
                _eager=eager_block,
                _compiled=compiled_block,
            ):
                if h_active.shape[0] < 4:
                    return _eager(
                        h_active,
                        global_state,
                        ray_bits,
                        active_flat_indices,
                        active_flat_lookup,
                    )
                for tensor in (
                    h_active,
                    global_state,
                    ray_bits,
                    active_flat_indices,
                    active_flat_lookup,
                ):
                    torch._dynamo.mark_dynamic(tensor, 0)
                return _compiled(
                    h_active,
                    global_state,
                    ray_bits,
                    active_flat_indices,
                    active_flat_lookup,
                )

            block.forward_compact = spatially_specialized_block

        if isinstance(model, PersistentRayKlentNet):
            for mixer in model.ray_mixers:
                if not isinstance(mixer, PersistentRayMixer):
                    continue
                eager_mixer = mixer.forward_compact
                compiled_mixer = torch.compile(eager_mixer)

                def spatially_specialized_mixer(
                    h_active,
                    ray_state,
                    ray_bits,
                    active_flat_indices,
                    active_flat_lookup,
                    *,
                    _eager=eager_mixer,
                    _compiled=compiled_mixer,
                ):
                    if h_active.shape[0] < 4:
                        return _eager(
                            h_active,
                            ray_state,
                            ray_bits,
                            active_flat_indices,
                            active_flat_lookup,
                        )
                    for tensor in (
                        h_active,
                        ray_bits,
                        active_flat_indices,
                        active_flat_lookup,
                    ):
                        torch._dynamo.mark_dynamic(tensor, 0)
                    if ray_state is not None:
                        torch._dynamo.mark_dynamic(ray_state, 0)
                    return _compiled(
                        h_active,
                        ray_state,
                        ray_bits,
                        active_flat_indices,
                        active_flat_lookup,
                    )

                mixer.forward_compact = spatially_specialized_mixer
        return

    model._forward_batch_core = torch.compile(
        model._forward_batch_core,
        dynamic=True,
    )
    fit_compile_kwargs: dict[str, object] = {"dynamic": True}
    if fit_max_autotune:
        # Tune the large GEMMs once while compiling the selected seed shape.
        # Runtime autotuning is deliberately disabled: ragged S4 row counts
        # otherwise create an unbounded stream of candidate kernels and
        # compiler workers. The resulting graph remains dynamically shaped.
        fit_compile_kwargs["options"] = {
            "max_autotune": True,
            "triton.autotune_at_compile_time": True,
        }
    model._forward_fit_core = torch.compile(
        model._forward_fit_core,
        **fit_compile_kwargs,
    )
    model._fit_compile_seed_nodes = (
        max(0, int(fit_compile_seed_nodes)) if fit_max_autotune else 0
    )
    model._fit_compile_seeded = False


def load_production_axis_weights(
    model: RasterKlentNet,
    checkpoint: Mapping[str, Any],
) -> ConversionReport:
    """Load the production relational trunk plus policy/Q heads into KLENT."""

    source = extract_state_dict(checkpoint)
    converted, report = convert_strix_axis_state_dict(source, model)
    if report.missing_in_source or report.shape_mismatches:
        raise ValueError(
            "production checkpoint is incompatible with dense KLENT: "
            f"missing={report.missing_in_source}, "
            f"shape_mismatches={report.shape_mismatches}"
        )
    target_keys = set(model.state_dict())
    not_copied = target_keys - set(report.copied)
    allowed_initialized = (
        {
            key
            for key in not_copied
            if key.startswith("ray_mixers.")
        }
        if isinstance(model, PersistentRayKlentNet)
        else set()
    )
    unexpectedly_not_copied = tuple(
        sorted(not_copied - allowed_initialized)
    )
    if unexpectedly_not_copied:
        raise ValueError(
            "production checkpoint did not initialize every dense KLENT "
            f"parameter: {unexpectedly_not_copied}"
        )
    model.load_state_dict(converted, strict=True)
    return report


def load_production_graph_weights(
    model: KlentNet,
    checkpoint: Mapping[str, Any],
) -> tuple[str, ...]:
    """Strictly load the production graph trunk plus policy/Q heads.

    Production checkpoints also contain value and optional auxiliary heads.
    KLENT deliberately ignores those source-only tensors while requiring every
    graph KLENT parameter to exist with exactly the expected shape.
    """

    source = extract_state_dict(checkpoint)
    target = model.state_dict()
    converted_source = dict(source)
    q_weight_key = "q_head.mlp.2.weight"
    q_bias_key = "q_head.mlp.2.bias"
    if (
        model.critic_type == "categorical"
        and q_weight_key in source
        and q_bias_key in source
        and q_weight_key in target
        and q_bias_key in target
        and tuple(source[q_weight_key].shape)
        == (1, target[q_weight_key].shape[1])
        and tuple(target[q_weight_key].shape)
        == (3, source[q_weight_key].shape[1])
        and tuple(source[q_bias_key].shape) == (1,)
        and tuple(target[q_bias_key].shape) == (3,)
    ):
        # Map the old scalar preactivation x to (x, -x, 0).  Around the origin
        # its committed mass is nearly action-independent, so acting-Q mass
        # normalization approximately recovers tanh(x), while the finite zero
        # logit lets intermediate-return targets train immediately.  Using a
        # huge negative zero-class bias would preserve raw Q more exactly but
        # would leave that class effectively frozen at the small revival LR.
        old_weight = source[q_weight_key]
        old_bias = source[q_bias_key]
        converted_source[q_weight_key] = torch.cat(
            (old_weight, -old_weight, torch.zeros_like(old_weight)),
            dim=0,
        )
        converted_source[q_bias_key] = torch.cat(
            (old_bias, -old_bias, torch.zeros_like(old_bias)),
            dim=0,
        )
    missing = tuple(sorted(set(target) - set(converted_source)))
    mismatches = tuple(
        sorted(
            f"{key} {tuple(converted_source[key].shape)} -> "
            f"{tuple(target[key].shape)}"
            for key in set(target) & set(converted_source)
            if tuple(converted_source[key].shape) != tuple(target[key].shape)
        )
    )
    if missing or mismatches:
        raise ValueError(
            "production checkpoint is incompatible with graph KLENT: "
            f"missing={missing}, shape_mismatches={mismatches}"
        )
    converted = {
        key: converted_source[key].to(dtype=value.dtype)
        for key, value in target.items()
    }
    model.load_state_dict(converted, strict=True)
    return tuple(sorted(converted))


def load_dense_klent_graft(
    model: PersistentRayKlentNet,
    checkpoint: Mapping[str, Any],
) -> tuple[str, ...]:
    """Load a dense KLENT base while retaining initialized ray parameters."""

    if not model.config.exact_graft_init:
        raise ValueError(
            "dense KLENT grafting requires model.exact_graft_init=true"
        )
    source = extract_state_dict(checkpoint)
    target = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    base_keys = {
        key for key in target if not key.startswith("ray_mixers.")
    }
    missing = tuple(sorted(base_keys - set(source)))
    mismatches = tuple(
        sorted(
            f"{key} {tuple(source[key].shape)} -> {tuple(target[key].shape)}"
            for key in base_keys & set(source)
            if tuple(source[key].shape) != tuple(target[key].shape)
        )
    )
    if missing or mismatches:
        raise ValueError(
            "dense KLENT checkpoint is incompatible with persistent-ray "
            f"graft: missing={missing}, shape_mismatches={mismatches}"
        )
    for key in base_keys:
        target[key] = source[key].to(dtype=target[key].dtype)
    model.load_state_dict(target, strict=True)
    return tuple(sorted(base_keys))
