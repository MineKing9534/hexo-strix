"""KLENT policy/Q network built on the existing HeXO representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hexo_a0.config import ModelConfig
from hexo_a0.model import PolicyHead, QHead, RepresentationNetwork
from hexo_axis_models import AxisGineCompatNet, AxisGineConfig
from hexo_axis_models.checkpoint import (
    ConversionReport,
    convert_strix_axis_state_dict,
    extract_state_dict,
)


@dataclass(frozen=True)
class BatchOutput:
    """Flat legal-action outputs and their per-position lengths."""

    policy_logits: Tensor
    q_values: Tensor
    legal_counts: Tensor


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
        self.q_head = QHead(head_dim, config.q_hidden)
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start KLENT at a uniform policy with zero action values."""

        policy_out = self.policy_head.mlp[-1]
        q_out = self.q_head.mlp[-2]
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
        policy_logits = self.policy_head.mlp(legal_embeddings).squeeze(-1)
        q_values = self.q_head.mlp(legal_embeddings).squeeze(-1)

        legal_counts = torch.zeros(
            batch.num_graphs, dtype=torch.long, device=embeddings.device
        )
        legal_counts.scatter_add_(
            0, batch.batch, batch.legal_mask.to(dtype=torch.long)
        )
        return BatchOutput(policy_logits, q_values, legal_counts)

    def forward_batch(self, batch) -> BatchOutput:
        """Evaluate all legal actions in a PyG batch."""

        # Keep the dynamic-shape nonzero outside the compiled GNN core, matching
        # the production AlphaZero training path.
        legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
        return self._forward_batch_core(batch, legal_idx=legal_idx)


def is_dense_axis_config(config: object) -> bool:
    """Whether a KLENT model uses the dense raster execution path."""

    return getattr(config, "architecture", "graph") == "dense_axis"


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


class DenseAxisKlentNet(AxisGineCompatNet):
    """KLENT policy/Q network using the dense axis-line representation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(_dense_axis_config(config))
        # KLENT derives V from its improved policy and action Q values. The
        # production value and horizon heads therefore do not belong to this
        # network or its optimizer.
        del self.value_head
        del self.horizon_value_heads
        self.reset_output_heads()

    def reset_output_heads(self) -> None:
        """Start a from-scratch dense KLENT model at uniform policy / zero Q."""

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
        ).squeeze(-1)
        return torch.tanh(output) if bool(getattr(head, "tanh", False)) else output

    def _forward_batch_core(
        self,
        batch,
        *,
        legal_idx: Tensor | None = None,
    ) -> BatchOutput:
        del legal_idx
        representation, _active, _legal, _stones = self.forward_features(
            batch.planes,
            batch.scalars,
            batch.active_mask,
            batch.ray_mask,
            batch.active_flat_indices,
        )
        flat = representation.permute(0, 2, 3, 1).reshape(
            -1, representation.shape[1]
        )
        legal_embeddings = flat.index_select(
            0, batch.legal_flat_indices.to(torch.long)
        )
        policy_logits = self._legal_head(self.policy_head, legal_embeddings)
        q_values = self._legal_head(self.q_head, legal_embeddings)
        legal_counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
        return BatchOutput(policy_logits, q_values, legal_counts)

    def forward_batch(self, batch) -> BatchOutput:
        return self._forward_batch_core(batch)


def make_klent_net(config: ModelConfig) -> KlentNet | DenseAxisKlentNet:
    """Construct the configured KLENT execution backend."""

    if is_dense_axis_config(config):
        return DenseAxisKlentNet(config)
    return KlentNet(config)


def compile_klent_forward(
    model: KlentNet | DenseAxisKlentNet,
) -> None:
    """Compile the fit/inference core using backend-appropriate shape policy."""

    if isinstance(model, DenseAxisKlentNet):
        # Inductor defaults to one compiler process per CPU on this machine
        # (32 here). Each dense AOTAutograd block compiler can consume hundreds
        # of MiB, so first-fit compilation otherwise rivals the model itself
        # for system memory. Respect a stricter caller setting, but bound the
        # dense default to four concurrent compiler workers.
        import torch._dynamo.config as dynamo_config
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = min(
            int(inductor_config.compile_threads),
            4,
        )
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
            eager_block = block.forward
            # The block is graph-clean (one graph, zero breaks), so fullgraph
            # does not improve fusion. Leaving it disabled lets Dynamo fall
            # back to eager if a truly unusual crop exhausts the bounded
            # specialization cache instead of turning that event into a fatal
            # FailOnRecompileLimitHit exception.
            compiled_block = torch.compile(eager_block)

            def spatially_specialized_block(
                h,
                global_state,
                active_mask,
                ray_mask,
                active_flat_indices,
                *,
                _eager=eager_block,
                _compiled=compiled_block,
            ):
                if h.shape[0] < 4:
                    return _eager(
                        h,
                        global_state,
                        active_mask,
                        ray_mask,
                        active_flat_indices,
                    )
                for tensor in (
                    h,
                    global_state,
                    active_mask,
                    ray_mask,
                    active_flat_indices,
                ):
                    torch._dynamo.mark_dynamic(tensor, 0)
                return _compiled(
                    h,
                    global_state,
                    active_mask,
                    ray_mask,
                    active_flat_indices,
                )

            block.forward = spatially_specialized_block
        return

    model._forward_batch_core = torch.compile(
        model._forward_batch_core,
        dynamic=True,
    )


def load_production_axis_weights(
    model: DenseAxisKlentNet,
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
    if set(report.copied) != target_keys:
        not_copied = tuple(sorted(target_keys - set(report.copied)))
        raise ValueError(
            "production checkpoint did not initialize every dense KLENT "
            f"parameter: {not_copied}"
        )
    model.load_state_dict(converted, strict=True)
    return report
