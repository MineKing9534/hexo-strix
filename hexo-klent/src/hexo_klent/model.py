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
        q_values = torch.tanh(
            F.linear(
                F.relu(q_hidden),
                q_second.weight,
                q_second.bias,
            ).squeeze(-1)
        )

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
        return BatchOutput(policy_logits, q_values, legal_counts)

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
        q_values = self.q_head.mlp(chosen_embeddings).squeeze(-1)

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
        return BatchOutput(policy_logits, q_values, legal_counts)

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

    def _configure_klent_outputs(self) -> None:
        # KLENT derives V from its improved policy and action Q values. The
        # production value and horizon heads therefore do not belong to this
        # network or its optimizer.
        del self.value_head
        del self.horizon_value_heads
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
        ).squeeze(-1)
        return (
            torch.tanh(output)
            if bool(getattr(head, "tanh", False))
            else output
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
        q_values = self._legal_head(
            self.q_head,
            legal_embeddings,
        )
        legal_counts = batch.legal_offsets[1:] - batch.legal_offsets[:-1]
        return BatchOutput(policy_logits, q_values, legal_counts)

    def forward_batch(self, batch) -> BatchOutput:
        return self._forward_batch_core(batch)


class DenseAxisKlentNet(_RasterKlentOutputMixin, AxisGineCompatNet):
    """KLENT policy/Q network using the compatibility dense axis trunk."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(_dense_axis_config(config))
        self._configure_klent_outputs()


class PersistentRayKlentNet(
    _RasterKlentOutputMixin,
    PersistentRayAxisNet,
):
    """KLENT dense trunk augmented with persistent six-ray latent state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(_persistent_ray_config(config))
        self._configure_klent_outputs()


RasterKlentNet = DenseAxisKlentNet | PersistentRayKlentNet


def make_klent_net(
    config: ModelConfig,
) -> KlentNet | RasterKlentNet:
    """Construct the configured KLENT execution backend."""

    if is_persistent_ray_config(config):
        return PersistentRayKlentNet(config)
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
    missing = tuple(sorted(set(target) - set(source)))
    mismatches = tuple(
        sorted(
            f"{key} {tuple(source[key].shape)} -> {tuple(target[key].shape)}"
            for key in set(target) & set(source)
            if tuple(source[key].shape) != tuple(target[key].shape)
        )
    )
    if missing or mismatches:
        raise ValueError(
            "production checkpoint is incompatible with graph KLENT: "
            f"missing={missing}, shape_mismatches={mismatches}"
        )
    converted = {
        key: source[key].to(dtype=value.dtype)
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
