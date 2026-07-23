"""KLENT policy/Q network built on the existing HeXO representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hexo_a0.config import ModelConfig
from hexo_a0.model import PolicyHead, QHead, RepresentationNetwork


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
