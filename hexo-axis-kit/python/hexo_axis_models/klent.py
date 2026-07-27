from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .ops import gather_dense_actions


class KlentPolicy(NamedTuple):
    probabilities: Tensor      # [total_legal]
    q_values: Tensor           # [total_legal]
    improved_logits: Tensor    # [total_legal], float32
    state_values: Tensor       # [B], sum_a pi_K(a|s) Q(s,a)
    graph_index: Tensor        # [total_legal]


def klent_policy_from_dense(
    policy_logits: Tensor,
    q_values: Tensor,
    legal_flat_indices: Tensor,
    legal_offsets: Tensor,
    *,
    alpha: float = 0.03,
    beta: float = 0.10,
) -> KlentPolicy:
    """Compute batched KLENT policy improvement over variable legal sets.

    All improvement math is float32 even when the network runs in BF16/FP16.
    ``legal_flat_indices`` and ``legal_offsets`` use the Rust HXR1 contract.
    """
    if alpha <= 0.0 or beta < 0.0:
        raise ValueError("KLENT requires alpha > 0 and beta >= 0")
    if legal_offsets.ndim != 1 or legal_offsets.numel() < 2:
        raise ValueError("legal_offsets must be [B+1]")
    counts = (legal_offsets[1:] - legal_offsets[:-1]).to(torch.long)
    if bool((counts <= 0).any()):
        raise ValueError("every non-terminal state must have at least one legal action")

    flat_logits = gather_dense_actions(policy_logits, legal_flat_indices).float()
    flat_q = gather_dense_actions(q_values, legal_flat_indices).float()
    batch_size = counts.numel()
    graph_index = torch.arange(batch_size, device=counts.device).repeat_interleave(counts)
    improved = (flat_q + beta * flat_logits) / (alpha + beta)

    max_per_graph = torch.full(
        (batch_size,), -torch.inf, device=improved.device, dtype=improved.dtype
    )
    max_per_graph.scatter_reduce_(
        0, graph_index, improved, reduce="amax", include_self=True
    )
    exponent = (improved - max_per_graph.index_select(0, graph_index)).exp()
    denominator = torch.zeros(batch_size, device=improved.device, dtype=improved.dtype)
    denominator.scatter_add_(0, graph_index, exponent)
    probabilities = exponent / denominator.index_select(0, graph_index)
    state_values = torch.zeros(batch_size, device=improved.device, dtype=improved.dtype)
    state_values.scatter_add_(0, graph_index, probabilities * flat_q)
    return KlentPolicy(probabilities, flat_q, improved, state_values, graph_index)


def sample_segmented_gumbel(
    probabilities: Tensor,
    graph_index: Tensor,
    legal_offsets: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample one categorical action per state without a Python state loop.

    Returns ``(flat_positions, local_action_indices)``. `flat_positions` indexes
    the concatenated legal arrays; local indices align with each state's sorted
    `GameState::legal_moves()` list.
    """
    if probabilities.ndim != 1 or graph_index.shape != probabilities.shape:
        raise ValueError("probabilities and graph_index must be aligned 1-D tensors")
    batch_size = legal_offsets.numel() - 1
    uniform = torch.rand(
        probabilities.shape,
        device=probabilities.device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_(1e-7, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    score = torch.log(probabilities.float().clamp_min(1e-20)) + gumbel

    maxima = torch.full((batch_size,), -torch.inf, device=score.device)
    maxima.scatter_reduce_(0, graph_index, score, reduce="amax", include_self=True)
    winners = score == maxima.index_select(0, graph_index)
    positions = torch.arange(score.numel(), device=score.device, dtype=torch.long)
    sentinel = torch.full_like(positions, score.numel())
    candidates = torch.where(winners, positions, sentinel)
    flat_positions = torch.full(
        (batch_size,), score.numel(), device=score.device, dtype=torch.long
    )
    flat_positions.scatter_reduce_(
        0, graph_index, candidates, reduce="amin", include_self=True
    )
    local_indices = flat_positions - legal_offsets[:-1].to(torch.long)
    return flat_positions, local_indices


def actor_aware_lambda_returns(
    rewards: Tensor,
    next_values: Tensor,
    same_actor_next: Tensor,
    terminals: Tensor,
    *,
    lam: float,
) -> Tensor:
    """HeXO-correct lambda returns for one temporally contiguous trajectory.

    ``next_values[t]`` is `V(s_{t+1})` from the next state's actor perspective.
    `same_actor_next[t]` is true after a player's first non-winning placement
    and false when control passes. Terminal transitions ignore both bootstraps.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0,1]")
    if not (
        rewards.ndim == next_values.ndim == same_actor_next.ndim == terminals.ndim == 1
    ):
        raise ValueError("all return inputs must be one-dimensional")
    if not (
        rewards.numel()
        == next_values.numel()
        == same_actor_next.numel()
        == terminals.numel()
    ):
        raise ValueError("all return inputs must have equal length")

    rewards_f = rewards.float()
    next_values_f = next_values.float()
    returns = torch.empty_like(rewards_f)
    for t in range(rewards.numel() - 1, -1, -1):
        if bool(terminals[t]):
            returns[t] = rewards_f[t]
            continue
        sign = 1.0 if bool(same_actor_next[t]) else -1.0
        next_return = next_values_f[t] if t == rewards.numel() - 1 else returns[t + 1]
        mixed = (1.0 - lam) * next_values_f[t] + lam * next_return
        returns[t] = rewards_f[t] + sign * mixed
    return returns
