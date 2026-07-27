from __future__ import annotations

import torch

from hexo_axis_models.klent import (
    actor_aware_lambda_returns,
    klent_policy_from_dense,
    sample_segmented_gumbel,
)


def test_segmented_klent_normalizes_and_values_match():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [2.0, 3.0]]])
    q = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, 0.0], [0.5, 0.9]]])
    legal = torch.tensor([0, 3, 5, 6, 7])
    offsets = torch.tensor([0, 2, 5])
    result = klent_policy_from_dense(logits, q, legal, offsets)
    assert torch.allclose(result.probabilities[:2].sum(), torch.tensor(1.0))
    assert torch.allclose(result.probabilities[2:].sum(), torch.tensor(1.0))
    expected0 = (result.probabilities[:2] * result.q_values[:2]).sum()
    expected1 = (result.probabilities[2:] * result.q_values[2:]).sum()
    torch.testing.assert_close(result.state_values, torch.stack([expected0, expected1]))


def test_segmented_gumbel_returns_local_indices_in_range():
    probs = torch.tensor([0.4, 0.6, 0.2, 0.3, 0.5])
    graph = torch.tensor([0, 0, 1, 1, 1])
    offsets = torch.tensor([0, 2, 5])
    generator = torch.Generator().manual_seed(123)
    flat, local = sample_segmented_gumbel(probs, graph, offsets, generator=generator)
    assert flat.shape == (2,)
    assert 0 <= int(local[0]) < 2
    assert 0 <= int(local[1]) < 3


def test_actor_aware_lambda_return_preserves_mid_turn_sign():
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    next_values = torch.zeros(4)
    # actor states: P2 -> P2 -> P1 -> P1 -> terminal
    same_actor_next = torch.tensor([True, False, True, False])
    terminal = torch.tensor([False, False, False, True])
    returns = actor_aware_lambda_returns(
        rewards, next_values, same_actor_next, terminal, lam=1.0
    )
    torch.testing.assert_close(returns, torch.tensor([-1.0, -1.0, 1.0, 1.0]))
