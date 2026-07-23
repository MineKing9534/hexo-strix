import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import graph_batch_fn_from_model_config
from hexo_klent.model import KlentNet, improved_policy


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        policy_hidden=8,
        q_hidden=4,
        graph_type="axis",
        prune_empty_edges=True,
        relative_stone_encoding=True,
        axis_relational=True,
        axis_window=2,
        compact_stone_onehot=True,
        node_coords=False,
    )


def test_improved_policy_matches_closed_form():
    logits = torch.tensor([0.0, 1.0, -1.0])
    q_values = torch.tensor([0.2, -0.1, 0.4])

    actual = improved_policy(
        logits, q_values, alpha=0.03, beta=0.1
    )
    expected = torch.softmax(
        (0.1 * logits + q_values) / 0.13, dim=0
    )

    torch.testing.assert_close(actual, expected)


def test_zero_initialized_heads_produce_uniform_policy_and_zero_q():
    config = tiny_model_config()
    model = KlentNet(config)
    game_config = hexo_rs.GameConfig(2, 1, 4)
    states = [hexo_rs.GameState(game_config) for _ in range(2)]
    graphs = graph_batch_fn_from_model_config(config)(states)
    output = model.forward_batch(Batch.from_data_list(graphs))

    assert output.legal_counts.tolist() == [6, 6]
    torch.testing.assert_close(output.policy_logits, torch.zeros(12))
    torch.testing.assert_close(output.q_values, torch.zeros(12))
    for logits in output.policy_logits.split([6, 6]):
        torch.testing.assert_close(
            torch.softmax(logits, dim=0), torch.full((6,), 1 / 6)
        )


def test_axis_relational_core_compiles_as_one_full_graph():
    """The production KLENT fit core must no longer break per GNN layer."""

    config = tiny_model_config()
    model = KlentNet(config).eval()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    states = [hexo_rs.GameState(game_config) for _ in range(2)]
    graphs = graph_batch_fn_from_model_config(config)(states)
    batch = Batch.from_data_list(graphs)
    legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)

    with torch.no_grad():
        eager = model._forward_batch_core(batch, legal_idx=legal_idx)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model._forward_batch_core)(
        batch, legal_idx=legal_idx
    )
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    compiled_core = torch.compile(
        model._forward_batch_core,
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    with torch.no_grad():
        compiled = compiled_core(batch, legal_idx=legal_idx)

    torch.testing.assert_close(compiled.policy_logits, eager.policy_logits)
    torch.testing.assert_close(compiled.q_values, eager.q_values)
    torch.testing.assert_close(compiled.legal_counts, eager.legal_counts)
