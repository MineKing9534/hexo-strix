import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.graph import graph_batch_fn_from_model_config
from hexo_klent.batching import _packed_ranges, prepare_graph_batches
from hexo_klent.model import KlentNet

from test_klent_model import tiny_model_config


def test_packed_ranges_keep_greedy_boundaries_and_oversize_items():
    assert _packed_ranges([], 10) == []
    assert _packed_ranges([4, 5, 6, 2], 10) == [(0, 2), (2, 4)]
    assert _packed_ranges([12, 3, 4], 10) == [(0, 1), (1, 3)]
    assert _packed_ranges([4, 5, 6], 0) == [(0, 3)]


def test_prepared_axis_batches_match_native_lean_graphs_and_outputs():
    config = tiny_model_config()
    game_config = hexo_rs.GameConfig(6, 1, 2**32 - 1)
    game = hexo_rs.GameState(game_config)
    states = [game.clone()]
    for _ in range(3):
        q, r = game.legal_moves()[0]
        game.apply_move(q, r)
        if not game.is_terminal():
            states.append(game.clone())

    native_graphs = graph_batch_fn_from_model_config(config)(states)
    edge_counts = [
        int(graph.edge_index.shape[1] + graph.global_edge_index.shape[1])
        for graph in native_graphs
    ]
    edge_budget = max(edge_counts)
    expected_ranges = _packed_ranges(edge_counts, edge_budget)
    prepared = prepare_graph_batches(
        states,
        model_config=config,
        edge_budget=edge_budget,
    )
    assert [
        (state_slice.start, state_slice.stop)
        for _batch, state_slice in prepared
    ] == expected_ranges

    torch.manual_seed(4)
    model = KlentNet(config).eval()
    with torch.no_grad():
        model.policy_head.mlp[-1].weight.normal_()
        model.policy_head.mlp[-1].bias.normal_()
        model.q_head.mlp[-2].weight.normal_()
        model.q_head.mlp[-2].bias.normal_()

    for actual_batch, state_slice in prepared:
        native = Batch.from_data_list(native_graphs[state_slice])
        assert torch.equal(native.x, actual_batch.x)
        assert torch.equal(native.edge_index, actual_batch.edge_index)
        assert torch.equal(native.edge_type, actual_batch.edge_type)
        assert torch.equal(native.edge_dist, actual_batch.edge_dist)
        assert torch.equal(
            native.global_edge_index, actual_batch.global_edge_index
        )
        assert torch.equal(native.legal_mask, actual_batch.legal_mask)
        assert torch.equal(native.batch, actual_batch.batch)
        with torch.inference_mode():
            expected = model.forward_batch(native)
            actual = model.forward_batch(actual_batch)
        assert torch.equal(actual.policy_logits, expected.policy_logits)
        assert torch.equal(actual.q_values, expected.q_values)
        assert torch.equal(actual.legal_counts, expected.legal_counts)
