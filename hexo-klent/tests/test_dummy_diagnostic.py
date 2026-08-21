from __future__ import annotations

import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.graph import graph_fn_from_model_config
from hexo_klent.config import KlentModelConfig
from hexo_klent.dummy_diagnostic import (
    cluster_summary,
    move_feature_index,
    relational_forward,
)
from hexo_klent.model import KlentNet


def _model_and_graph():
    config = KlentModelConfig(
        hidden_dim=16,
        num_layers=2,
        policy_hidden=12,
        q_hidden=8,
        graph_type="axis",
        pre_norm=True,
        use_jk=True,
        jk_mode="cat",
        axis_relational=True,
        axis_window=8,
        prune_empty_edges=True,
        threat_features=True,
        relative_stone_encoding=True,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )
    model = KlentNet(config).eval()
    rust_config = hexo_rs.GameConfig(6, 8, 100)
    game = hexo_rs.GameState(rust_config)
    game.apply_move(1, 0)
    game.apply_move(0, 1)
    graph = graph_fn_from_model_config(config)(game)
    return model, config, graph


def test_manual_and_dead_final_forward_match_model_outputs():
    model, _config, graph = _model_and_graph()
    with torch.inference_mode():
        actual = model.forward_batch(Batch.from_data_list([graph]))
        manual = relational_forward(model, graph, model_config=_config)
        optimized = relational_forward(
            model,
            graph,
            model_config=_config,
            skip_dead_final=True,
        )

    torch.testing.assert_close(manual[0], actual.policy_logits)
    torch.testing.assert_close(manual[1], actual.q_values)
    torch.testing.assert_close(optimized[0], actual.policy_logits)
    torch.testing.assert_close(optimized[1], actual.q_values)
    torch.testing.assert_close(optimized[2], manual[2])


def test_move_feature_index_matches_native_lean_schema():
    _model, config, graph = _model_and_graph()
    index = move_feature_index(config)
    assert index == 2
    assert set(graph.x[:, index].tolist()) <= {0.5, 1.0}


def test_cluster_summary_resamples_games_not_positions():
    summary = cluster_summary(
        [(0, 0.0), (0, 0.0), (1, 1.0), (1, 1.0)],
        seed=4,
        bootstrap_samples=200,
    )
    assert summary["games"] == 2
    assert summary["positions"] == 4
    assert summary["mean"] == 0.5
    assert summary["median"] == 0.5
