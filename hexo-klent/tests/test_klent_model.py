import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import graph_batch_fn_from_model_config
from hexo_klent.batching import prepare_graph_batches
from hexo_klent.config import KlentModelConfig
from hexo_klent.model import (
    DenseAxisKlentNet,
    KlentNet,
    compile_klent_forward,
    improved_policy,
    load_production_axis_weights,
)


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


def test_dense_conversion_matches_graph_policy_and_q_in_fp32():
    graph_config = tiny_model_config()
    graph_config.threat_features = True
    graph_config.axis_window = 5
    graph_config.use_jk = True
    graph_config.jk_mode = "cat"
    dense_config = KlentModelConfig(
        **vars(graph_config),
        architecture="dense_axis",
        dense_ray_radius=5,
    )
    torch.manual_seed(17)
    graph_model = KlentNet(graph_config).eval()
    with torch.no_grad():
        graph_model.policy_head.mlp[-1].weight.normal_()
        graph_model.policy_head.mlp[-1].bias.normal_()
        graph_model.q_head.mlp[-2].weight.normal_()
        graph_model.q_head.mlp[-2].bias.normal_()
    dense_model = DenseAxisKlentNet(dense_config).eval()
    report = load_production_axis_weights(
        dense_model,
        {"model_state_dict": graph_model.state_dict()},
    )
    assert len(report.copied) == len(dense_model.state_dict())
    assert not report.missing_in_source
    assert not report.shape_mismatches

    game_config = hexo_rs.GameConfig(6, 2, 2**32 - 1)
    game = hexo_rs.GameState(game_config)
    states = [game.clone()]
    for _ in range(5):
        q, r = game.legal_moves()[0]
        game.apply_move(q, r)
        if not game.is_terminal():
            states.append(game.clone())

    graph_batch, _aux = __import__(
        "hexo_a0.graph", fromlist=["axis_states_to_batch"]
    ).axis_states_to_batch(
        states,
        prune_empty_edges=True,
        threat_features=True,
        relative_stones=True,
    )
    from hexo_klent.batching import _native_axis_batch

    graph_batch = _native_axis_batch(
        x=graph_batch.x,
        edge_index=graph_batch.edge_index,
        edge_attr=graph_batch.edge_attr,
        legal_mask=graph_batch.legal_mask,
        batch_index=graph_batch.batch,
        num_graphs=graph_batch.num_graphs,
        model_config=graph_config,
    )
    dense_batches = prepare_graph_batches(
        states,
        model_config=dense_config,
        edge_budget=0,
    )
    with torch.inference_mode():
        expected = graph_model.forward_batch(graph_batch)
        actual_parts = [
            dense_model.forward_batch(dense_batch)
            for dense_batch, _state_slice in dense_batches
        ]
    actual_policy = torch.cat(
        [output.policy_logits for output in actual_parts]
    )
    actual_q = torch.cat([output.q_values for output in actual_parts])
    actual_counts = torch.cat(
        [output.legal_counts for output in actual_parts]
    )

    torch.testing.assert_close(
        actual_policy,
        expected.policy_logits,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        actual_q,
        expected.q_values,
        rtol=1e-4,
        atol=1e-4,
    )
    assert torch.equal(actual_counts, expected.legal_counts)


def test_dense_compile_allows_bucket_variants_without_fullgraph(monkeypatch):
    """Dense compilation must retain Dynamo's non-fatal eager fallback."""

    import torch._dynamo.config as dynamo_config

    graph_config = tiny_model_config()
    dense_config = KlentModelConfig(
        **vars(graph_config),
        architecture="dense_axis",
        dense_ray_radius=2,
    )
    model = DenseAxisKlentNet(dense_config).eval()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    states = [hexo_rs.GameState(game_config) for _ in range(4)]
    [(batch, _state_slice)] = prepare_graph_batches(
        states,
        model_config=dense_config,
        edge_budget=0,
    )

    compile_kwargs = []

    def fake_compile(eager, **kwargs):
        compile_kwargs.append(kwargs)
        return eager

    old_limit = dynamo_config.recompile_limit
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        compile_klent_forward(model)
        assert dynamo_config.recompile_limit >= 32
        with torch.inference_mode():
            output = model.forward_batch(batch)
    finally:
        dynamo_config.recompile_limit = old_limit

    assert compile_kwargs == [{}]
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.q_values).all()
