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
    PersistentRayKlentNet,
    compile_klent_forward,
    improved_policy,
    load_dense_klent_graft,
    load_production_axis_weights,
    make_klent_net,
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


def test_joint_policy_q_projection_matches_separate_heads():
    config = tiny_model_config()
    torch.manual_seed(31)
    model = KlentNet(config).eval()
    with torch.no_grad():
        for head in (model.policy_head, model.q_head):
            for parameter in head.parameters():
                parameter.normal_()

    game_config = hexo_rs.GameConfig(2, 1, 4)
    states = [hexo_rs.GameState(game_config) for _ in range(2)]
    batch = Batch.from_data_list(
        graph_batch_fn_from_model_config(config)(states)
    )
    legal_idx = batch.legal_mask.nonzero(as_tuple=False).squeeze(1)
    with torch.no_grad():
        embeddings = model.representation(
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            edge_type=batch.edge_type,
            edge_dist=batch.edge_dist,
            global_edge_index=batch.global_edge_index,
        )
        legal_embeddings = embeddings.index_select(0, legal_idx)
        expected_policy = model.policy_head.mlp(
            legal_embeddings
        ).squeeze(-1)
        expected_q = model.q_head.mlp(legal_embeddings).squeeze(-1)
        actual = model._forward_batch_core(batch, legal_idx=legal_idx)

    torch.testing.assert_close(actual.policy_logits, expected_policy)
    torch.testing.assert_close(actual.q_values, expected_q)


def test_fit_forward_only_evaluates_chosen_action_q():
    config = tiny_model_config()
    torch.manual_seed(37)
    model = KlentNet(config).eval()
    with torch.no_grad():
        for head in (model.policy_head, model.q_head):
            for parameter in head.parameters():
                parameter.normal_()

    game_config = hexo_rs.GameConfig(2, 1, 4)
    states = [hexo_rs.GameState(game_config) for _ in range(2)]
    batch = Batch.from_data_list(
        graph_batch_fn_from_model_config(config)(states)
    )
    chosen = torch.tensor([1, 8])

    with torch.no_grad():
        full = model.forward_batch(batch)
        fit = model.forward_fit(batch, chosen)

    torch.testing.assert_close(fit.policy_logits, full.policy_logits)
    torch.testing.assert_close(
        fit.q_values,
        full.q_values.index_select(0, chosen),
    )
    torch.testing.assert_close(fit.legal_counts, full.legal_counts)


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

    chosen = torch.tensor([1, 8])
    fit_explanation = torch._dynamo.explain(model._forward_fit_core)(
        batch,
        chosen=chosen,
        legal_idx=legal_idx,
    )
    assert fit_explanation.graph_count == 1
    assert fit_explanation.graph_break_count == 0

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


def test_persistent_ray_graft_preserves_dense_klent_function():
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
    persistent_config = KlentModelConfig(
        **vars(graph_config),
        architecture="persistent_ray_axis",
        dense_ray_radius=5,
        ray_channels=6,
        ray_update_hidden=12,
        exact_graft_init=True,
    )
    torch.manual_seed(29)
    dense_model = DenseAxisKlentNet(dense_config).eval()
    with torch.no_grad():
        dense_model.policy_head.fc2.weight.normal_()
        dense_model.policy_head.fc2.bias.normal_()
        dense_model.q_head.fc2.weight.normal_()
        dense_model.q_head.fc2.bias.normal_()
    persistent_model = make_klent_net(persistent_config).eval()
    assert isinstance(persistent_model, PersistentRayKlentNet)
    copied = load_dense_klent_graft(
        persistent_model,
        {"model_state_dict": dense_model.state_dict()},
    )
    assert set(copied) == set(dense_model.state_dict())

    game = hexo_rs.GameState(
        hexo_rs.GameConfig(6, 2, 2**32 - 1)
    )
    states = [game.clone()]
    for _ in range(4):
        q, r = game.legal_moves()[0]
        game.apply_move(q, r)
        if not game.is_terminal():
            states.append(game.clone())
    dense_batches = prepare_graph_batches(
        states,
        model_config=dense_config,
        edge_budget=0,
    )
    persistent_batches = prepare_graph_batches(
        states,
        model_config=persistent_config,
        edge_budget=0,
    )
    with torch.inference_mode():
        dense_outputs = [
            dense_model.forward_batch(batch)
            for batch, _state_slice in dense_batches
        ]
        persistent_outputs = [
            persistent_model.forward_batch(batch)
            for batch, _state_slice in persistent_batches
        ]
    for dense_output, persistent_output in zip(
        dense_outputs,
        persistent_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            persistent_output.policy_logits,
            dense_output.policy_logits,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            persistent_output.q_values,
            dense_output.q_values,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.equal(
            persistent_output.legal_counts,
            dense_output.legal_counts,
        )


def test_persistent_klent_heads_never_materialize_dense_features(
    monkeypatch,
):
    graph_config = tiny_model_config()
    config = KlentModelConfig(
        **vars(graph_config),
        architecture="persistent_ray_axis",
        dense_ray_radius=2,
        ray_channels=4,
        ray_update_hidden=8,
    )
    model = PersistentRayKlentNet(config).eval()
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 4))
        for _ in range(4)
    ]
    [(batch, _state_slice)] = prepare_graph_batches(
        states,
        model_config=config,
        edge_budget=0,
    )
    called = False
    active_forward = model.forward_active_features

    def tracked_active(*args, **kwargs):
        nonlocal called
        called = True
        return active_forward(*args, **kwargs)

    def forbidden_dense(*_args, **_kwargs):
        raise AssertionError("KLENT must not scatter compact JK features")

    monkeypatch.setattr(
        model,
        "forward_active_features",
        tracked_active,
    )
    monkeypatch.setattr(model, "forward_features", forbidden_dense)
    with torch.inference_mode():
        output = model.forward_batch(batch)

    assert called
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.q_values).all()


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


def test_graph_compile_max_autotunes_only_fit_core(monkeypatch):
    import torch._inductor.config as inductor_config

    model = KlentNet(tiny_model_config())
    compile_kwargs = []

    def fake_compile(eager, **kwargs):
        compile_kwargs.append(kwargs)
        return eager

    old_threads = inductor_config.compile_threads
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        compile_klent_forward(
            model,
            fit_max_autotune=True,
            fit_compile_seed_nodes=16_384,
        )
    finally:
        inductor_config.compile_threads = old_threads

    assert compile_kwargs == [
        {"dynamic": True},
        {
            "dynamic": True,
            "options": {
                "max_autotune": True,
                "triton.autotune_at_compile_time": True,
            },
        },
    ]
    assert model._fit_compile_seed_nodes == 16_384
    assert model._fit_compile_seeded is False


def test_persistent_compile_specializes_blocks_and_ray_mixers(monkeypatch):
    import torch._dynamo.config as dynamo_config

    graph_config = tiny_model_config()
    persistent_config = KlentModelConfig(
        **vars(graph_config),
        architecture="persistent_ray_axis",
        dense_ray_radius=2,
        ray_channels=4,
        ray_update_hidden=8,
    )
    model = PersistentRayKlentNet(persistent_config).eval()
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 4))
        for _ in range(4)
    ]
    [(batch, _state_slice)] = prepare_graph_batches(
        states,
        model_config=persistent_config,
        edge_budget=0,
    )
    compiled = []

    def fake_compile(eager, **kwargs):
        compiled.append((eager, kwargs))
        return eager

    old_limit = dynamo_config.recompile_limit
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        compile_klent_forward(model)
        with torch.inference_mode():
            output = model.forward_batch(batch)
    finally:
        dynamo_config.recompile_limit = old_limit

    assert len(compiled) == 2
    assert all(kwargs == {} for _eager, kwargs in compiled)
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.q_values).all()
