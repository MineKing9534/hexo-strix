import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import graph_batch_fn_from_model_config
from hexo_klent.batching import prepare_graph_batches
from hexo_klent.config import KlentModelConfig
from hexo_klent.hex_axial_cnn import (
    GatedDilatedHexConv,
    HexAxialAttention,
    HexConv2d,
    _three_way_softmax,
)
from hexo_klent.model import (
    DenseAxisKlentNet,
    HexAxialCNNKlentNet,
    HexD6DilatedCNNKlentNet,
    HexDilatedCNNKlentNet,
    KlentNet,
    PersistentRayKlentNet,
    compile_klent_forward,
    improved_policy,
    load_dense_klent_graft,
    load_production_axis_weights,
    make_klent_net,
    convert_hex_dilated_to_d6,
    graft_hex_d6_depth,
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


def test_explicit_three_way_softmax_matches_generic_outputs_and_gradients():
    torch.manual_seed(59)
    logits = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn_like(logits)

    actual = _three_way_softmax(logits)
    actual_gradient = torch.autograd.grad(
        (actual * upstream).sum(),
        logits,
        retain_graph=True,
    )[0]
    expected = torch.softmax(logits, dim=1)
    expected_gradient = torch.autograd.grad(
        (expected * upstream).sum(),
        logits,
    )[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


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


def test_hex_axial_cnn_forward_backward_uses_raster_batches():
    config = KlentModelConfig(
        architecture="hex_axial_cnn",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        policy_hidden=8,
        q_hidden=8,
        axial_attention_radius=2,
        axial_attention_layers=[1],
        dropout=0.0,
        prune_empty_edges=True,
    )
    model = make_klent_net(config)
    assert isinstance(model, HexAxialCNNKlentNet)
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(4, 2, 20))
        for _ in range(3)
    ]
    [(batch, state_slice)] = prepare_graph_batches(
        states,
        model_config=config,
        edge_budget=0,
    )

    output = model.forward_batch(batch)
    loss = output.policy_logits.square().mean() + output.q_values.square().mean()
    loss.backward()

    assert state_slice == slice(0, 3)
    assert batch.planes.shape == (3, 8, 9, 9)
    assert output.legal_counts.tolist() == [18, 18, 18]
    assert output.policy_logits.shape == (54,)
    assert output.q_values.shape == (54,)
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.q_values).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_hex_dilated_cnn_forward_backward_uses_raster_batches():
    config = KlentModelConfig(
        architecture="hex_dilated_cnn",
        hidden_dim=16,
        num_layers=4,
        num_heads=4,
        policy_hidden=8,
        q_hidden=8,
        cnn_dilations=[1, 2, 4, 8],
        dropout=0.0,
        prune_empty_edges=True,
    )
    model = make_klent_net(config)
    assert isinstance(model, HexDilatedCNNKlentNet)
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(4, 2, 20))
        for _ in range(3)
    ]
    [(batch, state_slice)] = prepare_graph_batches(
        states,
        model_config=config,
        edge_budget=0,
    )

    output = model.forward_batch(batch)
    loss = output.policy_logits.square().mean() + output.q_values.square().mean()
    loss.backward()

    assert state_slice == slice(0, 3)
    assert output.legal_counts.tolist() == [18, 18, 18]
    assert output.policy_logits.shape == (54,)
    assert output.q_values.shape == (54,)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_hex_d6_dilated_cnn_forward_backward_uses_raster_batches():
    config = KlentModelConfig(
        architecture="hex_d6_dilated_cnn",
        hidden_dim=16,
        num_layers=4,
        num_heads=4,
        policy_hidden=8,
        q_hidden=8,
        cnn_dilations=[1, 2, 4, 8],
        dropout=0.0,
        prune_empty_edges=True,
    )
    model = make_klent_net(config)
    assert isinstance(model, HexD6DilatedCNNKlentNet)
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(4, 2, 20))
        for _ in range(3)
    ]
    [(batch, state_slice)] = prepare_graph_batches(
        states,
        model_config=config,
        edge_budget=0,
    )

    output = model.forward_batch(batch)
    loss = output.policy_logits.square().mean() + output.q_values.square().mean()
    loss.backward()

    assert state_slice == slice(0, 3)
    assert output.legal_counts.tolist() == [18, 18, 18]
    assert output.policy_logits.shape == (54,)
    assert output.q_values.shape == (54,)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_hex_d6_dilated_cnn_is_equivariant_under_all_twelve_symmetries():
    from hexo_klent.distill import (
        DistillationExample,
        _transform_distillation_example,
    )

    config = KlentModelConfig(
        architecture="hex_d6_dilated_cnn",
        hidden_dim=12,
        num_layers=3,
        num_heads=3,
        policy_hidden=7,
        q_hidden=5,
        cnn_dilations=[1, 2, 3],
        dropout=0.0,
        prune_empty_edges=True,
    )
    torch.manual_seed(20260812)
    model = make_klent_net(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.15)

    state = hexo_rs.GameState(hexo_rs.GameConfig(6, 8, 300))
    for q, r in [(1, 0), (2, -1), (-1, 2), (3, -2), (-2, 0)]:
        state.apply_move(q, r)
    legal_count = len(state.legal_moves())
    example = DistillationExample(
        state,
        torch.arange(legal_count, dtype=torch.float32),
        torch.zeros(legal_count),
    )
    [(original_batch, _)] = prepare_graph_batches(
        [state],
        model_config=config,
        edge_budget=0,
    )
    with torch.inference_mode():
        original = model.forward_batch(original_batch)

    for transform_index in range(12):
        transformed = _transform_distillation_example(example, transform_index)
        permutation = transformed.policy_logits.to(torch.long)
        [(transformed_batch, _)] = prepare_graph_batches(
            [transformed.state],
            model_config=config,
            edge_budget=0,
        )
        with torch.inference_mode():
            actual = model.forward_batch(transformed_batch)
        torch.testing.assert_close(
            actual.policy_logits,
            original.policy_logits.index_select(0, permutation),
            rtol=4.0e-4,
            atol=2.0e-5,
        )
        torch.testing.assert_close(
            actual.q_values,
            original.q_values.index_select(0, permutation),
            rtol=4.0e-4,
            atol=2.0e-5,
        )


def test_hex_d6_initial_policy_ignores_non_d6_square_crop_corners():
    from hexo_klent.distill import _D6_COORD_TRANSFORMS

    config = KlentModelConfig(
        architecture="hex_d6_dilated_cnn",
        hidden_dim=12,
        num_layers=4,
        num_heads=3,
        policy_hidden=7,
        q_hidden=5,
        cnn_dilations=[1, 2, 4, 8],
        dropout=0.0,
        prune_empty_edges=True,
    )
    torch.manual_seed(20260813)
    model = make_klent_net(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.15)

    # The engine supplies the fixed centre opening. Its radius-eight legal
    # hex is embedded in a 17x17 axial rectangle with inactive square corners.
    # Every member of a D6 orbit must nevertheless receive the same output.
    state = hexo_rs.GameState(hexo_rs.GameConfig(6, 8, 500))
    legal = [tuple(coord) for coord in state.legal_moves()]
    [(batch, _)] = prepare_graph_batches(
        [state],
        model_config=config,
        edge_budget=0,
    )
    with torch.inference_mode():
        output = model.forward_batch(batch)
    logits = dict(zip(legal, output.policy_logits.tolist(), strict=True))
    q_values = dict(zip(legal, output.q_values.tolist(), strict=True))

    seen: set[tuple[int, int]] = set()
    for coord in legal:
        if coord in seen:
            continue
        orbit = sorted(
            {transform(*coord) for transform in _D6_COORD_TRANSFORMS}
            & logits.keys()
        )
        seen.update(orbit)
        expected_logits = torch.full(
            (len(orbit),), logits[orbit[0]], dtype=torch.float32
        )
        expected_q = torch.full(
            (len(orbit),), q_values[orbit[0]], dtype=torch.float32
        )
        torch.testing.assert_close(
            torch.tensor([logits[item] for item in orbit]),
            expected_logits,
            rtol=0.0,
            atol=1.0e-5,
        )
        torch.testing.assert_close(
            torch.tensor([q_values[item] for item in orbit]),
            expected_q,
            rtol=0.0,
            atol=1.0e-5,
        )


def test_dilated_cnn_conversion_projects_spatial_parameters_to_d6_orbits():
    source_config = KlentModelConfig(
        architecture="hex_dilated_cnn",
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        policy_hidden=3,
        q_hidden=2,
        cnn_dilations=[2],
        dropout=0.0,
    )
    target_config = dataclasses.replace(
        source_config,
        architecture="hex_d6_dilated_cnn",
    )
    source = HexDilatedCNNKlentNet(source_config)
    target = HexD6DilatedCNNKlentNet(target_config)
    old = source.backbone.blocks[0].axis_conv
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_()
        old.axis_conv.weight.copy_(
            torch.arange(old.axis_conv.weight.numel(), dtype=torch.float32).reshape_as(
                old.axis_conv.weight
            )
        )
        old.axis_conv.bias.copy_(torch.arange(12, dtype=torch.float32))
        old.axis_gate.weight.copy_(
            torch.arange(12, dtype=torch.float32).reshape(3, 4, 1, 1)
        )
        old.axis_gate.bias.copy_(torch.tensor([3.0, 6.0, 12.0]))

    report = convert_hex_dilated_to_d6(source, target)
    new = target.backbone.blocks[0].axis_conv
    weights = old.axis_conv.weight.reshape(4, 3, 3, 3)
    expected_centres = weights[:, :, 1, 1].mean(dim=1)
    expected_endpoints = torch.stack(
        (
            weights[:, 0, 1, 0],
            weights[:, 0, 1, 2],
            weights[:, 1, 0, 1],
            weights[:, 1, 2, 1],
            weights[:, 2, 0, 2],
            weights[:, 2, 2, 0],
        ),
        dim=1,
    ).mean(dim=1)

    torch.testing.assert_close(new.main_center, expected_centres)
    torch.testing.assert_close(new.main_neighbor, expected_endpoints)
    torch.testing.assert_close(new.main_bias, old.axis_conv.bias.reshape(4, 3).mean(1))
    torch.testing.assert_close(new.axis_gate, torch.zeros(4))
    torch.testing.assert_close(
        target.policy_head[0].weight,
        source.policy_head[0].weight,
    )
    assert report.projected_blocks == 1


def test_d6_depth_graft_exactly_preserves_the_shallower_network():
    source_config = KlentModelConfig(
        architecture="hex_d6_dilated_cnn",
        hidden_dim=8,
        num_layers=2,
        num_heads=1,
        policy_hidden=5,
        q_hidden=3,
        cnn_dilations=[1, 2],
        dropout=0.0,
    )
    target_config = dataclasses.replace(
        source_config,
        num_layers=4,
        cnn_dilations=[1, 2, 4, 8],
    )
    source = HexD6DilatedCNNKlentNet(source_config).eval()
    target = HexD6DilatedCNNKlentNet(target_config).eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0.0, 0.2)

    planes = torch.randn(3, 8, 11, 13)
    scalars = torch.randn(3, 5)
    active_mask = torch.zeros(3, 1, 11, 13, dtype=torch.bool)
    active_mask[:, :, 1:-1, 1:-1] = True
    with torch.inference_mode():
        expected = source.backbone(planes, scalars, active_mask)
    report = graft_hex_d6_depth(source, target)
    with torch.inference_mode():
        actual = target.backbone(planes, scalars, active_mask)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert report.source_blocks == 2
    assert report.target_blocks == 4
    assert all(
        torch.count_nonzero(block.layer_scale) == 0
        for block in target.backbone.blocks[2:]
    )


def test_gated_dilated_hex_conv_axes_use_only_valid_hex_directions():
    layer = GatedDilatedHexConv(1, dilation=1)
    with torch.no_grad():
        layer.axis_conv.weight.fill_(1.0)
        layer.axis_conv.bias.zero_()
        layer.axis_gate.weight.zero_()
        layer.axis_gate.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
    inputs = torch.zeros((1, 1, 5, 5))
    inputs[0, 0, 2, 1] = 3.0  # q-axis neighbour
    inputs[0, 0, 1, 2] = 5.0  # r-axis neighbour
    inputs[0, 0, 1, 1] = 7.0  # invalid square-only diagonal

    output = layer(inputs)

    torch.testing.assert_close(output[0, 0, 2, 2], torch.tensor(3.0))


def test_hex_axial_line_pack_round_trip():
    tensor = torch.arange(2 * 3 * 7 * 11).reshape(2, 3, 7, 11)
    for axis in range(3):
        lines = HexAxialAttention._pack_axis(tensor, axis)
        restored = HexAxialAttention._unpack_axis(
            lines,
            axis,
            height=7,
            width=11,
        )
        assert torch.equal(restored, tensor)


def test_hex_axial_offset_attention_matches_unfold_reference():
    torch.manual_seed(47)
    layer = HexAxialAttention(8, heads=2, radius=2, dropout=0.0).eval()
    with torch.no_grad():
        layer.relative_bias.normal_()
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 3, 7, 8)
    v = torch.randn(2, 3, 7, 8)
    active = torch.rand(2, 3, 7) > 0.25

    actual = layer._attend(q, k, v, active)

    merged = 2 * 3
    q_heads = q.reshape(merged, 7, 2, 4)
    k_heads = k.reshape(merged, 7, 2, 4)
    v_heads = v.reshape(merged, 7, 2, 4)
    padded_k = F.pad(k_heads, (0, 0, 0, 0, 2, 2))
    padded_v = F.pad(v_heads, (0, 0, 0, 0, 2, 2))
    k_windows = padded_k.unfold(1, 5, 1).permute(0, 1, 2, 4, 3)
    v_windows = padded_v.unfold(1, 5, 1).permute(0, 1, 2, 4, 3)
    active_flat = active.reshape(merged, 7)
    active_windows = F.pad(active_flat, (2, 2), value=False).unfold(1, 5, 1)
    logits = torch.einsum("mlhd,mlhwd->mlhw", q_heads, k_windows) / 2.0
    distances = torch.arange(-2, 3).abs()
    logits = logits + layer.relative_bias.index_select(1, distances)[None, None]
    logits = logits.masked_fill(
        ~active_windows[:, :, None, :],
        torch.finfo(logits.dtype).min,
    )
    weights = torch.softmax(logits, dim=-1)
    expected = torch.einsum(
        "mlhw,mlhwd->mlhd",
        weights,
        v_windows,
    ).reshape(2, 3, 7, 8)
    expected = expected * active[..., None]

    torch.testing.assert_close(actual, expected)


def test_hex_convolution_excludes_square_only_corner_neighbours():
    layer = HexConv2d(1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    inputs = torch.zeros((1, 1, 5, 5))
    inputs[0, 0, 1, 1] = 7.0
    inputs[0, 0, 1, 2] = 3.0

    output = layer(inputs)

    # The upper-left square diagonal is not a one-step axial neighbour.
    assert output[0, 0, 2, 2].item() == 3.0


def test_hex_axial_cnn_is_invariant_to_extra_inactive_crop_padding():
    config = KlentModelConfig(
        architecture="hex_axial_cnn",
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        policy_hidden=8,
        q_hidden=8,
        axial_attention_radius=2,
        axial_attention_layers=[1],
        dropout=0.0,
        prune_empty_edges=True,
    )
    torch.manual_seed(43)
    model = HexAxialCNNKlentNet(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    state = hexo_rs.GameState(hexo_rs.GameConfig(4, 2, 20))
    [(batch, _state_slice)] = prepare_graph_batches(
        [state],
        model_config=config,
        edge_budget=0,
    )
    pad = 4
    old_width = batch.planes.shape[-1]
    new_width = old_width + 2 * pad
    legal_row = batch.legal_flat_indices // old_width
    legal_column = batch.legal_flat_indices % old_width
    padded = SimpleNamespace(
        planes=F.pad(batch.planes, (pad, pad, pad, pad)),
        scalars=batch.scalars,
        active_mask=F.pad(batch.active_mask, (pad, pad, pad, pad)),
        legal_offsets=batch.legal_offsets,
        legal_flat_indices=(legal_row + pad) * new_width + legal_column + pad,
    )

    with torch.inference_mode():
        original = model.forward_batch(batch)
        expanded = model.forward_batch(padded)

    torch.testing.assert_close(expanded.policy_logits, original.policy_logits)
    torch.testing.assert_close(expanded.q_values, original.q_values)
    assert torch.equal(expanded.legal_counts, original.legal_counts)


@pytest.mark.parametrize(
    "architecture",
    ["hex_dilated_cnn", "hex_d6_dilated_cnn"],
)
def test_hex_dilated_cnn_is_invariant_to_extra_inactive_crop_padding(
    architecture,
):
    config = KlentModelConfig(
        architecture=architecture,
        hidden_dim=16,
        num_layers=4,
        num_heads=4,
        policy_hidden=8,
        q_hidden=8,
        cnn_dilations=[1, 2, 4, 8],
        dropout=0.0,
        prune_empty_edges=True,
    )
    torch.manual_seed(53)
    model = make_klent_net(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    state = hexo_rs.GameState(hexo_rs.GameConfig(4, 2, 20))
    [(batch, _state_slice)] = prepare_graph_batches(
        [state],
        model_config=config,
        edge_budget=0,
    )
    pad = 8
    old_width = batch.planes.shape[-1]
    new_width = old_width + 2 * pad
    legal_row = batch.legal_flat_indices // old_width
    legal_column = batch.legal_flat_indices % old_width
    padded = SimpleNamespace(
        planes=F.pad(batch.planes, (pad, pad, pad, pad)),
        scalars=batch.scalars,
        active_mask=F.pad(batch.active_mask, (pad, pad, pad, pad)),
        legal_offsets=batch.legal_offsets,
        legal_flat_indices=(legal_row + pad) * new_width + legal_column + pad,
    )

    with torch.inference_mode():
        original = model.forward_batch(batch)
        expanded = model.forward_batch(padded)

    torch.testing.assert_close(expanded.policy_logits, original.policy_logits)
    torch.testing.assert_close(expanded.q_values, original.q_values)
    assert torch.equal(expanded.legal_counts, original.legal_counts)


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


@pytest.mark.parametrize(
    "architecture",
    ["hex_dilated_cnn", "hex_d6_dilated_cnn"],
)
def test_dilated_cnn_compile_uses_one_dynamic_raster_graph(
    architecture,
    monkeypatch,
):
    config = KlentModelConfig(
        architecture=architecture,
        hidden_dim=8,
        num_layers=2,
        num_heads=1,
        policy_hidden=8,
        q_hidden=4,
        cnn_dilations=[1, 2],
        dropout=0.0,
    )
    model = make_klent_net(config).eval()
    compile_kwargs = []

    def fake_compile(eager, **kwargs):
        compile_kwargs.append(kwargs)
        return eager

    monkeypatch.setattr(torch, "compile", fake_compile)
    compile_klent_forward(model)

    assert compile_kwargs == [{"dynamic": True}]


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
