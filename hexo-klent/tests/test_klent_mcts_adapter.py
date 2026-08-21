import dataclasses
import math

import pytest
import torch
from torch_geometric.data import Batch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.evaluate import make_eval_fn, sample_opening
from hexo_a0.graph import (
    graph_batch_fn_from_model_config,
    graph_fn_from_model_config,
)
from hexo_a0.head_to_head import load_checkpoint
from hexo_klent.config import AlgorithmConfig, KlentModelConfig
from hexo_klent.mcts_adapter import KlentMCTSAdapter
from hexo_klent.model import BatchOutput, KlentNet, improved_policy, make_klent_net


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
        axis_window=8,
        compact_stone_onehot=True,
        node_coords=False,
    )


def tiny_dense_model_config() -> KlentModelConfig:
    return KlentModelConfig(
        architecture="dense_axis",
        dense_ray_radius=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        policy_hidden=8,
        q_hidden=4,
        pre_norm=True,
        dropout=0.0,
        graph_type="axis",
        prune_empty_edges=True,
        threat_features=True,
        relative_stone_encoding=True,
        axis_relational=True,
        axis_window=8,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )


def test_adapter_derives_value_from_learned_policy_q_expectation():
    config = tiny_model_config()
    network = KlentNet(config)
    q_output = network.q_head.mlp[-2]
    with torch.no_grad():
        q_output.bias.fill_(math.atanh(0.25))
    adapter = KlentMCTSAdapter(network, AlgorithmConfig())
    game_config = hexo_rs.GameConfig(6, 1, 12)
    states = [hexo_rs.GameState(game_config) for _ in range(2)]
    graphs = graph_batch_fn_from_model_config(config)(states)

    logits, values = adapter.forward_batch(Batch.from_data_list(graphs))

    assert [chunk.numel() for chunk in logits] == [6, 6]
    torch.testing.assert_close(values, torch.full((2,), 0.25))


def test_adapter_does_not_reimprove_policy_for_test_time_value():
    adapter = KlentMCTSAdapter(KlentNet(tiny_model_config()), AlgorithmConfig())
    output = BatchOutput(
        policy_logits=torch.tensor([2.0, -1.0]),
        q_values=torch.tensor([-0.5, 0.75]),
        legal_counts=torch.tensor([2]),
    )

    [value] = adapter._state_values(output)
    learned_policy = torch.softmax(output.policy_logits, dim=0)
    expected = torch.dot(learned_policy, output.q_values)
    reimproved = improved_policy(
        output.policy_logits,
        output.q_values,
        alpha=adapter.algorithm.alpha,
        beta=adapter.algorithm.beta,
    )

    torch.testing.assert_close(value, expected)
    assert not torch.isclose(value, torch.dot(reimproved, output.q_values))


@pytest.mark.parametrize(
    "architecture",
    [
        "graph",
        "dense_axis",
        "persistent_ray_axis",
        "hex_axial_cnn",
        "hex_dilated_cnn",
        "hex_d6_dilated_cnn",
    ],
)
@pytest.mark.parametrize("critic", ["scalar", "categorical"])
def test_head_to_head_loader_runs_klent_checkpoint_through_rust_mcts(
    tmp_path,
    architecture,
    critic,
):
    config = (
        tiny_dense_model_config()
        if architecture in {
            "dense_axis",
            "persistent_ray_axis",
            "hex_axial_cnn",
            "hex_dilated_cnn",
            "hex_d6_dilated_cnn",
        }
        else tiny_model_config()
    )
    if not isinstance(config, KlentModelConfig):
        config = KlentModelConfig(**dataclasses.asdict(config))
    config.critic = critic
    if architecture == "persistent_ray_axis":
        config.architecture = architecture
        config.ray_channels = 4
        config.ray_update_hidden = 8
    elif architecture == "hex_axial_cnn":
        config.architecture = architecture
        config.axial_attention_radius = 2
        config.axial_attention_layers = [0]
    elif architecture in {"hex_dilated_cnn", "hex_d6_dilated_cnn"}:
        config.architecture = architecture
        config.num_layers = 2
        config.cnn_dilations = [1, 2]
    algorithm = AlgorithmConfig()
    network = make_klent_net(config)
    checkpoint_path = tmp_path / f"klent-{architecture}-{critic}.pt"
    torch.save(
        {
            "format": "hexo-klent-v1",
            "iteration": 7,
            "model_state_dict": network.state_dict(),
            "model_config": dataclasses.asdict(config),
            "config": {"algorithm": dataclasses.asdict(algorithm)},
        },
        checkpoint_path,
    )

    loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
    graph_fn = graph_fn_from_model_config(loaded.model_config)
    eval_fn = make_eval_fn(
        loaded.model,
        torch.device("cpu"),
        graph_type=loaded.model_config.graph_type,
        prune_empty_edges=loaded.model_config.prune_empty_edges,
        threat_features=loaded.model_config.threat_features,
        relative_stones=loaded.model_config.relative_stone_encoding,
        graph_fn=graph_fn,
        model_config=loaded.model_config,
    )
    game_config = hexo_rs.GameConfig(6, 1, 12)
    game = hexo_rs.GameState(game_config)
    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=4,
        m_actions=2,
        c_visit=50,
        c_scale=1.0,
        disable_forcing_solver=True,
    )

    action, improved = hexo_rs.gumbel_mcts(
        game, eval_fn, mcts_config, seed=3
    )

    assert loaded.train_steps == 7
    assert action in game.legal_moves()
    assert len(improved) == game.legal_move_count()
    assert sum(improved) == pytest.approx(1.0)

    opening = sample_opening(
        loaded.model,
        game_config,
        torch.device("cpu"),
        k=2,
        temperature=0.5,
        seed=4,
        model_config=loaded.model_config,
    )
    assert len(opening) == 2
