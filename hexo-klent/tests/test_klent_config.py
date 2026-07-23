from pathlib import Path

import pytest

from hexo_klent.config import load_config


FIXTURE = Path(__file__).parent / "fixtures" / "config.toml"


def test_config_fixture_loads_algorithm_collection_and_model():
    config = load_config(FIXTURE)

    assert config.algorithm.alpha == pytest.approx(0.03)
    assert config.algorithm.beta == pytest.approx(0.1)
    assert config.algorithm.trace_decay == pytest.approx(0.8824969025845955)
    assert config.model.axis_relational is True
    assert config.model.node_coords is False
    assert config.collection.positions_per_iteration == 16
    assert config.collection.parallel_games == 4
    assert config.collection.workers == 2
    assert config.training.grad_accumulation is True
    assert config.training.prefetch_batches is False
    assert config.run.compile is False


def test_config_fixture_loads_game_and_named_opponents():
    config = load_config(FIXTURE)

    assert config.game.win_length == 6
    assert config.game.placement_radius == 2
    assert config.game.rollout_horizon == 12
    assert [
        (
            opponent.name,
            opponent.kind,
            opponent.checkpoint,
            opponent.lag_iterations,
            opponent.games,
            opponent.depth,
            opponent.placement_radius,
            opponent.mcts_simulations,
            opponent.mcts_actions,
            opponent.opponent_mcts_simulations,
            opponent.opponent_mcts_actions,
        )
        for opponent in config.evaluation.opponents
    ] == [
        ("random", "random", "", 0, 4, 0, 0, 0, 16, 0, 16),
        ("sealbot_raw", "sealbot", "", 0, 2, 1, 8, 0, 16, 0, 16),
        (
            "sealbot_mcts4/2",
            "sealbot",
            "",
            0,
            2,
            1,
            8,
            4,
            2,
            0,
            16,
        ),
    ]
    assert config.run.output_dir == "test-output"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[algorithm]\nalpah = 0.03\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown \[algorithm\] keys: alpah"):
        load_config(path)


def test_parallel_games_cannot_exceed_position_budget(tmp_path):
    path = tmp_path / "too-many-lanes.toml"
    path.write_text(
        "[collection]\n"
        "positions_per_iteration = 3\n"
        "parallel_games = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="parallel_games cannot exceed.*positions_per_iteration",
    ):
        load_config(path)


def test_sealbot_opponent_requires_fixed_depth(tmp_path):
    path = tmp_path / "sealbot-without-depth.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "sealbot"\n'
        "games = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="SealBot evaluation depth must be positive",
    ):
        load_config(path)


def test_checkpoint_opponent_requires_a_path(tmp_path):
    path = tmp_path / "checkpoint-without-path.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "checkpoint"\n'
        "games = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="checkpoint evaluation opponent requires a checkpoint path",
    ):
        load_config(path)


def test_checkpoint_opponent_loads_fixed_checkpoint_path(tmp_path):
    path = tmp_path / "checkpoint.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'name = "w1_anchor"\n'
        'kind = "checkpoint"\n'
        'checkpoint = "runs/klent/reference-w1/checkpoints/final.pt"\n'
        "games = 8\n"
        "mcts_simulations = 24\n"
        "mcts_actions = 8\n"
        "opponent_mcts_simulations = 48\n"
        "opponent_mcts_actions = 12\n",
        encoding="utf-8",
    )

    [opponent] = load_config(path).evaluation.opponents

    assert opponent.name == "w1_anchor"
    assert opponent.kind == "checkpoint"
    assert (
        opponent.checkpoint
        == "runs/klent/reference-w1/checkpoints/final.pt"
    )
    assert opponent.games == 8
    assert opponent.mcts_simulations == 24
    assert opponent.mcts_actions == 8
    assert opponent.opponent_mcts_simulations == 48
    assert opponent.opponent_mcts_actions == 12


def test_opponent_side_mcts_is_checkpoint_specific(tmp_path):
    path = tmp_path / "random-opponent-mcts.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "random"\n'
        "games = 4\n"
        "opponent_mcts_simulations = 8\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="opponent-side MCTS settings are only supported",
    ):
        load_config(path)


def test_lagged_opponent_requires_positive_lag(tmp_path):
    path = tmp_path / "lagged-without-lag.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "lagged"\n'
        "games = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="lagged evaluation opponent requires positive lag_iterations",
    ):
        load_config(path)


def test_lagged_opponent_loads_shared_search_budget(tmp_path):
    path = tmp_path / "lagged.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'name = "lag_250"\n'
        'kind = "lagged"\n'
        "lag_iterations = 250\n"
        "games = 16\n"
        "mcts_simulations = 24\n"
        "mcts_actions = 8\n",
        encoding="utf-8",
    )

    [opponent] = load_config(path).evaluation.opponents

    assert opponent.kind == "lagged"
    assert opponent.lag_iterations == 250
    assert opponent.mcts_simulations == 24
    assert opponent.mcts_actions == 8
