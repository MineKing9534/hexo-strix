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
    assert config.training.policy_diagnostic_samples == 2048
    assert config.training.fit_max_autotune is False
    assert config.training.fit_compile_seed_nodes == 0
    assert config.training.learning_rate_warmup_iterations == 0
    assert config.training.learning_rate_warmup_start_iteration == 0
    assert config.training.learning_rate_warmup_start_factor == pytest.approx(
        0.1
    )
    assert config.training.critic_head_only is False
    assert config.training.heads_only is False
    assert config.training.search_q_teacher_samples == 0
    assert config.training.search_q_teacher_checkpoint == ""
    assert config.training.search_q_teacher_simulations == 32
    assert config.training.search_q_teacher_actions == 8
    assert config.training.search_q_teacher_root_batch_size == 16
    assert config.training.search_q_teacher_refit_epochs == 0
    assert config.run.compile is False


def test_config_fixture_loads_game_and_named_opponents():
    config = load_config(FIXTURE)

    assert config.game.win_length == 6
    assert config.game.placement_radius == 2
    assert config.game.rollout_horizon == 12
    assert config.evaluation.opening_plies == 8
    assert config.evaluation.opening_temperature == pytest.approx(0.5)
    assert config.evaluation.opening_generator == "alternate"
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


def test_policy_diagnostic_sample_count_cannot_be_negative(tmp_path):
    path = tmp_path / "negative-policy-diagnostics.toml"
    path.write_text(
        "[training]\npolicy_diagnostic_samples = -1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="policy_diagnostic_samples cannot be negative",
    ):
        load_config(path)


def test_protected_training_modes_are_mutually_exclusive(tmp_path):
    path = tmp_path / "conflicting-protected-modes.toml"
    path.write_text(
        "[training]\ncritic_head_only = true\nheads_only = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)


def test_search_q_teacher_requires_a_checkpoint_when_enabled(tmp_path):
    path = tmp_path / "missing-search-q-teacher.toml"
    path.write_text(
        "[training]\nsearch_q_teacher_samples = 16\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="search_q_teacher_samples requires search_q_teacher_checkpoint",
    ):
        load_config(path)


def test_search_q_refit_requires_search_q_samples(tmp_path):
    path = tmp_path / "search-q-refit-without-labels.toml"
    path.write_text(
        "[training]\nsearch_q_teacher_refit_epochs = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="search_q_teacher_refit_epochs requires search_q_teacher_samples",
    ):
        load_config(path)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        (
            "learning_rate_warmup_iterations",
            "-1",
            "learning_rate_warmup_iterations cannot be negative",
        ),
        (
            "learning_rate_warmup_start_iteration",
            "-1",
            "learning_rate_warmup_start_iteration cannot be negative",
        ),
        (
            "learning_rate_warmup_start_factor",
            "1.1",
            "learning_rate_warmup_start_factor must be in",
        ),
    ],
)
def test_learning_rate_warmup_settings_are_validated(
    tmp_path,
    setting,
    value,
    message,
):
    path = tmp_path / "bad-warmup.toml"
    path.write_text(
        f"[training]\n{setting} = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_dense_axis_backend_loads_strict_compatible_schema(tmp_path):
    path = tmp_path / "dense.toml"
    path.write_text(
        "[model]\n"
        'architecture = "dense_axis"\n'
        "dense_ray_radius = 5\n"
        'graph_type = "axis"\n'
        "axis_relational = true\n"
        "threat_features = true\n"
        "relative_stone_encoding = true\n"
        "compact_stone_onehot = true\n"
        "node_coords = false\n"
        'moves_scope = "node"\n'
        "pre_norm = true\n"
        "dropout = 0.0\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.architecture == "dense_axis"
    assert config.model.dense_ray_radius == 5


def test_dense_axis_rejects_incomplete_ray_radius(tmp_path):
    path = tmp_path / "dense-short-rays.toml"
    path.write_text(
        "[model]\n"
        'architecture = "dense_axis"\n'
        "dense_ray_radius = 4\n"
        'graph_type = "axis"\n'
        "axis_relational = true\n"
        "threat_features = true\n"
        "relative_stone_encoding = true\n"
        "compact_stone_onehot = true\n"
        "node_coords = false\n"
        'moves_scope = "node"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dense_ray_radius to cover"):
        load_config(path)


def test_persistent_ray_backend_loads_branch_configuration(tmp_path):
    path = tmp_path / "persistent-ray.toml"
    path.write_text(
        "[model]\n"
        'architecture = "persistent_ray_axis"\n'
        "dense_ray_radius = 5\n"
        "ray_channels = 10\n"
        "ray_update_hidden = 32\n"
        "ray_branch_scale = 0.5\n"
        "exact_graft_init = true\n"
        "ray_after_layers = [0, 2]\n"
        "num_layers = 3\n"
        'graph_type = "axis"\n'
        "axis_relational = true\n"
        "threat_features = true\n"
        "relative_stone_encoding = true\n"
        "compact_stone_onehot = true\n"
        "node_coords = false\n"
        'moves_scope = "node"\n'
        "pre_norm = true\n"
        "dropout = 0.0\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.architecture == "persistent_ray_axis"
    assert config.model.ray_channels == 10
    assert config.model.ray_update_hidden == 32
    assert config.model.ray_branch_scale == 0.5
    assert config.model.ray_after_layers == [0, 2]


def test_persistent_ray_rejects_invalid_branch_layer(tmp_path):
    path = tmp_path / "persistent-ray-invalid-layer.toml"
    path.write_text(
        "[model]\n"
        'architecture = "persistent_ray_axis"\n'
        "num_layers = 2\n"
        "ray_after_layers = [2]\n"
        'graph_type = "axis"\n'
        "axis_relational = true\n"
        "threat_features = true\n"
        "relative_stone_encoding = true\n"
        "compact_stone_onehot = true\n"
        "node_coords = false\n"
        'moves_scope = "node"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid layer indices"):
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


def test_best_so_far_opponent_loads_initial_checkpoint_and_threshold(tmp_path):
    path = tmp_path / "best.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'name = "incumbent"\n'
        'kind = "best_so_far"\n'
        'checkpoint = "checkpoint_000010.pt"\n'
        "best_promotion_win_rate = 0.60\n"
        "games = 16\n",
        encoding="utf-8",
    )

    [opponent] = load_config(path).evaluation.opponents

    assert opponent.kind == "best_so_far"
    assert opponent.checkpoint == "checkpoint_000010.pt"
    assert opponent.best_promotion_win_rate == pytest.approx(0.60)


def test_best_so_far_opponent_allows_a_distinct_incumbent_search_budget(
    tmp_path,
):
    path = tmp_path / "best-budget.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'name = "incumbent"\n'
        'kind = "best_so_far"\n'
        'checkpoint = "checkpoint_000010.pt"\n'
        "games = 16\n"
        "mcts_simulations = 24\n"
        "mcts_actions = 8\n"
        "opponent_mcts_simulations = 64\n"
        "opponent_mcts_actions = 16\n",
        encoding="utf-8",
    )

    [opponent] = load_config(path).evaluation.opponents

    assert opponent.opponent_mcts_simulations == 64
    assert opponent.opponent_mcts_actions == 16


def test_best_so_far_opponent_rejects_invalid_promotion_threshold(tmp_path):
    path = tmp_path / "best-invalid.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "best_so_far"\n'
        'checkpoint = "checkpoint_000010.pt"\n'
        "best_promotion_win_rate = 0.49\n"
        "games = 16\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="best_so_far promotion win rate must be in",
    ):
        load_config(path)


def test_checkpoint_paired_openings_require_even_games(tmp_path):
    path = tmp_path / "odd-checkpoint-games.toml"
    path.write_text(
        "[[evaluation.opponents]]\n"
        'kind = "checkpoint"\n'
        'checkpoint = "anchor.pt"\n'
        "games = 3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="paired-opening checkpoint evaluation requires an even",
    ):
        load_config(path)


def test_checkpoint_paired_openings_can_be_disabled(tmp_path):
    path = tmp_path / "legacy-checkpoint-games.toml"
    path.write_text(
        "[evaluation]\n"
        "opening_plies = 0\n"
        "[[evaluation.opponents]]\n"
        'kind = "checkpoint"\n'
        'checkpoint = "anchor.pt"\n'
        "games = 3\n",
        encoding="utf-8",
    )

    config = load_config(path)
    assert config.evaluation.opening_plies == 0
    assert config.evaluation.opponents[0].games == 3
