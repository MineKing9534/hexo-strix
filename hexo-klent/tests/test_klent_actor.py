import math

import pytest
import torch

from hexo_a0.config import ModelConfig
from hexo_klent.actor import (
    SharedInferenceActors,
    Trajectory,
    TrajectoryStep,
    _actor_worker_main,
    _collect_with_inference,
    collect_games,
    collect_games_parallel,
    flatten_trajectories,
    terminal_played_q_calibration,
)
from hexo_klent.batching import raster_shape
from hexo_klent.config import AlgorithmConfig, GameConfig
from hexo_klent.model import KlentNet


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


def test_collect_games_drains_to_terminal_games_with_fresh_targets():
    model_config = tiny_model_config()
    model = KlentNet(model_config)

    trajectories, stats = collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        algorithm=AlgorithmConfig(),
        positions=12,
        parallel_games=2,
        inference_batch_size=2,
        device=torch.device("cpu"),
        seed=7,
    )
    samples = flatten_trajectories(trajectories)

    assert stats.games == len(trajectories)
    assert stats.positions == len(samples)
    assert stats.positions >= 12
    assert stats.discarded_positions == 0
    assert stats.p1_wins + stats.p2_wins + stats.truncations == stats.games
    assert all(sample.return_target is not None for sample in samples)
    assert all(
        sample.played_q is not None and math.isfinite(sample.played_q)
        for sample in samples
    )
    assert all(
        sample.target_policy.sum().item() == pytest.approx(1.0)
        for sample in samples
    )
    assert all(
        sample.target_policy.numel() == sample.state.legal_move_count()
        for sample in samples
    )
    entropies = [
        float(
            -(
                sample.target_policy
                * sample.target_policy.clamp_min(1e-12).log()
            ).sum()
        )
        for sample in samples
    ]
    assert stats.mean_entropy == pytest.approx(
        sum(entropies) / len(samples)
    )
    assert stats.mean_normalized_entropy == pytest.approx(
        sum(
            entropy / math.log(sample.target_policy.numel())
            if sample.target_policy.numel() > 1
            else 0.0
            for entropy, sample in zip(entropies, samples, strict=True)
        )
        / len(samples)
    )
    assert stats.mean_target_top1_probability == pytest.approx(
        sum(float(sample.target_policy.max()) for sample in samples)
        / len(samples)
    )
    assert 0.0 <= stats.mean_prior_normalized_entropy <= 1.0
    assert 0.0 < stats.mean_prior_top1_probability <= 1.0
    assert stats.mean_legal_actions == pytest.approx(
        sum(sample.target_policy.numel() for sample in samples)
        / len(samples)
    )
    assert stats.mean_q_span == pytest.approx(0.0)


def test_collect_games_confines_play_to_d6_symmetric_finite_board():
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    board_radius = 2

    trajectories, stats = collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=2, placement_radius=2, rollout_horizon=8
        ),
        algorithm=AlgorithmConfig(),
        positions=12,
        parallel_games=2,
        inference_batch_size=2,
        board_radius=board_radius,
        device=torch.device("cpu"),
        seed=17,
    )

    assert stats.positions >= 12
    for sample in flatten_trajectories(trajectories):
        assert sample.state.board_radius() == board_radius
        height, width = raster_shape(sample.state)
        assert height == width
        for q, r in sample.state.legal_moves():
            assert max(abs(q), abs(r), abs(q + r)) <= board_radius


def test_terminal_played_q_calibration_uses_terminal_player_outcomes():
    def step(player: str, q: float | None) -> TrajectoryStep:
        return TrajectoryStep(
            state=object(),
            target_policy=torch.tensor([1.0]),
            action_index=0,
            player=player,
            state_value=0.0,
            played_q=q,
        )

    trajectories = [
        Trajectory(
            [step("P1", 0.8), step("P2", -0.6)],
            winner="P1",
        ),
        Trajectory(
            [step("P1", -0.2), step("P2", 0.4)],
            winner="P2",
        ),
        # Neither an unfinished game nor a legacy step without played_q may
        # contaminate the genuinely terminal calibration sample.
        Trajectory([step("P1", 100.0)], truncated=True),
        Trajectory([step("P1", None)], winner="P1"),
    ]

    metrics = terminal_played_q_calibration(trajectories, opening_plies=1)

    assert metrics["played_q_outcome_positions"] == 4
    assert metrics["played_q_outcome_mse"] == pytest.approx(0.3)
    assert metrics["played_q_outcome_mae"] == pytest.approx(0.5)
    assert metrics["played_q_outcome_bias"] == pytest.approx(0.1)
    assert metrics["played_q_outcome_sign_accuracy"] == 1.0
    assert metrics["played_q_outcome_calibration_slope"] == pytest.approx(
        2.0 / 1.16
    )
    assert metrics["opening_played_q_outcome_positions"] == 2
    assert metrics["opening_played_q_outcome_mse"] == pytest.approx(0.34)
    assert metrics["opening_played_q_outcome_sign_accuracy"] == 1.0
    assert metrics["opening_played_q_outcome_calibration_slope"] == pytest.approx(
        2.0
    )


def test_collection_model_forward_runs_in_inference_mode():
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    original_forward = model.forward_batch
    inference_states = []

    def recording_forward(batch):
        inference_states.append(
            (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        )
        return original_forward(batch)

    model.forward_batch = recording_forward
    collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=2,
            placement_radius=1,
            rollout_horizon=2,
        ),
        algorithm=AlgorithmConfig(),
        positions=4,
        parallel_games=2,
        inference_batch_size=2,
        device=torch.device("cpu"),
        seed=11,
    )

    assert inference_states
    assert all(
        not grad_enabled and inference_enabled
        for grad_enabled, inference_enabled in inference_states
    )


def test_rollout_horizon_discards_the_entire_nonterminal_game():
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    with torch.no_grad():
        model.q_head.mlp[-2].bias.fill_(math.atanh(0.25))

    trajectories, stats = collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=2
        ),
        algorithm=AlgorithmConfig(),
        positions=4,
        parallel_games=2,
        inference_batch_size=2,
        device=torch.device("cpu"),
        seed=17,
    )

    assert stats.truncations == 2
    assert stats.horizon_truncations == 2
    assert stats.chunk_truncations == 0
    assert stats.p1_wins == stats.p2_wins == 0
    assert stats.positions == 0
    assert stats.discarded_positions == 4
    assert stats.mean_abs_bootstrap_value == 0.0
    assert trajectories == []


def test_distillation_can_retain_horizon_capped_teacher_prefixes():
    model_config = tiny_model_config()
    model = KlentNet(model_config)

    trajectories, stats = collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=2
        ),
        algorithm=AlgorithmConfig(),
        positions=4,
        parallel_games=2,
        inference_batch_size=2,
        device=torch.device("cpu"),
        seed=17,
        retain_horizon_truncations=True,
    )

    assert stats.truncations == 2
    assert stats.horizon_truncations == 2
    assert stats.spatial_truncations == 0
    assert stats.chunk_truncations == 0
    assert stats.positions == 4
    assert stats.discarded_positions == 0
    assert len(trajectories) == 2
    assert all(trajectory.truncated for trajectory in trajectories)
    assert all(trajectory.winner is None for trajectory in trajectories)
    assert all(
        step.return_target is None
        for trajectory in trajectories
        for step in trajectory.steps
    )


def test_position_budget_drains_live_games_to_terminal_results():
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    with torch.no_grad():
        model.q_head.mlp[-2].bias.fill_(math.atanh(0.25))

    trajectories, stats = collect_games(
        model,
        model_config=model_config,
        game_config=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=10
        ),
        algorithm=AlgorithmConfig(),
        positions=3,
        parallel_games=2,
        inference_batch_size=2,
        device=torch.device("cpu"),
        seed=19,
    )

    assert stats.positions >= 3
    assert stats.positions == sum(len(item.steps) for item in trajectories)
    assert stats.discarded_positions == 0
    assert stats.games == 2
    assert stats.truncations == 0
    assert stats.horizon_truncations == 0
    assert stats.chunk_truncations == 0
    assert stats.mean_abs_bootstrap_value == 0.0
    assert all(trajectory.winner in {"P1", "P2"} for trajectory in trajectories)
    assert all(not trajectory.truncated for trajectory in trajectories)
    assert all(
        abs(trajectory.steps[-1].return_target) == pytest.approx(1.0)
        for trajectory in trajectories
    )


def test_dense_spatial_boundary_discards_wandering_games():
    def frontier_inference(states):
        logits = []
        q_values = []
        for state in states:
            legal = state.legal_moves()
            frontier = max(
                range(len(legal)),
                key=lambda index: (
                    abs(legal[index][0]) + abs(legal[index][1]),
                    legal[index],
                ),
            )
            state_logits = torch.full((len(legal),), -100.0)
            state_logits[frontier] = 100.0
            logits.append(state_logits)
            q_values.append(torch.full((len(legal),), 0.25))
        return logits, q_values

    trajectories, stats = _collect_with_inference(
        frontier_inference,
        game_config=GameConfig(
            win_length=6,
            placement_radius=6,
            rollout_horizon=20,
        ),
        algorithm=AlgorithmConfig(),
        positions=2,
        parallel_games=2,
        dense_position_cell_limit=17 * 17,
        seed=29,
        worker_processes=1,
        retain_horizon_truncations=True,
    )

    assert stats.positions == 0
    assert stats.discarded_positions == 2
    assert stats.truncations == 2
    assert stats.horizon_truncations == 0
    assert stats.spatial_truncations == 2
    assert stats.chunk_truncations == 0
    assert stats.max_dense_position_cells == 17 * 17
    assert trajectories == []
    assert stats.mean_abs_bootstrap_value == 0.0


def test_completed_trajectory_streaming_preserves_collection_semantics():
    def zero_inference(states):
        return (
            [torch.zeros(state.legal_move_count()) for state in states],
            [torch.zeros(state.legal_move_count()) for state in states],
        )

    arguments = dict(
        game_config=GameConfig(
            win_length=2,
            placement_radius=1,
            rollout_horizon=10,
        ),
        algorithm=AlgorithmConfig(),
        positions=20,
        parallel_games=2,
        dense_position_cell_limit=0,
        seed=37,
        worker_processes=1,
    )
    expected, expected_stats = _collect_with_inference(
        zero_inference,
        **arguments,
    )
    chunks = []
    retained, actual_stats = _collect_with_inference(
        zero_inference,
        completed_callback=chunks.append,
        completed_chunk_positions=3,
        **arguments,
    )
    actual = [
        trajectory
        for chunk in chunks
        for trajectory in chunk
    ]

    assert retained == []
    assert len(chunks) > 1
    assert [len(item.steps) for item in actual] == [
        len(item.steps) for item in expected
    ]
    assert actual_stats.positions == expected_stats.positions
    assert actual_stats.games == expected_stats.games
    assert actual_stats.p1_wins == expected_stats.p1_wins
    assert actual_stats.p2_wins == expected_stats.p2_wins
    assert actual_stats.truncations == expected_stats.truncations
    assert actual_stats.mean_game_length == pytest.approx(
        expected_stats.mean_game_length
    )
    assert actual_stats.mean_abs_return == pytest.approx(
        expected_stats.mean_abs_return
    )
    for expected_step, actual_step in zip(
        flatten_trajectories(expected),
        flatten_trajectories(actual),
        strict=True,
    ):
        assert actual_step.state.placed_stones() == (
            expected_step.state.placed_stones()
        )
        assert actual_step.return_target == pytest.approx(
            expected_step.return_target
        )
        torch.testing.assert_close(
            actual_step.target_policy,
            expected_step.target_policy,
        )


def test_parallel_collection_uses_spawn_workers_and_merges_shards():
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    with torch.no_grad():
        model.q_head.mlp[-2].bias.fill_(math.atanh(0.25))
    # Force several acknowledged result chunks per actor in this small test.
    actors = SharedInferenceActors(2, result_chunk_positions=1)
    try:
        for seed in (23, 24):
            trajectories, stats = collect_games_parallel(
                model,
                model_config=model_config,
                game_config=GameConfig(
                    win_length=2, placement_radius=1, rollout_horizon=10
                ),
                algorithm=AlgorithmConfig(),
                positions=8,
                parallel_games=4,
                inference_batch_size=4,
                inference_edge_budget=100,
                workers=2,
                batch_timeout_ms=2.0,
                device=torch.device("cpu"),
                seed=seed,
                actors=actors,
            )

            assert len(trajectories) == stats.games
            assert stats.positions >= 8
            assert stats.positions == sum(
                len(trajectory.steps) for trajectory in trajectories
            )
            assert stats.discarded_positions == 0
            assert stats.truncations == 0
            assert stats.horizon_truncations == 0
            assert stats.chunk_truncations == 0
            assert stats.mean_abs_bootstrap_value == 0.0
            assert all(
                trajectory.winner in {"P1", "P2"}
                and not trajectory.truncated
                for trajectory in trajectories
            )
            assert stats.worker_processes == 2
    finally:
        actors.close()


def test_actor_worker_ignores_process_group_sigint(monkeypatch):
    installed_handlers = []

    class IdleTaskQueue:
        @staticmethod
        def get():
            return None

    class RunningEvent:
        @staticmethod
        def is_set():
            return False

    monkeypatch.setattr(
        "hexo_klent.actor.signal.signal",
        lambda signum, handler: installed_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _threads: None)

    _actor_worker_main(
        0,
        IdleTaskQueue(),
        object(),
        object(),
        RunningEvent(),
    )

    import signal

    assert installed_handlers == [(signal.SIGINT, signal.SIG_IGN)]


def test_parent_interrupt_can_reap_busy_parallel_actors(monkeypatch):
    model_config = tiny_model_config()
    model = KlentNet(model_config)
    actors = SharedInferenceActors(2)
    processes = actors._processes

    def interrupt_inference(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(actors, "_serve_requests", interrupt_inference)
    try:
        with pytest.raises(KeyboardInterrupt):
            actors.collect(
                model,
                model_config=model_config,
                game_config=GameConfig(
                    win_length=6,
                    placement_radius=1,
                    rollout_horizon=20,
                ),
                algorithm=AlgorithmConfig(),
                positions=40,
                parallel_games=4,
                inference_batch_size=4,
                inference_edge_budget=100,
                batch_timeout_ms=2.0,
                device=torch.device("cpu"),
                precision="float32",
                seed=31,
            )
    finally:
        actors.close()

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode is not None for process in processes)
