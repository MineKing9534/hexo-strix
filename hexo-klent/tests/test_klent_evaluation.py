import dataclasses
from contextlib import contextmanager

import pytest
import torch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.model import HeXONet
from hexo_klent import evaluation
from hexo_klent.config import AlgorithmConfig, GameConfig
from hexo_klent.evaluation import (
    CheckpointOpponentCache,
    EvaluationStats,
    evaluate_vs_checkpoint,
    evaluate_vs_lagged,
    evaluate_vs_random,
    resolve_lagged_checkpoint,
)
from hexo_klent.mcts_adapter import KlentMCTSAdapter
from hexo_klent.model import KlentNet


def _tiny_model_config() -> ModelConfig:
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


def test_batched_greedy_evaluation_completes_both_model_sides():
    model_config = ModelConfig(
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
    stats = evaluate_vs_random(
        KlentNet(model_config),
        model_config=model_config,
        game_config=GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=4
        ),
        games=4,
        device=torch.device("cpu"),
        seed=9,
    )

    assert stats.games == 4
    assert stats.wins == stats.losses == 0
    assert stats.truncations == 4
    assert stats.decided_rate == 0.0
    assert stats.win_rate_decided == 0.0


def test_random_evaluation_can_route_model_moves_through_mcts():
    model_config = ModelConfig(
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
    stats = evaluate_vs_random(
        KlentNet(model_config),
        model_config=model_config,
        game_config=GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=2
        ),
        games=2,
        device=torch.device("cpu"),
        seed=11,
        algorithm=AlgorithmConfig(),
        mcts_simulations=4,
        mcts_actions=2,
    )

    assert stats.games == 2
    assert stats.truncations == 2


def test_mcts_evaluation_honors_configured_inference_precision(monkeypatch):
    model_config = ModelConfig(
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
    events = []

    @contextmanager
    def recording_autocast(device, precision):
        events.append(("enter", device.type, precision))
        try:
            yield
        finally:
            events.append(("exit", device.type, precision))

    def fake_make_eval_fn(*args, **kwargs):
        del args, kwargs

        def model_eval_fn(states):
            events.append(("evaluate", len(states)))
            return ([[0.0, 0.0]], [0.0])

        return model_eval_fn

    def fake_gumbel_mcts(game, eval_fn, config, seed):
        del config, seed
        eval_fn([game])
        return ((0, 0), [0.25, 0.75])

    monkeypatch.setattr(evaluation, "_autocast", recording_autocast)
    monkeypatch.setattr(
        "hexo_a0.evaluate.make_eval_fn",
        fake_make_eval_fn,
    )
    monkeypatch.setattr("hexo_rs.gumbel_mcts", fake_gumbel_mcts)

    choose = evaluation._mcts_move_selector(
        KlentNet(model_config),
        model_config=model_config,
        algorithm=AlgorithmConfig(),
        simulations=4,
        actions=2,
        device=torch.device("cpu"),
        precision="bf16",
        seed=11,
    )

    class Game:
        @staticmethod
        def legal_moves():
            return [(0, 0), (1, 0)]

    assert choose is not None
    assert choose(Game()) == (1, 0)
    assert events == [
        ("enter", "cpu", "bf16"),
        ("evaluate", 1),
        ("exit", "cpu", "bf16"),
    ]


@pytest.mark.parametrize("checkpoint_kind", ["klent", "a0"])
def test_checkpoint_evaluation_accepts_cross_format_opponents(
    tmp_path,
    checkpoint_kind,
):
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / f"{checkpoint_kind}.pt"
    if checkpoint_kind == "klent":
        opponent = KlentNet(model_config)
        checkpoint = {
            "format": "hexo-klent-v1",
            "iteration": 7,
            "model_state_dict": opponent.state_dict(),
            "model_config": dataclasses.asdict(model_config),
            "config": {
                "algorithm": dataclasses.asdict(AlgorithmConfig())
            },
        }
    else:
        opponent = HeXONet(model_config)
        checkpoint = {
            "train_steps": 11,
            "model_state_dict": opponent.state_dict(),
            "model_config": dataclasses.asdict(model_config),
        }
    torch.save(checkpoint, checkpoint_path)

    stats = evaluate_vs_checkpoint(
        KlentNet(model_config),
        model_config=model_config,
        game_config=GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=4
        ),
        games=4,
        checkpoint=str(checkpoint_path),
        device=torch.device("cpu"),
        seed=13,
        opponent_mcts_simulations=4,
        opponent_mcts_actions=2,
        opening_plies=2,
    )

    assert stats.games == 4
    assert stats.wins == stats.losses == 0
    assert stats.truncations == 4
    assert stats.mean_opponent_depth == 0.0
    assert stats.opening_pairs == 2


def test_checkpoint_evaluation_replays_paired_openings_and_disables_noise(
    monkeypatch,
    tmp_path,
):
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / "anchor.pt"
    opponent = HeXONet(model_config)
    torch.save(
        {
            "train_steps": 11,
            "model_state_dict": opponent.state_dict(),
            "model_config": dataclasses.asdict(model_config),
        },
        checkpoint_path,
    )

    original_game_state = hexo_rs.GameState
    created_games = []

    class RecordingGame:
        def __init__(self, config):
            self.inner = original_game_state(config)
            self.applied = []
            created_games.append(self)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def apply_move(self, q, r):
            self.applied.append((int(q), int(r)))
            return self.inner.apply_move(q, r)

    sample_calls = []

    def fake_sample_opening(
        generator,
        rust_config,
        device,
        k,
        temperature,
        seed,
        model_config,
    ):
        del device, temperature
        probe = original_game_state(rust_config)
        opening = []
        for ply in range(k):
            legal = probe.legal_moves()
            q, r = legal[(seed + ply) % len(legal)]
            probe.apply_move(q, r)
            opening.append((int(q), int(r)))
        sample_calls.append((generator, model_config, seed, opening))
        return opening

    noise_settings = []

    def fake_candidate_selector(*args, **kwargs):
        del args
        noise_settings.append(("candidate", kwargs["disable_gumbel_noise"]))
        return None

    def fake_opponent_selector(*args, **kwargs):
        del args
        noise_settings.append(("opponent", kwargs["disable_gumbel_noise"]))
        return None

    def greedy_candidate(_model, states, **kwargs):
        del kwargs
        return [torch.zeros(len(state.legal_moves())) for state in states]

    def greedy_opponent(_model, states, **kwargs):
        del kwargs
        return (
            [torch.zeros(len(state.legal_moves())) for state in states],
            [torch.tensor(0.0) for _state in states],
        )

    monkeypatch.setattr(hexo_rs, "GameState", RecordingGame)
    monkeypatch.setattr("hexo_a0.evaluate.sample_opening", fake_sample_opening)
    monkeypatch.setattr(evaluation, "_mcts_move_selector", fake_candidate_selector)
    monkeypatch.setattr(
        evaluation,
        "_compatible_mcts_move_selector",
        fake_opponent_selector,
    )
    monkeypatch.setattr(evaluation, "_klent_policy_chunks", greedy_candidate)
    monkeypatch.setattr(evaluation, "_compatible_outputs", greedy_opponent)

    stats = evaluate_vs_checkpoint(
        KlentNet(model_config),
        model_config=model_config,
        game_config=GameConfig(
            win_length=6,
            placement_radius=1,
            rollout_horizon=4,
        ),
        games=4,
        checkpoint=str(checkpoint_path),
        device=torch.device("cpu"),
        seed=13,
        opening_plies=2,
        opening_temperature=0.5,
        opening_generator="alternate",
    )

    assert [call[2] for call in sample_calls] == [13, 14]
    assert isinstance(sample_calls[0][0], HeXONet)
    assert isinstance(sample_calls[1][0], KlentMCTSAdapter)
    assert created_games[0].applied[:2] == created_games[1].applied[:2]
    assert created_games[2].applied[:2] == created_games[3].applied[:2]
    assert created_games[0].applied[:2] != created_games[2].applied[:2]
    assert noise_settings == [("candidate", True), ("opponent", True)]
    assert stats.opening_pairs == 2
    assert stats.frac_unique_opening == 1.0


def test_checkpoint_cache_loads_a_fixed_model_once(monkeypatch, tmp_path):
    model_config = _tiny_model_config()
    checkpoint_path = tmp_path / "anchor.pt"
    opponent = KlentNet(model_config)
    torch.save(
        {
            "format": "hexo-klent-v1",
            "iteration": 7,
            "model_state_dict": opponent.state_dict(),
            "model_config": dataclasses.asdict(model_config),
            "config": {
                "algorithm": dataclasses.asdict(AlgorithmConfig())
            },
        },
        checkpoint_path,
    )
    from hexo_a0 import head_to_head

    original_load = head_to_head.load_checkpoint
    loads = []

    def recording_load(path, device):
        loads.append((path, device))
        return original_load(path, device)

    monkeypatch.setattr(head_to_head, "load_checkpoint", recording_load)
    cache = CheckpointOpponentCache()
    kwargs = {
        "model_config": model_config,
        "game_config": GameConfig(
            win_length=6, placement_radius=1, rollout_horizon=2
        ),
        "games": 2,
        "checkpoint": str(checkpoint_path),
        "device": torch.device("cpu"),
        "checkpoint_cache": cache,
    }

    evaluate_vs_checkpoint(KlentNet(model_config), **kwargs)
    evaluate_vs_checkpoint(KlentNet(model_config), **kwargs)

    assert loads == [(checkpoint_path.resolve(), torch.device("cpu"))]


def test_checkpoint_cache_bounds_moving_lagged_models(tmp_path):
    model_config = _tiny_model_config()
    opponent = KlentNet(model_config)
    checkpoint = {
        "format": "hexo-klent-v1",
        "iteration": 7,
        "model_state_dict": opponent.state_dict(),
        "model_config": dataclasses.asdict(model_config),
        "config": {"algorithm": dataclasses.asdict(AlgorithmConfig())},
    }
    paths = []
    for iteration in (100, 200, 300):
        path = tmp_path / f"checkpoint_{iteration:06d}.pt"
        torch.save(checkpoint, path)
        paths.append(path)

    cache = CheckpointOpponentCache(max_entries=2)
    for path in paths:
        with cache.activate(path, torch.device("cpu")):
            pass

    assert list(cache._loaded) == [
        paths[1].resolve(),
        paths[2].resolve(),
    ]


def test_lagged_checkpoint_prefers_branch_then_resume_history(tmp_path):
    branch = tmp_path / "branch"
    source = tmp_path / "source"
    branch.mkdir()
    source.mkdir()
    source_target = source / "checkpoint_000750.pt"
    source_target.touch()

    assert resolve_lagged_checkpoint(
        iteration=1000,
        lag_iterations=250,
        checkpoint_dirs=(branch, source),
    ) == source_target

    branch_target = branch / source_target.name
    branch_target.touch()
    assert resolve_lagged_checkpoint(
        iteration=1000,
        lag_iterations=250,
        checkpoint_dirs=(branch, source),
    ) == branch_target


def test_lagged_opponent_uses_one_shared_mcts_budget(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    target = checkpoint_dir / "checkpoint_000900.pt"
    target.touch()
    captured = {}
    expected = EvaluationStats(
        games=4,
        wins=2,
        losses=2,
        truncations=0,
        decided_rate=1.0,
        win_rate_decided=0.5,
        mean_game_length=12.0,
        mean_opponent_depth=0.0,
    )

    def fake_evaluate(model, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(evaluation, "evaluate_vs_checkpoint", fake_evaluate)
    model_config = _tiny_model_config()
    actual = evaluate_vs_lagged(
        KlentNet(model_config),
        model_config=model_config,
        game_config=GameConfig(),
        games=4,
        device=torch.device("cpu"),
        iteration=1000,
        checkpoint_dirs=(checkpoint_dir,),
        lag_iterations=100,
        algorithm=AlgorithmConfig(),
        mcts_simulations=24,
        mcts_actions=8,
    )

    assert actual == expected
    assert captured["checkpoint"] == str(target)
    assert captured["mcts_simulations"] == 24
    assert captured["mcts_actions"] == 8
    assert captured["opponent_mcts_simulations"] == 24
    assert captured["opponent_mcts_actions"] == 8
