import dataclasses
import json
import resource
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_klent.actor import TrajectoryStep
from hexo_klent.config import (
    CollectionConfig,
    Config,
    EvaluationConfig,
    EvaluationOpponentConfig,
    GameConfig,
    KlentModelConfig,
    RunConfig,
    TrainingConfig,
)
from hexo_klent.model import KlentNet, PersistentRayKlentNet
from hexo_klent.evaluation import EvaluationStats
from hexo_klent.trainer import (
    Trainer,
    _CudaCachePressureGate,
    _attach_search_q_teacher_labels,
    _critic_head_only_scope,
    _ensure_compiler_nofile_limit,
    _flat_training_targets,
    _gradient_clip_statistics,
    _heads_only_scope,
    _learning_rate_for_iteration,
    _optimizer_update_statistics,
    _prepared_training_batches,
    _release_cuda_cache,
    _refit_search_q_head,
    _seed_fit_compilation,
    _segmented_log_softmax,
    train_epoch,
)
from hexo_klent.search_q_teacher import SearchQLabels


def test_learning_rate_warmup_reaches_target_on_fifth_generation():
    training = TrainingConfig(
        learning_rate=2e-4,
        learning_rate_warmup_iterations=5,
        learning_rate_warmup_start_factor=0.1,
    )

    assert [
        _learning_rate_for_iteration(training, iteration)
        for iteration in range(1, 8)
    ] == pytest.approx(
        [2e-5, 6.5e-5, 1.1e-4, 1.55e-4, 2e-4, 2e-4, 2e-4]
    )


def test_learning_rate_warmup_can_start_after_resumed_iteration():
    training = TrainingConfig(
        learning_rate=2e-4,
        learning_rate_warmup_iterations=5,
        learning_rate_warmup_start_iteration=125,
        learning_rate_warmup_start_factor=0.1,
    )

    assert [
        _learning_rate_for_iteration(training, iteration)
        for iteration in (125, 126, 127, 128, 129, 130, 131)
    ] == pytest.approx(
        [2e-4, 2e-5, 6.5e-5, 1.1e-4, 1.55e-4, 2e-4, 2e-4]
    )


def test_run_honors_tui_pause_only_between_committed_iterations(tmp_path):
    events = []

    class Display:
        def begin_run(self, current_iteration, stop_at):
            events.append(("begin", current_iteration, stop_at))

        def wait_if_paused(self, iteration, *, on_pause):
            events.append(("pause", iteration))
            on_pause()
            events.append(("cache_released", iteration))
            return True

        def complete(self, checkpoint):
            events.append(("complete", checkpoint.name))

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        run=SimpleNamespace(iterations=2),
    )
    trainer.iteration = 0
    trainer.device = torch.device("cpu")
    trainer.display = Display()

    def run_iteration():
        trainer.iteration += 1
        events.append(("iteration", trainer.iteration))

    trainer.run_iteration = run_iteration
    trainer.close = lambda: events.append("close")
    trainer.save_checkpoint = lambda *, final: (
        tmp_path / ("final.pt" if final else "checkpoint.pt")
    )

    trainer.run()

    assert events == [
        ("begin", 0, 2),
        ("iteration", 1),
        ("pause", 1),
        ("cache_released", 1),
        ("iteration", 2),
        "close",
        ("complete", "final.pt"),
    ]


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


def test_segmented_log_softmax_matches_independent_values_and_gradients():
    flat = torch.tensor(
        [1.0, -2.0, 0.5, 4.0, -3.0, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    segment_ids = torch.tensor([0, 0, 1, 1, 1, 2])
    expected = torch.cat(
        [
            F.log_softmax(flat[:2], dim=0),
            F.log_softmax(flat[2:5], dim=0),
            F.log_softmax(flat[5:], dim=0),
        ]
    )
    actual = _segmented_log_softmax(
        flat,
        segment_ids,
        torch.tensor([2, 3, 1]),
        3,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    weights = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.6, 1.0])
    expected_gradient = torch.autograd.grad(
        (expected * weights).sum(), flat, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad((actual * weights).sum(), flat)[0]
    assert torch.allclose(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


def test_compiler_nofile_limit_is_raised_to_safe_soft_limit(monkeypatch):
    applied = []
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _kind: (1_024, 524_288),
    )
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda kind, limits: applied.append((kind, limits)),
    )

    assert _ensure_compiler_nofile_limit() == (65_536, 524_288)
    assert applied == [(resource.RLIMIT_NOFILE, (65_536, 524_288))]


def test_flat_training_targets_preserve_variable_action_alignment():
    samples = [
        TrajectoryStep(
            state=object(),
            target_policy=torch.tensor([0.1, 0.7, 0.2]),
            action_index=2,
            player="P1",
            state_value=0.0,
            return_target=0.5,
        ),
        TrajectoryStep(
            state=object(),
            target_policy=torch.tensor([0.9, 0.1]),
            action_index=0,
            player="P2",
            state_value=0.0,
            return_target=-0.25,
        ),
    ]

    (
        target_policy,
        segment_ids,
        chosen,
        target_q,
        target_top1,
        segment_lengths,
    ) = _flat_training_targets(samples)

    assert target_policy.tolist() == pytest.approx([0.1, 0.7, 0.2, 0.9, 0.1])
    assert segment_ids.tolist() == [0, 0, 0, 1, 1]
    assert chosen.tolist() == [2, 3]
    assert target_q.tolist() == pytest.approx([0.5, -0.25])
    assert target_top1 == 1
    assert segment_lengths.tolist() == [3, 2]


def test_flat_training_targets_append_unplayed_search_q_labels():
    sample = TrajectoryStep(
        state=object(),
        target_policy=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        action_index=2,
        player="P1",
        state_value=0.0,
        return_target=-0.5,
        auxiliary_q_action_indices=torch.tensor([0, 3]),
        auxiliary_q_targets=torch.tensor([0.75, -0.25]),
    )

    _policy, _segments, chosen, target_q, _top1, _lengths = (
        _flat_training_targets([sample])
    )

    assert chosen.tolist() == [2, 0, 3]
    assert target_q.tolist() == pytest.approx([-0.5, 0.75, -0.25])


def test_search_q_labels_keep_terminal_target_authoritative_for_played_move():
    sample = TrajectoryStep(
        state=object(),
        target_policy=torch.full((4,), 0.25),
        action_index=1,
        player="P1",
        state_value=0.0,
        return_target=1.0,
    )

    metrics = _attach_search_q_teacher_labels(
        [sample],
        [
            SearchQLabels(
                action_indices=torch.tensor([0, 1, 3]),
                targets=torch.tensor([0.25, -0.75, 0.5]),
                legal_actions=4,
            )
        ],
    )

    assert sample.auxiliary_q_action_indices.tolist() == [0, 3]
    assert sample.auxiliary_q_targets.tolist() == pytest.approx([0.25, 0.5])
    packed = _flat_training_targets([sample])
    assert packed[2].tolist() == [1, 0, 3]
    assert packed[3].tolist() == pytest.approx([1.0, 0.25, 0.5])
    assert metrics == pytest.approx(
        {
            "search_q_teacher_states": 1.0,
            "search_q_teacher_visited_labels": 3.0,
            "search_q_teacher_auxiliary_labels": 2.0,
            "search_q_teacher_mean_auxiliary_labels": 2.0,
            "search_q_teacher_visited_coverage": 0.75,
            "search_q_teacher_mean_abs_target": 0.375,
        }
    )


def test_search_q_head_refit_does_not_move_policy_or_shared_trunk():
    config = tiny_model_config()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    samples = [
        TrajectoryStep(
            state=hexo_rs.GameState(game_config),
            target_policy=torch.full((6,), 1 / 6),
            action_index=index,
            player="P1",
            state_value=0.0,
            return_target=(-1.0) ** index,
            auxiliary_q_action_indices=torch.tensor([2, 3]),
            auxiliary_q_targets=torch.tensor([0.8, -0.6]),
        )
        for index in range(2)
    ]
    model = KlentNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    metrics = _refit_search_q_head(
        model,
        optimizer,
        samples,
        model_config=config,
        device=torch.device("cpu"),
        precision="float32",
        batch_size=2,
        edge_budget=0,
        epochs=1,
        max_grad_norm=0.0,
        seed=7,
    )

    changed_q = False
    for name, value in model.state_dict().items():
        if name.startswith("q_head."):
            changed_q = changed_q or not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert changed_q
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert metrics["search_q_teacher_refit_states"] == 2.0
    assert metrics["search_q_teacher_refit_q_labels"] == 6.0
    assert metrics["search_q_teacher_refit_optimizer_steps"] == 1.0
    assert metrics["search_q_teacher_refit_nonfinite_optimizer_steps"] == 0.0
    assert metrics["search_q_teacher_refit_q_loss"] > 0.0


def test_critic_head_only_epoch_preserves_policy_trunk_and_optimizer_state():
    config = tiny_model_config()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    target_policy = torch.tensor([0.45, 0.2, 0.1, 0.1, 0.1, 0.05])
    samples = [
        TrajectoryStep(
            state=hexo_rs.GameState(game_config),
            target_policy=target_policy,
            action_index=index,
            player="P1",
            state_value=0.0,
            return_target=(-1.0) ** index,
        )
        for index in range(2)
    ]
    model = KlentNet(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=1e-3
    )
    arguments = dict(
        samples=samples,
        model_config=config,
        device=torch.device("cpu"),
        precision="float32",
        batch_size=2,
        edge_budget=0,
        grad_accumulation=True,
        q_loss_weight=1.0,
        max_grad_norm=0.0,
        seed=7,
        prefetch_batches=False,
    )

    # Populate AdamW moments for the complete model before the protected pass,
    # matching a warm-up resumed from an already-trained graft.
    train_epoch(model, optimizer, optimize_policy=True, **arguments)
    optimizer.zero_grad(set_to_none=True)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    q_parameter_ids = {id(parameter) for parameter in model.q_head.parameters()}
    frozen_optimizer_state = {
        id(parameter): {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state.get(parameter, {}).items()
        }
        for parameter in model.parameters()
        if id(parameter) not in q_parameter_ids
    }

    with _critic_head_only_scope(model):
        metrics = train_epoch(
            model,
            optimizer,
            optimize_policy=False,
            **arguments,
        )

    changed_q = False
    for name, value in model.state_dict().items():
        if name.startswith("q_head."):
            changed_q = changed_q or not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert changed_q
    assert all(parameter.requires_grad for parameter in model.parameters())
    for parameter in model.parameters():
        if id(parameter) in q_parameter_ids:
            continue
        actual = optimizer.state.get(parameter, {})
        expected = frozen_optimizer_state[id(parameter)]
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(
                    actual_value, expected_value, rtol=0, atol=0
                )
            else:
                assert actual_value == expected_value
    assert metrics["policy_updates_enabled"] == 0.0
    assert metrics["total_loss"] == pytest.approx(metrics["q_loss"])


def test_heads_only_epoch_preserves_trunk_and_optimizer_state():
    config = tiny_model_config()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    samples = [
        TrajectoryStep(
            state=hexo_rs.GameState(game_config),
            target_policy=torch.tensor(
                [0.45, 0.2, 0.1, 0.1, 0.1, 0.05]
            ),
            action_index=index,
            player="P1",
            state_value=0.0,
            return_target=(-1.0) ** index,
        )
        for index in range(2)
    ]
    model = KlentNet(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=1e-3
    )
    arguments = dict(
        samples=samples,
        model_config=config,
        device=torch.device("cpu"),
        precision="float32",
        batch_size=2,
        edge_budget=0,
        grad_accumulation=True,
        q_loss_weight=1.0,
        max_grad_norm=0.0,
        seed=7,
        prefetch_batches=False,
        optimize_policy=True,
    )

    # Populate every AdamW slot before checking the protected pass.
    train_epoch(model, optimizer, **arguments)
    optimizer.zero_grad(set_to_none=True)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    head_parameter_ids = {
        id(parameter)
        for head in (model.policy_head, model.q_head)
        for parameter in head.parameters()
    }
    frozen_optimizer_state = {
        id(parameter): {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state.get(parameter, {}).items()
        }
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    }

    with _heads_only_scope(model):
        metrics = train_epoch(model, optimizer, **arguments)

    changed_policy = False
    changed_q = False
    for name, value in model.state_dict().items():
        if name.startswith("policy_head."):
            changed_policy = changed_policy or not torch.equal(
                value, before[name]
            )
        elif name.startswith("q_head."):
            changed_q = changed_q or not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert changed_policy
    assert changed_q
    assert all(parameter.requires_grad for parameter in model.parameters())
    for parameter in model.parameters():
        if id(parameter) in head_parameter_ids:
            continue
        actual = optimizer.state.get(parameter, {})
        expected = frozen_optimizer_state[id(parameter)]
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(
                    actual_value, expected_value, rtol=0, atol=0
                )
            else:
                assert actual_value == expected_value
    assert metrics["policy_updates_enabled"] == 1.0
    assert metrics["total_loss"] == pytest.approx(
        metrics["policy_loss"] + metrics["q_loss"]
    )


def test_gradient_clip_statistics_report_frequency_distribution_and_scale():
    metrics = _gradient_clip_statistics(
        torch.tensor([0.5, 1.0, 2.0, 4.0]),
        max_grad_norm=1.0,
    )

    assert metrics["mean_grad_norm"] == pytest.approx(1.875)
    assert metrics["grad_norm_p50"] == pytest.approx(1.5)
    assert metrics["grad_norm_p95"] == pytest.approx(3.7)
    assert metrics["grad_norm_max"] == pytest.approx(4.0)
    assert metrics["clipped_optimizer_steps"] == 2.0
    assert metrics["clip_fraction"] == pytest.approx(0.5)
    assert metrics["mean_clip_scale"] == pytest.approx(0.6875)


def test_optimizer_update_statistics_report_norms_and_relative_movement():
    metrics = _optimizer_update_statistics(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([1e-4, 2e-4, 3e-4, 4e-4]),
    )

    assert metrics["mean_parameter_update_norm"] == pytest.approx(0.25)
    assert metrics["parameter_update_norm_p95"] == pytest.approx(0.385)
    assert metrics["mean_update_to_weight_ratio"] == pytest.approx(2.5e-4)
    assert metrics["update_to_weight_ratio_p95"] == pytest.approx(3.85e-4)


def test_release_cuda_cache_reports_and_releases_reserved_memory(monkeypatch):
    gib = 1024**3
    empty_cache_calls = []
    reset_calls = []
    reserved = iter([12 * gib, 3 * gib])

    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 2 * gib,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: next(reserved),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 7 * gib,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: 14 * gib,
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_calls.append(device),
    )

    metrics = _release_cuda_cache(
        torch.device("cuda"),
        phase="training",
    )

    assert empty_cache_calls == [True]
    assert reset_calls == [torch.device("cuda")]
    assert metrics == pytest.approx(
        {
            "memory/training_allocated_gib": 2.0,
            "memory/training_reserved_before_gib": 12.0,
            "memory/training_reserved_after_gib": 3.0,
            "memory/training_cache_released_gib": 9.0,
            "memory/training_peak_allocated_gib": 7.0,
            "memory/training_peak_reserved_gib": 14.0,
        }
    )


def test_release_cuda_cache_is_a_noop_on_cpu(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("CPU cache release called torch.cuda"),
    )

    assert (
        _release_cuda_cache(torch.device("cpu"), phase="training")
        == {}
    )


def test_cuda_cache_pressure_gate_clears_only_over_threshold(monkeypatch):
    gib = 1024**3
    reserved = iter([10 * gib, 2 * gib, 7 * gib])
    empty_cache_calls = []

    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: next(reserved),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: 1 * gib,
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    gate = _CudaCachePressureGate(
        torch.device("cuda"),
        threshold_bytes=8 * gib,
        check_every=2,
    )

    assert gate.step() is False
    assert gate.step() is True
    assert gate.step() is False
    assert gate.step() is False

    assert empty_cache_calls == [True]
    assert gate.metrics() == pytest.approx(
        {
            "allocator_pressure_checks": 2.0,
            "allocator_pressure_clears": 1.0,
            "allocator_pressure_released_gib": 8.0,
            "allocator_pressure_max_reserved_gib": 10.0,
            "allocator_pressure_threshold_gib": 8.0,
            "allocator_pressure_check_every": 2.0,
        }
    )


def test_gradient_clip_statistics_are_neutral_when_clipping_is_disabled():
    metrics = _gradient_clip_statistics(
        torch.tensor([2.0, 4.0]),
        max_grad_norm=0.0,
    )

    assert metrics["clipped_optimizer_steps"] == 0.0
    assert metrics["clip_fraction"] == 0.0
    assert metrics["mean_clip_scale"] == 1.0


def test_accumulated_microbatches_match_unsplit_outer_batch(monkeypatch):
    class FakeBatch:
        def __init__(self, features):
            self.features = features

        def to(self, device):
            self.features = self.features.to(device)
            return self

    class TinyPolicyQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = nn.Linear(2, 1)
            self.q = nn.Linear(2, 1)

        def forward_batch(self, batch):
            return SimpleNamespace(
                policy_logits=self.policy(batch.features).squeeze(-1),
                q_values=self.q(batch.features).squeeze(-1),
            )

    samples = [
        TrajectoryStep(
            state=torch.tensor(
                [[float(index), 1.0], [float(index) + 0.5, -1.0]]
            ),
            target_policy=torch.tensor([0.25, 0.75]),
            action_index=index % 2,
            player="P1",
            state_value=0.0,
            return_target=(-1.0) ** index * 0.4,
        )
        for index in range(4)
    ]
    # Make the two edge-budgeted microbatches contain different Q-label
    # densities. Exact state-weighting would now be wrong; the accumulated
    # gradient must instead reconstruct the six-label outer Q population.
    samples[0].auxiliary_q_action_indices = torch.tensor([1])
    samples[0].auxiliary_q_targets = torch.tensor([0.7])
    samples[3].auxiliary_q_action_indices = torch.tensor([0])
    samples[3].auxiliary_q_targets = torch.tensor([-0.8])

    def fake_prepared(
        shuffled,
        *,
        edge_budget,
        **_kwargs,
    ):
        ranges = [(0, len(shuffled))]
        if edge_budget > 0:
            ranges = [(0, 3), (3, len(shuffled))]
        prepared = []
        for start, stop in ranges:
            packed = shuffled[start:stop]
            prepared.append(
                (
                    FakeBatch(
                        torch.cat([sample.state for sample in packed])
                    ),
                    packed,
                    _flat_training_targets(packed),
                )
            )
        yield prepared

    monkeypatch.setattr(
        "hexo_klent.trainer._prepared_training_batches",
        fake_prepared,
    )
    torch.manual_seed(17)
    unsplit = TinyPolicyQ()
    accumulated = TinyPolicyQ()
    accumulated.load_state_dict(unsplit.state_dict())
    per_microbatch = TinyPolicyQ()
    per_microbatch.load_state_dict(unsplit.state_dict())
    unsplit_optimizer = torch.optim.SGD(unsplit.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(
        accumulated.parameters(), lr=0.05
    )
    per_microbatch_optimizer = torch.optim.SGD(
        per_microbatch.parameters(), lr=0.05
    )
    arguments = dict(
        samples=samples,
        model_config=object(),
        device=torch.device("cpu"),
        precision="float32",
        batch_size=4,
        grad_accumulation=True,
        q_loss_weight=0.7,
        max_grad_norm=0.0,
        seed=3,
        prefetch_batches=False,
    )

    unsplit_metrics = train_epoch(
        unsplit,
        unsplit_optimizer,
        edge_budget=0,
        **arguments,
    )
    accumulated_metrics = train_epoch(
        accumulated,
        accumulated_optimizer,
        edge_budget=1,
        **arguments,
    )
    per_microbatch_metrics = train_epoch(
        per_microbatch,
        per_microbatch_optimizer,
        edge_budget=1,
        **{**arguments, "grad_accumulation": False},
    )

    for actual, expected in zip(
        accumulated.parameters(), unsplit.parameters()
    ):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert accumulated_metrics["policy_loss"] == pytest.approx(
        unsplit_metrics["policy_loss"], abs=1e-6
    )
    assert accumulated_metrics["q_loss"] == pytest.approx(
        unsplit_metrics["q_loss"], abs=1e-6
    )
    assert accumulated_metrics["optimizer_steps"] == 1.0
    assert accumulated_metrics["microbatches"] == 2.0
    assert accumulated_metrics["mean_optimizer_batch_size"] == 4.0
    assert accumulated_metrics["mean_microbatch_size"] == 2.0
    assert accumulated_metrics["mean_microbatches_per_step"] == 2.0
    assert accumulated_metrics["q_labels"] == 6.0
    assert accumulated_metrics["mean_q_labels_per_example"] == pytest.approx(
        1.5
    )
    assert per_microbatch_metrics["optimizer_steps"] == 2.0
    assert per_microbatch_metrics["microbatches"] == 2.0
    assert per_microbatch_metrics["mean_optimizer_batch_size"] == 2.0
    assert per_microbatch_metrics["mean_microbatches_per_step"] == 1.0

    guarded = TinyPolicyQ()
    guarded.load_state_dict(unsplit.state_dict())
    guarded_optimizer = torch.optim.SGD(guarded.parameters(), lr=0.05)
    real_clip_grad_norm = torch.nn.utils.clip_grad_norm_
    clip_calls = 0

    def reject_first_group(parameters, max_norm):
        nonlocal clip_calls
        materialized = list(parameters)
        clip_calls += 1
        if clip_calls == 1:
            return torch.tensor(float("nan"))
        return real_clip_grad_norm(materialized, max_norm)

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        reject_first_group,
    )
    guarded_metrics = train_epoch(
        guarded,
        guarded_optimizer,
        edge_budget=1,
        **{
            **arguments,
            "grad_accumulation": False,
            "max_grad_norm": 1.0,
        },
    )

    assert guarded_metrics["examples"] == 4.0
    assert guarded_metrics["updated_examples"] == 1.0
    assert guarded_metrics["skipped_nonfinite_examples"] == 3.0
    assert guarded_metrics["attempted_optimizer_steps"] == 2.0
    assert guarded_metrics["nonfinite_optimizer_steps"] == 1.0
    assert guarded_metrics["optimizer_steps"] == 1.0
    assert all(
        torch.isfinite(parameter).all() for parameter in guarded.parameters()
    )


def test_fit_compile_seed_does_not_update_model_or_optimizer():
    config = tiny_model_config()
    game_config = hexo_rs.GameConfig(2, 1, 4)
    samples = [
        TrajectoryStep(
            state=hexo_rs.GameState(game_config),
            target_policy=torch.full((6,), 1 / 6),
            action_index=index,
            player="P1",
            state_value=0.0,
            return_target=0.25,
        )
        for index in range(2)
    ]
    [prepared] = list(
        _prepared_training_batches(
            samples,
            batch_size=2,
            model_config=config,
            edge_budget=0,
            prefetch=False,
        )
    )
    model = KlentNet(config)
    model._fit_compile_seed_nodes = 1
    model._fit_compile_seeded = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    calls = 0
    eager_forward_fit = model.forward_fit

    def tracked_forward_fit(batch, chosen):
        nonlocal calls
        calls += 1
        return eager_forward_fit(batch, chosen)

    model.forward_fit = tracked_forward_fit
    for _ in range(2):
        _seed_fit_compilation(
            model,
            optimizer,
            prepared,
            device=torch.device("cpu"),
            precision="float32",
            q_loss_weight=1.0,
        )

    assert calls == 1
    assert model._fit_compile_seeded is True
    assert optimizer.state_dict()["state"] == {}
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_tiny_cpu_iteration_collects_fits_and_checkpoints(tmp_path):
    class RecordingDisplay:
        def __init__(self):
            self.phases = []
            self.metrics = []

        def set_phase(self, phase, iteration, detail=""):
            self.phases.append((phase, iteration, detail))

        def update_metrics(self, metrics):
            self.metrics.append(metrics)

    display = RecordingDisplay()
    config = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=6,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=16,
            edge_budget=0,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path),
            checkpoint_interval=0,
            seed=11,
        ),
    )
    trainer = Trainer(config, tensorboard=False, display=display)

    metrics = trainer.run_iteration()
    checkpoint = trainer.save_checkpoint(final=True)

    assert metrics["collection/positions"] > 0
    assert metrics["collection/chunk_truncations"] == 0.0
    assert (
        metrics["training/examples"] == metrics["collection/positions"]
    )
    assert metrics["collection/positions_per_second"] > 0
    assert metrics["training/examples"] > 0
    assert metrics["training/microbatches"] >= 1
    assert metrics["training/optimizer_steps"] >= 1
    assert (
        metrics["training/microbatches"]
        >= metrics["training/optimizer_steps"]
    )
    assert metrics["training/elapsed_seconds"] > 0
    assert metrics["training/examples_per_second"] > 0
    assert metrics["training/policy_excess_kl"] == pytest.approx(
        max(
            0.0,
            metrics["training/policy_loss"]
            - metrics["collection/mean_entropy"],
        )
    )
    assert (
        metrics["training/policy_diagnostic_examples"]
        == metrics["training/examples"]
    )
    assert metrics["training/policy_diagnostic_seconds"] >= 0.0
    assert metrics["training/policy_target_kl_before"] == pytest.approx(
        metrics["collection/mean_reverse_kl"]
    )
    expected_target_progress = (
        0.0
        if metrics["training/policy_target_kl_before"] <= 1e-12
        else 1.0
        - metrics["training/policy_target_kl_after"]
        / metrics["training/policy_target_kl_before"]
    )
    assert metrics["training/policy_target_progress"] == pytest.approx(
        expected_target_progress
    )
    assert (
        metrics["training/trunk_gradient_diagnostic_examples"]
        == metrics["training/policy_diagnostic_examples"]
    )
    assert metrics["training/trunk_gradient_diagnostic_seconds"] >= 0
    assert metrics["training/policy_trunk_grad_norm"] >= 0
    assert metrics["training/q_trunk_grad_norm"] >= 0
    assert -1 <= metrics["training/policy_q_trunk_grad_cosine"] <= 1
    assert 0 <= metrics["training/clip_fraction"] <= 1
    assert 0 < metrics["training/mean_clip_scale"] <= 1
    assert (
        metrics["training/clipped_optimizer_steps"]
        <= metrics["training/optimizer_steps"]
    )
    assert (
        metrics["training/grad_norm_p50"]
        <= metrics["training/grad_norm_p95"]
        <= metrics["training/grad_norm_max"]
    )
    assert metrics["training/mean_parameter_update_norm"] > 0
    assert (
        metrics["training/parameter_update_norm_p95"]
        >= metrics["training/mean_parameter_update_norm"]
    )
    assert metrics["training/mean_update_to_weight_ratio"] > 0
    assert metrics["training/update_to_weight_ratio_p95"] > 0
    assert checkpoint.exists()
    assert [phase for phase, _iteration, _detail in display.phases] == [
        "COLLECT",
        "FIT",
        "COMMIT",
    ]
    assert display.metrics == [metrics]
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["iteration"] == 1.0


    resumed = Trainer(config, tensorboard=False, resume=checkpoint)
    assert resumed.iteration == 1
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    restored_optimizer = resumed.optimizer.state_dict()
    config.training.learning_rate = 5e-4
    lower_lr_resume = Trainer(
        config,
        tensorboard=False,
        resume=checkpoint,
        resume_configured_lr=True,
    )
    assert lower_lr_resume.iteration == 1
    assert lower_lr_resume.optimizer.param_groups[0]["lr"] == pytest.approx(
        5e-4
    )
    lower_lr_optimizer = lower_lr_resume.optimizer.state_dict()
    assert (
        lower_lr_optimizer["state"].keys()
        == restored_optimizer["state"].keys()
    )
    for parameter_id, restored_state in restored_optimizer["state"].items():
        lower_lr_state = lower_lr_optimizer["state"][parameter_id]
        assert lower_lr_state.keys() == restored_state.keys()
        for state_name, restored_value in restored_state.items():
            lower_lr_value = lower_lr_state[state_name]
            if isinstance(restored_value, torch.Tensor):
                torch.testing.assert_close(lower_lr_value, restored_value)
            else:
                assert lower_lr_value == restored_value
    restored_group = restored_optimizer["param_groups"][0]
    lower_lr_group = lower_lr_optimizer["param_groups"][0]
    assert lower_lr_group["lr"] == pytest.approx(5e-4)
    assert {
        key: value
        for key, value in lower_lr_group.items()
        if key != "lr"
    } == {
        key: value
        for key, value in restored_group.items()
        if key != "lr"
    }
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_tiny_critic_head_only_iteration_freezes_policy_and_trunk(tmp_path):
    config = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=6,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=16,
            edge_budget=0,
            learning_rate=1e-3,
            weight_decay=1e-4,
            critic_head_only=True,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path),
            checkpoint_interval=0,
            seed=13,
        ),
    )
    trainer = Trainer(config, tensorboard=False)
    before = {
        name: value.detach().clone()
        for name, value in trainer.model.state_dict().items()
    }

    metrics = trainer.run_iteration()
    trainer.close()

    changed_q = False
    for name, value in trainer.model.state_dict().items():
        if name.startswith("q_head."):
            changed_q = changed_q or not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert changed_q
    assert metrics["training/critic_head_only"] == 1.0
    assert metrics["training/policy_updates_enabled"] == 0.0
    assert metrics["training/trunk_gradient_diagnostic_examples"] == 0.0
    assert metrics["training/q_trunk_grad_norm"] == 0.0
    assert metrics["training/policy_trunk_grad_norm"] == 0.0
    assert metrics["training/policy_target_kl_after"] == pytest.approx(
        metrics["training/policy_target_kl_before"], abs=1e-8
    )
    assert metrics["training/policy_target_top1_agreement_after"] == (
        metrics["training/policy_target_top1_agreement_before"]
    )


def test_tiny_heads_only_iteration_freezes_shared_trunk(tmp_path):
    config = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=6,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=16,
            edge_budget=0,
            learning_rate=1e-3,
            weight_decay=1e-4,
            heads_only=True,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path),
            checkpoint_interval=0,
            seed=17,
        ),
    )
    trainer = Trainer(config, tensorboard=False)
    before = {
        name: value.detach().clone()
        for name, value in trainer.model.state_dict().items()
    }

    metrics = trainer.run_iteration()
    trainer.close()

    changed_policy = False
    changed_q = False
    for name, value in trainer.model.state_dict().items():
        if name.startswith("policy_head."):
            changed_policy = changed_policy or not torch.equal(
                value, before[name]
            )
        elif name.startswith("q_head."):
            changed_q = changed_q or not torch.equal(value, before[name])
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert changed_policy
    assert changed_q
    assert metrics["training/critic_head_only"] == 0.0
    assert metrics["training/heads_only"] == 1.0
    assert metrics["training/policy_updates_enabled"] == 1.0
    assert metrics["training/shared_trunk_updates_enabled"] == 0.0
    assert metrics["training/trunk_gradient_diagnostic_examples"] == 0.0
    assert metrics["training/q_trunk_grad_norm"] == 0.0
    assert metrics["training/policy_trunk_grad_norm"] == 0.0


def test_learning_rate_warmup_is_applied_and_resumes_by_iteration(tmp_path):
    config = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=4,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=4,
            edge_budget=0,
            learning_rate=2e-4,
            learning_rate_warmup_iterations=5,
            learning_rate_warmup_start_factor=0.1,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=2,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path),
            checkpoint_interval=0,
            seed=13,
        ),
    )
    trainer = Trainer(config, tensorboard=False)

    first = trainer.run_iteration()
    checkpoint = trainer.save_checkpoint()
    trainer.close()

    assert first["training/learning_rate"] == pytest.approx(2e-5)

    resumed = Trainer(
        config,
        tensorboard=False,
        resume=checkpoint,
    )
    second = resumed.run_iteration()

    assert second["training/learning_rate"] == pytest.approx(6.5e-5)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(6.5e-5)
    resumed.close()


def test_persistent_trainer_grafts_dense_klent_checkpoint(tmp_path):
    base_model = tiny_model_config()
    base_model.threat_features = True
    dense_model = KlentModelConfig(
        **vars(base_model),
        architecture="dense_axis",
        dense_ray_radius=2,
    )
    source_config = Config(
        model=dense_model,
        game=GameConfig(
            win_length=2,
            placement_radius=1,
            rollout_horizon=4,
        ),
        collection=CollectionConfig(
            positions_per_iteration=4,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(batch_size=4),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path / "dense"),
            checkpoint_interval=0,
        ),
    )
    source = Trainer(source_config, tensorboard=False)
    source.iteration = 7
    source_checkpoint = source.save_checkpoint(final=True)
    dense_state = {
        key: value.detach().clone()
        for key, value in source.model.state_dict().items()
    }
    source.close()

    persistent_model = KlentModelConfig(
        **vars(base_model),
        architecture="persistent_ray_axis",
        dense_ray_radius=2,
        ray_channels=4,
        ray_update_hidden=8,
        exact_graft_init=True,
    )
    target_config = dataclasses.replace(
        source_config,
        model=persistent_model,
        run=dataclasses.replace(
            source_config.run,
            output_dir=str(tmp_path / "persistent"),
        ),
    )
    target = Trainer(
        target_config,
        tensorboard=False,
        init_from=source_checkpoint,
    )

    assert isinstance(target.model, PersistentRayKlentNet)
    assert target.iteration == 0
    assert target.initial_checkpoint is not None
    assert target.initial_checkpoint["iteration"] == 7
    assert target.initial_checkpoint["graft"] == "persistent_ray_axis"
    target_state = target.model.state_dict()
    for key, value in dense_state.items():
        torch.testing.assert_close(target_state[key], value)
    assert any(
        key.startswith("ray_mixers.") for key in target_state
    )

    target.run_iteration()
    checkpoint = target.save_checkpoint(final=True)
    trained_state = {
        key: value.detach().clone()
        for key, value in target.model.state_dict().items()
    }
    optimizer_state = target.optimizer.state_dict()
    assert optimizer_state["state"]
    target.close()

    resumed = Trainer(
        target_config,
        tensorboard=False,
        resume=checkpoint,
    )
    assert resumed.iteration == 1
    for key, value in trained_state.items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key],
            value,
        )
    resumed_optimizer = resumed.optimizer.state_dict()
    assert (
        resumed_optimizer["param_groups"]
        == optimizer_state["param_groups"]
    )
    assert resumed_optimizer["state"].keys() == optimizer_state["state"].keys()
    for parameter_id, expected_state in optimizer_state["state"].items():
        actual_state = resumed_optimizer["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()
        for state_name, expected_value in expected_state.items():
            actual_value = actual_state[state_name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    resumed.close()


def test_persistent_trainer_grafts_graph_klent_checkpoint(tmp_path):
    graph_model = tiny_model_config()
    graph_model.threat_features = True
    graph_model.axis_window = 5
    graph_model.use_jk = True
    graph_model.jk_mode = "cat"
    source_config = Config(
        model=graph_model,
        game=GameConfig(
            win_length=2,
            placement_radius=1,
            rollout_horizon=4,
        ),
        collection=CollectionConfig(
            positions_per_iteration=4,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(batch_size=4),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path / "graph"),
            checkpoint_interval=0,
        ),
    )
    source = Trainer(source_config, tensorboard=False)
    source.iteration = 25
    source_checkpoint = source.save_checkpoint(final=True)
    source.close()

    persistent_model = KlentModelConfig(
        **vars(graph_model),
        architecture="persistent_ray_axis",
        dense_ray_radius=5,
        ray_channels=4,
        ray_update_hidden=8,
        exact_graft_init=True,
    )
    target_config = dataclasses.replace(
        source_config,
        model=persistent_model,
        run=dataclasses.replace(
            source_config.run,
            output_dir=str(tmp_path / "persistent"),
        ),
    )
    target = Trainer(
        target_config,
        tensorboard=False,
        init_from=source_checkpoint,
    )

    assert isinstance(target.model, PersistentRayKlentNet)
    assert target.iteration == 0
    assert target.initial_checkpoint is not None
    assert target.initial_checkpoint["iteration"] == 25
    assert target.initial_checkpoint["graft"] == "persistent_ray_axis"
    assert target.initial_checkpoint["source_architecture"] == "graph"
    assert target.initial_checkpoint["copied_tensors"] > 0
    assert any(
        key.startswith("ray_mixers.")
        for key in target.model.state_dict()
    )
    target.close()


def test_graph_trainer_initializes_from_production_checkpoint(tmp_path):
    model_config = tiny_model_config()
    model_config.threat_features = True
    model_config.use_jk = True
    model_config.jk_mode = "cat"
    source_model = KlentNet(model_config)
    with torch.no_grad():
        for index, parameter in enumerate(source_model.parameters(), start=1):
            parameter.fill_(index / 100.0)
    source_state = {
        key: value.detach().clone()
        for key, value in source_model.state_dict().items()
    }
    source_state["value_head.unused"] = torch.ones(3)
    source_checkpoint = tmp_path / "production.pt"
    torch.save(
        {
            "model_state_dict": source_state,
            "model_config": {
                **dataclasses.asdict(model_config),
                "q_head": True,
            },
            "train_steps": 1234,
        },
        source_checkpoint,
    )

    config = Config(
        model=KlentModelConfig(**dataclasses.asdict(model_config)),
        game=GameConfig(
            win_length=2,
            placement_radius=1,
            rollout_horizon=4,
        ),
        collection=CollectionConfig(
            positions_per_iteration=4,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(batch_size=4),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=1,
            device="cpu",
            precision="float32",
            output_dir=str(tmp_path / "graph"),
            checkpoint_interval=0,
        ),
    )
    target = Trainer(
        config,
        tensorboard=False,
        init_from=source_checkpoint,
    )

    assert isinstance(target.model, KlentNet)
    assert target.iteration == 0
    assert not target.optimizer.state_dict()["state"]
    assert target.initial_checkpoint is not None
    assert target.initial_checkpoint["train_steps"] == 1234
    assert target.initial_checkpoint["graft"] == "graph"
    assert target.initial_checkpoint["copied_tensors"] == len(source_state) - 1
    for key, value in target.model.state_dict().items():
        torch.testing.assert_close(value, source_state[key])
    target.close()


def test_lagged_evaluation_uses_resume_checkpoint_history(tmp_path):
    source_dir = tmp_path / "source"
    branch_dir = tmp_path / "branch"
    base = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=6,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=16,
            edge_budget=0,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=6,
            device="cpu",
            precision="float32",
            output_dir=str(source_dir),
            checkpoint_interval=0,
            seed=11,
        ),
    )
    source = Trainer(base, tensorboard=False)
    source.iteration = 5
    checkpoint = source.save_checkpoint()
    source.close()

    branch_config = dataclasses.replace(
        base,
        evaluation=EvaluationConfig(
            interval=1,
            opening_plies=2,
            opponents=[
                EvaluationOpponentConfig(
                    name="lag_1",
                    kind="lagged",
                    lag_iterations=1,
                    games=2,
                )
            ],
        ),
        run=dataclasses.replace(
            base.run,
            output_dir=str(branch_dir),
        ),
    )
    branch = Trainer(
        branch_config,
        tensorboard=False,
        resume=checkpoint,
    )

    metrics = branch.run_iteration()
    branch_checkpoint = branch.save_checkpoint()
    branch.close()

    assert metrics["evaluation/lag_1/games"] == 2.0
    assert metrics["evaluation/lag_1/configured_lag_iterations"] == 1.0
    assert metrics["evaluation/lag_1/opponent_iteration"] == 5.0
    assert metrics["evaluation/lag_1/mcts_simulations"] == 0.0
    assert metrics["evaluation/lag_1/opponent_mcts_simulations"] == 0.0
    assert metrics["evaluation/lag_1/opening_pairs"] == 1.0
    assert metrics["evaluation/lag_1/opening_plies"] == 2.0

    resumed_branch = Trainer(
        branch_config,
        tensorboard=False,
        resume=branch_checkpoint,
    )
    assert source.checkpoint_dir.resolve() in (
        resumed_branch._checkpoint_history_dirs
    )
    resumed_branch.close()


def test_best_so_far_evaluation_promotes_and_persists_across_rounds(
    tmp_path, monkeypatch
):
    import hexo_klent.trainer as trainer_module

    source_dir = tmp_path / "source"
    branch_dir = tmp_path / "branch"
    base = Config(
        model=tiny_model_config(),
        game=GameConfig(
            win_length=2, placement_radius=1, rollout_horizon=4
        ),
        collection=CollectionConfig(
            positions_per_iteration=6,
            parallel_games=2,
            inference_batch_size=2,
        ),
        training=TrainingConfig(
            batch_size=16,
            edge_budget=0,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=EvaluationConfig(interval=0, opponents=[]),
        run=RunConfig(
            iterations=12,
            device="cpu",
            precision="float32",
            output_dir=str(source_dir),
            checkpoint_interval=0,
            seed=17,
        ),
    )
    source = Trainer(base, tensorboard=False)
    source.iteration = 10
    initial_best = source.save_checkpoint()
    source.close()

    branch_config = dataclasses.replace(
        base,
        evaluation=EvaluationConfig(
            interval=1,
            opening_plies=2,
            opponents=[
                EvaluationOpponentConfig(
                    name="best",
                    kind="best_so_far",
                    checkpoint=str(initial_best),
                    best_promotion_win_rate=0.55,
                    games=2,
                )
            ],
        ),
        run=dataclasses.replace(
            base.run,
            output_dir=str(branch_dir),
            checkpoint_interval=0,
        ),
    )
    evaluated_checkpoints = []
    results = iter(((2, 0), (0, 2)))

    def fake_evaluate(kind, _model, **kwargs):
        evaluated_checkpoints.append((kind, kwargs["checkpoint"]))
        wins, losses = next(results)
        return EvaluationStats(
            games=2,
            wins=wins,
            losses=losses,
            truncations=0,
            decided_rate=1.0,
            win_rate_decided=wins / 2,
            mean_game_length=2.0,
            mean_opponent_depth=0.0,
            opening_pairs=1,
            frac_unique_opening=1.0,
        )

    monkeypatch.setattr(trainer_module, "evaluate_opponent", fake_evaluate)
    branch = Trainer(
        branch_config,
        tensorboard=False,
        resume=initial_best,
    )

    promoted_metrics = branch.run_iteration()
    promoted_checkpoint = branch.checkpoint_dir / "checkpoint_000011.pt"
    state_path = branch.output_dir / "best_so_far" / "best.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert promoted_metrics["evaluation/best/promoted"] == 1.0
    assert promoted_metrics["evaluation/best/opponent_iteration"] == 10.0
    assert promoted_checkpoint.is_file()
    assert state["iteration"] == 11
    assert Path(state["checkpoint"]) == promoted_checkpoint.resolve()

    rejected_metrics = branch.run_iteration()
    unchanged_state = json.loads(state_path.read_text(encoding="utf-8"))
    branch.close()

    assert rejected_metrics["evaluation/best/promoted"] == 0.0
    assert rejected_metrics["evaluation/best/opponent_iteration"] == 11.0
    assert Path(unchanged_state["checkpoint"]) == promoted_checkpoint.resolve()
    assert evaluated_checkpoints == [
        ("checkpoint", str(initial_best.resolve())),
        ("checkpoint", str(promoted_checkpoint.resolve())),
    ]
