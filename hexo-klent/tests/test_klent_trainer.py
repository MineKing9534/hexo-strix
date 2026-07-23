import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hexo_a0.config import ModelConfig
from hexo_klent.actor import TrajectoryStep
from hexo_klent.config import (
    CollectionConfig,
    Config,
    EvaluationConfig,
    EvaluationOpponentConfig,
    GameConfig,
    RunConfig,
    TrainingConfig,
)
from hexo_klent.trainer import (
    Trainer,
    _flat_training_targets,
    _gradient_clip_statistics,
    _segmented_log_softmax,
    train_epoch,
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
    actual = _segmented_log_softmax(flat, segment_ids, 3)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    weights = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.6, 1.0])
    expected_gradient = torch.autograd.grad(
        (expected * weights).sum(), flat, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad((actual * weights).sum(), flat)[0]
    assert torch.allclose(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


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

    target_policy, segment_ids, chosen, target_q, target_top1 = (
        _flat_training_targets(samples)
    )

    assert target_policy.tolist() == pytest.approx([0.1, 0.7, 0.2, 0.9, 0.1])
    assert segment_ids.tolist() == [0, 0, 0, 1, 1]
    assert chosen.tolist() == [2, 3]
    assert target_q.tolist() == pytest.approx([0.5, -0.25])
    assert target_top1 == 1


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
    assert per_microbatch_metrics["optimizer_steps"] == 2.0
    assert per_microbatch_metrics["microbatches"] == 2.0
    assert per_microbatch_metrics["mean_optimizer_batch_size"] == 2.0
    assert per_microbatch_metrics["mean_microbatches_per_step"] == 1.0


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

    assert metrics["collection/positions"] == 6.0
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
    assert not checkpoint.with_suffix(".pt.tmp").exists()


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

    resumed_branch = Trainer(
        branch_config,
        tensorboard=False,
        resume=branch_checkpoint,
    )
    assert source.checkpoint_dir.resolve() in (
        resumed_branch._checkpoint_history_dirs
    )
    resumed_branch.close()
