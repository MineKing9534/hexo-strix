import copy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import hexo_rs
import hexo_klent.distill as distill_module
from hexo_klent.config import Config, KlentModelConfig
from hexo_klent.distill import (
    DistillationExample,
    _initialize_student_weights,
    _save_distilled_checkpoint,
    _transform_distillation_example,
    distillation_losses,
    label_teacher_outputs,
)
from hexo_klent.model import (
    BatchOutput,
    HexAxialCNNKlentNet,
    HexD6DilatedCNNKlentNet,
    HexDilatedCNNKlentNet,
    make_klent_net,
)
from hexo_klent.trainer import Trainer


def _tiny_hex_config(tmp_path=None) -> Config:
    config = Config(
        model=KlentModelConfig(
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
    )
    config.run.device = "cpu"
    config.run.precision = "float32"
    config.run.compile = False
    config.collection.workers = 1
    if tmp_path is not None:
        config.run.output_dir = str(tmp_path / "run")
    return config


def test_distillation_losses_are_zero_for_identical_teacher_outputs():
    output = BatchOutput(
        policy_logits=torch.tensor([1.0, -1.0, 0.5, 0.25, -0.75]),
        q_values=torch.tensor([0.2, -0.4, 0.7, 0.1, -0.2]),
        legal_counts=torch.tensor([2, 3]),
    )
    game = hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 4))
    examples = [
        DistillationExample(
            game,
            output.policy_logits[:2].clone(),
            output.q_values[:2].clone(),
        ),
        DistillationExample(
            game,
            output.policy_logits[2:].clone(),
            output.q_values[2:].clone(),
        ),
    ]

    policy_kl, q_mse, top1 = distillation_losses(
        output,
        examples,
        temperature=1.0,
    )

    torch.testing.assert_close(policy_kl, torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(q_mse, torch.tensor(0.0))
    torch.testing.assert_close(top1, torch.tensor(1.0))


def test_segmented_distillation_losses_and_gradients_match_state_loop():
    torch.manual_seed(67)
    counts = [2, 5, 3]
    total = sum(counts)
    student_logits = torch.randn(total, requires_grad=True)
    student_q = torch.randn(total, requires_grad=True)
    teacher_logits = torch.randn(total)
    teacher_q = torch.randn(total)
    output = BatchOutput(
        policy_logits=student_logits,
        q_values=student_q,
        legal_counts=torch.tensor(counts),
    )
    game = hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 4))
    examples = []
    offset = 0
    for count in counts:
        examples.append(
            DistillationExample(
                game,
                teacher_logits[offset : offset + count],
                teacher_q[offset : offset + count],
            )
        )
        offset += count

    temperature = 1.7
    actual = distillation_losses(
        output,
        examples,
        temperature=temperature,
    )
    actual_grad = torch.autograd.grad(
        actual[0] + actual[1],
        (student_logits, student_q),
    )

    reference_policy = []
    reference_q = []
    reference_top1 = []
    offset = 0
    for count in counts:
        student_policy = F.log_softmax(
            student_logits[offset : offset + count] / temperature,
            dim=0,
        )
        teacher_policy = F.softmax(
            teacher_logits[offset : offset + count] / temperature,
            dim=0,
        )
        reference_policy.append(
            F.kl_div(student_policy, teacher_policy, reduction="sum")
            * temperature**2
        )
        reference_q.append(
            F.mse_loss(
                student_q[offset : offset + count],
                teacher_q[offset : offset + count],
            )
        )
        reference_top1.append(
            (
                student_logits[offset : offset + count].argmax()
                == teacher_logits[offset : offset + count].argmax()
            ).float()
        )
        offset += count
    expected = (
        torch.stack(reference_policy).mean(),
        torch.stack(reference_q).mean(),
        torch.stack(reference_top1).mean(),
    )
    expected_grad = torch.autograd.grad(
        expected[0] + expected[1],
        (student_logits, student_q),
    )

    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_retained_teacher_labels_are_valid_backward_targets():
    teacher_config = KlentModelConfig(
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
    teacher = make_klent_net(teacher_config)
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 8))
        for _ in range(2)
    ]
    examples = label_teacher_outputs(
        states,
        teacher,
        teacher_model_config=teacher_config,
        device=torch.device("cpu"),
        precision="float32",
        edge_budget=0,
    )
    counts = torch.tensor([example.policy_logits.numel() for example in examples])
    student_logits = torch.zeros(int(counts.sum()), requires_grad=True)
    student_q = torch.zeros(int(counts.sum()), requires_grad=True)
    output = BatchOutput(student_logits, student_q, counts)

    policy_kl, q_mse, _top1 = distillation_losses(
        output,
        examples,
        temperature=1.0,
    )
    (policy_kl + q_mse).backward()

    assert student_logits.grad is not None
    assert student_q.grad is not None


def test_teacher_labelling_bounds_graph_collation_chunks(monkeypatch):
    teacher_config = KlentModelConfig(
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
    teacher = make_klent_net(teacher_config)
    states = [
        hexo_rs.GameState(hexo_rs.GameConfig(2, 1, 8))
        for _ in range(5)
    ]
    original_prepare = distill_module.prepare_graph_batches
    observed_sizes = []

    def recording_prepare(state_chunk, **kwargs):
        observed_sizes.append(len(state_chunk))
        return original_prepare(state_chunk, **kwargs)

    monkeypatch.setattr(
        distill_module,
        "prepare_graph_batches",
        recording_prepare,
    )
    examples = label_teacher_outputs(
        states,
        teacher,
        teacher_model_config=teacher_config,
        device=torch.device("cpu"),
        precision="float32",
        edge_budget=0,
        batch_size=2,
    )

    assert observed_sizes == [2, 2, 1]
    assert len(examples) == len(states)


def test_student_optimizer_batches_span_raster_shape_groups(monkeypatch):
    def examples(count):
        return [
            DistillationExample(object(), torch.zeros(1), torch.zeros(1))
            for _ in range(count)
        ]

    groups = [
        [(example, 0) for example in examples(count)]
        for count in (3, 1, 5)
    ]
    monkeypatch.setattr(
        distill_module,
        "_example_groups",
        lambda _examples, *, seed, augment_symmetries=False: groups,
    )
    monkeypatch.setattr(
        distill_module,
        "prepare_graph_batches",
        lambda states, **_kwargs: [(object(), slice(0, len(states)))],
    )

    batches = list(
        distill_module._iter_student_batches(
            [example for group in groups for example, _transform in group],
            model_config=object(),
            batch_size=4,
            cell_budget=0,
            seed=0,
        )
    )

    assert [(len(selected), boundary) for _, selected, boundary in batches] == [
        (3, False),
        (1, True),
        (4, True),
        (1, True),
    ]


@pytest.mark.parametrize("transform_index", range(12))
def test_d6_distillation_transform_preserves_and_permutes_actions(
    transform_index,
):
    state = hexo_rs.GameState(hexo_rs.GameConfig(6, 8, 300))
    state.apply_move(1, 0)
    state.apply_move(2, 0)
    state.apply_move(0, 1)
    legal = state.legal_moves()
    policy = torch.arange(len(legal), dtype=torch.float32)
    q_values = -policy
    example = DistillationExample(state, policy, q_values)

    transformed = _transform_distillation_example(example, transform_index)
    expected_by_coord = {
        distill_module._transform_coord(coord, transform_index): index
        for index, coord in enumerate(legal)
    }
    expected_indices = torch.tensor(
        [expected_by_coord[coord] for coord in transformed.state.legal_moves()],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        transformed.policy_logits,
        policy.index_select(0, expected_indices),
    )
    torch.testing.assert_close(
        transformed.q_values,
        q_values.index_select(0, expected_indices),
    )
    assert set(transformed.state.legal_moves()) == set(expected_by_coord)


def test_symmetry_schedule_covers_every_d6_orientation_per_twelve_epochs(
    monkeypatch,
):
    examples = [
        DistillationExample(object(), torch.zeros(1), torch.zeros(1))
        for _ in range(3)
    ]
    monkeypatch.setattr(
        distill_module,
        "_transformed_raster_shape",
        lambda _state, _transform: (9, 9),
    )

    observed = []
    for epoch in range(12):
        groups = distill_module._example_groups(
            examples,
            seed=17 + epoch,
            augment_symmetries=True,
        )
        observed.append(
            {
                id(example): transform
                for group in groups
                for example, transform in group
            }
        )

    for example in examples:
        assert {epoch[id(example)] for epoch in observed} == set(range(12))


def test_distillation_targets_require_every_configured_metric():
    metrics = {
        "validation_policy_kl": 1.4,
        "validation_q_mse": 0.25,
        "validation_top1_agreement": 0.3,
    }

    assert distill_module._validation_targets_reached(
        metrics,
        target_policy_kl=1.5,
        target_q_mse=0.3,
        target_top1=0.25,
    )
    assert not distill_module._validation_targets_reached(
        metrics,
        target_policy_kl=1.3,
        target_q_mse=0.3,
        target_top1=0.25,
    )
    assert not distill_module._validation_targets_reached(
        metrics,
        target_policy_kl=None,
        target_q_mse=None,
        target_top1=None,
    )


@pytest.mark.parametrize(
    (
        "restore_best_fit",
        "selected_epoch",
        "selected_weight_index",
        "validation_policy_kl",
        "fitted_train_policy_kl",
    ),
    [
        (False, 4.0, 6, 0.81, 0.3),
        (True, 2.0, 2, 0.8, 0.4),
    ],
)
def test_distillation_plateau_selects_requested_epoch(
    monkeypatch,
    restore_best_fit,
    selected_epoch,
    selected_weight_index,
    validation_policy_kl,
    fitted_train_policy_kl,
):
    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward_batch(self, _batch):
            values = torch.stack((self.weight, -self.weight))
            return BatchOutput(
                policy_logits=values,
                q_values=values,
                legal_counts=torch.tensor([2]),
            )

    example = DistillationExample(
        object(),
        torch.tensor([1.0, -1.0]),
        torch.tensor([0.5, -0.5]),
    )
    examples = [example]
    monkeypatch.setattr(
        distill_module,
        "_iter_student_batches",
        lambda *_args, **_kwargs: iter([(object(), examples, True)]),
    )
    monkeypatch.setattr(
        distill_module,
        "move_batch_to_device",
        lambda batch, _device: batch,
    )
    validation_sequence = iter(
        [
            (1.0, 1.0, 0.1),
            (0.5, 0.5, 0.3),
            (0.8, 0.9, 0.2),
            (0.4, 0.4, 0.4),
            (0.805, 0.9, 0.21),
            (0.35, 0.35, 0.45),
            (0.81, 0.9, 0.22),
            (0.3, 0.3, 0.5),
        ]
    )
    evaluated_weights = []

    def fake_evaluate(student, *_args, **_kwargs):
        evaluated_weights.append(student.weight.detach().clone())
        policy_kl, q_mse, top1 = next(validation_sequence)
        return {
            "validation_positions": 1.0,
            "validation_policy_kl": policy_kl,
            "validation_q_mse": q_mse,
            "validation_top1_agreement": top1,
        }

    monkeypatch.setattr(
        distill_module,
        "evaluate_distillation",
        fake_evaluate,
    )
    config = _tiny_hex_config()
    config.training.weight_decay = 0.0
    model = TinyStudent()

    metrics, _optimizer = distill_module.train_distillation(
        model,
        examples,
        examples,
        config=config,
        device=torch.device("cpu"),
        precision="float32",
        epochs=10,
        batch_size=1,
        temperature=1.0,
        policy_weight=1.0,
        q_weight=1.0,
        learning_rate=0.05,
        seed=0,
        early_stop_patience=2,
        early_stop_min_delta=0.01,
        restore_best_fit=restore_best_fit,
        training_probe_positions=1,
        epoch_callback=lambda epoch, _model, _metrics: {
            "strength_eval_epoch": float(epoch)
        },
    )

    assert metrics["epochs_completed"] == 4.0
    assert metrics["best_epoch"] == 2.0
    assert metrics["selected_epoch"] == selected_epoch
    assert metrics["restored_best_fit"] == float(restore_best_fit)
    assert metrics["strength_eval_epoch"] == selected_epoch
    assert metrics["total_optimizer_steps"] == 4.0
    assert metrics["stopped_on_plateau"] == 1.0
    assert metrics["validation_policy_kl"] == validation_policy_kl
    assert metrics["fitted_train_policy_kl"] == fitted_train_policy_kl
    assert metrics["policy_kl_generalization_gap"] == pytest.approx(
        validation_policy_kl - fitted_train_policy_kl
    )
    torch.testing.assert_close(model.weight, evaluated_weights[selected_weight_index])


def test_distilled_checkpoint_strictly_resumes_with_fresh_optimizer(tmp_path):
    config = _tiny_hex_config(tmp_path)
    model = make_klent_net(config.model)
    assert isinstance(model, HexAxialCNNKlentNet)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"teacher": True}, teacher_path)
    checkpoint = Path(config.run.output_dir) / "checkpoints" / "checkpoint_000000.pt"

    _save_distilled_checkpoint(
        model,
        config=config,
        output_path=checkpoint,
        teacher_path=teacher_path,
        teacher_iteration=244,
        student_path=None,
        metrics={"validation_policy_kl": 0.1},
    )
    trainer = Trainer(config, tensorboard=False, resume=checkpoint)
    try:
        assert trainer.iteration == 0
        assert isinstance(trainer.model, HexAxialCNNKlentNet)
        assert trainer.initial_checkpoint["graft"] == "distilled_hex_axial_cnn"
        assert trainer.optimizer.state_dict()["state"] == {}
        for expected, actual in zip(
            model.state_dict().values(),
            trainer.model.state_dict().values(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
    finally:
        trainer.close()


def test_distillation_can_strictly_continue_student_weights(tmp_path):
    config = _tiny_hex_config(tmp_path)
    source = make_klent_net(config.model)
    assert isinstance(source, HexAxialCNNKlentNet)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0.0, 0.2)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"teacher": True}, teacher_path)
    checkpoint = tmp_path / "source" / "checkpoints" / "checkpoint_000000.pt"
    _save_distilled_checkpoint(
        source,
        config=config,
        output_path=checkpoint,
        teacher_path=teacher_path,
        teacher_iteration=220,
        student_path=None,
        metrics={},
    )
    target = make_klent_net(config.model)
    assert isinstance(target, HexAxialCNNKlentNet)

    _initialize_student_weights(
        target,
        config=config,
        checkpoint=checkpoint,
        device=torch.device("cpu"),
    )

    for expected, actual in zip(
        source.state_dict().values(),
        target.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_distillation_can_project_dilated_student_into_exact_d6(tmp_path):
    source_config = _tiny_hex_config(tmp_path)
    source_config.model.architecture = "hex_dilated_cnn"
    source_config.model.axial_attention_layers = []
    source_config.model.cnn_dilations = [1, 2]
    source = make_klent_net(source_config.model)
    assert isinstance(source, HexDilatedCNNKlentNet)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0.0, 0.2)
    teacher_path = tmp_path / "teacher-d6.pt"
    torch.save({"teacher": True}, teacher_path)
    checkpoint = tmp_path / "source-d6" / "checkpoints" / "checkpoint_000000.pt"
    _save_distilled_checkpoint(
        source,
        config=source_config,
        output_path=checkpoint,
        teacher_path=teacher_path,
        teacher_iteration=220,
        student_path=None,
        metrics={},
    )

    target_config = copy.deepcopy(source_config)
    target_config.model.architecture = "hex_d6_dilated_cnn"
    target = make_klent_net(target_config.model)
    assert isinstance(target, HexD6DilatedCNNKlentNet)
    _initialize_student_weights(
        target,
        config=target_config,
        checkpoint=checkpoint,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        target.policy_head[0].weight,
        source.policy_head[0].weight,
    )
    assert torch.count_nonzero(
        target.backbone.blocks[0].axis_conv.main_neighbor
    ) > 0


def test_distillation_can_graft_exact_d6_student_into_deeper_model(tmp_path):
    source_config = _tiny_hex_config(tmp_path)
    source_config.model.architecture = "hex_d6_dilated_cnn"
    source_config.model.axial_attention_layers = []
    source_config.model.cnn_dilations = [1, 2]
    source = make_klent_net(source_config.model)
    assert isinstance(source, HexD6DilatedCNNKlentNet)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0.0, 0.2)
    teacher_path = tmp_path / "teacher-depth-graft.pt"
    torch.save({"teacher": True}, teacher_path)
    checkpoint = tmp_path / "source-depth-graft" / "checkpoint_000000.pt"
    _save_distilled_checkpoint(
        source,
        config=source_config,
        output_path=checkpoint,
        teacher_path=teacher_path,
        teacher_iteration=220,
        student_path=None,
        metrics={},
    )

    target_config = copy.deepcopy(source_config)
    target_config.model.num_layers = 4
    target_config.model.cnn_dilations = [1, 2, 4, 8]
    target = make_klent_net(target_config.model)
    assert isinstance(target, HexD6DilatedCNNKlentNet)
    _initialize_student_weights(
        target,
        config=target_config,
        checkpoint=checkpoint,
        device=torch.device("cpu"),
    )

    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, expected in source_state.items():
        torch.testing.assert_close(target_state[name], expected)
    assert all(
        torch.count_nonzero(block.layer_scale) == 0
        for block in target.backbone.blocks[2:]
    )
