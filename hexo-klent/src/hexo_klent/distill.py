"""Distill a KLENT graph teacher into a dense hex-axial CNN checkpoint."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from hexo_klent.actor import (
    _autocast,
    collect_games_parallel,
    flatten_trajectories,
)
from hexo_klent.batching import (
    move_batch_to_device,
    prepare_graph_batches,
    raster_shape,
    raster_shape_from_coords,
)
from hexo_klent.config import Config
from hexo_klent.mcts_adapter import load_checkpoint
from hexo_klent.model import (
    BatchOutput,
    HexD6DilatedCNNKlentNet,
    HexDilatedCNNKlentNet,
    HexCNNKlentNet,
    convert_hex_dilated_to_d6,
    compile_klent_forward,
    graft_hex_d6_depth,
    make_klent_net,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistillationExample:
    """One state plus the graph teacher's complete legal-action outputs."""

    state: object
    policy_logits: Tensor
    q_values: Tensor


# The same dihedral action used by the production D6 graph augmentation.  The
# graph teacher is invariant to these transforms by construction; the planar
# CNN is not, so distillation must expose every orientation explicitly.
_D6_COORD_TRANSFORMS: tuple[
    Callable[[int, int], tuple[int, int]], ...
] = (
    lambda q, r: (q, r),
    lambda q, r: (-r, q + r),
    lambda q, r: (-q - r, q),
    lambda q, r: (-q, -r),
    lambda q, r: (r, -q - r),
    lambda q, r: (q + r, -q),
    lambda q, r: (r, q),
    lambda q, r: (-q, q + r),
    lambda q, r: (-q - r, r),
    lambda q, r: (-r, -q),
    lambda q, r: (q, -q - r),
    lambda q, r: (q + r, -r),
)

_ExampleRef = tuple[DistillationExample, int]


def _transform_coord(
    coord: tuple[int, int], transform_index: int
) -> tuple[int, int]:
    if not 0 <= transform_index < len(_D6_COORD_TRANSFORMS):
        raise ValueError(f"invalid D6 transform index {transform_index}")
    return _D6_COORD_TRANSFORMS[transform_index](*coord)


def _transformed_raster_shape(
    state: object,
    transform_index: int,
) -> tuple[int, int]:
    if transform_index == 0:
        return raster_shape(state)
    coords = [
        _transform_coord(coord, transform_index)
        for coord, _player in state.placed_stones()
    ]
    coords.extend(
        _transform_coord(coord, transform_index)
        for coord in state.legal_moves()
    )
    return raster_shape_from_coords(coords)


def _transform_distillation_example(
    example: DistillationExample,
    transform_index: int,
) -> DistillationExample:
    """Transform a state and permute its teacher labels into legal-move order."""

    if transform_index == 0:
        return example

    import hexo_rs

    state = example.state
    current_player = state.current_player()
    if current_player is None:
        raise ValueError("cannot D6-transform a terminal distillation state")
    transformed_stones = [
        (_transform_coord(coord, transform_index), player)
        for coord, player in state.placed_stones()
    ]
    transformed_state = hexo_rs.GameState.from_state(
        transformed_stones,
        current_player,
        state.moves_remaining_this_turn(),
        state.config(),
    )
    source_by_coord = {
        _transform_coord(coord, transform_index): source_index
        for source_index, coord in enumerate(state.legal_moves())
    }
    transformed_legal = transformed_state.legal_moves()
    if len(source_by_coord) != len(transformed_legal) or any(
        coord not in source_by_coord for coord in transformed_legal
    ):
        raise RuntimeError("D6 transform changed the legal-action set")
    permutation = torch.tensor(
        [source_by_coord[coord] for coord in transformed_legal],
        dtype=torch.long,
    )
    return DistillationExample(
        state=transformed_state,
        policy_logits=example.policy_logits.index_select(0, permutation),
        q_values=example.q_values.index_select(0, permutation),
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    """Commit one monitorable JSON snapshot without partial writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_student_weights(
    model: HexCNNKlentNet,
    *,
    config: Config,
    checkpoint: Path,
    device: torch.device,
) -> None:
    """Strictly restore a representation-compatible distilled CNN."""

    initialized = load_checkpoint(checkpoint, device)
    if not isinstance(initialized.model.network, HexCNNKlentNet):
        raise TypeError("initial student checkpoint is not a native hex CNN")
    expected_model_config = dataclasses.asdict(config.model)
    actual_model_config = dataclasses.asdict(initialized.model_config)
    convert_to_d6 = isinstance(model, HexD6DilatedCNNKlentNet) and isinstance(
        initialized.model.network,
        HexDilatedCNNKlentNet,
    )
    graft_d6_depth = (
        isinstance(model, HexD6DilatedCNNKlentNet)
        and isinstance(initialized.model.network, HexD6DilatedCNNKlentNet)
        and len(model.backbone.blocks)
        > len(initialized.model.network.backbone.blocks)
    )
    comparable_model_config = dict(actual_model_config)
    if convert_to_d6:
        comparable_model_config["architecture"] = "hex_d6_dilated_cnn"
    if graft_d6_depth:
        comparable_model_config["num_layers"] = expected_model_config["num_layers"]
        comparable_model_config["cnn_dilations"] = expected_model_config[
            "cnn_dilations"
        ]
    if comparable_model_config != expected_model_config:
        raise ValueError(
            "initial student model configuration is not representation-"
            "compatible with the distillation target"
        )
    if convert_to_d6:
        report = convert_hex_dilated_to_d6(
            initialized.model.network,
            model,
        )
        logger.info(
            "projected hex_dilated_cnn onto exact D6 orbits: "
            "blocks=%d copied_tensors=%d parameters=%d->%d",
            report.projected_blocks,
            report.copied_tensors,
            report.source_parameters,
            report.target_parameters,
        )
        return
    if graft_d6_depth:
        report = graft_hex_d6_depth(
            initialized.model.network,
            model,
        )
        logger.info(
            "grafted exact-D6 CNN depth with identity residual blocks: "
            "blocks=%d->%d copied_tensors=%d parameters=%d->%d",
            report.source_blocks,
            report.target_blocks,
            report.copied_tensors,
            report.source_parameters,
            report.target_parameters,
        )
        return
    model.load_state_dict(
        initialized.model.network.state_dict(),
        strict=True,
    )


def _split_output(output: BatchOutput) -> list[tuple[Tensor, Tensor]]:
    counts = [int(value) for value in output.legal_counts.detach().cpu()]
    return list(
        zip(
            output.policy_logits.split(counts),
            output.q_values.split(counts),
            strict=True,
        )
    )


def _segment_log_softmax(values: Tensor, counts: Tensor) -> Tensor:
    """Log-softmax each contiguous variable-length state segment."""

    maxima = torch.segment_reduce(
        values,
        "max",
        lengths=counts,
    ).detach()
    repeated_maxima = torch.repeat_interleave(
        maxima,
        counts,
        output_size=values.numel(),
    )
    centered = values - repeated_maxima
    normalizers = torch.segment_reduce(
        centered.exp(),
        "sum",
        lengths=counts,
    )
    return centered - torch.repeat_interleave(
        normalizers.log(),
        counts,
        output_size=values.numel(),
    )


def label_teacher_outputs(
    states: list[object],
    teacher,
    *,
    teacher_model_config,
    device: torch.device,
    precision: str,
    edge_budget: int,
    batch_size: int = 256,
) -> list[DistillationExample]:
    """Evaluate every legal action once and retain compact CPU labels."""

    if batch_size <= 0:
        raise ValueError("teacher labelling batch size must be positive")
    examples: list[DistillationExample | None] = [None] * len(states)
    was_training = teacher.training
    teacher.eval()
    try:
        # ``inference_mode`` would mark the retained CPU labels as inference
        # tensors, which autograd then rejects when they participate in the
        # student's loss. ``no_grad`` avoids teacher graphs while producing
        # ordinary immutable label tensors.
        with torch.no_grad():
            # Graph collation is intentionally bounded before entering the
            # batch builder. Passing the complete distillation corpus here can
            # materialize tens of GiB of ragged nodes/edges even when the GPU
            # edge budget later splits it into small device fragments.
            for chunk_start in range(0, len(states), batch_size):
                state_chunk = states[chunk_start : chunk_start + batch_size]
                for batch, state_slice in prepare_graph_batches(
                    state_chunk,
                    model_config=teacher_model_config,
                    edge_budget=edge_budget,
                ):
                    batch = move_batch_to_device(batch, device)
                    with _autocast(device, precision):
                        output = teacher.forward_batch(batch)
                    chunks = _split_output(output)
                    selected_states = state_chunk[state_slice]
                    if len(chunks) != len(selected_states):
                        raise RuntimeError(
                            "teacher output count does not match input states"
                        )
                    slice_start = 0 if state_slice.start is None else state_slice.start
                    for offset, (state, (logits, q_values)) in enumerate(
                        zip(selected_states, chunks, strict=True)
                    ):
                        legal_count = int(state.legal_move_count())
                        if logits.numel() != legal_count:
                            raise RuntimeError(
                                "teacher legal-action ordering/count disagrees "
                                f"with state: {logits.numel()} != {legal_count}"
                            )
                        examples[
                            chunk_start + slice_start + offset
                        ] = DistillationExample(
                            state=state,
                            policy_logits=logits.detach().float().cpu(),
                            q_values=q_values.detach().float().cpu(),
                        )
    finally:
        teacher.train(was_training)
    if any(example is None for example in examples):
        raise RuntimeError("teacher labelling left an incomplete state range")
    return list(examples)


def distillation_losses(
    output: BatchOutput,
    examples: list[DistillationExample],
    *,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return state-balanced policy KL, full-action Q MSE, and top-1 rate."""

    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    if output.legal_counts.numel() != len(examples):
        raise RuntimeError("student output count does not match examples")
    teacher_count_values = [
        example.policy_logits.numel() for example in examples
    ]
    teacher_counts = torch.tensor(
        teacher_count_values,
        dtype=torch.long,
        device=output.policy_logits.device,
    )
    total_teacher_actions = sum(teacher_count_values)
    if (
        total_teacher_actions != output.policy_logits.numel()
        or total_teacher_actions != output.q_values.numel()
    ):
        raise RuntimeError(
            "student and teacher legal-action totals differ: "
            f"{output.policy_logits.numel()} != {total_teacher_actions}"
        )
    teacher_logits = torch.cat(
        [example.policy_logits for example in examples]
    ).to(
        device=output.policy_logits.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    teacher_q = torch.cat([example.q_values for example in examples]).to(
        device=output.q_values.device,
        dtype=torch.float32,
        non_blocking=True,
    )
    student_logits = output.policy_logits.float()
    student_q = output.q_values.float()

    teacher_log_policy = _segment_log_softmax(
        teacher_logits / temperature,
        teacher_counts,
    )
    student_log_policy = _segment_log_softmax(
        student_logits / temperature,
        teacher_counts,
    )
    teacher_policy = teacher_log_policy.exp()
    per_action_kl = teacher_policy * (
        teacher_log_policy - student_log_policy
    )
    # The T^2 factor preserves gradient scale across temperatures. Reduce
    # actions within each state before averaging so states with more legal
    # moves do not receive more weight.
    policy_loss = torch.segment_reduce(
        per_action_kl,
        "sum",
        lengths=teacher_counts,
    ).mean() * temperature**2
    q_loss = torch.segment_reduce(
        (student_q - teacher_q).square(),
        "mean",
        lengths=teacher_counts,
    ).mean()

    repeated_student_max = torch.repeat_interleave(
        torch.segment_reduce(
            student_logits.detach(),
            "max",
            lengths=teacher_counts,
        ),
        teacher_counts,
        output_size=student_logits.numel(),
    )
    repeated_teacher_max = torch.repeat_interleave(
        torch.segment_reduce(
            teacher_logits,
            "max",
            lengths=teacher_counts,
        ),
        teacher_counts,
        output_size=teacher_logits.numel(),
    )
    shared_top = (
        (student_logits.detach() == repeated_student_max)
        & (teacher_logits == repeated_teacher_max)
    ).to(torch.float32)
    top1 = torch.segment_reduce(
        shared_top,
        "max",
        lengths=teacher_counts,
    ).mean()
    return (
        policy_loss,
        q_loss,
        top1,
    )


def _example_groups(
    examples: list[DistillationExample],
    *,
    seed: int,
    augment_symmetries: bool = False,
) -> list[list[_ExampleRef]]:
    groups: dict[tuple[int, int], list[_ExampleRef]] = defaultdict(list)
    for source_index, example in enumerate(examples):
        # Advancing ``seed`` by one each epoch makes every example visit every
        # D6 orientation exactly once in any consecutive twelve epochs.  This
        # is balanced and more reproducible than independent random sampling.
        transform_index = (
            (source_index + seed) % len(_D6_COORD_TRANSFORMS)
            if augment_symmetries
            else 0
        )
        shape = _transformed_raster_shape(example.state, transform_index)
        groups[shape].append((example, transform_index))
    generator = random.Random(seed)
    values = list(groups.values())
    for group in values:
        generator.shuffle(group)
    generator.shuffle(values)
    return values


def _iter_student_batches(
    examples: list[DistillationExample],
    *,
    model_config,
    batch_size: int,
    cell_budget: int,
    seed: int,
    augment_symmetries: bool = False,
):
    groups = _example_groups(
        examples,
        seed=seed,
        augment_symmetries=augment_symmetries,
    )
    logical_batch_examples = 0
    for group_index, group in enumerate(groups):
        start = 0
        while start < len(group):
            # Raster shapes cannot share one dense tensor, but they can share
            # one optimizer update. Fill the remaining logical batch capacity
            # before crossing into the next shape so a six-position overflow
            # crop does not receive the same AdamW weight as 2,048 ordinary
            # positions.
            remaining = batch_size - logical_batch_examples
            end = min(start + remaining, len(group))
            outer_refs = group[start:end]
            # Materialize at most one logical batch of transformed states at a
            # time.  Holding all augmented GameStates would needlessly inflate
            # host/unified memory on the APU.
            outer = [
                _transform_distillation_example(example, transform_index)
                for example, transform_index in outer_refs
            ]
            states = [example.state for example in outer]
            prepared = prepare_graph_batches(
                states,
                model_config=model_config,
                edge_budget=cell_budget,
            )
            logical_batch_examples += len(outer)
            final_examples = (
                group_index == len(groups) - 1 and end == len(group)
            )
            optimizer_boundary = (
                logical_batch_examples == batch_size or final_examples
            )
            for index, (batch, state_slice) in enumerate(prepared):
                yield (
                    batch,
                    outer[state_slice],
                    optimizer_boundary and index == len(prepared) - 1,
                )
            if optimizer_boundary:
                logical_batch_examples = 0
            start = end


def evaluate_distillation(
    model: HexCNNKlentNet,
    examples: list[DistillationExample],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    cell_budget: int,
    temperature: float,
) -> dict[str, float]:
    """Measure the held-out teacher/student agreement."""

    totals = torch.zeros(4, dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch, selected, _optimizer_boundary in _iter_student_batches(
                examples,
                model_config=model_config,
                batch_size=batch_size,
                cell_budget=cell_budget,
                seed=0,
            ):
                batch = move_batch_to_device(batch, device)
                with _autocast(device, precision):
                    output = model.forward_batch(batch)
                    policy_loss, q_loss, top1 = distillation_losses(
                        output,
                        selected,
                        temperature=temperature,
                    )
                count = len(selected)
                totals += torch.tensor(
                    [
                        float(policy_loss),
                        float(q_loss),
                        float(top1),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=device,
                ) * count
    finally:
        model.train(was_training)
    policy_kl, q_mse, top1, count = totals.cpu().tolist()
    denominator = max(count, 1.0)
    return {
        "validation_positions": count,
        "validation_policy_kl": policy_kl / denominator,
        "validation_q_mse": q_mse / denominator,
        "validation_top1_agreement": top1 / denominator,
    }


def _validation_objective(
    metrics: dict[str, float],
    *,
    policy_weight: float,
    q_weight: float,
) -> float:
    """Return the held-out counterpart of the optimized training loss."""

    return (
        policy_weight * metrics["validation_policy_kl"]
        + q_weight * metrics["validation_q_mse"]
    )


def _validation_targets_reached(
    metrics: dict[str, float],
    *,
    target_policy_kl: float | None,
    target_q_mse: float | None,
    target_top1: float | None,
) -> bool:
    """Require every explicitly configured held-out target to be satisfied."""

    checks = []
    if target_policy_kl is not None:
        checks.append(metrics["validation_policy_kl"] <= target_policy_kl)
    if target_q_mse is not None:
        checks.append(metrics["validation_q_mse"] <= target_q_mse)
    if target_top1 is not None:
        checks.append(metrics["validation_top1_agreement"] >= target_top1)
    return bool(checks) and all(checks)


def _validate_stopping_parameters(
    *,
    early_stop_patience: int,
    early_stop_min_delta: float,
    target_policy_kl: float | None,
    target_q_mse: float | None,
    target_top1: float | None,
) -> None:
    if early_stop_patience < 0:
        raise ValueError("distillation early-stop patience cannot be negative")
    if early_stop_min_delta < 0.0:
        raise ValueError("distillation early-stop minimum delta cannot be negative")
    if target_policy_kl is not None and target_policy_kl < 0.0:
        raise ValueError("distillation target policy KL cannot be negative")
    if target_q_mse is not None and target_q_mse < 0.0:
        raise ValueError("distillation target Q MSE cannot be negative")
    if target_top1 is not None and not 0.0 <= target_top1 <= 1.0:
        raise ValueError("distillation target top-1 must be between zero and one")


def train_distillation(
    model: HexCNNKlentNet,
    train_examples: list[DistillationExample],
    validation_examples: list[DistillationExample],
    *,
    config: Config,
    device: torch.device,
    precision: str,
    epochs: int,
    batch_size: int,
    temperature: float,
    policy_weight: float,
    q_weight: float,
    learning_rate: float,
    seed: int,
    augment_symmetries: bool = False,
    progress_path: Path | None = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    restore_best_fit: bool = False,
    target_policy_kl: float | None = None,
    target_q_mse: float | None = None,
    target_top1: float | None = None,
    training_probe_positions: int = 0,
    epoch_callback: Callable[
        [int, HexCNNKlentNet, dict[str, float]], dict[str, float]
    ]
    | None = None,
    history_path: Path | None = None,
) -> tuple[dict[str, float], torch.optim.Optimizer]:
    """Fit the CNN to frozen graph outputs without KLENT return targets."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("distillation epochs and batch size must be positive")
    if policy_weight < 0.0 or q_weight < 0.0 or policy_weight + q_weight <= 0.0:
        raise ValueError("distillation loss weights must be non-negative and non-zero")
    if learning_rate <= 0.0:
        raise ValueError("distillation learning rate must be positive")
    if training_probe_positions < 0:
        raise ValueError("distillation training probe cannot be negative")
    _validate_stopping_parameters(
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        target_policy_kl=target_policy_kl,
        target_q_mse=target_q_mse,
        target_top1=target_top1,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    started_at = time.monotonic()
    training_probe = train_examples[:training_probe_positions]
    model.train()
    last_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    best_state: dict[str, Tensor] | None = None
    best_objective = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    target_reached = False
    stopped_on_plateau = False
    total_optimizer_steps = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        optimizer_steps = 0
        outer_batch_examples = 0
        epoch_positions = 0
        epoch_started_at = time.monotonic()
        last_progress_at = epoch_started_at
        for batch, selected, optimizer_boundary in _iter_student_batches(
            train_examples,
            model_config=config.model,
            batch_size=batch_size,
            cell_budget=config.training.edge_budget,
            seed=seed + epoch,
            augment_symmetries=augment_symmetries,
        ):
            batch = move_batch_to_device(batch, device)
            with _autocast(device, precision):
                output = model.forward_batch(batch)
                policy_loss, q_loss, top1 = distillation_losses(
                    output,
                    selected,
                    temperature=temperature,
                )
                total_loss = policy_weight * policy_loss + q_weight * q_loss
            count = len(selected)
            epoch_positions += count
            # One logical optimizer update may span raster shapes. Cell-budget
            # splits only bound memory; normalize their gradients by the
            # complete logical-batch population before stepping.
            outer_batch_examples += count
            (total_loss * count).backward()
            totals += torch.tensor(
                [
                    float(policy_loss.detach()),
                    float(q_loss.detach()),
                    float(top1.detach()),
                    1.0,
                ],
                dtype=torch.float64,
                device=device,
            ) * count
            if optimizer_boundary:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(outer_batch_examples)
                if config.training.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.training.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                outer_batch_examples = 0
                now = time.monotonic()
                if now - last_progress_at >= 30.0:
                    elapsed = max(now - epoch_started_at, 1e-9)
                    logger.info(
                        "distill epoch %d/%d progress=%d/%d (%.1f%%) "
                        "steps=%d %.0f pos/s",
                        epoch + 1,
                        epochs,
                        epoch_positions,
                        len(train_examples),
                        100.0 * epoch_positions / max(len(train_examples), 1),
                        optimizer_steps,
                        epoch_positions / elapsed,
                    )
                    last_progress_at = now
        if outer_batch_examples:
            raise RuntimeError("distillation batch iterator missed update boundary")
        total_optimizer_steps += optimizer_steps
        policy_kl, q_mse, top1, count = totals.cpu().tolist()
        denominator = max(count, 1.0)
        last_metrics = {
            "epoch": float(epoch + 1),
            "train_positions": count,
            "train_policy_kl": policy_kl / denominator,
            "train_q_mse": q_mse / denominator,
            "train_top1_agreement": top1 / denominator,
            "optimizer_steps": float(optimizer_steps),
        }
        logger.info(
            "distill epoch %d/%d positions=%d policy_kl=%.5f "
            "q_mse=%.5f top1=%.1f%% steps=%d",
            epoch + 1,
            epochs,
            int(count),
            last_metrics["train_policy_kl"],
            last_metrics["train_q_mse"],
            100.0 * last_metrics["train_top1_agreement"],
            optimizer_steps,
        )
        last_metrics.update(
            evaluate_distillation(
                model,
                validation_examples,
                model_config=config.model,
                device=device,
                precision=precision,
                batch_size=batch_size,
                cell_budget=config.training.edge_budget,
                temperature=temperature,
            )
        )
        if training_probe:
            fitted_train = evaluate_distillation(
                model,
                training_probe,
                model_config=config.model,
                device=device,
                precision=precision,
                batch_size=batch_size,
                cell_budget=config.training.edge_budget,
                temperature=temperature,
            )
            last_metrics.update(
                {
                    "fitted_train_positions": fitted_train[
                        "validation_positions"
                    ],
                    "fitted_train_policy_kl": fitted_train[
                        "validation_policy_kl"
                    ],
                    "fitted_train_q_mse": fitted_train["validation_q_mse"],
                    "fitted_train_top1_agreement": fitted_train[
                        "validation_top1_agreement"
                    ],
                }
            )
            last_metrics["policy_kl_generalization_gap"] = (
                last_metrics["validation_policy_kl"]
                - last_metrics["fitted_train_policy_kl"]
            )
            last_metrics["q_mse_generalization_gap"] = (
                last_metrics["validation_q_mse"]
                - last_metrics["fitted_train_q_mse"]
            )
            last_metrics["top1_generalization_gap"] = (
                last_metrics["fitted_train_top1_agreement"]
                - last_metrics["validation_top1_agreement"]
            )
        validation_objective = _validation_objective(
            last_metrics,
            policy_weight=policy_weight,
            q_weight=q_weight,
        )
        last_metrics["validation_objective"] = validation_objective
        last_metrics["total_optimizer_steps"] = float(total_optimizer_steps)
        last_metrics["elapsed_seconds"] = time.monotonic() - started_at
        epochs_completed = epoch + 1
        if epoch_callback is not None:
            callback_metrics = epoch_callback(
                epochs_completed,
                model,
                dict(last_metrics),
            )
            last_metrics.update(callback_metrics)
        if validation_objective < best_objective - early_stop_min_delta:
            best_objective = validation_objective
            best_epoch = epochs_completed
            best_metrics = dict(last_metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        target_reached = _validation_targets_reached(
            last_metrics,
            target_policy_kl=target_policy_kl,
            target_q_mse=target_q_mse,
            target_top1=target_top1,
        )
        stopped_on_plateau = (
            early_stop_patience > 0
            and epochs_without_improvement >= early_stop_patience
        )
        logger.info(
            "distill validation epoch %d/%d objective=%.5f policy_kl=%.5f "
            "q_mse=%.5f top1=%.1f%% best=%d stale=%d",
            epoch + 1,
            epochs,
            validation_objective,
            last_metrics["validation_policy_kl"],
            last_metrics["validation_q_mse"],
            100.0 * last_metrics["validation_top1_agreement"],
            best_epoch,
            epochs_without_improvement,
        )
        if training_probe:
            logger.info(
                "distill fitted-train probe policy_kl=%.5f q_mse=%.5f "
                "top1=%.1f%% gaps=(%+.5f, %+.5f, %+.1fpp)",
                last_metrics["fitted_train_policy_kl"],
                last_metrics["fitted_train_q_mse"],
                100.0 * last_metrics["fitted_train_top1_agreement"],
                last_metrics["policy_kl_generalization_gap"],
                last_metrics["q_mse_generalization_gap"],
                100.0 * last_metrics["top1_generalization_gap"],
            )
        if progress_path is not None:
            _write_json_atomic(
                progress_path,
                {
                    "status": "fitting",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "best_epoch": best_epoch,
                    "best_validation_objective": best_objective,
                    "epochs_without_improvement": epochs_without_improvement,
                    "target_reached": target_reached,
                    "stopped_on_plateau": stopped_on_plateau,
                    "metrics": last_metrics,
                },
            )
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as history:
                history.write(
                    json.dumps(
                        {
                            "epoch": epochs_completed,
                            "best_epoch": best_epoch,
                            "epochs_without_improvement": (
                                epochs_without_improvement
                            ),
                            "target_reached": target_reached,
                            "stopped_on_plateau": stopped_on_plateau,
                            "metrics": last_metrics,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if target_reached:
            logger.info(
                "distillation validation target reached at epoch %d",
                epochs_completed,
            )
            break
        if stopped_on_plateau:
            logger.info(
                "distillation validation objective plateaued for %d epochs",
                early_stop_patience,
            )
            break
    if best_state is None:
        raise RuntimeError("distillation completed without validation metrics")
    if restore_best_fit and best_epoch != epochs_completed:
        model.load_state_dict(best_state, strict=True)
        logger.info("restored best distillation weights from epoch %d", best_epoch)
    selected_epoch = best_epoch if restore_best_fit else epochs_completed
    result_metrics = dict(best_metrics if restore_best_fit else last_metrics)
    best_elapsed_seconds = best_metrics.get("elapsed_seconds", 0.0)
    result_metrics.update(
        {
            "epochs_requested": float(epochs),
            "epochs_completed": float(epochs_completed),
            "best_epoch": float(best_epoch),
            "best_validation_objective": best_objective,
            "best_epoch_elapsed_seconds": best_elapsed_seconds,
            "selected_epoch": float(selected_epoch),
            "restored_best_fit": float(restore_best_fit),
            "elapsed_seconds": time.monotonic() - started_at,
            "total_optimizer_steps": float(total_optimizer_steps),
            "target_reached": float(target_reached),
            "stopped_on_plateau": float(stopped_on_plateau),
            "early_stop_patience": float(early_stop_patience),
            "early_stop_min_delta": early_stop_min_delta,
            "symmetry_augmentation": float(augment_symmetries),
        }
    )
    if target_policy_kl is not None:
        result_metrics["target_policy_kl"] = target_policy_kl
    if target_q_mse is not None:
        result_metrics["target_q_mse"] = target_q_mse
    if target_top1 is not None:
        result_metrics["target_top1"] = target_top1
    return result_metrics, optimizer


def _save_distilled_checkpoint(
    model: HexCNNKlentNet,
    *,
    config: Config,
    output_path: Path,
    teacher_path: Path,
    teacher_iteration: int | str,
    student_path: Path | None,
    metrics: dict[str, float],
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite distilled checkpoint {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # KLENT starts with fresh AdamW moments; the supervised distillation
    # optimizer is deliberately not carried into the on-policy run.
    fresh_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    provenance = {
        "graft": f"distilled_{config.model.architecture}",
        "checkpoint": str(teacher_path),
        "sha256": _checkpoint_sha256(teacher_path),
        "source_iteration": teacher_iteration,
        "metrics": metrics,
    }
    if student_path is not None:
        provenance.update(
            {
                "student_checkpoint": str(student_path),
                "student_sha256": _checkpoint_sha256(student_path),
            }
        )
    payload = {
        "format": "hexo-klent-v1",
        "iteration": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": fresh_optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "model_config": dataclasses.asdict(config.model),
        "initial_checkpoint": provenance,
        "checkpoint_history_dirs": [str(output_path.parent.resolve())],
        "distillation": provenance,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    metrics_path = output_path.parent.parent / "distillation.json"
    metrics_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def distill_checkpoint(
    config: Config,
    teacher_checkpoint: str | Path,
    *,
    positions: int,
    epochs: int,
    batch_size: int,
    validation_positions: int,
    device_str: str | None,
    precision: str | None,
    output: str | Path | None,
    temperature: float,
    policy_weight: float,
    q_weight: float,
    learning_rate: float,
    parallel_games: int | None,
    student_compile: bool | None,
    student_checkpoint: str | Path | None,
    seed: int | None,
    teacher_horizon: int | None = None,
    augment_symmetries: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    restore_best_fit: bool = False,
    strength_eval_interval: int = 0,
    strength_eval_games: int = 32,
    strength_eval_mcts_simulations: int = 24,
    strength_eval_mcts_actions: int = 8,
    target_policy_kl: float | None = None,
    target_q_mse: float | None = None,
    target_top1: float | None = None,
) -> Path:
    """Collect from a graph teacher, distill, and emit generation-zero state."""

    if config.model.architecture not in {
        "hex_axial_cnn",
        "hex_dilated_cnn",
        "hex_d6_dilated_cnn",
    }:
        raise ValueError(
            "distillation target must use a native hex CNN architecture"
        )
    if positions <= 0 or validation_positions < 0:
        raise ValueError("distillation positions must be positive")
    if teacher_horizon is not None and teacher_horizon <= 0:
        raise ValueError("distillation teacher horizon must be positive")
    _validate_stopping_parameters(
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        target_policy_kl=target_policy_kl,
        target_q_mse=target_q_mse,
        target_top1=target_top1,
    )
    if strength_eval_interval < 0:
        raise ValueError("distillation strength eval interval cannot be negative")
    if strength_eval_interval > 0:
        if strength_eval_games <= 0 or strength_eval_games % 2 != 0:
            raise ValueError(
                "distillation paired strength eval games must be positive and even"
            )
        if strength_eval_mcts_simulations < 0:
            raise ValueError(
                "distillation strength eval MCTS simulations cannot be negative"
            )
        if strength_eval_mcts_actions <= 0:
            raise ValueError(
                "distillation strength eval MCTS actions must be positive"
            )
    collection_parallel_games = (
        config.collection.parallel_games
        if parallel_games is None
        else parallel_games
    )
    if collection_parallel_games <= 0:
        raise ValueError("distillation parallel games must be positive")
    if collection_parallel_games > positions:
        raise ValueError(
            "distillation parallel games cannot exceed the position budget"
        )
    device = torch.device(device_str or config.run.device)
    precision = precision or config.run.precision
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable")
    output_path = (
        Path(output).expanduser()
        if output is not None
        else Path(config.run.output_dir) / "checkpoints" / "checkpoint_000000.pt"
    ).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite distilled checkpoint {output_path}"
        )
    progress_path = output_path.parent.parent / "distillation-progress.json"
    teacher_path = Path(teacher_checkpoint).expanduser().resolve()
    student_path = (
        None
        if student_checkpoint is None
        else Path(student_checkpoint).expanduser().resolve()
    )
    _write_json_atomic(
        progress_path,
        {
            "status": "collecting",
            "teacher_checkpoint": str(teacher_path),
            "student_checkpoint": (
                None if student_path is None else str(student_path)
            ),
            "positions_requested": positions,
            "teacher_horizon": teacher_horizon,
            "augment_symmetries": augment_symmetries,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "restore_best_fit": restore_best_fit,
            "strength_eval_interval": strength_eval_interval,
            "strength_eval_games": strength_eval_games,
            "strength_eval_mcts_simulations": (
                strength_eval_mcts_simulations
            ),
            "strength_eval_mcts_actions": strength_eval_mcts_actions,
            "target_policy_kl": target_policy_kl,
            "target_q_mse": target_q_mse,
            "target_top1": target_top1,
        },
    )
    loaded = load_checkpoint(teacher_path, device)
    teacher = loaded.model.network
    teacher_model_config = loaded.model_config
    teacher_iteration = loaded.iteration
    if getattr(teacher_model_config, "architecture", "graph") != "graph":
        logger.warning(
            "distillation teacher architecture is %s rather than graph",
            getattr(teacher_model_config, "architecture", "graph"),
        )
    if config.run.compile and device.type == "cuda":
        logger.info("enabling compiled graph-teacher inference")
        compile_klent_forward(teacher)
    collection_seed = config.run.seed if seed is None else seed
    teacher_game_config = (
        config.game
        if teacher_horizon is None
        else dataclasses.replace(
            config.game,
            rollout_horizon=teacher_horizon,
        )
    )
    logger.info(
        "collecting %d teacher positions with %d lanes from %s at iteration "
        "%s%s",
        positions,
        collection_parallel_games,
        teacher_path,
        teacher_iteration,
        (
            ""
            if teacher_horizon is None
            else f" (retaining {teacher_horizon}-placement prefixes)"
        ),
    )
    collection_started_at = time.monotonic()
    trajectories, stats = collect_games_parallel(
        teacher,
        model_config=teacher_model_config,
        game_config=teacher_game_config,
        algorithm=loaded.algorithm,
        positions=positions,
        parallel_games=collection_parallel_games,
        inference_batch_size=config.collection.inference_batch_size,
        inference_edge_budget=config.collection.inference_edge_budget,
        dense_position_cell_limit=0,
        board_radius=config.collection.board_radius,
        workers=config.collection.workers,
        batch_timeout_ms=config.collection.batch_timeout_ms,
        device=device,
        precision=precision,
        seed=collection_seed,
        retain_horizon_truncations=teacher_horizon is not None,
    )
    collection_elapsed = time.monotonic() - collection_started_at
    states = [step.state for step in flatten_trajectories(trajectories)]
    if not states:
        raise RuntimeError("teacher collection produced no usable positions")
    shape_counts = Counter(raster_shape(state) for state in states)
    logger.info(
        "student raster shapes: %s",
        ", ".join(
            f"{height}x{width}={count}"
            for (height, width), count in sorted(shape_counts.items())
        ),
    )
    # Sorting once by crop shape makes both labelling and student fitting
    # deterministic while retaining every selected teacher position.
    states.sort(key=raster_shape)
    logger.info(
        "labelling %d teacher positions (%d discarded safety positions)",
        len(states),
        stats.discarded_positions,
    )
    _write_json_atomic(
        progress_path,
        {
            "status": "labelling",
            "collected_positions": len(states),
            "discarded_positions": stats.discarded_positions,
            "teacher_horizon": teacher_horizon,
            "retained_horizon_trajectories": (
                stats.horizon_truncations if teacher_horizon is not None else 0
            ),
            "raster_shapes": {
                f"{height}x{width}": count
                for (height, width), count in sorted(shape_counts.items())
            },
        },
    )
    labelling_started_at = time.monotonic()
    examples = label_teacher_outputs(
        states,
        teacher,
        teacher_model_config=teacher_model_config,
        device=device,
        precision=precision,
        edge_budget=config.collection.inference_edge_budget,
        batch_size=config.collection.inference_batch_size,
    )
    labelling_elapsed = time.monotonic() - labelling_started_at
    logger.info(
        "teacher stages complete: collection=%.1fs labelling=%.1fs",
        collection_elapsed,
        labelling_elapsed,
    )
    # The complete CPU labels are now independent of the teacher. Release its
    # network and compiled allocator blocks before constructing the student;
    # retaining both models needlessly raises the unified-memory high-water
    # mark on this APU.
    del teacher
    del loaded
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    holdout = min(validation_positions, max(0, len(examples) - 1))
    generator = random.Random(collection_seed)
    generator.shuffle(examples)
    validation_examples = examples[:holdout]
    train_examples = examples[holdout:]
    if not validation_examples:
        validation_examples = train_examples[: min(len(train_examples), 128)]
    model = make_klent_net(config.model).to(device)
    if not isinstance(model, HexCNNKlentNet):
        raise TypeError("configured target did not construct a hex CNN")
    if student_path is not None:
        _initialize_student_weights(
            model,
            config=config,
            checkpoint=student_path,
            device=device,
        )
        logger.info("initialized student weights from %s", student_path)
    compile_student = (
        config.run.compile if student_compile is None else student_compile
    )
    if compile_student and device.type == "cuda":
        logger.info("enabling compiled hex-CNN distillation FIT")
        compile_klent_forward(model)
    elif config.run.compile and device.type == "cuda":
        logger.info(
            "using eager hex-CNN distillation FIT; saved KLENT config retains "
            "run.compile=true"
        )
    strength_cache = None
    epoch_checkpoints: dict[int, Path] = {}
    if student_path is not None:
        epoch_checkpoints[0] = student_path

    def evaluate_strength(
        epoch: int,
        candidate: HexCNNKlentNet,
        fit_metrics: dict[str, float],
    ) -> dict[str, float]:
        nonlocal strength_cache
        if strength_eval_interval <= 0 or epoch % strength_eval_interval != 0:
            return {}

        from hexo_klent.evaluation import (
            CheckpointOpponentCache,
            evaluate_vs_checkpoint,
        )

        if strength_cache is None:
            strength_cache = CheckpointOpponentCache(max_entries=2)
        eval_seed = (
            0 if collection_seed is None else collection_seed
        ) + epoch * 1_000_003
        opponents: list[tuple[str, Path, int | None]] = [
            ("teacher", teacher_path, None)
        ]
        lagged_epoch = epoch - strength_eval_interval
        lagged_path = epoch_checkpoints.get(lagged_epoch)
        if lagged_path is not None:
            opponents.append(("lagged", lagged_path, lagged_epoch))

        strength_metrics: dict[str, float] = {
            "strength_eval_epoch": float(epoch),
        }
        for opponent_index, (name, checkpoint, opponent_epoch) in enumerate(
            opponents
        ):
            logger.info(
                "distill strength epoch %d: %d games vs %s (%s)",
                epoch,
                strength_eval_games,
                name,
                checkpoint,
            )
            stats = evaluate_vs_checkpoint(
                candidate,
                model_config=config.model,
                game_config=config.game,
                games=strength_eval_games,
                checkpoint=str(checkpoint),
                device=device,
                precision=precision,
                seed=eval_seed + opponent_index * 500_009,
                algorithm=config.algorithm,
                mcts_simulations=strength_eval_mcts_simulations,
                mcts_actions=strength_eval_mcts_actions,
                checkpoint_cache=strength_cache,
                opponent_mcts_simulations=strength_eval_mcts_simulations,
                opponent_mcts_actions=strength_eval_mcts_actions,
                opening_plies=config.evaluation.opening_plies,
                opening_temperature=config.evaluation.opening_temperature,
                opening_generator=config.evaluation.opening_generator,
            )
            prefix = f"strength_{name}"
            strength_metrics.update(
                {
                    f"{prefix}_{key}": float(value)
                    for key, value in dataclasses.asdict(stats).items()
                }
            )
            if opponent_epoch is not None:
                strength_metrics[f"{prefix}_opponent_epoch"] = float(
                    opponent_epoch
                )
            logger.info(
                "distill strength epoch %d vs %s: W-L-T=%d-%d-%d "
                "win/decided=%.3f",
                epoch,
                name,
                stats.wins,
                stats.losses,
                stats.truncations,
                stats.win_rate_decided,
            )

        epoch_checkpoint = (
            output_path.parent.parent
            / "distillation_epochs"
            / f"epoch_{epoch:04d}"
            / "checkpoints"
            / "checkpoint_000000.pt"
        )
        checkpoint_metrics = dict(fit_metrics)
        checkpoint_metrics.update(strength_metrics)
        _save_distilled_checkpoint(
            candidate,
            config=config,
            output_path=epoch_checkpoint,
            teacher_path=teacher_path,
            teacher_iteration=teacher_iteration,
            student_path=student_path,
            metrics=checkpoint_metrics,
        )
        epoch_checkpoints[epoch] = epoch_checkpoint
        strength_metrics["strength_checkpoint_epoch"] = float(epoch)
        return strength_metrics

    try:
        metrics, _distill_optimizer = train_distillation(
            model,
            train_examples,
            validation_examples,
            config=config,
            device=device,
            precision=precision,
            epochs=epochs,
            batch_size=batch_size,
            temperature=temperature,
            policy_weight=policy_weight,
            q_weight=q_weight,
            learning_rate=learning_rate,
            seed=0 if collection_seed is None else collection_seed,
            augment_symmetries=augment_symmetries,
            progress_path=progress_path,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            restore_best_fit=restore_best_fit,
            target_policy_kl=target_policy_kl,
            target_q_mse=target_q_mse,
            target_top1=target_top1,
            training_probe_positions=validation_positions,
            epoch_callback=evaluate_strength,
            history_path=(
                output_path.parent.parent / "distillation-metrics.jsonl"
            ),
        )
    finally:
        if strength_cache is not None:
            strength_cache.clear()
    metrics.update(
        {
            "collected_games": float(stats.games),
            "collected_positions": float(len(states)),
            "collection_parallel_games": float(collection_parallel_games),
            "teacher_collection_horizon": float(
                teacher_game_config.rollout_horizon
            ),
            "retained_horizon_trajectories": float(
                stats.horizon_truncations if teacher_horizon is not None else 0
            ),
            "discarded_positions": float(stats.discarded_positions),
            "collection_elapsed_seconds": collection_elapsed,
            "teacher_labelling_elapsed_seconds": labelling_elapsed,
            "teacher_iteration": (
                float(teacher_iteration)
                if isinstance(teacher_iteration, int)
                else math.nan
            ),
            "symmetry_augmentation": float(augment_symmetries),
        }
    )
    _save_distilled_checkpoint(
        model,
        config=config,
        output_path=output_path,
        teacher_path=teacher_path,
        teacher_iteration=teacher_iteration,
        student_path=student_path,
        metrics=metrics,
    )
    _write_json_atomic(
        progress_path,
        {
            "status": "complete",
            "checkpoint": str(output_path),
            "metrics": metrics,
        },
    )
    logger.info("saved distilled KLENT checkpoint to %s", output_path)
    return output_path
