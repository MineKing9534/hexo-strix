"""Synchronous collect-then-fit KLENT training loop."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from torch.nn import functional as F

from hexo_klent.actor import (
    SharedInferenceActors,
    TrajectoryStep,
    _autocast,
    collect_games_parallel,
    flatten_trajectories,
    terminal_played_q_calibration,
)
from hexo_klent.batching import (
    move_batch_to_device,
    prepare_graph_batches,
    raster_shape,
)
from hexo_klent.config import Config, KlentModelConfig, TrainingConfig
from hexo_klent.evaluation import (
    CheckpointOpponentCache,
    evaluate_opponent,
)
from hexo_klent.model import (
    DenseAxisKlentNet,
    KlentNet,
    PersistentRayKlentNet,
    compile_klent_forward,
    is_dense_axis_config,
    load_dense_klent_graft,
    load_production_axis_weights,
    load_production_graph_weights,
    make_klent_net,
)
from hexo_klent.search_q_teacher import (
    FixedCheckpointSearchQTeacher,
    SearchQLabels,
)

if TYPE_CHECKING:
    from hexo_klent.tui import TrainingDashboard

logger = logging.getLogger(__name__)

_GIB = float(1024**3)
_CACHE_CLEAR_RESERVED_MB_ENV = "HEXO_CACHE_CLEAR_RESERVED_MB"
_CACHE_CLEAR_CHECK_EVERY_ENV = "HEXO_CACHE_CLEAR_CHECK_EVERY"
_DEFAULT_CACHE_CLEAR_RESERVED_MB = 8192.0
_DEFAULT_CACHE_CLEAR_CHECK_EVERY = 128
_MIN_COMPILER_NOFILE = 65_536
_BEST_SO_FAR_FORMAT = "hexo-klent-best-so-far-v1"


def _checkpoint_iteration_from_path(path: str | Path) -> int | None:
    """Return the KLENT generation encoded in a checkpoint filename."""

    match = re.fullmatch(r"checkpoint_(\d+)\.pt", Path(path).name)
    return int(match.group(1)) if match is not None else None


def _ensure_compiler_nofile_limit() -> tuple[int, int]:
    """Give Inductor/Triton enough descriptors for compile-time autotuning."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        target = max(soft, _MIN_COMPILER_NOFILE)
    else:
        target = min(hard, max(soft, _MIN_COMPILER_NOFILE))
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft = target
    logger.info("compiler_nofile soft=%d hard=%d", soft, hard)
    return soft, hard


def _learning_rate_for_iteration(
    training: TrainingConfig,
    iteration: int,
) -> float:
    """Return the generation-level linear warm-up learning rate."""

    if iteration <= 0:
        raise ValueError("iteration must be positive")
    relative_iteration = (
        iteration - training.learning_rate_warmup_start_iteration
    )
    # A delayed warm-up is intended for a stage resumed after the configured
    # iteration. If this config is inspected or run before that boundary, do
    # not unexpectedly throttle the preceding stage.
    if relative_iteration <= 0:
        return training.learning_rate
    warmup = training.learning_rate_warmup_iterations
    if warmup <= 1 or relative_iteration >= warmup:
        return training.learning_rate
    progress = (relative_iteration - 1) / (warmup - 1)
    factor = training.learning_rate_warmup_start_factor + progress * (
        1.0 - training.learning_rate_warmup_start_factor
    )
    return training.learning_rate * factor


class _CudaCachePressureGate:
    """Release inactive CUDA/ROCm blocks during a long fit under pressure."""

    def __init__(
        self,
        device: torch.device,
        *,
        threshold_bytes: int,
        check_every: int,
    ) -> None:
        self.device = device
        self.threshold_bytes = max(0, int(threshold_bytes))
        self.check_every = max(1, int(check_every))
        self.enabled = (
            device.type == "cuda" and self.threshold_bytes > 0
        )
        self.steps = 0
        self.checks = 0
        self.clears = 0
        self.released_bytes = 0
        self.max_reserved_bytes = 0

    @classmethod
    def from_environment(
        cls,
        device: torch.device,
    ) -> "_CudaCachePressureGate":
        try:
            threshold_mb = float(
                os.environ.get(
                    _CACHE_CLEAR_RESERVED_MB_ENV,
                    str(_DEFAULT_CACHE_CLEAR_RESERVED_MB),
                )
            )
        except ValueError:
            logger.warning(
                "invalid %s=%r; using %.0f MB",
                _CACHE_CLEAR_RESERVED_MB_ENV,
                os.environ.get(_CACHE_CLEAR_RESERVED_MB_ENV),
                _DEFAULT_CACHE_CLEAR_RESERVED_MB,
            )
            threshold_mb = _DEFAULT_CACHE_CLEAR_RESERVED_MB
        try:
            check_every = int(
                os.environ.get(
                    _CACHE_CLEAR_CHECK_EVERY_ENV,
                    str(_DEFAULT_CACHE_CLEAR_CHECK_EVERY),
                )
            )
        except ValueError:
            logger.warning(
                "invalid %s=%r; using %d",
                _CACHE_CLEAR_CHECK_EVERY_ENV,
                os.environ.get(_CACHE_CLEAR_CHECK_EVERY_ENV),
                _DEFAULT_CACHE_CLEAR_CHECK_EVERY,
            )
            check_every = _DEFAULT_CACHE_CLEAR_CHECK_EVERY
        if check_every <= 0:
            logger.warning(
                "%s must be positive; using %d",
                _CACHE_CLEAR_CHECK_EVERY_ENV,
                _DEFAULT_CACHE_CLEAR_CHECK_EVERY,
            )
            check_every = _DEFAULT_CACHE_CLEAR_CHECK_EVERY
        return cls(
            device,
            threshold_bytes=int(max(0.0, threshold_mb) * 1_000_000),
            check_every=check_every,
        )

    def step(self) -> bool:
        """Check reserve at the configured cadence and clear above threshold."""

        if not self.enabled:
            return False
        self.steps += 1
        if self.steps % self.check_every:
            return False

        self.checks += 1
        reserved_before = torch.cuda.memory_reserved(self.device)
        self.max_reserved_bytes = max(
            self.max_reserved_bytes,
            reserved_before,
        )
        if reserved_before <= self.threshold_bytes:
            return False

        allocated = torch.cuda.memory_allocated(self.device)
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved(self.device)
        released = max(0, reserved_before - reserved_after)
        self.clears += 1
        self.released_bytes += released
        logger.info(
            "cuda_cache pressure_clear=%d allocated=%.2fGiB "
            "reserved=%.2f->%.2fGiB released=%.2fGiB",
            self.clears,
            allocated / _GIB,
            reserved_before / _GIB,
            reserved_after / _GIB,
            released / _GIB,
        )
        return True

    def metrics(self) -> dict[str, float]:
        """Return stable training metrics on accelerator and CPU runs."""

        return {
            "allocator_pressure_checks": float(self.checks),
            "allocator_pressure_clears": float(self.clears),
            "allocator_pressure_released_gib": (
                self.released_bytes / _GIB
            ),
            "allocator_pressure_max_reserved_gib": (
                self.max_reserved_bytes / _GIB
            ),
            "allocator_pressure_threshold_gib": (
                self.threshold_bytes / _GIB
            ),
            "allocator_pressure_check_every": float(self.check_every),
        }


def _release_cuda_cache(
    device: torch.device,
    *,
    phase: str,
) -> dict[str, float]:
    """Release inactive allocator blocks and report phase memory pressure.

    ROCm exposes unified-memory allocations through the CUDA-compatible torch
    API. Variable dense crop shapes can leave large inactive blocks reserved
    by its caching allocator even though the tensors themselves are dead.
    Collection, fitting, and evaluation are synchronous phases, so their
    boundaries are safe points to return those blocks to the system.
    """

    if device.type != "cuda":
        return {}

    allocated = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    released = max(0, reserved_before - reserved_after)
    logger.info(
        "cuda_cache phase=%s allocated=%.2fGiB reserved=%.2f->%.2fGiB "
        "released=%.2fGiB peak_allocated=%.2fGiB peak_reserved=%.2fGiB",
        phase,
        allocated / _GIB,
        reserved_before / _GIB,
        reserved_after / _GIB,
        released / _GIB,
        peak_allocated / _GIB,
        peak_reserved / _GIB,
    )
    prefix = f"memory/{phase}"
    return {
        f"{prefix}_allocated_gib": allocated / _GIB,
        f"{prefix}_reserved_before_gib": reserved_before / _GIB,
        f"{prefix}_reserved_after_gib": reserved_after / _GIB,
        f"{prefix}_cache_released_gib": released / _GIB,
        f"{prefix}_peak_allocated_gib": peak_allocated / _GIB,
        f"{prefix}_peak_reserved_gib": peak_reserved / _GIB,
    }


def _segmented_log_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_lengths: torch.Tensor,
    max_segment_length: int,
) -> torch.Tensor:
    """Compute stable log-softmaxes over contiguous flat segments.

    ``scatter_reduce(..., reduce="amax")`` intermittently segfaults in the
    ROCm runtime on this workload, as does ``segment_reduce``.  Legal actions
    are already contiguous, so gather them into a compact padded matrix, use
    the mature row-wise log-softmax kernel, and gather the valid prefixes back.
    """

    if (
        logits.ndim != 1
        or segment_ids.ndim != 1
        or segment_lengths.ndim != 1
    ):
        raise ValueError(
            "logits, segment_ids, and segment_lengths must be one-dimensional"
        )
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have equal length")
    if segment_lengths.numel() <= 0:
        raise ValueError("segment_lengths must not be empty")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")

    num_segments = segment_lengths.numel()
    offsets = segment_lengths.cumsum(0) - segment_lengths
    columns = torch.arange(
        max_segment_length,
        dtype=segment_lengths.dtype,
        device=segment_lengths.device,
    )
    valid = columns.unsqueeze(0) < segment_lengths.unsqueeze(1)
    padded_indices = offsets.unsqueeze(1) + columns.unsqueeze(0)
    padded_logits = logits.index_select(
        0,
        padded_indices.clamp_max(logits.numel() - 1).reshape(-1),
    ).reshape(num_segments, max_segment_length)
    padded_log_policy = F.log_softmax(
        padded_logits.masked_fill(~valid, -torch.inf),
        dim=1,
    )
    flat_offsets = offsets.index_select(0, segment_ids)
    within_segment = torch.arange(
        logits.numel(),
        dtype=segment_ids.dtype,
        device=segment_ids.device,
    ) - flat_offsets
    return padded_log_policy[segment_ids, within_segment]


def _segmented_argmax(
    values: torch.Tensor,
    segment_lengths: torch.Tensor,
    max_segment_length: int,
) -> torch.Tensor:
    """Return relative argmax indices for contiguous flat segments."""

    if values.ndim != 1 or segment_lengths.ndim != 1:
        raise ValueError("values and segment_lengths must be one-dimensional")
    if segment_lengths.numel() <= 0:
        raise ValueError("segment_lengths must not be empty")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    if int(segment_lengths.sum().item()) != values.numel():
        raise ValueError("segment lengths must sum to the number of values")

    offsets = segment_lengths.cumsum(0) - segment_lengths
    columns = torch.arange(
        max_segment_length,
        dtype=segment_lengths.dtype,
        device=segment_lengths.device,
    )
    valid = columns.unsqueeze(0) < segment_lengths.unsqueeze(1)
    padded_indices = offsets.unsqueeze(1) + columns.unsqueeze(0)
    padded_values = values.index_select(
        0,
        padded_indices.clamp_max(values.numel() - 1).reshape(-1),
    ).reshape(segment_lengths.numel(), max_segment_length)
    return padded_values.masked_fill(~valid, -torch.inf).argmax(dim=1)


def _flat_training_targets(
    samples: list[TrajectoryStep],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """Pack variable-length policy and selected-action targets on CPU."""

    counts = torch.tensor(
        [sample.target_policy.numel() for sample in samples],
        dtype=torch.long,
    )
    if bool((counts <= 0).any()):
        raise ValueError("every KLENT sample must have a legal action")

    target_policy = torch.cat(
        [sample.target_policy.to(dtype=torch.float32) for sample in samples]
    )
    segment_ids = torch.repeat_interleave(
        torch.arange(len(samples), dtype=torch.long),
        counts,
        output_size=target_policy.numel(),
    )
    offsets = counts.cumsum(0) - counts
    chosen_chunks: list[torch.Tensor] = []
    target_q_chunks: list[torch.Tensor] = []
    for sample_index, sample in enumerate(samples):
        count = int(counts[sample_index].item())
        if not 0 <= sample.action_index < count:
            raise ValueError("KLENT sample action index is outside its policy")
        if sample.return_target is None:
            raise ValueError("KLENT sample has no return target")
        relative_indices = torch.tensor(
            [sample.action_index], dtype=torch.long
        )
        q_targets = torch.tensor(
            [sample.return_target], dtype=torch.float32
        )

        auxiliary_indices = sample.auxiliary_q_action_indices
        auxiliary_targets = sample.auxiliary_q_targets
        if (auxiliary_indices is None) != (auxiliary_targets is None):
            raise ValueError(
                "auxiliary Q action indices and targets must be paired"
            )
        if auxiliary_indices is not None and auxiliary_targets is not None:
            auxiliary_indices = auxiliary_indices.detach().to(
                device="cpu", dtype=torch.long
            ).reshape(-1)
            auxiliary_targets = auxiliary_targets.detach().to(
                device="cpu", dtype=torch.float32
            ).reshape(-1)
            if auxiliary_indices.numel() != auxiliary_targets.numel():
                raise ValueError(
                    "auxiliary Q action indices and targets must align"
                )
            if bool(
                ((auxiliary_indices < 0) | (auxiliary_indices >= count)).any()
            ):
                raise ValueError("auxiliary Q action index is outside its policy")
            if bool((auxiliary_indices == sample.action_index).any()):
                raise ValueError(
                    "played action must not be duplicated in auxiliary Q labels"
                )
            if auxiliary_indices.unique().numel() != auxiliary_indices.numel():
                raise ValueError("auxiliary Q action indices must be unique")
            if not bool(torch.isfinite(auxiliary_targets).all()):
                raise ValueError("auxiliary Q targets must be finite")
            relative_indices = torch.cat((relative_indices, auxiliary_indices))
            q_targets = torch.cat((q_targets, auxiliary_targets))

        chosen_chunks.append(offsets[sample_index] + relative_indices)
        target_q_chunks.append(q_targets)

    chosen = torch.cat(chosen_chunks)
    target_q = torch.cat(target_q_chunks)
    target_top1 = sum(
        int(sample.target_policy.argmax().item()) == sample.action_index
        for sample in samples
    )
    return target_policy, segment_ids, chosen, target_q, target_top1, counts


def _prepare_training_chunk(
    samples: list[TrajectoryStep],
    *,
    model_config,
    edge_budget: int,
):
    """Build, edge-pack, and collate one outer sample chunk on CPU."""

    if is_dense_axis_config(model_config):
        # Collection order interleaves games at different crop buckets. A
        # stable local sort turns those into useful dense batches without
        # changing the outer shuffle or any target/state correspondence.
        samples = sorted(samples, key=lambda sample: raster_shape(sample.state))

    prepared = []
    for batch, state_slice in prepare_graph_batches(
        [sample.state for sample in samples],
        model_config=model_config,
        edge_budget=edge_budget,
    ):
        packed_samples = samples[state_slice]
        prepared.append(
            (
                batch,
                packed_samples,
                _flat_training_targets(packed_samples),
            )
        )
    return prepared


def _prepared_training_batches(
    samples: list[TrajectoryStep],
    *,
    batch_size: int,
    model_config,
    edge_budget: int,
    prefetch: bool,
):
    """Yield edge-packed microbatches grouped by configured outer batch."""

    chunks = [
        samples[start : start + batch_size]
        for start in range(0, len(samples), batch_size)
    ]
    if not prefetch or len(chunks) <= 1:
        for chunk in chunks:
            yield _prepare_training_chunk(
                chunk,
                model_config=model_config,
                edge_budget=edge_budget,
            )
        return

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="hexo-klent-prefetch",
    ) as executor:
        future = executor.submit(
            _prepare_training_chunk,
            chunks[0],
            model_config=model_config,
            edge_budget=edge_budget,
        )
        for chunk in chunks[1:]:
            prepared = future.result()
            future = executor.submit(
                _prepare_training_chunk,
                chunk,
                model_config=model_config,
                edge_budget=edge_budget,
            )
            yield prepared
        yield future.result()


def _policy_diagnostic_slice(
    samples: list[TrajectoryStep],
    limit: int,
) -> list[TrajectoryStep]:
    """Select a deterministic collection-wide slice for policy tracking."""

    if limit <= 0 or not samples:
        return []
    if len(samples) <= limit:
        return list(samples)

    stride = len(samples) / limit
    return [
        samples[min(len(samples) - 1, int((index + 0.5) * stride))]
        for index in range(limit)
    ]


def _attach_search_q_teacher_labels(
    samples: list[TrajectoryStep],
    labels: list[SearchQLabels],
) -> dict[str, float]:
    """Attach unplayed, visited-action search labels to selected samples."""

    if len(samples) != len(labels):
        raise ValueError("search-Q labels must align one-to-one with samples")
    visited_labels = 0
    auxiliary_labels = 0
    legal_actions = 0
    absolute_target_sum = 0.0
    for sample, label in zip(samples, labels, strict=True):
        expected_legal = sample.target_policy.numel()
        if label.legal_actions != expected_legal:
            raise ValueError(
                "search-Q label legal-action count does not match sample"
            )
        action_indices = label.action_indices.detach().to(
            device="cpu", dtype=torch.long
        ).reshape(-1)
        targets = label.targets.detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(-1)
        if action_indices.numel() != targets.numel():
            raise ValueError("search-Q action indices and targets must align")
        if action_indices.unique().numel() != action_indices.numel():
            raise ValueError("search-Q action indices must be unique")
        if bool(((action_indices < 0) | (action_indices >= expected_legal)).any()):
            raise ValueError("search-Q action index is outside its policy")
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("search-Q targets must be finite")

        visited_labels += action_indices.numel()
        legal_actions += expected_legal
        keep = action_indices != sample.action_index
        action_indices = action_indices[keep]
        targets = targets[keep]
        sample.auxiliary_q_action_indices = action_indices
        sample.auxiliary_q_targets = targets
        auxiliary_labels += action_indices.numel()
        absolute_target_sum += float(targets.abs().sum().item())

    return {
        "search_q_teacher_states": float(len(samples)),
        "search_q_teacher_visited_labels": float(visited_labels),
        "search_q_teacher_auxiliary_labels": float(auxiliary_labels),
        "search_q_teacher_mean_auxiliary_labels": (
            auxiliary_labels / max(len(samples), 1)
        ),
        "search_q_teacher_visited_coverage": (
            visited_labels / max(legal_actions, 1)
        ),
        "search_q_teacher_mean_abs_target": (
            absolute_target_sum / max(auxiliary_labels, 1)
        ),
    }


def _measure_auxiliary_q_diagnostics(
    model: KlentNet | DenseAxisKlentNet | PersistentRayKlentNet,
    samples: list[TrajectoryStep],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    edge_budget: int,
) -> dict[str, float]:
    """Measure current Q predictions on the fixed teacher's unplayed labels."""

    selected = [
        sample
        for sample in samples
        if sample.auxiliary_q_action_indices is not None
        and sample.auxiliary_q_action_indices.numel() > 0
    ]
    if not selected:
        return {
            "search_q_teacher_q_labels": 0.0,
            "search_q_teacher_q_mse": 0.0,
            "search_q_teacher_q_mae": 0.0,
            "search_q_teacher_q_correlation": 0.0,
        }

    was_training = model.training
    totals = torch.zeros(8, device=device, dtype=torch.float64)
    fit_q_is_selected = isinstance(model, KlentNet)
    model.eval()
    try:
        with torch.inference_mode():
            for prepared in _prepared_training_batches(
                selected,
                batch_size=batch_size,
                model_config=model_config,
                edge_budget=edge_budget,
                prefetch=False,
            ):
                for batch_cpu, packed_samples, packed_targets in prepared:
                    segment_lengths_cpu = packed_targets[-1]
                    offsets = segment_lengths_cpu.cumsum(0) - segment_lengths_cpu
                    chosen = torch.cat(
                        [
                            offsets[index]
                            + sample.auxiliary_q_action_indices.to(
                                dtype=torch.long
                            )
                            for index, sample in enumerate(packed_samples)
                        ]
                    ).to(device)
                    target_q = torch.cat(
                        [
                            sample.auxiliary_q_targets.to(dtype=torch.float32)
                            for sample in packed_samples
                        ]
                    ).to(device)
                    batch = move_batch_to_device(batch_cpu, device)
                    with _autocast(device, precision):
                        if fit_q_is_selected:
                            output = model.forward_fit(batch, chosen)
                            predicted_q = output.q_values
                        else:
                            output = model.forward_batch(batch)
                            predicted_q = output.q_values.index_select(0, chosen)
                    predicted_q = predicted_q.float()
                    difference = predicted_q - target_q
                    count = float(target_q.numel())
                    totals.add_(
                        torch.stack(
                            (
                                torch.tensor(count, device=device),
                                predicted_q.sum(),
                                target_q.sum(),
                                predicted_q.square().sum(),
                                target_q.square().sum(),
                                (predicted_q * target_q).sum(),
                                difference.square().sum(),
                                difference.abs().sum(),
                            )
                        ).to(torch.float64)
                    )
                    del batch, chosen, target_q, output, predicted_q, difference
    finally:
        model.train(was_training)

    values = totals.cpu().tolist()
    count, sum_x, sum_y, sum_x2, sum_y2, sum_xy, sum_sq, sum_abs = values
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = max(0.0, sum_x2 - sum_x * sum_x / count)
    variance_y = max(0.0, sum_y2 - sum_y * sum_y / count)
    denominator = math.sqrt(variance_x * variance_y)
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    return {
        "search_q_teacher_q_labels": count,
        "search_q_teacher_q_mse": sum_sq / count,
        "search_q_teacher_q_mae": sum_abs / count,
        "search_q_teacher_q_correlation": max(-1.0, min(1.0, correlation)),
    }


def _stored_policy_target_kl(
    samples: list[TrajectoryStep],
) -> float | None:
    """Return collection-time target/prior KL when every sample records it."""

    values = [sample.target_prior_kl for sample in samples]
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def _measure_policy_target_diagnostics(
    model: KlentNet,
    samples: list[TrajectoryStep],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    edge_budget: int,
) -> tuple[float, float]:
    """Measure target KL and target/current top-1 agreement without updates."""

    if not samples:
        raise ValueError("policy target diagnostic requires samples")

    was_training = model.training
    total_kl = torch.zeros((), device=device)
    top1_agreements = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    model.eval()
    try:
        with torch.inference_mode():
            for prepared_outer_batch in _prepared_training_batches(
                samples,
                batch_size=batch_size,
                model_config=model_config,
                edge_budget=edge_budget,
                prefetch=False,
            ):
                for batch_cpu, packed_samples, packed_targets in (
                    prepared_outer_batch
                ):
                    (
                        target_policy_cpu,
                        segment_ids_cpu,
                        _chosen_cpu,
                        _target_q_cpu,
                        _target_top1,
                        segment_lengths_cpu,
                    ) = packed_targets
                    batch = move_batch_to_device(batch_cpu, device)
                    target_policy = target_policy_cpu.to(device)
                    segment_ids = segment_ids_cpu.to(device)
                    segment_lengths = segment_lengths_cpu.to(device)
                    count = len(packed_samples)
                    with _autocast(device, precision):
                        output = model.forward_batch(batch)
                    if output.policy_logits.numel() != target_policy.numel():
                        raise RuntimeError(
                            "stored target policies no longer match legal moves"
                        )
                    log_policy = _segmented_log_softmax(
                        output.policy_logits.float(),
                        segment_ids,
                        segment_lengths,
                        int(segment_lengths_cpu.max().item()),
                    )
                    max_segment_length = int(
                        segment_lengths_cpu.max().item()
                    )
                    target_argmax = _segmented_argmax(
                        target_policy,
                        segment_lengths,
                        max_segment_length,
                    )
                    current_argmax = _segmented_argmax(
                        output.policy_logits.float(),
                        segment_lengths,
                        max_segment_length,
                    )
                    top1_agreements.add_(
                        (target_argmax == current_argmax).sum()
                    )
                    total_kl.add_(
                        (
                            target_policy
                            * (
                                target_policy.clamp_min(1e-12).log()
                                - log_policy
                            )
                        ).sum()
                    )
                    seen += count
                    del (
                        batch,
                        target_policy,
                        segment_ids,
                        segment_lengths,
                        output,
                        log_policy,
                        target_argmax,
                        current_argmax,
                    )
    finally:
        model.train(was_training)

    values = torch.stack((total_kl, top1_agreements.float())).cpu().tolist()
    mean_kl = float(values[0]) / seen
    if math.isfinite(mean_kl):
        mean_kl = max(0.0, mean_kl)
    return mean_kl, float(values[1]) / seen


def _measure_policy_q_trunk_gradients(
    model: KlentNet | DenseAxisKlentNet | PersistentRayKlentNet,
    samples: list[TrajectoryStep],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    edge_budget: int,
    q_loss_weight: float,
) -> dict[str, float]:
    """Compare policy and weighted-Q gradients across the diagnostic slice.

    This deliberately excludes both output heads. The cosine therefore
    measures whether the two losses agree about the shared representation,
    which is the only route by which the Q objective can rewrite policy
    features. Policy gradients are accumulated by state population and Q
    gradients by supervised-action population, matching the two independently
    normalized losses used during FIT. Using the complete deterministic slice
    avoids mistaking one small graph-shape-dependent microbatch for a
    generation-wide interaction, while leaving optimizer and model unchanged.
    """

    if not samples:
        raise ValueError("trunk gradient diagnostic requires samples")

    excluded = {
        id(parameter)
        for head in (model.policy_head, model.q_head)
        for parameter in head.parameters()
    }
    trunk_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    if not trunk_parameters:
        raise RuntimeError("model has no shared policy/Q trunk parameters")

    started_at = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        policy_gradient_sums = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in trunk_parameters
        ]
        q_gradient_sums = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in trunk_parameters
        ]
        seen = 0
        q_seen = 0
        fit_q_is_selected = isinstance(model, KlentNet)
        for prepared in _prepared_training_batches(
            samples,
            batch_size=batch_size,
            model_config=model_config,
            edge_budget=edge_budget,
            prefetch=False,
        ):
            for batch_cpu, packed_samples, packed_targets in prepared:
                (
                    target_policy_cpu,
                    segment_ids_cpu,
                    chosen_cpu,
                    target_q_cpu,
                    _target_top1,
                    segment_lengths_cpu,
                ) = packed_targets
                batch = move_batch_to_device(batch_cpu, device)
                target_policy = target_policy_cpu.to(device)
                segment_ids = segment_ids_cpu.to(device)
                chosen = chosen_cpu.to(device)
                target_q = target_q_cpu.to(device)
                segment_lengths = segment_lengths_cpu.to(device)
                count = len(packed_samples)
                q_count = target_q.numel()

                def component_losses() -> tuple[torch.Tensor, torch.Tensor]:
                    # Independent calls are intentional. Compiled AOTAutograd
                    # may donate saved buffers to backward, which forbids
                    # retaining one compiled graph for two autograd.grad calls
                    # on ROCm/CUDA.
                    with _autocast(device, precision):
                        if fit_q_is_selected:
                            output = model.forward_fit(batch, chosen)
                        else:
                            output = model.forward_batch(batch)
                        log_policy = _segmented_log_softmax(
                            output.policy_logits.float(),
                            segment_ids,
                            segment_lengths,
                            int(segment_lengths_cpu.max().item()),
                        )
                        policy_loss = (
                            -(target_policy * log_policy).sum() / count
                        )
                        predicted_q = (
                            output.q_values
                            if fit_q_is_selected
                            else output.q_values.index_select(0, chosen)
                        ).float()
                        weighted_q_loss = q_loss_weight * F.mse_loss(
                            predicted_q,
                            target_q,
                        )
                    return policy_loss, weighted_q_loss

                policy_loss, unused_q_loss = component_losses()
                del unused_q_loss
                policy_gradients = torch.autograd.grad(
                    policy_loss,
                    trunk_parameters,
                    allow_unused=True,
                )
                del policy_loss
                unused_policy_loss, weighted_q_loss = component_losses()
                del unused_policy_loss
                q_gradients = torch.autograd.grad(
                    weighted_q_loss,
                    trunk_parameters,
                    allow_unused=True,
                )
                del weighted_q_loss
                for index, (policy_gradient, q_gradient) in enumerate(
                    zip(policy_gradients, q_gradients, strict=True)
                ):
                    if policy_gradient is not None:
                        policy_gradient_sums[index].add_(
                            policy_gradient.detach().float(),
                            alpha=count,
                        )
                    if q_gradient is not None:
                        q_gradient_sums[index].add_(
                            q_gradient.detach().float(),
                            alpha=q_count,
                        )
                seen += count
                q_seen += q_count
                del (
                    batch,
                    target_policy,
                    segment_ids,
                    chosen,
                    target_q,
                    segment_lengths,
                    policy_gradients,
                    q_gradients,
                )

        if seen != len(samples):
            raise RuntimeError(
                "trunk gradient diagnostic did not consume every sample: "
                f"seen={seen}, expected={len(samples)}"
            )
        policy_sq = torch.zeros((), device=device)
        q_sq = torch.zeros((), device=device)
        dot = torch.zeros((), device=device)
        inverse_seen = 1.0 / seen
        inverse_q_seen = 1.0 / q_seen
        for policy_gradient, q_gradient in zip(
            policy_gradient_sums,
            q_gradient_sums,
            strict=True,
        ):
            policy_gradient.mul_(inverse_seen)
            q_gradient.mul_(inverse_q_seen)
            policy_sq.add_(policy_gradient.square().sum())
            q_sq.add_(q_gradient.square().sum())
            dot.add_((policy_gradient * q_gradient).sum())

        policy_norm = policy_sq.sqrt()
        q_norm = q_sq.sqrt()
        denominator = policy_norm * q_norm
        cosine = torch.where(
            denominator > 0,
            dot / denominator,
            torch.zeros_like(dot),
        ).clamp(-1.0, 1.0)
        values = torch.stack((policy_norm, q_norm, cosine)).cpu().tolist()
    finally:
        model.train(was_training)

    return {
        "trunk_gradient_diagnostic_examples": float(seen),
        "trunk_gradient_diagnostic_q_labels": float(q_seen),
        "trunk_gradient_diagnostic_seconds": time.monotonic() - started_at,
        "policy_trunk_grad_norm": float(values[0]),
        "q_trunk_grad_norm": float(values[1]),
        "policy_q_trunk_grad_cosine": float(values[2]),
    }


def _gradient_clip_statistics(
    grad_norms: torch.Tensor,
    max_grad_norm: float,
) -> dict[str, float]:
    """Summarize pre-clipping optimizer-step gradient norms."""

    if grad_norms.ndim != 1 or grad_norms.numel() == 0:
        raise ValueError("grad_norms must be a non-empty one-dimensional tensor")
    values = grad_norms.detach().float().cpu()
    if max_grad_norm > 0:
        clipped = values > max_grad_norm
        clip_scales = torch.clamp(max_grad_norm / values, max=1.0)
    else:
        clipped = torch.zeros_like(values, dtype=torch.bool)
        clip_scales = torch.ones_like(values)
    return {
        "mean_grad_norm": float(values.mean().item()),
        "grad_norm_p50": float(torch.quantile(values, 0.50).item()),
        "grad_norm_p95": float(torch.quantile(values, 0.95).item()),
        "grad_norm_max": float(values.max().item()),
        "clipped_optimizer_steps": float(clipped.sum().item()),
        "clip_fraction": float(clipped.float().mean().item()),
        "mean_clip_scale": float(clip_scales.mean().item()),
    }


def _global_l2_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Return one L2 norm across a homogeneous list of device tensors."""

    if not tensors:
        raise ValueError("cannot measure an empty tensor list")
    per_tensor = torch._foreach_norm(tensors, 2.0)
    return torch.linalg.vector_norm(
        torch.stack([value.float() for value in per_tensor])
    )


def _optimizer_update_statistics(
    update_norms: torch.Tensor,
    update_to_weight_ratios: torch.Tensor,
) -> dict[str, float]:
    """Summarize exact parameter movement across optimizer steps."""

    if (
        update_norms.ndim != 1
        or update_to_weight_ratios.ndim != 1
        or update_norms.numel() == 0
        or update_norms.shape != update_to_weight_ratios.shape
    ):
        raise ValueError("optimizer update samples must be paired 1-D tensors")
    norms = update_norms.detach().float().cpu()
    ratios = update_to_weight_ratios.detach().float().cpu()
    return {
        "mean_parameter_update_norm": float(norms.mean().item()),
        "parameter_update_norm_p95": float(
            torch.quantile(norms, 0.95).item()
        ),
        "mean_update_to_weight_ratio": float(ratios.mean().item()),
        "update_to_weight_ratio_p95": float(
            torch.quantile(ratios, 0.95).item()
        ),
    }


def _seed_fit_compilation(
    model: KlentNet,
    optimizer: torch.optim.Optimizer,
    prepared_outer_batch,
    *,
    device: torch.device,
    precision: str,
    q_loss_weight: float,
) -> None:
    """Compile FIT at a representative graph size without updating weights."""

    target_nodes = int(getattr(model, "_fit_compile_seed_nodes", 0))
    if target_nodes <= 0 or bool(
        getattr(model, "_fit_compile_seeded", False)
    ):
        return
    batch_cpu, packed_samples, packed_targets = min(
        prepared_outer_batch,
        key=lambda item: abs(int(item[0].x.shape[0]) - target_nodes),
    )
    (
        target_policy_cpu,
        segment_ids_cpu,
        chosen_cpu,
        target_q_cpu,
        _target_top1,
        segment_lengths_cpu,
    ) = packed_targets
    count = len(packed_samples)
    optimizer.zero_grad(set_to_none=True)
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        # A compile seed must not consume dropout or other model RNG relative
        # to the real exactly-once epoch.
        with torch.random.fork_rng(devices=fork_devices):
            batch = move_batch_to_device(batch_cpu, device)
            target_policy = target_policy_cpu.to(device)
            segment_ids = segment_ids_cpu.to(device)
            chosen = chosen_cpu.to(device)
            target_q = target_q_cpu.to(device)
            segment_lengths = segment_lengths_cpu.to(device)
            with _autocast(device, precision):
                output = model.forward_fit(batch, chosen)
                if output.policy_logits.numel() != target_policy.numel():
                    raise RuntimeError(
                        "stored target policies no longer match legal moves"
                    )
                log_policy = _segmented_log_softmax(
                    output.policy_logits.float(),
                    segment_ids,
                    segment_lengths,
                    int(segment_lengths_cpu.max().item()),
                )
                policy_loss = -(target_policy * log_policy).sum() / count
                q_loss = F.mse_loss(output.q_values.float(), target_q)
                total_loss = policy_loss + q_loss_weight * q_loss
            total_loss.backward()
        model._fit_compile_seeded = True
    finally:
        # No optimizer step: parameters and optimizer state are bit-for-bit
        # unchanged, and the real epoch still sees every example exactly once.
        optimizer.zero_grad(set_to_none=True)


def train_epoch(
    model: KlentNet,
    optimizer: torch.optim.Optimizer,
    samples: list[TrajectoryStep],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    edge_budget: int,
    grad_accumulation: bool,
    q_loss_weight: float,
    max_grad_norm: float,
    seed: int | None,
    prefetch_batches: bool = True,
    optimize_policy: bool = True,
) -> dict[str, float]:
    """Fit every fresh on-policy position exactly once."""

    started_at = time.monotonic()
    if not samples:
        raise ValueError("cannot train on an empty self-play batch")
    if any(sample.return_target is None for sample in samples):
        raise ValueError("all samples must have return targets")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    totals = {
        "policy_loss": torch.zeros((), device=device),
        "q_loss": torch.zeros((), device=device),
    }
    grad_norms: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    update_to_weight_ratios: list[torch.Tensor] = []
    tracked_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if not tracked_parameters:
        raise ValueError("optimizer has no trainable parameters")
    # Reuse one model-sized snapshot across all steps. This records exact
    # AdamW movement without retaining one model copy per optimizer update.
    parameter_snapshot = [
        torch.empty_like(
            parameter,
            memory_format=torch.preserve_format,
        )
        for parameter in tracked_parameters
    ]
    target_top1_total = 0
    seen = 0
    updated_examples = 0
    q_labels = 0
    updated_q_labels = 0
    skipped_nonfinite_examples = 0
    microbatches = 0
    attempted_optimizer_steps = 0
    nonfinite_optimizer_steps = 0
    optimizer_steps = 0
    model.train()
    cache_pressure = _CudaCachePressureGate.from_environment(device)

    if (
        isinstance(model, KlentNet)
        and int(getattr(model, "_fit_compile_seed_nodes", 0)) > 0
        and not bool(getattr(model, "_fit_compile_seeded", False))
    ):
        # Build the seed without prefetch so compilation cannot retain the next
        # outer graph batch while Inductor benchmarks candidate GEMMs.
        seed_outer_batch = next(
            _prepared_training_batches(
                shuffled[:batch_size],
                batch_size=batch_size,
                model_config=model_config,
                edge_budget=edge_budget,
                prefetch=False,
            )
        )
        _seed_fit_compilation(
            model,
            optimizer,
            seed_outer_batch,
            device=device,
            precision=precision,
            q_loss_weight=q_loss_weight,
        )
        del seed_outer_batch

    for prepared_outer_batch in (
        _prepared_training_batches(
            shuffled,
            batch_size=batch_size,
            model_config=model_config,
            edge_budget=edge_budget,
            prefetch=prefetch_batches and device.type == "cuda",
        )
    ):
        optimizer_groups = (
            [prepared_outer_batch]
            if grad_accumulation
            else [[microbatch] for microbatch in prepared_outer_batch]
        )
        for optimizer_group in optimizer_groups:
            group_examples = sum(
                len(packed_samples)
                for _batch, packed_samples, _targets in optimizer_group
            )
            if group_examples <= 0:
                raise RuntimeError("prepared an empty optimizer batch")
            group_q_labels = sum(
                int(targets[3].numel())
                for _batch, _packed_samples, targets in optimizer_group
            )
            if group_q_labels <= 0:
                raise RuntimeError("prepared an optimizer batch without Q labels")
            optimizer.zero_grad(set_to_none=True)
            group_totals = {
                "policy_loss": torch.zeros((), device=device),
                "q_loss": torch.zeros((), device=device),
            }
            group_target_top1 = 0

            for batch_cpu, packed_samples, packed_targets in optimizer_group:
                (
                    target_policy_cpu,
                    segment_ids_cpu,
                    chosen_cpu,
                    target_q_cpu,
                    target_top1,
                    segment_lengths_cpu,
                ) = packed_targets
                batch = move_batch_to_device(batch_cpu, device)
                target_policy = target_policy_cpu.to(device)
                segment_ids = segment_ids_cpu.to(device)
                chosen = chosen_cpu.to(device)
                target_q = target_q_cpu.to(device)
                segment_lengths = segment_lengths_cpu.to(device)
                count = len(packed_samples)
                q_count = target_q.numel()
                with _autocast(device, precision):
                    fit_q_is_selected = isinstance(model, KlentNet)
                    if fit_q_is_selected:
                        output = model.forward_fit(batch, chosen)
                    else:
                        output = model.forward_batch(batch)
                    if output.policy_logits.numel() != target_policy.numel():
                        raise RuntimeError(
                            "stored target policies no longer match legal moves"
                        )
                    log_policy = _segmented_log_softmax(
                        output.policy_logits.float(),
                        segment_ids,
                        segment_lengths,
                        int(segment_lengths_cpu.max().item()),
                    )
                    policy_loss = (
                        -(target_policy * log_policy).sum() / count
                    )
                    predicted_q = (
                        output.q_values
                        if fit_q_is_selected
                        else output.q_values.index_select(0, chosen)
                    ).float()
                    q_loss = F.mse_loss(predicted_q, target_q)
                    total_loss = q_loss_weight * q_loss
                    if optimize_policy:
                        total_loss = policy_loss + total_loss

                # Policy is a mean over states while Q is a mean over labelled
                # actions. Reconstruct both outer-batch populations exactly;
                # in ordinary KLENT each state has one Q label, reducing to
                # the historical total-loss scaling bit for bit.
                if q_count == count and group_q_labels == group_examples:
                    backward_loss = total_loss * (count / group_examples)
                else:
                    backward_loss = (
                        q_loss_weight
                        * q_loss
                        * (q_count / group_q_labels)
                    )
                    if optimize_policy:
                        backward_loss = (
                            policy_loss * (count / group_examples)
                            + backward_loss
                        )
                backward_loss.backward()
                seen += count
                q_labels += q_count
                microbatches += 1
                group_totals["policy_loss"].add_(
                    policy_loss.detach() * count
                )
                group_totals["q_loss"].add_(q_loss.detach() * q_count)
                group_target_top1 += target_top1
                del (
                    batch,
                    target_policy,
                    segment_ids,
                    segment_lengths,
                    chosen,
                    target_q,
                    output,
                    log_policy,
                    policy_loss,
                    predicted_q,
                    q_loss,
                    total_loss,
                    backward_loss,
                )
                cache_pressure.step()

            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            else:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().norm()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ]
                    )
                )
            attempted_optimizer_steps += 1
            finite_group = bool(
                torch.isfinite(
                    torch.stack(
                        (
                            grad_norm.detach().float(),
                            group_totals["policy_loss"].float(),
                            group_totals["q_loss"].float(),
                        )
                    )
                )
                .all()
                .item()
            )
            if not finite_group:
                nonfinite_optimizer_steps += 1
                skipped_nonfinite_examples += group_examples
                optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    "discarding non-finite optimizer group step=%d examples=%d",
                    attempted_optimizer_steps,
                    group_examples,
                )
                continue

            for name, value in group_totals.items():
                totals[name].add_(value)
            target_top1_total += group_target_top1
            updated_examples += group_examples
            updated_q_labels += group_q_labels
            with torch.no_grad():
                torch._foreach_copy_(
                    parameter_snapshot,
                    tracked_parameters,
                )
                parameter_norm = _global_l2_norm(parameter_snapshot)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_sub_(
                    parameter_snapshot,
                    tracked_parameters,
                )
                update_norm = _global_l2_norm(parameter_snapshot)
                update_to_weight_ratio = update_norm / parameter_norm.clamp_min(
                    torch.finfo(parameter_norm.dtype).tiny
                )
            optimizer_steps += 1
            grad_norms.append(grad_norm.detach())
            update_norms.append(update_norm.detach())
            update_to_weight_ratios.append(
                update_to_weight_ratio.detach()
            )

    if optimizer_steps == 0:
        raise RuntimeError(
            "all optimizer groups had non-finite losses or gradients; "
            "no parameters were updated"
        )

    summary = torch.cat(
        (torch.stack([totals["policy_loss"], totals["q_loss"]]),
         torch.stack(grad_norms))
    ).cpu()
    policy_loss_total, q_loss_total = summary[:2].tolist()
    gradient_stats = _gradient_clip_statistics(
        summary[2:],
        max_grad_norm,
    )
    update_stats = _optimizer_update_statistics(
        torch.stack(update_norms),
        torch.stack(update_to_weight_ratios),
    )
    elapsed_seconds = time.monotonic() - started_at

    mean_policy_loss = policy_loss_total / updated_examples
    mean_q_loss = q_loss_total / updated_q_labels
    return {
        "examples": float(seen),
        "updated_examples": float(updated_examples),
        "q_labels": float(q_labels),
        "updated_q_labels": float(updated_q_labels),
        "mean_q_labels_per_example": q_labels / seen,
        "skipped_nonfinite_examples": float(skipped_nonfinite_examples),
        "microbatches": float(microbatches),
        "attempted_optimizer_steps": float(attempted_optimizer_steps),
        "nonfinite_optimizer_steps": float(nonfinite_optimizer_steps),
        "optimizer_steps": float(optimizer_steps),
        "mean_microbatch_size": seen / microbatches,
        "mean_optimizer_batch_size": updated_examples / optimizer_steps,
        "mean_microbatches_per_step": (
            microbatches / attempted_optimizer_steps
        ),
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": seen / elapsed_seconds,
        "policy_loss": mean_policy_loss,
        "q_loss": mean_q_loss,
        "total_loss": (
            (mean_policy_loss if optimize_policy else 0.0)
            + q_loss_weight * mean_q_loss
        ),
        "policy_updates_enabled": float(optimize_policy),
        **gradient_stats,
        **update_stats,
        "played_action_target_top1": target_top1_total / updated_examples,
        **cache_pressure.metrics(),
    }


@contextmanager
def _output_heads_only_scope(
    model: KlentNet | DenseAxisKlentNet | PersistentRayKlentNet,
    *,
    include_policy: bool,
) -> Iterator[list[torch.nn.Parameter]]:
    """Temporarily restrict training to selected output heads.

    AdamW skips parameters whose gradient is ``None``, including weight decay
    and moment updates. Restoring ``requires_grad`` afterwards therefore
    preserves both frozen trunk weights and their existing optimizer state.
    """

    q_head = getattr(model, "q_head", None)
    if not isinstance(q_head, torch.nn.Module):
        raise ValueError("head-only training requires an action-Q head")
    trainable_parameters = list(q_head.parameters())
    if include_policy:
        policy_head = getattr(model, "policy_head", None)
        if not isinstance(policy_head, torch.nn.Module):
            raise ValueError("heads-only training requires a policy head")
        trainable_parameters.extend(policy_head.parameters())
    if not trainable_parameters:
        raise ValueError("action-Q head has no trainable parameters")
    trainable_parameter_ids = {
        id(parameter) for parameter in trainable_parameters
    }
    original_requires_grad = {
        id(parameter): parameter.requires_grad
        for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable_parameter_ids)
    try:
        yield trainable_parameters
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])


@contextmanager
def _critic_head_only_scope(
    model: KlentNet | DenseAxisKlentNet | PersistentRayKlentNet,
) -> Iterator[list[torch.nn.Parameter]]:
    """Temporarily make the action-Q head the only trainable module."""

    with _output_heads_only_scope(model, include_policy=False) as parameters:
        yield parameters


@contextmanager
def _heads_only_scope(
    model: KlentNet | DenseAxisKlentNet | PersistentRayKlentNet,
) -> Iterator[list[torch.nn.Parameter]]:
    """Temporarily train both output heads but not their shared trunk."""

    with _output_heads_only_scope(model, include_policy=True) as parameters:
        yield parameters


def _refit_search_q_head(
    model: KlentNet,
    optimizer: torch.optim.Optimizer,
    samples: list[TrajectoryStep],
    *,
    model_config,
    device: torch.device,
    precision: str,
    batch_size: int,
    edge_budget: int,
    epochs: int,
    max_grad_norm: float,
    seed: int | None,
) -> dict[str, float]:
    """Give the Q head a final sparse correction without moving the trunk.

    The ordinary joint FIT already consumes these labels and remains the only
    path by which they may teach shared features. This short final pass freezes
    every non-Q parameter, preventing fixed-teacher supervision from directly
    changing policy logits while restoring action ranking that the much larger
    on-policy objective may have overwritten later in the shuffled epoch.
    """

    if epochs <= 0:
        raise ValueError("search-Q refit epochs must be positive")
    selected = [
        sample
        for sample in samples
        if sample.auxiliary_q_action_indices is not None
        and sample.auxiliary_q_action_indices.numel() > 0
    ]
    if not selected:
        raise ValueError("search-Q refit requires auxiliary Q labels")
    if not isinstance(model, KlentNet):
        raise ValueError("search-Q head refit currently requires graph KLENT")

    started_at = time.monotonic()
    was_training = model.training
    loss_sum = torch.zeros((), device=device)
    updated_labels = 0
    optimizer_steps = 0
    nonfinite_optimizer_steps = 0
    grad_norms: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    update_ratios: list[torch.Tensor] = []
    model.eval()
    try:
        scope = _critic_head_only_scope(model)
        q_parameters = scope.__enter__()
        parameter_snapshot = [
            torch.empty_like(parameter, memory_format=torch.preserve_format)
            for parameter in q_parameters
        ]
        for epoch in range(epochs):
            shuffled = list(selected)
            epoch_seed = None if seed is None else seed + epoch
            random.Random(epoch_seed).shuffle(shuffled)
            for prepared in _prepared_training_batches(
                shuffled,
                batch_size=batch_size,
                model_config=model_config,
                edge_budget=edge_budget,
                prefetch=False,
            ):
                group_q_labels = sum(
                    int(targets[3].numel())
                    for _batch, _packed_samples, targets in prepared
                )
                optimizer.zero_grad(set_to_none=True)
                group_loss_sum = torch.zeros((), device=device)
                for batch_cpu, _packed_samples, packed_targets in prepared:
                    chosen_cpu = packed_targets[2]
                    target_q_cpu = packed_targets[3]
                    chosen = chosen_cpu.to(device)
                    target_q = target_q_cpu.to(device)
                    batch = move_batch_to_device(batch_cpu, device)
                    with _autocast(device, precision):
                        output = model.forward_fit(batch, chosen)
                        q_loss = F.mse_loss(
                            output.q_values.float(), target_q
                        )
                    q_count = target_q.numel()
                    (q_loss * (q_count / group_q_labels)).backward()
                    group_loss_sum.add_(q_loss.detach() * q_count)
                    del batch, chosen, target_q, output, q_loss

                if max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        q_parameters, max_grad_norm
                    )
                else:
                    grad_norm = torch.linalg.vector_norm(
                        torch.stack(
                            [
                                parameter.grad.detach().norm()
                                for parameter in q_parameters
                                if parameter.grad is not None
                            ]
                        )
                    )
                finite = bool(
                    torch.isfinite(
                        torch.stack(
                            (grad_norm.detach().float(), group_loss_sum.float())
                        )
                    ).all().item()
                )
                if not finite:
                    nonfinite_optimizer_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    logger.warning(
                        "discarding non-finite search-Q refit group labels=%d",
                        group_q_labels,
                    )
                    continue

                with torch.no_grad():
                    torch._foreach_copy_(parameter_snapshot, q_parameters)
                    parameter_norm = _global_l2_norm(parameter_snapshot)
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_sub_(parameter_snapshot, q_parameters)
                    update_norm = _global_l2_norm(parameter_snapshot)
                    update_ratio = update_norm / parameter_norm.clamp_min(
                        torch.finfo(parameter_norm.dtype).tiny
                    )
                loss_sum.add_(group_loss_sum)
                updated_labels += group_q_labels
                optimizer_steps += 1
                grad_norms.append(grad_norm.detach())
                update_norms.append(update_norm.detach())
                update_ratios.append(update_ratio.detach())
    finally:
        optimizer.zero_grad(set_to_none=True)
        if "scope" in locals():
            scope.__exit__(None, None, None)
        model.train(was_training)

    if optimizer_steps == 0:
        raise RuntimeError("all search-Q refit optimizer groups were non-finite")
    gradient_stats = _gradient_clip_statistics(
        torch.stack(grad_norms), max_grad_norm
    )
    update_stats = _optimizer_update_statistics(
        torch.stack(update_norms), torch.stack(update_ratios)
    )
    metrics = {
        "search_q_teacher_refit_epochs": float(epochs),
        "search_q_teacher_refit_states": float(len(selected)),
        "search_q_teacher_refit_q_labels": float(updated_labels),
        "search_q_teacher_refit_optimizer_steps": float(optimizer_steps),
        "search_q_teacher_refit_nonfinite_optimizer_steps": float(
            nonfinite_optimizer_steps
        ),
        "search_q_teacher_refit_q_loss": float(
            loss_sum.cpu().item() / updated_labels
        ),
        "search_q_teacher_refit_seconds": time.monotonic() - started_at,
    }
    metrics.update(
        {
            f"search_q_teacher_refit_{key}": value
            for key, value in gradient_stats.items()
        }
    )
    metrics.update(
        {
            f"search_q_teacher_refit_{key}": value
            for key, value in update_stats.items()
        }
    )
    return metrics


class Trainer:
    """Own the model, optimizer, artifacts, and reference KLENT iteration."""

    def __init__(
        self,
        config: Config,
        *,
        tensorboard: bool = True,
        resume: str | Path | None = None,
        resume_configured_lr: bool = False,
        init_from: str | Path | None = None,
        display: TrainingDashboard | None = None,
    ) -> None:
        if resume is not None and init_from is not None:
            raise ValueError("resume and init_from are mutually exclusive")
        if resume_configured_lr and resume is None:
            raise ValueError("resume_configured_lr requires a resume checkpoint")
        self.config = config
        self.display = display
        self.device = torch.device(config.run.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"run.device={config.run.device!r}, but CUDA/ROCm is unavailable"
            )
        if config.run.seed is not None:
            random.seed(config.run.seed)
            torch.manual_seed(config.run.seed)

        self.model = make_klent_net(config.model).to(self.device)
        if (
            config.run.compile
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            # Keep KLENT's inductor artifacts separate from the production
            # trainer/inference subprocess caches on the shared APU.
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR",
                "/tmp/torchinductor_hexo/klent",
            )
            _ensure_compiler_nofile_limit()
            compile_klent_forward(
                self.model,
                fit_max_autotune=config.training.fit_max_autotune,
                fit_compile_seed_nodes=(
                    config.training.fit_compile_seed_nodes
                ),
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.iteration = 0
        self.output_dir = Path(config.run.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.writer = None
        self._actors = None
        self._checkpoint_opponents = CheckpointOpponentCache()
        self._search_q_teacher: FixedCheckpointSearchQTeacher | None = None
        self._checkpoint_history_dirs = [self.checkpoint_dir.resolve()]
        self.initial_checkpoint: dict[str, object] | None = None
        try:
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(self.output_dir / "tensorboard")
            if resume is not None:
                self.load_checkpoint(
                    resume,
                    use_configured_learning_rate=resume_configured_lr,
                )
            elif init_from is not None:
                self.initialize_from_production(init_from)
            if config.collection.workers > 1:
                self._actors = SharedInferenceActors(
                    config.collection.workers
                )
        except BaseException:
            self.close()
            raise

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        use_configured_learning_rate: bool = False,
    ) -> None:
        path = Path(path).expanduser().resolve()
        source_checkpoint_dir = path.parent
        if source_checkpoint_dir not in self._checkpoint_history_dirs:
            self._checkpoint_history_dirs.append(source_checkpoint_dir)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        stored_history = checkpoint.get("checkpoint_history_dirs", ())
        if isinstance(stored_history, (list, tuple)):
            for directory in stored_history:
                if not isinstance(directory, str):
                    continue
                resolved = Path(directory).expanduser().resolve()
                if resolved not in self._checkpoint_history_dirs:
                    self._checkpoint_history_dirs.append(resolved)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if use_configured_learning_rate:
            configured_lr = self.config.training.learning_rate
            restored_lrs = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
            for group in self.optimizer.param_groups:
                group["lr"] = configured_lr
                if "initial_lr" in group:
                    group["initial_lr"] = configured_lr
            logger.info(
                "overrode resumed optimizer learning rate %s -> %.6g",
                restored_lrs,
                configured_lr,
            )
        self.iteration = int(checkpoint["iteration"])
        initial_checkpoint = checkpoint.get("initial_checkpoint")
        if isinstance(initial_checkpoint, dict):
            self.initial_checkpoint = initial_checkpoint
        logger.info("resumed %s at iteration %d", path, self.iteration)

    def _get_search_q_teacher(self) -> FixedCheckpointSearchQTeacher:
        """Lazily load and retain the configured fixed search-Q teacher."""

        if self._search_q_teacher is None:
            training = self.config.training
            self._search_q_teacher = FixedCheckpointSearchQTeacher(
                training.search_q_teacher_checkpoint,
                device=self.device,
                precision=self.config.run.precision,
                simulations=training.search_q_teacher_simulations,
                actions=training.search_q_teacher_actions,
                root_batch_size=training.search_q_teacher_root_batch_size,
            )
            logger.info(
                "loaded search-Q teacher %s sims=%d actions=%d roots=%d",
                training.search_q_teacher_checkpoint,
                training.search_q_teacher_simulations,
                training.search_q_teacher_actions,
                training.search_q_teacher_root_batch_size,
            )
        return self._search_q_teacher

    def initialize_from_production(self, path: str | Path) -> None:
        """Initialize KLENT from a compatible trained checkpoint."""

        if not isinstance(
            self.model,
            (KlentNet, DenseAxisKlentNet, PersistentRayKlentNet),
        ):
            raise ValueError(
                "checkpoint initialization requires a graph or dense-axis "
                "model architecture"
            )

        path = Path(path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"initial checkpoint {path} is not a dict")

        compatibility_fields = (
            "hidden_dim",
            "num_layers",
            "num_heads",
            "conv_type",
            "pre_norm",
            "dropout",
            "use_layer_scale",
            "use_jk",
            "jk_mode",
            "policy_hidden",
            "q_hidden",
            "graph_type",
            "prune_empty_edges",
            "threat_features",
            "relative_stone_encoding",
            "axis_relational",
            "axis_window",
            "compact_stone_onehot",
            "node_coords",
            "moves_scope",
        )
        if checkpoint.get("format") == "hexo-klent-v1":
            if not isinstance(self.model, PersistentRayKlentNet):
                raise ValueError(
                    "a KLENT checkpoint may be used with --init-from only "
                    "to graft persistent_ray_axis from graph or dense_axis"
                )
            raw_config = checkpoint.get("model_config", {})
            if not isinstance(raw_config, dict):
                raise ValueError(
                    "KLENT initial checkpoint has no model_config"
                )
            known = {
                field.name
                for field in dataclasses.fields(KlentModelConfig)
            }
            source_config = KlentModelConfig(
                **{
                    key: value
                    for key, value in raw_config.items()
                    if key in known
                }
            )
            source_architecture = source_config.architecture
            if source_architecture not in {"graph", "dense_axis"}:
                raise ValueError(
                    "persistent-ray graft source must use model.architecture="
                    "'graph' or 'dense_axis'"
                )
            fields = compatibility_fields
            if source_architecture == "dense_axis":
                fields = (*fields, "dense_ray_radius")
            mismatches = [
                f"{name}: source={getattr(source_config, name)!r}, "
                f"target={getattr(self.config.model, name)!r}"
                for name in fields
                if getattr(source_config, name)
                != getattr(self.config.model, name)
            ]
            if mismatches:
                raise ValueError(
                    f"{source_architecture} KLENT checkpoint does not match "
                    "persistent-ray "
                    "target: " + "; ".join(mismatches)
                )
            if source_architecture == "dense_axis":
                copied = load_dense_klent_graft(self.model, checkpoint)
            else:
                if not self.model.config.exact_graft_init:
                    raise ValueError(
                        "graph KLENT checkpoint grafting requires "
                        "model.exact_graft_init=true"
                    )
                copied = load_production_axis_weights(
                    self.model,
                    checkpoint,
                ).copied
            self.initial_checkpoint = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "iteration": checkpoint.get("iteration"),
                "copied_tensors": len(copied),
                "graft": "persistent_ray_axis",
                "source_architecture": source_architecture,
            }
            logger.info(
                "initialized persistent-ray KLENT from %s %s "
                "(%d tensors, source iteration=%s); optimizer starts fresh",
                source_architecture,
                path,
                len(copied),
                checkpoint.get("iteration", "?"),
            )
            return

        from hexo_a0.config import model_config_from_checkpoint

        if (
            isinstance(self.model, PersistentRayKlentNet)
            and not self.model.config.exact_graft_init
        ):
            raise ValueError(
                "production checkpoint grafting requires "
                "model.exact_graft_init=true"
            )
        source_config = model_config_from_checkpoint(checkpoint)
        if not bool(getattr(source_config, "q_head", False)):
            raise ValueError("production checkpoint does not contain a trained Q head")
        mismatches = [
            f"{name}: source={getattr(source_config, name)!r}, "
            f"target={getattr(self.config.model, name)!r}"
            for name in compatibility_fields
            if getattr(source_config, name) != getattr(self.config.model, name)
        ]
        if mismatches:
            raise ValueError(
                "production checkpoint architecture does not match KLENT: "
                + "; ".join(mismatches)
            )
        if isinstance(self.model, KlentNet):
            copied = load_production_graph_weights(self.model, checkpoint)
            graft = "graph"
        else:
            report = load_production_axis_weights(self.model, checkpoint)
            copied = report.copied
            graft = (
                "persistent_ray_axis"
                if isinstance(self.model, PersistentRayKlentNet)
                else "dense_axis"
            )
        self.initial_checkpoint = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train_steps": checkpoint.get("train_steps"),
            "copied_tensors": len(copied),
            "graft": graft,
        }
        logger.info(
            "initialized %s KLENT from %s (%d tensors, train_steps=%s); "
            "optimizer starts fresh",
            graft,
            path,
            len(copied),
            checkpoint.get("train_steps", "?"),
        )

    def save_checkpoint(self, *, final: bool = False) -> Path:
        name = (
            "final.pt"
            if final
            else f"checkpoint_{self.iteration:06d}.pt"
        )
        path = self.checkpoint_dir / name
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format": "hexo-klent-v1",
                "iteration": self.iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": dataclasses.asdict(self.config),
                "model_config": dataclasses.asdict(self.config.model),
                "initial_checkpoint": self.initial_checkpoint,
                "checkpoint_history_dirs": [
                    str(path) for path in self._checkpoint_history_dirs
                ],
            },
            temporary_path,
        )
        temporary_path.replace(path)
        return path

    def _best_so_far_state_path(self, opponent_name: str) -> Path:
        safe_name = opponent_name.replace("/", "__")
        return self.output_dir / "best_so_far" / f"{safe_name}.json"

    def _write_best_so_far_state(
        self,
        opponent_name: str,
        checkpoint: str | Path,
        iteration: int | None,
    ) -> dict[str, object]:
        path = self._best_so_far_state_path(opponent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        state: dict[str, object] = {
            "format": _BEST_SO_FAR_FORMAT,
            "name": opponent_name,
            "checkpoint": str(checkpoint_path),
            "iteration": iteration,
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return state

    def _load_best_so_far_state(
        self,
        opponent_name: str,
        initial_checkpoint: str | Path,
    ) -> dict[str, object]:
        path = self._best_so_far_state_path(opponent_name)
        if not path.exists():
            checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"best_so_far initial checkpoint not found: "
                    f"{checkpoint_path}"
                )
            return self._write_best_so_far_state(
                opponent_name,
                checkpoint_path,
                _checkpoint_iteration_from_path(checkpoint_path),
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != _BEST_SO_FAR_FORMAT:
            raise ValueError(f"invalid best_so_far state: {path}")
        if raw.get("name") != opponent_name:
            raise ValueError(
                f"best_so_far state name mismatch in {path}: "
                f"{raw.get('name')!r}"
            )
        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"best_so_far checkpoint not found: {checkpoint!r}"
            )
        iteration = raw.get("iteration")
        if iteration is not None and not isinstance(iteration, int):
            raise ValueError(
                f"invalid best_so_far iteration in {path}: {iteration!r}"
            )
        return raw

    def run(self, iterations: int | None = None) -> None:
        stop_at = (
            self.config.run.iterations
            if iterations is None
            else self.iteration + iterations
        )
        if self.display is not None:
            self.display.begin_run(self.iteration, stop_at)
        try:
            while self.iteration < stop_at:
                self.run_iteration()
                if (
                    self.display is not None
                    and self.iteration < stop_at
                ):
                    def release_for_pause() -> None:
                        _release_cuda_cache(
                            self.device,
                            phase="pause",
                        )
                        logger.info(
                            "training paused after committed iteration %d",
                            self.iteration,
                        )

                    paused = self.display.wait_if_paused(
                        self.iteration,
                        on_pause=release_for_pause,
                    )
                    if paused:
                        logger.info(
                            "training resumed at iteration %d",
                            self.iteration,
                        )
        finally:
            self.close()
        final_path = self.save_checkpoint(final=True)
        if self.display is not None:
            self.display.complete(final_path)
        logger.info("saved final checkpoint to %s", final_path)

    def close(self) -> None:
        """Release persistent collector processes and metric writers."""

        actors, self._actors = self._actors, None
        writer, self.writer = self.writer, None
        search_q_teacher, self._search_q_teacher = (
            self._search_q_teacher,
            None,
        )
        try:
            if actors is not None:
                actors.close()
        finally:
            if writer is not None:
                writer.close()
            self._checkpoint_opponents.clear()
            del search_q_teacher

    def run_iteration(self) -> dict[str, float]:
        next_iteration = self.iteration + 1
        iteration_started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        seed = (
            None
            if self.config.run.seed is None
            else self.config.run.seed + next_iteration
        )

        if self.display is not None:
            self.display.set_phase(
                "COLLECT",
                next_iteration,
                (
                    f"{self.config.collection.positions_per_iteration:,} "
                    "fresh positions"
                ),
            )
        trajectories, collection = collect_games_parallel(
            self.model,
            model_config=self.config.model,
            game_config=self.config.game,
            algorithm=self.config.algorithm,
            positions=self.config.collection.positions_per_iteration,
            parallel_games=self.config.collection.parallel_games,
            inference_batch_size=self.config.collection.inference_batch_size,
            inference_edge_budget=(
                self.config.collection.inference_edge_budget
            ),
            dense_position_cell_limit=(
                self.config.collection.dense_position_cell_limit
            ),
            workers=self.config.collection.workers,
            batch_timeout_ms=self.config.collection.batch_timeout_ms,
            device=self.device,
            precision=self.config.run.precision,
            seed=seed,
            actors=self._actors,
        )
        memory_metrics = _release_cuda_cache(
            self.device,
            phase="collection",
        )
        critic_calibration = terminal_played_q_calibration(trajectories)
        samples = flatten_trajectories(trajectories)
        search_q_samples: list[TrajectoryStep] = []
        search_q_metrics: dict[str, float] = {}
        search_q_diagnostic_seconds = 0.0
        if self.config.training.search_q_teacher_samples > 0:
            search_q_samples = _policy_diagnostic_slice(
                samples,
                self.config.training.search_q_teacher_samples,
            )
            if self.display is not None:
                self.display.set_phase(
                    "FIT",
                    next_iteration,
                    (
                        f"fixed search-Q labels // "
                        f"{len(search_q_samples):,} roots"
                    ),
                )
            search_started = time.monotonic()
            search_labels = self._get_search_q_teacher().label(
                [sample.state for sample in search_q_samples],
                seed=seed,
            )
            search_q_metrics.update(
                _attach_search_q_teacher_labels(
                    search_q_samples,
                    search_labels,
                )
            )
            search_q_metrics["search_q_teacher_search_seconds"] = (
                time.monotonic() - search_started
            )
            diagnostic_started = time.monotonic()
            before = _measure_auxiliary_q_diagnostics(
                self.model,
                search_q_samples,
                model_config=self.config.model,
                device=self.device,
                precision=self.config.run.precision,
                batch_size=self.config.training.batch_size,
                edge_budget=self.config.training.edge_budget,
            )
            search_q_diagnostic_seconds += (
                time.monotonic() - diagnostic_started
            )
            search_q_metrics["search_q_teacher_q_labels"] = before.pop(
                "search_q_teacher_q_labels"
            )
            search_q_metrics.update(
                {f"{key}_before": value for key, value in before.items()}
            )
        diagnostic_samples = _policy_diagnostic_slice(
            samples,
            self.config.training.policy_diagnostic_samples,
        )
        policy_diagnostic_seconds = 0.0
        policy_target_kl_collection = _stored_policy_target_kl(
            diagnostic_samples
        )
        policy_target_kl_before = None
        policy_target_top1_before = None
        if diagnostic_samples:
            diagnostic_started = time.monotonic()
            (
                policy_target_kl_before,
                policy_target_top1_before,
            ) = _measure_policy_target_diagnostics(
                self.model,
                diagnostic_samples,
                model_config=self.config.model,
                device=self.device,
                precision=self.config.run.precision,
                batch_size=self.config.training.batch_size,
                edge_budget=self.config.training.edge_budget,
            )
            policy_diagnostic_seconds += time.monotonic() - diagnostic_started
        if self.display is not None:
            fit_detail = f"one epoch across {len(samples):,} fresh examples"
            if self.config.training.critic_head_only:
                fit_detail = (
                    f"Q-head-only MC calibration across {len(samples):,} "
                    "fresh examples"
                )
            elif self.config.training.heads_only:
                fit_detail = (
                    f"policy/Q heads only across {len(samples):,} "
                    "fresh examples"
                )
            self.display.set_phase(
                "FIT",
                next_iteration,
                fit_detail,
            )
        if self.config.training.learning_rate_warmup_iterations > 0:
            scheduled_learning_rate = _learning_rate_for_iteration(
                self.config.training,
                next_iteration,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = scheduled_learning_rate
        critic_head_only = self.config.training.critic_head_only
        heads_only = self.config.training.heads_only
        if critic_head_only:
            fit_scope = _critic_head_only_scope(self.model)
        elif heads_only:
            fit_scope = _heads_only_scope(self.model)
        else:
            fit_scope = nullcontext()
        with fit_scope:
            training = train_epoch(
                self.model,
                self.optimizer,
                samples,
                model_config=self.config.model,
                device=self.device,
                precision=self.config.run.precision,
                batch_size=self.config.training.batch_size,
                edge_budget=self.config.training.edge_budget,
                grad_accumulation=self.config.training.grad_accumulation,
                q_loss_weight=self.config.training.q_loss_weight,
                max_grad_norm=self.config.training.max_grad_norm,
                seed=seed,
                prefetch_batches=self.config.training.prefetch_batches,
                optimize_policy=not critic_head_only,
            )
        training["critic_head_only"] = float(critic_head_only)
        training["heads_only"] = float(heads_only)
        training["shared_trunk_updates_enabled"] = float(
            not critic_head_only and not heads_only
        )
        learning_rates = {
            float(group["lr"]) for group in self.optimizer.param_groups
        }
        if len(learning_rates) != 1:
            raise RuntimeError(
                "KLENT metrics require one shared optimizer learning rate"
            )
        training["learning_rate"] = learning_rates.pop()
        if search_q_samples:
            diagnostic_started = time.monotonic()
            after_fit = _measure_auxiliary_q_diagnostics(
                self.model,
                search_q_samples,
                model_config=self.config.model,
                device=self.device,
                precision=self.config.run.precision,
                batch_size=self.config.training.batch_size,
                edge_budget=self.config.training.edge_budget,
            )
            search_q_diagnostic_seconds += (
                time.monotonic() - diagnostic_started
            )
            after_labels = after_fit.pop("search_q_teacher_q_labels")
            if after_labels != search_q_metrics["search_q_teacher_q_labels"]:
                raise RuntimeError(
                    "search-Q diagnostic label population changed during FIT"
                )
            search_q_metrics.update(
                {f"{key}_after_fit": value for key, value in after_fit.items()}
            )
            before_mse = search_q_metrics[
                "search_q_teacher_q_mse_before"
            ]
            after_fit_mse = search_q_metrics[
                "search_q_teacher_q_mse_after_fit"
            ]
            search_q_metrics["search_q_teacher_q_mse_fit_progress"] = (
                0.0
                if before_mse <= 1e-12
                else 1.0 - after_fit_mse / before_mse
            )
            final_diagnostics = after_fit
            refit_epochs = self.config.training.search_q_teacher_refit_epochs
            if refit_epochs > 0:
                if self.display is not None:
                    self.display.set_phase(
                        "FIT",
                        next_iteration,
                        "sparse fixed-search Q-head correction",
                    )
                search_q_metrics.update(
                    _refit_search_q_head(
                        self.model,
                        self.optimizer,
                        search_q_samples,
                        model_config=self.config.model,
                        device=self.device,
                        precision=self.config.run.precision,
                        batch_size=self.config.training.batch_size,
                        edge_budget=self.config.training.edge_budget,
                        epochs=refit_epochs,
                        max_grad_norm=self.config.training.max_grad_norm,
                        seed=seed,
                    )
                )
                diagnostic_started = time.monotonic()
                after_refit = _measure_auxiliary_q_diagnostics(
                    self.model,
                    search_q_samples,
                    model_config=self.config.model,
                    device=self.device,
                    precision=self.config.run.precision,
                    batch_size=self.config.training.batch_size,
                    edge_budget=self.config.training.edge_budget,
                )
                search_q_diagnostic_seconds += (
                    time.monotonic() - diagnostic_started
                )
                refit_labels = after_refit.pop(
                    "search_q_teacher_q_labels"
                )
                if refit_labels != after_labels:
                    raise RuntimeError(
                        "search-Q diagnostic label population changed during "
                        "Q-head refit"
                    )
                search_q_metrics.update(
                    {
                        f"{key}_after_refit": value
                        for key, value in after_refit.items()
                    }
                )
                after_refit_mse = search_q_metrics[
                    "search_q_teacher_q_mse_after_refit"
                ]
                search_q_metrics["search_q_teacher_q_mse_refit_progress"] = (
                    0.0
                    if after_fit_mse <= 1e-12
                    else 1.0 - after_refit_mse / after_fit_mse
                )
                final_diagnostics = after_refit
            search_q_metrics.update(
                {f"{key}_after": value for key, value in final_diagnostics.items()}
            )
            after_mse = search_q_metrics[
                "search_q_teacher_q_mse_after"
            ]
            search_q_metrics["search_q_teacher_q_mse_progress"] = (
                0.0 if before_mse <= 1e-12 else 1.0 - after_mse / before_mse
            )
            search_q_metrics["search_q_teacher_diagnostic_seconds"] = (
                search_q_diagnostic_seconds
            )
            training.update(search_q_metrics)
        if diagnostic_samples:
            if critic_head_only or heads_only:
                # The compiled FIT graph was specialized while the trunk (and
                # optionally policy head) was intentionally frozen. Asking it
                # for a hypothetical trunk gradient after restoring
                # requires_grad would both recompile and misrepresent the
                # update that just occurred. Record the actual protected
                # condition explicitly.
                training.update(
                    {
                        "trunk_gradient_diagnostic_examples": 0.0,
                        "trunk_gradient_diagnostic_q_labels": 0.0,
                        "trunk_gradient_diagnostic_seconds": 0.0,
                        "policy_trunk_grad_norm": 0.0,
                        "q_trunk_grad_norm": 0.0,
                        "policy_q_trunk_grad_cosine": 0.0,
                    }
                )
            else:
                training.update(
                    _measure_policy_q_trunk_gradients(
                        self.model,
                        diagnostic_samples,
                        model_config=self.config.model,
                        device=self.device,
                        precision=self.config.run.precision,
                        batch_size=self.config.training.batch_size,
                        edge_budget=self.config.training.edge_budget,
                        q_loss_weight=self.config.training.q_loss_weight,
                    )
                )
            diagnostic_started = time.monotonic()
            (
                policy_target_kl_after,
                policy_target_top1_after,
            ) = _measure_policy_target_diagnostics(
                self.model,
                diagnostic_samples,
                model_config=self.config.model,
                device=self.device,
                precision=self.config.run.precision,
                batch_size=self.config.training.batch_size,
                edge_budget=self.config.training.edge_budget,
            )
            policy_diagnostic_seconds += time.monotonic() - diagnostic_started
            if policy_target_kl_before is None:
                raise RuntimeError("policy target KL before fit was not measured")
            if policy_target_top1_before is None:
                raise RuntimeError(
                    "policy target top-1 agreement before fit was not measured"
                )
            policy_target_progress = (
                0.0
                if policy_target_kl_before <= 1e-12
                else 1.0
                - policy_target_kl_after / policy_target_kl_before
            )
            training.update(
                {
                    "policy_diagnostic_examples": float(
                        len(diagnostic_samples)
                    ),
                    "policy_diagnostic_seconds": policy_diagnostic_seconds,
                    "policy_target_kl_before": policy_target_kl_before,
                    "policy_target_kl_after": policy_target_kl_after,
                    "policy_target_progress": policy_target_progress,
                    "policy_target_top1_agreement_before": (
                        policy_target_top1_before
                    ),
                    "policy_target_top1_agreement_after": (
                        policy_target_top1_after
                    ),
                    "policy_target_top1_agreement_delta": (
                        policy_target_top1_after
                        - policy_target_top1_before
                    ),
                }
            )
            if policy_target_kl_collection is not None:
                training.update(
                    {
                        "policy_target_kl_collection": (
                            policy_target_kl_collection
                        ),
                        "policy_target_kl_sync_gap": (
                            policy_target_kl_before
                            - policy_target_kl_collection
                        ),
                    }
                )
        memory_metrics.update(
            _release_cuda_cache(
                self.device,
                phase="training",
            )
        )

        metrics: dict[str, float] = {
            "iteration": float(next_iteration),
            **{
                f"collection/{key}": float(value)
                for key, value in dataclasses.asdict(collection).items()
            },
            "collection/positions_per_second": (
                collection.positions / collection.elapsed_seconds
            ),
            **{
                f"collection/{key}": value
                for key, value in critic_calibration.items()
            },
            **{f"training/{key}": value for key, value in training.items()},
            **memory_metrics,
        }
        # Cross-entropy includes the improved target policy's irreducible
        # entropy. Subtracting it leaves the average forward KL from the
        # stored target to the evolving model as each sample is fitted.
        metrics["training/policy_excess_kl"] = max(
            0.0,
            training["policy_loss"] - collection.mean_entropy,
        )

        evaluation = self.config.evaluation
        pending_best_promotions: list[str] = []
        if (
            evaluation.interval > 0
            and evaluation.opponents
            and next_iteration % evaluation.interval == 0
        ):
            for opponent_index, opponent in enumerate(
                evaluation.opponents
            ):
                opponent_name = opponent.name or opponent.kind
                evaluation_kind = opponent.kind
                evaluation_checkpoint = opponent.checkpoint
                best_state: dict[str, object] | None = None
                if opponent.kind == "best_so_far":
                    best_state = self._load_best_so_far_state(
                        opponent_name,
                        opponent.checkpoint,
                    )
                    evaluation_kind = "checkpoint"
                    evaluation_checkpoint = str(best_state["checkpoint"])
                if self.display is not None:
                    self.display.set_phase(
                        "EVAL",
                        next_iteration,
                        (
                            f"{opponent_name} // {opponent.games} games"
                        ),
                    )
                opponent_game_config = dataclasses.replace(
                    self.config.game,
                    placement_radius=(
                        opponent.placement_radius
                        or self.config.game.placement_radius
                    ),
                )
                try:
                    result = evaluate_opponent(
                        evaluation_kind,
                        self.model,
                        model_config=self.config.model,
                        game_config=opponent_game_config,
                        games=opponent.games,
                        depth=opponent.depth,
                        algorithm=self.config.algorithm,
                        mcts_simulations=opponent.mcts_simulations,
                        mcts_actions=opponent.mcts_actions,
                        device=self.device,
                        precision=self.config.run.precision,
                        checkpoint=evaluation_checkpoint,
                        checkpoint_cache=self._checkpoint_opponents,
                        opponent_mcts_simulations=(
                            opponent.opponent_mcts_simulations
                        ),
                        opponent_mcts_actions=opponent.opponent_mcts_actions,
                        iteration=next_iteration,
                        checkpoint_dirs=tuple(
                            self._checkpoint_history_dirs
                        ),
                        lag_iterations=opponent.lag_iterations,
                        opening_plies=evaluation.opening_plies,
                        opening_temperature=evaluation.opening_temperature,
                        opening_generator=evaluation.opening_generator,
                        seed=(
                            None
                            if seed is None
                            else seed + opponent_index * 1_000_003
                        ),
                    )
                except FileNotFoundError as error:
                    logger.warning(
                        "skipping %s evaluation: %s",
                        opponent.kind,
                        error,
                    )
                    continue
                prefix = f"evaluation/{opponent_name}"
                metrics.update(
                    {
                        f"{prefix}/{key}": float(value)
                        for key, value in dataclasses.asdict(result).items()
                    }
                )
                if opponent.depth > 0:
                    metrics[f"{prefix}/configured_depth"] = float(
                        opponent.depth
                    )
                metrics[f"{prefix}/placement_radius"] = float(
                    opponent_game_config.placement_radius
                )
                metrics[f"{prefix}/mcts_simulations"] = float(
                    opponent.mcts_simulations
                )
                metrics[f"{prefix}/mcts_actions"] = float(
                    opponent.mcts_actions
                )
                if opponent.kind in {
                    "checkpoint",
                    "lagged",
                    "best_so_far",
                }:
                    metrics[f"{prefix}/opening_plies"] = float(
                        evaluation.opening_plies
                    )
                    metrics[f"{prefix}/opening_temperature"] = float(
                        evaluation.opening_temperature
                    )
                    metrics[f"{prefix}/opponent_mcts_simulations"] = float(
                        (
                            opponent.mcts_simulations
                            if opponent.kind == "lagged"
                            else opponent.opponent_mcts_simulations
                        )
                    )
                    metrics[f"{prefix}/opponent_mcts_actions"] = float(
                        (
                            opponent.mcts_actions
                            if opponent.kind == "lagged"
                            else opponent.opponent_mcts_actions
                        )
                    )
                if opponent.kind == "lagged":
                    metrics[f"{prefix}/configured_lag_iterations"] = float(
                        opponent.lag_iterations
                    )
                    metrics[f"{prefix}/opponent_iteration"] = float(
                        next_iteration - opponent.lag_iterations
                    )
                if opponent.kind == "best_so_far":
                    assert best_state is not None
                    best_iteration = best_state.get("iteration")
                    if isinstance(best_iteration, int):
                        metrics[f"{prefix}/opponent_iteration"] = float(
                            best_iteration
                        )
                    threshold = opponent.best_promotion_win_rate
                    promoted = (
                        result.wins + result.losses > 0
                        and result.win_rate_decided >= threshold
                    )
                    metrics[f"{prefix}/promotion_win_rate"] = float(
                        threshold
                    )
                    metrics[f"{prefix}/promoted"] = float(promoted)
                    if promoted:
                        pending_best_promotions.append(opponent_name)
                        logger.info(
                            "best_so_far=%s promotion pending: iteration=%d "
                            "win_rate_decided=%.3f threshold=%.3f",
                            opponent_name,
                            next_iteration,
                            result.win_rate_decided,
                            threshold,
                        )
                logger.info(
                    "evaluation=%s games=%d wins=%d losses=%d "
                    "truncations=%d win_rate_decided=%.3f",
                    opponent_name,
                    result.games,
                    result.wins,
                    result.losses,
                    result.truncations,
                    result.win_rate_decided,
                )
            metrics.update(
                _release_cuda_cache(
                    self.device,
                    phase="evaluation",
                )
            )

        if self.display is not None:
            self.display.set_phase(
                "COMMIT",
                next_iteration,
                "persisting metrics + checkpoint state",
            )
        self.iteration = next_iteration
        metrics["iteration_seconds"] = time.monotonic() - iteration_started
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        if self.writer is not None:
            for key, value in metrics.items():
                if key != "iteration":
                    self.writer.add_scalar(key, value, self.iteration)
            self.writer.flush()

        interval = self.config.run.checkpoint_interval
        checkpoint_due = interval > 0 and self.iteration % interval == 0
        committed_checkpoint: Path | None = None
        if checkpoint_due or pending_best_promotions:
            committed_checkpoint = self.save_checkpoint()
        if pending_best_promotions:
            assert committed_checkpoint is not None
            for opponent_name in pending_best_promotions:
                self._write_best_so_far_state(
                    opponent_name,
                    committed_checkpoint,
                    self.iteration,
                )
                logger.info(
                    "best_so_far=%s promoted checkpoint=%s",
                    opponent_name,
                    committed_checkpoint,
                )
        if self.display is not None:
            self.display.update_metrics(metrics)
        logger.info(
            "iteration=%d games=%d positions=%d discarded_positions=%d "
            "truncations=%d(horizon=%d spatial=%d chunk=%d) workers=%d "
            "policy=%.4f excess_kl=%.4f q=%.4f reverse_kl=%.4f "
            "collect=%.1fs fit=%.1fs total=%.1fs",
            self.iteration,
            collection.games,
            collection.positions,
            collection.discarded_positions,
            collection.truncations,
            collection.horizon_truncations,
            collection.spatial_truncations,
            collection.chunk_truncations,
            collection.worker_processes,
            training["policy_loss"],
            metrics["training/policy_excess_kl"],
            training["q_loss"],
            collection.mean_reverse_kl,
            collection.elapsed_seconds,
            training["elapsed_seconds"],
            metrics["iteration_seconds"],
        )
        return metrics
