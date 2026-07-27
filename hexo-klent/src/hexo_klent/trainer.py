"""Synchronous collect-then-fit KLENT training loop."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from hexo_klent.actor import (
    SharedInferenceActors,
    TrajectoryStep,
    _autocast,
    collect_games_parallel,
    flatten_trajectories,
)
from hexo_klent.batching import (
    move_batch_to_device,
    prepare_graph_batches,
    raster_shape,
)
from hexo_klent.config import Config
from hexo_klent.evaluation import (
    CheckpointOpponentCache,
    evaluate_opponent,
)
from hexo_klent.model import (
    DenseAxisKlentNet,
    KlentNet,
    compile_klent_forward,
    is_dense_axis_config,
    load_production_axis_weights,
    make_klent_net,
)

if TYPE_CHECKING:
    from hexo_klent.tui import TrainingDashboard

logger = logging.getLogger(__name__)

_GIB = float(1024**3)
_CACHE_CLEAR_RESERVED_MB_ENV = "HEXO_CACHE_CLEAR_RESERVED_MB"
_CACHE_CLEAR_CHECK_EVERY_ENV = "HEXO_CACHE_CLEAR_CHECK_EVERY"
_DEFAULT_CACHE_CLEAR_RESERVED_MB = 8192.0
_DEFAULT_CACHE_CLEAR_CHECK_EVERY = 128


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
    num_segments: int,
) -> torch.Tensor:
    """Compute independent stable log-softmaxes over flat segments."""

    if logits.ndim != 1 or segment_ids.ndim != 1:
        raise ValueError("logits and segment_ids must be one-dimensional")
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have equal length")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")

    # Detaching the per-segment shift is mathematically neutral and avoids
    # differentiating through the amax reduction. The remaining expression is
    # exactly the usual numerically stable log-softmax.
    maxima = torch.full(
        (num_segments,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    ).scatter_reduce(
        0,
        segment_ids,
        logits.detach(),
        reduce="amax",
        include_self=True,
    )
    shifted = logits - maxima.index_select(0, segment_ids)
    normalizers = torch.zeros(
        num_segments,
        dtype=logits.dtype,
        device=logits.device,
    ).scatter_add(0, segment_ids, shifted.exp())
    return shifted - normalizers.log().index_select(0, segment_ids)


def _flat_training_targets(
    samples: list[TrajectoryStep],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
    chosen = offsets + torch.tensor(
        [sample.action_index for sample in samples],
        dtype=torch.long,
    )
    if bool((chosen < offsets).any()) or bool((chosen >= offsets + counts).any()):
        raise ValueError("KLENT sample action index is outside its policy")

    target_q = torch.tensor(
        [sample.return_target for sample in samples],
        dtype=torch.float32,
    )
    target_top1 = sum(
        int(sample.target_policy.argmax().item()) == sample.action_index
        for sample in samples
    )
    return target_policy, segment_ids, chosen, target_q, target_top1


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
        "total_loss": torch.zeros((), device=device),
    }
    grad_norms: list[torch.Tensor] = []
    target_top1_total = 0
    seen = 0
    microbatches = 0
    optimizer_steps = 0
    model.train()
    cache_pressure = _CudaCachePressureGate.from_environment(device)

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
            optimizer.zero_grad(set_to_none=True)

            for batch_cpu, packed_samples, packed_targets in optimizer_group:
                (
                    target_policy_cpu,
                    segment_ids_cpu,
                    chosen_cpu,
                    target_q_cpu,
                    target_top1,
                ) = packed_targets
                batch = move_batch_to_device(batch_cpu, device)
                target_policy = target_policy_cpu.to(device)
                segment_ids = segment_ids_cpu.to(device)
                chosen = chosen_cpu.to(device)
                target_q = target_q_cpu.to(device)
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
                        count,
                    )
                    policy_loss = (
                        -(target_policy * log_policy).sum() / count
                    )
                    predicted_q = output.q_values.index_select(
                        0, chosen
                    ).float()
                    q_loss = F.mse_loss(predicted_q, target_q)
                    total_loss = policy_loss + q_loss_weight * q_loss

                # Each microbatch loss is already a per-example mean. Its
                # population share reconstructs the exact outer-batch mean.
                (total_loss * (count / group_examples)).backward()
                seen += count
                microbatches += 1
                totals["policy_loss"].add_(policy_loss.detach() * count)
                totals["q_loss"].add_(q_loss.detach() * count)
                totals["total_loss"].add_(total_loss.detach() * count)
                target_top1_total += target_top1
                del (
                    batch,
                    target_policy,
                    segment_ids,
                    chosen,
                    target_q,
                    output,
                    log_policy,
                    policy_loss,
                    predicted_q,
                    q_loss,
                    total_loss,
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
            optimizer.step()
            optimizer_steps += 1
            grad_norms.append(grad_norm.detach())

    summary = torch.cat(
        (
            torch.stack(
                [
                    totals["policy_loss"],
                    totals["q_loss"],
                    totals["total_loss"],
                ]
            ),
            torch.stack(grad_norms),
        )
    ).cpu()
    (
        policy_loss_total,
        q_loss_total,
        total_loss_total,
    ) = summary[:3].tolist()
    gradient_stats = _gradient_clip_statistics(
        summary[3:],
        max_grad_norm,
    )
    elapsed_seconds = time.monotonic() - started_at

    return {
        "examples": float(seen),
        "microbatches": float(microbatches),
        "optimizer_steps": float(optimizer_steps),
        "mean_microbatch_size": seen / microbatches,
        "mean_optimizer_batch_size": seen / optimizer_steps,
        "mean_microbatches_per_step": microbatches / optimizer_steps,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": seen / elapsed_seconds,
        "policy_loss": policy_loss_total / seen,
        "q_loss": q_loss_total / seen,
        "total_loss": total_loss_total / seen,
        **gradient_stats,
        "played_action_target_top1": target_top1_total / seen,
        **cache_pressure.metrics(),
    }


class Trainer:
    """Own the model, optimizer, artifacts, and reference KLENT iteration."""

    def __init__(
        self,
        config: Config,
        *,
        tensorboard: bool = True,
        resume: str | Path | None = None,
        init_from: str | Path | None = None,
        display: TrainingDashboard | None = None,
    ) -> None:
        if resume is not None and init_from is not None:
            raise ValueError("resume and init_from are mutually exclusive")
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
            compile_klent_forward(self.model)
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
        self._checkpoint_history_dirs = [self.checkpoint_dir.resolve()]
        self.initial_checkpoint: dict[str, object] | None = None
        try:
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(self.output_dir / "tensorboard")
            if resume is not None:
                self.load_checkpoint(resume)
            elif init_from is not None:
                self.initialize_from_production(init_from)
            if config.collection.workers > 1:
                self._actors = SharedInferenceActors(
                    config.collection.workers
                )
        except BaseException:
            self.close()
            raise

    def load_checkpoint(self, path: str | Path) -> None:
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
        self.iteration = int(checkpoint["iteration"])
        initial_checkpoint = checkpoint.get("initial_checkpoint")
        if isinstance(initial_checkpoint, dict):
            self.initial_checkpoint = initial_checkpoint
        logger.info("resumed %s at iteration %d", path, self.iteration)

    def initialize_from_production(self, path: str | Path) -> None:
        """Initialize dense KLENT from a production Axis-GINE Q-head model."""

        if not isinstance(self.model, DenseAxisKlentNet):
            raise ValueError(
                "production checkpoint conversion requires "
                "model.architecture='dense_axis'"
            )
        from hexo_a0.config import model_config_from_checkpoint

        path = Path(path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"production checkpoint {path} is not a dict")
        source_config = model_config_from_checkpoint(checkpoint)
        if not bool(getattr(source_config, "q_head", False)):
            raise ValueError("production checkpoint does not contain a trained Q head")
        compatibility_fields = (
            "hidden_dim",
            "num_layers",
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
        mismatches = [
            f"{name}: source={getattr(source_config, name)!r}, "
            f"target={getattr(self.config.model, name)!r}"
            for name in compatibility_fields
            if getattr(source_config, name) != getattr(self.config.model, name)
        ]
        if mismatches:
            raise ValueError(
                "production checkpoint architecture does not match dense KLENT: "
                + "; ".join(mismatches)
            )
        report = load_production_axis_weights(self.model, checkpoint)
        self.initial_checkpoint = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train_steps": checkpoint.get("train_steps"),
            "copied_tensors": len(report.copied),
        }
        logger.info(
            "initialized dense KLENT from %s (%d tensors, train_steps=%s); "
            "optimizer starts fresh",
            path,
            len(report.copied),
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
        try:
            if actors is not None:
                actors.close()
        finally:
            if writer is not None:
                writer.close()
            self._checkpoint_opponents.clear()

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
        samples = flatten_trajectories(trajectories)
        if self.display is not None:
            self.display.set_phase(
                "FIT",
                next_iteration,
                f"one epoch across {len(samples):,} fresh examples",
            )
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
        if (
            evaluation.interval > 0
            and evaluation.opponents
            and next_iteration % evaluation.interval == 0
        ):
            for opponent_index, opponent in enumerate(
                evaluation.opponents
            ):
                opponent_name = opponent.name or opponent.kind
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
                        opponent.kind,
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
                        checkpoint=opponent.checkpoint,
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
                if opponent.kind in {"checkpoint", "lagged"}:
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
        if interval > 0 and self.iteration % interval == 0:
            self.save_checkpoint()
        if self.display is not None:
            self.display.update_metrics(metrics)
        logger.info(
            "iteration=%d games=%d positions=%d "
            "truncations=%d(horizon=%d spatial=%d chunk=%d) workers=%d "
            "policy=%.4f excess_kl=%.4f q=%.4f reverse_kl=%.4f "
            "collect=%.1fs fit=%.1fs total=%.1fs",
            self.iteration,
            collection.games,
            collection.positions,
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
