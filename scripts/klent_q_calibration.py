#!/usr/bin/env python3
"""Compare played-action Q calibration between two KLENT checkpoints.

Each checkpoint first generates an independent on-policy self-play corpus.
Both checkpoints are then evaluated on every played action in both corpora.
This 2x2 design distinguishes a changed state distribution from a changed
critic. Only terminal games are scored: horizon-, spatial-, and position-
budget-truncated fragments have no observed outcome and are excluded.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexo_klent.actor import (
    SharedInferenceActors,
    Trajectory,
    TrajectoryStep,
    collect_games_parallel,
)
from hexo_klent.batching import move_batch_to_device, prepare_graph_batches
from hexo_klent.config import GameConfig, load_config
from hexo_klent.mcts_adapter import load_checkpoint
from hexo_klent.model import compile_klent_forward


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="KLENT checkpoint; pass exactly twice",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="short label; pass once per checkpoint (defaults to iteration)",
    )
    parser.add_argument("--positions", type=int, default=32_768)
    parser.add_argument("--parallel-games", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--inference-edge-budget", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-timeout-ms", type=float)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--eval-edge-budget", type=int)
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=1_000,
        help="diagnostic horizon; truncated games are still excluded",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "bf16"))
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        parser.error("pass exactly two --checkpoint arguments")
    if args.label is not None and len(args.label) != 2:
        parser.error("pass exactly two --label arguments")
    if args.positions <= 0 or args.parallel_games <= 0:
        parser.error("--positions and --parallel-games must be positive")
    if args.parallel_games > args.positions:
        parser.error("--parallel-games cannot exceed --positions")
    return args


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def _terminal_samples(
    trajectories: list[Trajectory],
) -> tuple[list[TrajectoryStep], np.ndarray, np.ndarray, list[int]]:
    samples: list[TrajectoryStep] = []
    outcomes: list[float] = []
    game_ids: list[int] = []
    game_lengths: list[int] = []
    for trajectory in trajectories:
        if trajectory.truncated or trajectory.winner not in {"P1", "P2"}:
            continue
        game_id = len(game_lengths)
        game_lengths.append(len(trajectory.steps))
        for step in trajectory.steps:
            samples.append(step)
            outcomes.append(1.0 if step.player == trajectory.winner else -1.0)
            game_ids.append(game_id)
    if not samples:
        raise RuntimeError("collection produced no completed-game positions")
    return (
        samples,
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(game_ids, dtype=np.int64),
        game_lengths,
    )


def _played_q(
    model,
    model_config,
    samples: list[TrajectoryStep],
    *,
    batch_size: int,
    edge_budget: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            states = [sample.state for sample in chunk]
            for batch_cpu, state_slice in prepare_graph_batches(
                states,
                model_config=model_config,
                edge_budget=edge_budget,
            ):
                packed = chunk[state_slice]
                counts = batch_cpu.legal_counts.to(dtype=torch.long)
                offsets = counts.cumsum(0) - counts
                chosen = offsets + torch.tensor(
                    [sample.action_index for sample in packed],
                    dtype=torch.long,
                )
                batch = move_batch_to_device(batch_cpu, device)
                chosen = chosen.to(device)
                with _autocast(device, precision):
                    output = model.forward_fit(batch, chosen)
                predictions.append(
                    output.q_values.float().cpu().numpy().astype(
                        np.float64, copy=False
                    )
                )
    result = np.concatenate(predictions)
    if result.shape != (len(samples),):
        raise RuntimeError(
            f"Q output shape {result.shape} does not match {len(samples)} samples"
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("played-action Q contains non-finite values")
    print(
        f"    scored {len(samples):,} actions in "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )
    return result


def _calibration_bins(q: np.ndarray, outcome: np.ndarray) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    edges = np.linspace(-1.0, 1.0, 11)
    indices = np.clip(np.digitize(q, edges[1:-1], right=False), 0, 9)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = indices == index
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(mask.sum()),
                "mean_q": float(q[mask].mean()) if mask.any() else None,
                "mean_outcome": (
                    float(outcome[mask].mean()) if mask.any() else None
                ),
            }
        )
    return bins


def _summary(q: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    error = q - outcome
    q_centered = q - q.mean()
    outcome_centered = outcome - outcome.mean()
    q_variance = float(np.dot(q_centered, q_centered))
    covariance = float(np.dot(q_centered, outcome_centered))
    correlation_denominator = math.sqrt(
        q_variance * float(np.dot(outcome_centered, outcome_centered))
    )
    bins = _calibration_bins(q, outcome)
    ece = sum(
        item["n"] / len(q) * abs(item["mean_q"] - item["mean_outcome"])
        for item in bins
        if item["n"]
    )
    return {
        "positions": len(q),
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "bias_q_minus_outcome": float(error.mean()),
        "mean_q": float(q.mean()),
        "mean_abs_q": float(np.abs(q).mean()),
        "mean_outcome": float(outcome.mean()),
        "sign_accuracy": float(np.mean(q * outcome > 0.0)),
        "pearson_r": (
            covariance / correlation_denominator
            if correlation_denominator > 0.0
            else None
        ),
        "calibration_intercept": (
            float(outcome.mean()) - covariance / q_variance * float(q.mean())
            if q_variance > 0.0
            else None
        ),
        "calibration_slope": covariance / q_variance if q_variance > 0.0 else None,
        "ece_10_fixed_bins": float(ece),
        "mean_q_on_winning_actions": float(q[outcome > 0].mean()),
        "mean_q_on_losing_actions": float(q[outcome < 0].mean()),
        "calibration_bins": bins,
    }


def _paired_cluster_delta(
    q_a: np.ndarray,
    q_b: np.ndarray,
    outcome: np.ndarray,
    game_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Game-cluster bootstrap of B-A error on one fixed corpus."""

    games = int(game_ids.max()) + 1
    counts = np.bincount(game_ids, minlength=games).astype(np.float64)
    mse_a = np.bincount(
        game_ids, weights=np.square(q_a - outcome), minlength=games
    )
    mse_b = np.bincount(
        game_ids, weights=np.square(q_b - outcome), minlength=games
    )
    mae_a = np.bincount(
        game_ids, weights=np.abs(q_a - outcome), minlength=games
    )
    mae_b = np.bincount(
        game_ids, weights=np.abs(q_b - outcome), minlength=games
    )

    def point(sum_a: np.ndarray, sum_b: np.ndarray) -> float:
        return float((sum_b.sum() - sum_a.sum()) / counts.sum())

    result = {
        "delta_mse_b_minus_a": point(mse_a, mse_b),
        "delta_mae_b_minus_a": point(mae_a, mae_b),
    }
    if samples <= 0:
        return result

    rng = np.random.default_rng(seed)
    boot_mse = np.empty(samples, dtype=np.float64)
    boot_mae = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, games, size=games)
        denominator = counts[selected].sum()
        boot_mse[index] = (
            mse_b[selected].sum() - mse_a[selected].sum()
        ) / denominator
        boot_mae[index] = (
            mae_b[selected].sum() - mae_a[selected].sum()
        ) / denominator
    result.update(
        {
            "delta_mse_ci95_low": float(np.quantile(boot_mse, 0.025)),
            "delta_mse_ci95_high": float(np.quantile(boot_mse, 0.975)),
            "delta_mae_ci95_low": float(np.quantile(boot_mae, 0.025)),
            "delta_mae_ci95_high": float(np.quantile(boot_mae, 0.975)),
        }
    )
    return result


def _print_matrix(results: dict[str, Any], labels: list[str]) -> None:
    print("\nPlayed-action Q versus terminal outcome (position weighted)")
    print(
        f"{'corpus':>12}  {'critic':>12}  {'n':>8}  {'mse':>7}  {'mae':>7}  "
        f"{'bias':>7}  {'sign':>7}  {'r':>7}  {'slope':>7}  {'ece':>7}"
    )
    for corpus_label in labels:
        for critic_label in labels:
            row = results["corpora"][corpus_label]["critics"][critic_label]
            print(
                f"{corpus_label:>12}  {critic_label:>12}  "
                f"{row['positions']:8,d}  {row['mse']:7.4f}  "
                f"{row['mae']:7.4f}  {row['bias_q_minus_outcome']:+7.4f}  "
                f"{row['sign_accuracy']:7.3f}  {row['pearson_r']:7.3f}  "
                f"{row['calibration_slope']:7.3f}  "
                f"{row['ece_10_fixed_bins']:7.4f}"
            )

    print(f"\nPaired critic delta on identical states ({labels[1]} - {labels[0]})")
    for corpus_label in labels:
        delta = results["corpora"][corpus_label]["paired_delta"]
        print(
            f"  {corpus_label}: delta MSE {delta['delta_mse_b_minus_a']:+.5f} "
            f"[95% {delta['delta_mse_ci95_low']:+.5f}, "
            f"{delta['delta_mse_ci95_high']:+.5f}], "
            f"delta MAE {delta['delta_mae_b_minus_a']:+.5f}"
        )
    print("  Negative delta means the second checkpoint is better calibrated.")


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    device = torch.device(args.device or config.run.device)
    precision = args.precision or config.run.precision
    compile_model = config.run.compile if args.compile is None else args.compile
    seed = config.run.seed if args.seed is None else args.seed
    if seed is None:
        seed = 0
    inference_batch_size = (
        args.inference_batch_size or config.collection.inference_batch_size
    )
    inference_edge_budget = (
        args.inference_edge_budget or config.collection.inference_edge_budget
    )
    eval_edge_budget = args.eval_edge_budget or config.training.edge_budget
    workers = args.workers or config.collection.workers
    batch_timeout_ms = (
        config.collection.batch_timeout_ms
        if args.batch_timeout_ms is None
        else args.batch_timeout_ms
    )
    game_config = dataclasses.replace(
        config.game,
        rollout_horizon=args.rollout_horizon,
    )
    if not isinstance(game_config, GameConfig):
        raise TypeError("unexpected game configuration")

    loaded = [load_checkpoint(path, device) for path in args.checkpoint]
    labels = args.label or [str(item.iteration) for item in loaded]
    if len(set(labels)) != 2:
        raise ValueError("checkpoint labels must be distinct")
    networks = [item.model.network for item in loaded]
    if compile_model:
        print("Compiling KLENT inference/FIT paths...", flush=True)
        for network in networks:
            compile_klent_forward(network)

    results: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "checkpoints": {
            label: {
                "path": str(path.resolve()),
                "iteration": str(item.iteration),
            }
            for label, path, item in zip(
                labels, args.checkpoint, loaded, strict=True
            )
        },
        "settings": {
            "positions_per_corpus": args.positions,
            "parallel_games": args.parallel_games,
            "workers": workers,
            "rollout_horizon": args.rollout_horizon,
            "seed": seed,
            "device": str(device),
            "precision": precision,
            "compile": compile_model,
        },
        "corpora": {},
    }

    actors = SharedInferenceActors(workers) if workers > 1 else None
    try:
        for corpus_index, corpus_label in enumerate(labels):
            print(
                f"\nCollecting {corpus_label} on-policy corpus "
                f"({args.positions:,} positions)...",
                flush=True,
            )
            trajectories, stats = collect_games_parallel(
                networks[corpus_index],
                model_config=loaded[corpus_index].model_config,
                game_config=game_config,
                algorithm=config.algorithm,
                positions=args.positions,
                parallel_games=args.parallel_games,
                inference_batch_size=inference_batch_size,
                inference_edge_budget=inference_edge_budget,
                workers=workers,
                batch_timeout_ms=batch_timeout_ms,
                device=device,
                precision=precision,
                seed=seed + corpus_index * 10_000_019,
                actors=actors,
            )
            samples, outcomes, game_ids, game_lengths = _terminal_samples(
                trajectories
            )
            excluded_positions = stats.positions - len(samples)
            print(
                f"  complete games={len(game_lengths):,}, "
                f"scored positions={len(samples):,}, "
                f"excluded truncated positions={excluded_positions:,}",
                flush=True,
            )

            predictions: dict[str, np.ndarray] = {}
            critic_results: dict[str, Any] = {}
            for critic_index, critic_label in enumerate(labels):
                print(f"  Evaluating critic {critic_label}...", flush=True)
                q = _played_q(
                    networks[critic_index],
                    loaded[critic_index].model_config,
                    samples,
                    batch_size=args.eval_batch_size,
                    edge_budget=eval_edge_budget,
                    device=device,
                    precision=precision,
                )
                predictions[critic_label] = q
                critic_results[critic_label] = _summary(q, outcomes)

            results["corpora"][corpus_label] = {
                "collection": dataclasses.asdict(stats),
                "completed_games": len(game_lengths),
                "completed_positions": len(samples),
                "excluded_truncated_positions": excluded_positions,
                "mean_completed_game_length": float(np.mean(game_lengths)),
                "critics": critic_results,
                "paired_delta": _paired_cluster_delta(
                    predictions[labels[0]],
                    predictions[labels[1]],
                    outcomes,
                    game_ids,
                    samples=args.bootstrap_samples,
                    seed=seed + corpus_index,
                ),
            }
            del trajectories, samples, outcomes, game_ids, predictions
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if actors is not None:
            actors.close()

    _print_matrix(results, labels)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
