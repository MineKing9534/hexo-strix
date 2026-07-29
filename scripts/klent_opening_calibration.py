#!/usr/bin/env python3
"""Measure KLENT opening Q calibration under stratified forced actions.

The ordinary on-policy calibration only observes actions selected by the
current policy. This diagnostic forces the first selectable move at each
hex-distance ring, cycles evenly through orientations within that ring, and
then resumes normal KLENT self-play. It compares the frozen opening Q with the
empirical terminal outcome from the opening player's perspective.
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

import torch

from hexo_klent.actor import Trajectory, _collect_with_inference
from hexo_klent.batching import (
    move_batch_to_device,
    order_states_for_batching,
    prepare_graph_batches,
    restore_state_order,
)
from hexo_klent.config import load_config
from hexo_klent.mcts_adapter import load_checkpoint
from hexo_klent.model import compile_klent_forward, improved_policy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--label", action="append")
    parser.add_argument("--positions-per-ring", type=int, default=8_192)
    parser.add_argument("--parallel-games", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--inference-edge-budget", type=int)
    parser.add_argument("--rollout-horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "bf16"))
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.label is not None and len(args.label) != len(args.checkpoint):
        parser.error("pass one --label per --checkpoint")
    if args.positions_per_ring <= 0 or args.parallel_games <= 0:
        parser.error("position and lane counts must be positive")
    if args.parallel_games > args.positions_per_ring:
        parser.error("--parallel-games cannot exceed --positions-per-ring")
    return args


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def _hex_distance(coord: tuple[int, int]) -> int:
    q, r = coord
    return max(abs(q), abs(r), abs(q + r))


def _network_infer(
    model,
    model_config,
    states: list[object],
    *,
    inference_batch_size: int,
    edge_budget: int,
    device: torch.device,
    precision: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logits: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    for start in range(0, len(states), inference_batch_size):
        chunk = states[start : start + inference_batch_size]
        ordered, source_indices = order_states_for_batching(chunk, model_config)
        ordered_logits: list[torch.Tensor] = []
        ordered_q: list[torch.Tensor] = []
        for batch_cpu, _state_slice in prepare_graph_batches(
            ordered,
            model_config=model_config,
            edge_budget=edge_budget,
        ):
            batch = move_batch_to_device(batch_cpu, device)
            with torch.inference_mode(), _autocast(device, precision):
                output = model.forward_batch(batch)
            counts = [int(item) for item in output.legal_counts.cpu().tolist()]
            ordered_logits.extend(output.policy_logits.split(counts))
            ordered_q.extend(output.q_values.split(counts))
        logits.extend(restore_state_order(ordered_logits, source_indices))
        q_values.extend(restore_state_order(ordered_q, source_indices))
    return logits, q_values


def _opening_outputs(
    model,
    model_config,
    algorithm,
    state,
    *,
    edge_budget: int,
    device: torch.device,
    precision: str,
) -> dict[tuple[int, int], dict[str, float]]:
    logits, q_values = _network_infer(
        model,
        model_config,
        [state],
        inference_batch_size=1,
        edge_budget=edge_budget,
        device=device,
        precision=precision,
    )
    logits_f = logits[0].float()
    q_f = q_values[0].float()
    raw = torch.softmax(logits_f, dim=0)
    target = improved_policy(
        logits_f,
        q_f,
        alpha=algorithm.alpha,
        beta=algorithm.beta,
    )
    return {
        tuple(coord): {
            "q": float(q_value),
            "raw_probability": float(raw_probability),
            "improved_probability": float(target_probability),
        }
        for coord, q_value, raw_probability, target_probability in zip(
            state.legal_moves(), q_f, raw, target, strict=True
        )
    }


def _wilson_outcome_interval(wins: int, games: int) -> tuple[float, float]:
    """Wilson score interval mapped from win probability to [-1, 1]."""

    if games <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    probability = wins / games
    denominator = 1.0 + z * z / games
    center = (probability + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / games
        + z * z / (4.0 * games * games)
    ) / denominator
    return 2.0 * (center - radius) - 1.0, 2.0 * (center + radius) - 1.0


def _ring_result(
    trajectories: list[Trajectory],
    opening: dict[tuple[int, int], dict[str, float]],
    ring: int,
) -> dict[str, Any]:
    games = 0
    wins = 0
    lengths = 0
    q_sum = 0.0
    raw_probability_sum = 0.0
    improved_probability_sum = 0.0
    orientation_counts: dict[str, int] = {}
    for trajectory in trajectories:
        if trajectory.truncated or trajectory.winner not in {"P1", "P2"}:
            continue
        first = trajectory.steps[0]
        legal = first.state.legal_moves()
        coord = tuple(legal[first.action_index])
        if _hex_distance(coord) != ring:
            raise RuntimeError(
                f"forced ring {ring} produced opening coordinate {coord}"
            )
        prediction = opening[coord]
        games += 1
        won = trajectory.winner == first.player
        wins += int(won)
        lengths += len(trajectory.steps)
        q_sum += prediction["q"]
        raw_probability_sum += prediction["raw_probability"]
        improved_probability_sum += prediction["improved_probability"]
        key = f"{coord[0]},{coord[1]}"
        orientation_counts[key] = orientation_counts.get(key, 0) + 1
    if games <= 0:
        raise RuntimeError(f"ring {ring} produced no completed games")
    empirical_outcome = 2.0 * wins / games - 1.0
    predicted_q = q_sum / games
    ci_low, ci_high = _wilson_outcome_interval(wins, games)
    return {
        "ring": ring,
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "mean_game_length": lengths / games,
        "predicted_q": predicted_q,
        "empirical_outcome": empirical_outcome,
        "outcome_ci95_low": ci_low,
        "outcome_ci95_high": ci_high,
        "q_minus_outcome": predicted_q - empirical_outcome,
        "mean_raw_action_probability": raw_probability_sum / games,
        "mean_improved_action_probability": (
            improved_probability_sum / games
        ),
        "orientation_counts": orientation_counts,
    }


def _weighted_regression(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    weights = torch.tensor([row["games"] for row in rows], dtype=torch.float64)
    predicted = torch.tensor(
        [row["predicted_q"] for row in rows], dtype=torch.float64
    )
    observed = torch.tensor(
        [row["empirical_outcome"] for row in rows], dtype=torch.float64
    )
    weight_sum = weights.sum()
    predicted_mean = (weights * predicted).sum() / weight_sum
    observed_mean = (weights * observed).sum() / weight_sum
    centered_predicted = predicted - predicted_mean
    centered_observed = observed - observed_mean
    covariance = (weights * centered_predicted * centered_observed).sum()
    predicted_variance = (weights * centered_predicted.square()).sum()
    observed_variance = (weights * centered_observed.square()).sum()
    denominator = torch.sqrt(predicted_variance * observed_variance)
    return {
        "weighted_mae": float(
            (weights * (predicted - observed).abs()).sum() / weight_sum
        ),
        "weighted_rmse": float(
            torch.sqrt(
                (weights * (predicted - observed).square()).sum() / weight_sum
            )
        ),
        "calibration_slope": (
            float(covariance / predicted_variance)
            if predicted_variance > 0
            else None
        ),
        "calibration_intercept": (
            float(observed_mean - covariance / predicted_variance * predicted_mean)
            if predicted_variance > 0
            else None
        ),
        "ring_rank_pearson": (
            float(covariance / denominator) if denominator > 0 else None
        ),
    }


def _print_checkpoint(label: str, result: dict[str, Any]) -> None:
    print(f"\n{label}: forced opening action calibration")
    print(
        f"{'ring':>4} {'games':>6} {'Q':>8} {'outcome':>8} "
        f"{'95% outcome CI':>20} {'Q-out':>8} {'raw mass':>9} {'pi* mass':>9}"
    )
    for row in result["rings"]:
        print(
            f"{row['ring']:4d} {row['games']:6d} {row['predicted_q']:+8.3f} "
            f"{row['empirical_outcome']:+8.3f} "
            f"[{row['outcome_ci95_low']:+.3f}, {row['outcome_ci95_high']:+.3f}] "
            f"{row['q_minus_outcome']:+8.3f} "
            f"{100.0 * result['opening_ring_mass'][str(row['ring'])]['raw']:8.2f}% "
            f"{100.0 * result['opening_ring_mass'][str(row['ring'])]['improved']:8.2f}%"
        )
    summary = result["summary"]
    print(
        "summary: "
        f"MAE={summary['weighted_mae']:.3f} "
        f"RMSE={summary['weighted_rmse']:.3f} "
        f"slope={summary['calibration_slope']} "
        f"ring-r={summary['ring_rank_pearson']}"
    )


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
    game_config = dataclasses.replace(
        config.game,
        rollout_horizon=args.rollout_horizon,
    )

    labels = args.label
    results: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "settings": {
            "positions_per_ring": args.positions_per_ring,
            "parallel_games": args.parallel_games,
            "rollout_horizon": args.rollout_horizon,
            "seed": seed,
            "device": str(device),
            "precision": precision,
            "compile": compile_model,
        },
        "checkpoints": {},
    }

    import hexo_rs

    rust_config = hexo_rs.GameConfig(
        win_length=game_config.win_length,
        placement_radius=game_config.placement_radius,
        max_moves=2**32 - 1,
    )
    opening_state = hexo_rs.GameState(rust_config)

    for checkpoint_index, checkpoint_path in enumerate(args.checkpoint):
        loaded = load_checkpoint(checkpoint_path, device)
        label = (
            labels[checkpoint_index]
            if labels is not None
            else f"gen{loaded.iteration}"
        )
        print(f"\nLoading {label}: {checkpoint_path}", flush=True)
        model = loaded.model.network
        if compile_model:
            compile_klent_forward(model)
        opening = _opening_outputs(
            model,
            loaded.model_config,
            loaded.algorithm,
            opening_state,
            edge_budget=inference_edge_budget,
            device=device,
            precision=precision,
        )
        ring_mass = {
            str(ring): {
                "raw": sum(
                    item["raw_probability"]
                    for coord, item in opening.items()
                    if _hex_distance(coord) == ring
                ),
                "improved": sum(
                    item["improved_probability"]
                    for coord, item in opening.items()
                    if _hex_distance(coord) == ring
                ),
            }
            for ring in range(1, game_config.placement_radius + 1)
        }

        ring_rows: list[dict[str, Any]] = []
        checkpoint_started = time.monotonic()
        for ring in range(1, game_config.placement_radius + 1):
            ring_coords = [
                coord for coord in opening if _hex_distance(coord) == ring
            ]
            forced_count = 0

            def infer(states, *, _ring_coords=ring_coords):
                nonlocal forced_count
                logits, q_values = _network_infer(
                    model,
                    loaded.model_config,
                    states,
                    inference_batch_size=inference_batch_size,
                    edge_budget=inference_edge_budget,
                    device=device,
                    precision=precision,
                )
                for index, state in enumerate(states):
                    if len(state.placed_stones()) != 1:
                        continue
                    forced_coord = _ring_coords[forced_count % len(_ring_coords)]
                    forced_count += 1
                    legal = list(state.legal_moves())
                    action_index = legal.index(forced_coord)
                    forced_logits = torch.full_like(logits[index], -100.0)
                    forced_logits[action_index] = 100.0
                    logits[index] = forced_logits
                    q_values[index] = torch.zeros_like(q_values[index])
                return logits, q_values

            print(f"  ring {ring}: collecting...", end="", flush=True)
            trajectories, stats = _collect_with_inference(
                infer,
                game_config=game_config,
                algorithm=loaded.algorithm,
                positions=args.positions_per_ring,
                parallel_games=args.parallel_games,
                dense_position_cell_limit=0,
                seed=seed + ring * 1_000_003,
                worker_processes=1,
            )
            row = _ring_result(trajectories, opening, ring)
            row["collection_positions"] = stats.positions
            row["excluded_truncated_games"] = stats.truncations
            ring_rows.append(row)
            print(
                f" {row['games']} games, Q={row['predicted_q']:+.3f}, "
                f"outcome={row['empirical_outcome']:+.3f}",
                flush=True,
            )
            del trajectories
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        checkpoint_result = {
            "path": str(checkpoint_path.resolve()),
            "iteration": str(loaded.iteration),
            "elapsed_seconds": time.monotonic() - checkpoint_started,
            "opening_ring_mass": ring_mass,
            "rings": ring_rows,
            "summary": _weighted_regression(ring_rows),
        }
        results["checkpoints"][label] = checkpoint_result
        _print_checkpoint(label, checkpoint_result)
        del loaded, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
