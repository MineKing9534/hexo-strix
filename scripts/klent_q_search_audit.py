#!/usr/bin/env python3
"""Audit KLENT's full action-Q ranking against a fixed search teacher.

Played-action return calibration can look excellent while the hundreds of
unplayed legal actions drift.  KLENT nevertheless uses every Q(s, a) both in
its analytic policy improvement and when reconstructing MCTS leaf values.
This audit therefore:

1. collects terminal-only on-policy games from each of two KLENT checkpoints;
2. samples identical-sized early, middle, and late state strata;
3. runs lockstep Gumbel MCTS from a fixed teacher checkpoint; and
4. compares both KLENT Q vectors with the teacher's *visited-child* search Q.

The teacher and all candidates see exactly the same states and legal-action
ordering.  Root/leaf forcing is disabled so the result measures neural search,
not differences in solver shortcuts.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexo_a0.evaluate import make_eval_fn
from hexo_a0.graph import game_to_axis_graph, game_to_graph
from hexo_a0.head_to_head import load_checkpoint as load_teacher_checkpoint
from hexo_klent.actor import (
    SharedInferenceActors,
    Trajectory,
    collect_games_parallel,
)
from hexo_klent.batching import (
    move_batch_to_device,
    order_states_for_batching,
    prepare_graph_batches,
    restore_state_order,
)
from hexo_klent.config import GameConfig, load_config
from hexo_klent.mcts_adapter import load_checkpoint as load_klent_checkpoint
from hexo_klent.model import compile_klent_forward, improved_policy


PHASES = ("early", "middle", "late")
METRICS = (
    "q_mse",
    "q_mae",
    "within_state_pearson",
    "spearman",
    "pair_accuracy",
    "q_best_agreement",
    "q_teacher_regret",
    "prior_best_agreement",
    "prior_teacher_regret",
    "target_best_agreement",
    "target_teacher_regret",
    "q_global_argmax_searched",
    "prior_global_argmax_searched",
    "target_global_argmax_searched",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="KLENT checkpoint; pass exactly twice",
    )
    parser.add_argument("--label", action="append")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--positions-per-corpus", type=int, default=16_384)
    parser.add_argument("--states-per-phase", type=int, default=32)
    parser.add_argument("--parallel-games", type=int, default=256)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--inference-edge-budget", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--eval-edge-budget", type=int)
    parser.add_argument("--search-sims", type=int, default=64)
    parser.add_argument("--search-actions", type=int, default=16)
    parser.add_argument("--search-root-batch-size", type=int, default=16)
    parser.add_argument("--rollout-horizon", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "bf16"))
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        parser.error("pass exactly two --checkpoint arguments")
    if args.label is not None and len(args.label) != 2:
        parser.error("pass exactly two --label arguments")
    for name in (
        "positions_per_corpus",
        "states_per_phase",
        "parallel_games",
        "eval_batch_size",
        "search_sims",
        "search_actions",
        "search_root_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def _select_phase_states(
    trajectories: list[Trajectory],
    *,
    per_phase: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Take one state per game, using disjoint games for each phase."""

    completed = [
        trajectory
        for trajectory in trajectories
        if not trajectory.truncated
        and trajectory.winner in {"P1", "P2"}
        and len(trajectory.steps) >= 3
    ]
    required = per_phase * len(PHASES)
    if len(completed) < required:
        raise RuntimeError(
            f"need {required} completed games for disjoint phase strata, "
            f"but collection produced {len(completed)}"
        )
    rng = random.Random(seed)
    rng.shuffle(completed)
    selected: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(PHASES):
        start_game = phase_index * per_phase
        for local_game, trajectory in enumerate(
            completed[start_game : start_game + per_phase]
        ):
            length = len(trajectory.steps)
            low = phase_index * length // 3
            high = (phase_index + 1) * length // 3
            high = max(low + 1, high)
            step_index = rng.randrange(low, min(length, high))
            selected.append(
                {
                    "state": trajectory.steps[step_index].state,
                    "phase": phase,
                    "game_index": start_game + local_game,
                    "step_index": step_index,
                    "game_length": length,
                }
            )
    return selected


def _network_outputs(
    model,
    model_config,
    states: list[object],
    *,
    batch_size: int,
    edge_budget: int,
    device: torch.device,
    precision: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ordered, source_indices = order_states_for_batching(states, model_config)
    ordered_logits: list[np.ndarray] = []
    ordered_q: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            for batch_cpu, _state_slice in prepare_graph_batches(
                chunk, model_config=model_config, edge_budget=edge_budget
            ):
                batch = move_batch_to_device(batch_cpu, device)
                with _autocast(device, precision):
                    output = model.forward_batch(batch)
                counts = [int(value) for value in output.legal_counts.cpu()]
                ordered_logits.extend(
                    value.float().cpu().numpy().astype(np.float64, copy=False)
                    for value in output.policy_logits.split(counts)
                )
                ordered_q.extend(
                    value.float().cpu().numpy().astype(np.float64, copy=False)
                    for value in output.q_values.split(counts)
                )
    return (
        restore_state_order(ordered_logits, source_indices),
        restore_state_order(ordered_q, source_indices),
    )


def _teacher_eval_fn(loaded, device: torch.device):
    config = loaded.model_config
    graph_type = config.graph_type
    prune = bool(getattr(config, "prune_empty_edges", False))
    threat = bool(getattr(config, "threat_features", False))
    relative = bool(getattr(config, "relative_stone_encoding", False))
    if graph_type == "axis":
        graph_fn = lambda state: game_to_axis_graph(
            state,
            prune_empty_edges=prune,
            threat_features=threat,
            relative_stones=relative,
        )
    else:
        graph_fn = lambda state: game_to_graph(
            state,
            threat_features=threat,
            relative_stones=relative,
        )
    return make_eval_fn(
        loaded.model,
        device,
        graph_type=graph_type,
        prune_empty_edges=prune,
        threat_features=threat,
        relative_stones=relative,
        graph_fn=graph_fn,
        model_config=config,
    )


def _search_teacher(
    states: list[object],
    eval_fn,
    *,
    sims: int,
    actions: int,
    root_batch_size: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    import hexo_rs

    if not hasattr(hexo_rs, "batched_gumbel_mcts_with_diagnostics"):
        raise RuntimeError(
            "hexo_rs lacks batched_gumbel_mcts_with_diagnostics; "
            "rebuild the local Rust extension"
        )
    config = hexo_rs.MCTSConfig(
        n_simulations=sims,
        m_actions=actions,
        c_visit=50,
        c_scale=1.0,
        disable_gumbel_noise=True,
        disable_forcing_solver=True,
    )
    results: list[dict[str, np.ndarray]] = []
    for start in range(0, len(states), root_batch_size):
        chunk = states[start : start + root_batch_size]
        chunk_results = hexo_rs.batched_gumbel_mcts_with_diagnostics(
            chunk, eval_fn, config, seed=seed + start
        )
        for state, result in zip(chunk, chunk_results, strict=True):
            action, policy, visits, q_values, priors, candidates = result
            legal_count = len(state.legal_moves())
            vectors = (policy, visits, q_values, priors)
            if any(len(vector) != legal_count for vector in vectors):
                raise RuntimeError("teacher diagnostic is not legal-move aligned")
            results.append(
                {
                    "action": np.asarray(action, dtype=np.int64),
                    "policy": np.asarray(policy, dtype=np.float64),
                    "visits": np.asarray(visits, dtype=np.int64),
                    "q": np.asarray(q_values, dtype=np.float64),
                    "prior": np.asarray(priors, dtype=np.float64),
                    "candidates": np.asarray(candidates, dtype=np.int64),
                }
            )
    return results


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


def _pair_accuracy(prediction: np.ndarray, teacher: np.ndarray) -> float:
    row, column = np.triu_indices(len(teacher), 1)
    teacher_delta = teacher[row] - teacher[column]
    prediction_delta = prediction[row] - prediction[column]
    informative = teacher_delta != 0.0
    if not informative.any():
        return 0.5
    products = teacher_delta[informative] * prediction_delta[informative]
    return float(np.mean(np.where(products > 0.0, 1.0, np.where(products == 0.0, 0.5, 0.0))))


def _state_metrics(
    logits: np.ndarray,
    q_values: np.ndarray,
    algorithm,
    teacher: dict[str, np.ndarray],
) -> dict[str, float]:
    searched = teacher["visits"] > 0
    if searched.sum() < 2:
        raise RuntimeError("teacher search visited fewer than two root actions")
    target = improved_policy(
        torch.from_numpy(logits),
        torch.from_numpy(q_values),
        alpha=algorithm.alpha,
        beta=algorithm.beta,
    ).numpy()
    teacher_q = teacher["q"][searched]
    predicted_q = q_values[searched]
    searched_logits = logits[searched]
    searched_target = target[searched]
    teacher_best = int(np.argmax(teacher_q))

    def selection_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
        selected = int(np.argmax(values))
        return {
            f"{prefix}_best_agreement": float(selected == teacher_best),
            f"{prefix}_teacher_regret": float(
                teacher_q[teacher_best] - teacher_q[selected]
            ),
        }

    metrics = {
        "searched_actions": float(searched.sum()),
        "q_mse": float(np.mean(np.square(predicted_q - teacher_q))),
        "q_mae": float(np.mean(np.abs(predicted_q - teacher_q))),
        "q_bias": float(np.mean(predicted_q - teacher_q)),
        "within_state_pearson": _correlation(predicted_q, teacher_q),
        "spearman": _correlation(
            _rankdata(predicted_q), _rankdata(teacher_q)
        ),
        "pair_accuracy": _pair_accuracy(predicted_q, teacher_q),
        "q_global_argmax_searched": float(searched[int(np.argmax(q_values))]),
        "prior_global_argmax_searched": float(searched[int(np.argmax(logits))]),
        "target_global_argmax_searched": float(searched[int(np.argmax(target))]),
    }
    metrics.update(selection_metrics(predicted_q, "q"))
    metrics.update(selection_metrics(searched_logits, "prior"))
    metrics.update(selection_metrics(searched_target, "target"))
    return metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _bootstrap_delta(
    rows_a: list[dict[str, float]],
    rows_b: list[dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if len(rows_a) != len(rows_b):
        raise ValueError("paired audit rows must align")
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        differences = np.asarray(
            [right[metric] - left[metric] for left, right in zip(rows_a, rows_b, strict=True)]
        )
        entry = {"mean_b_minus_a": float(differences.mean())}
        if samples > 0:
            indices = rng.integers(0, len(differences), size=(samples, len(differences)))
            means = differences[indices].mean(axis=1)
            entry.update(
                {
                    "ci95_low": float(np.quantile(means, 0.025)),
                    "ci95_high": float(np.quantile(means, 0.975)),
                }
            )
        result[metric] = entry
    return result


def _print_results(results: dict[str, Any], labels: list[str]) -> None:
    print("\nFull-action Q audit against fixed searched-child values")
    print(
        f"{'corpus':>10} {'phase':>8} {'model':>10} {'n':>4} "
        f"{'MSE':>7} {'rank-r':>7} {'pair':>7} {'Qbest':>7} "
        f"{'Qregret':>8} {'pi*best':>8} {'pi*reg':>8}"
    )
    for corpus in labels:
        for phase in (*PHASES, "all"):
            for label in labels:
                row = results["corpora"][corpus]["summaries"][phase][label]
                print(
                    f"{corpus:>10} {phase:>8} {label:>10} {int(row['states']):4d} "
                    f"{row['q_mse']:7.4f} {row['spearman']:7.3f} "
                    f"{row['pair_accuracy']:7.3f} "
                    f"{row['q_best_agreement']:7.3f} "
                    f"{row['q_teacher_regret']:8.4f} "
                    f"{row['target_best_agreement']:8.3f} "
                    f"{row['target_teacher_regret']:8.4f}"
                )
    print(f"\nPaired delta ({labels[1]} - {labels[0]}); positive regret/MSE is worse")
    for corpus in labels:
        delta = results["corpora"][corpus]["paired_delta"]["all"]
        print(
            f"  {corpus}: MSE {delta['q_mse']['mean_b_minus_a']:+.5f} "
            f"[{delta['q_mse']['ci95_low']:+.5f}, {delta['q_mse']['ci95_high']:+.5f}], "
            f"rank-r {delta['spearman']['mean_b_minus_a']:+.3f}, "
            f"Q regret {delta['q_teacher_regret']['mean_b_minus_a']:+.4f}, "
            f"pi* regret {delta['target_teacher_regret']['mean_b_minus_a']:+.4f}"
        )


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    device = torch.device(args.device or config.run.device)
    precision = args.precision or config.run.precision
    compile_models = config.run.compile if args.compile is None else args.compile
    workers = args.workers or config.collection.workers
    inference_batch_size = (
        args.inference_batch_size or config.collection.inference_batch_size
    )
    inference_edge_budget = (
        args.inference_edge_budget or config.collection.inference_edge_budget
    )
    eval_edge_budget = args.eval_edge_budget or config.training.edge_budget
    game_config = dataclasses.replace(
        config.game, rollout_horizon=args.rollout_horizon
    )
    if not isinstance(game_config, GameConfig):
        raise TypeError("unexpected game configuration")

    loaded = [load_klent_checkpoint(path, device) for path in args.checkpoint]
    labels = args.label or [f"gen{item.iteration}" for item in loaded]
    if len(set(labels)) != 2:
        raise ValueError("checkpoint labels must be distinct")
    networks = [item.model.network for item in loaded]
    if compile_models:
        print("Compiling KLENT inference paths...", flush=True)
        for network in networks:
            compile_klent_forward(network)

    teacher = load_teacher_checkpoint(args.teacher_checkpoint, device)
    teacher_eval = _teacher_eval_fn(teacher, device)
    results: dict[str, Any] = {
        "config": str(args.config.resolve()),
        "teacher": {
            "path": str(args.teacher_checkpoint.resolve()),
            "train_steps": str(teacher.train_steps),
        },
        "checkpoints": {
            label: {
                "path": str(path.resolve()),
                "iteration": str(item.iteration),
            }
            for label, path, item in zip(labels, args.checkpoint, loaded, strict=True)
        },
        "settings": {
            "positions_per_corpus": args.positions_per_corpus,
            "states_per_phase": args.states_per_phase,
            "search_sims": args.search_sims,
            "search_actions": args.search_actions,
            "search_root_batch_size": args.search_root_batch_size,
            "rollout_horizon": args.rollout_horizon,
            "device": str(device),
            "precision": precision,
            "seed": args.seed,
        },
        "corpora": {},
    }

    actors = SharedInferenceActors(workers) if workers > 1 else None
    try:
        for corpus_index, corpus_label in enumerate(labels):
            print(
                f"\nCollecting {corpus_label} terminal corpus "
                f"({args.positions_per_corpus:,} positions)...",
                flush=True,
            )
            trajectories, stats = collect_games_parallel(
                networks[corpus_index],
                model_config=loaded[corpus_index].model_config,
                game_config=game_config,
                algorithm=loaded[corpus_index].algorithm,
                positions=args.positions_per_corpus,
                parallel_games=args.parallel_games,
                inference_batch_size=inference_batch_size,
                inference_edge_budget=inference_edge_budget,
                workers=workers,
                batch_timeout_ms=config.collection.batch_timeout_ms,
                device=device,
                precision=precision,
                seed=args.seed + corpus_index * 10_000_019,
                actors=actors,
            )
            selected = _select_phase_states(
                trajectories,
                per_phase=args.states_per_phase,
                seed=args.seed + corpus_index,
            )
            states = [row["state"] for row in selected]
            print(
                f"  selected {len(states)} disjoint-game states; "
                f"searching with {args.search_sims}x{args.search_actions} teacher...",
                flush=True,
            )
            started = time.monotonic()
            teacher_rows = _search_teacher(
                states,
                teacher_eval,
                sims=args.search_sims,
                actions=args.search_actions,
                root_batch_size=args.search_root_batch_size,
                seed=args.seed + corpus_index * 1_000_003,
            )
            print(f"  teacher search completed in {time.monotonic() - started:.1f}s", flush=True)

            per_model_rows: dict[str, list[dict[str, float]]] = {}
            for model_index, label in enumerate(labels):
                logits, q_values = _network_outputs(
                    networks[model_index],
                    loaded[model_index].model_config,
                    states,
                    batch_size=args.eval_batch_size,
                    edge_budget=eval_edge_budget,
                    device=device,
                    precision=precision,
                )
                per_model_rows[label] = [
                    _state_metrics(
                        state_logits,
                        state_q,
                        loaded[model_index].algorithm,
                        teacher_row,
                    )
                    for state_logits, state_q, teacher_row in zip(
                        logits, q_values, teacher_rows, strict=True
                    )
                ]

            summaries: dict[str, dict[str, Any]] = {}
            deltas: dict[str, Any] = {}
            phases = [row["phase"] for row in selected]
            for phase in (*PHASES, "all"):
                indices = [
                    index
                    for index, item_phase in enumerate(phases)
                    if phase == "all" or item_phase == phase
                ]
                summaries[phase] = {}
                phase_rows: dict[str, list[dict[str, float]]] = {}
                for label in labels:
                    phase_rows[label] = [per_model_rows[label][index] for index in indices]
                    summaries[phase][label] = {
                        "states": len(indices),
                        **_mean_metrics(phase_rows[label]),
                    }
                deltas[phase] = _bootstrap_delta(
                    phase_rows[labels[0]],
                    phase_rows[labels[1]],
                    samples=args.bootstrap_samples,
                    seed=args.seed + corpus_index * 100 + len(indices),
                )

            results["corpora"][corpus_label] = {
                "collection": dataclasses.asdict(stats),
                "selected_states": [
                    {key: value for key, value in row.items() if key != "state"}
                    for row in selected
                ],
                "summaries": summaries,
                "paired_delta": deltas,
            }
            del trajectories, selected, states, teacher_rows, per_model_rows
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if actors is not None:
            actors.close()

    _print_results(results, labels)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
