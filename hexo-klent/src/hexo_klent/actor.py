"""Search-free KLENT self-play with shared parent-process inference."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from hexo_klent.batching import (
    move_batch_to_device,
    order_states_for_batching,
    prepare_graph_batches,
    raster_shape,
    restore_state_order,
)
from hexo_klent.config import AlgorithmConfig
from hexo_klent.model import KlentNet, improved_policy
from hexo_klent.returns import lambda_returns


@dataclass
class TrajectoryStep:
    """One frozen-policy decision and its eventual Q target."""

    state: object
    target_policy: Tensor
    action_index: int
    player: str
    state_value: float
    # Frozen critic prediction for the action sampled at this state.  Once the
    # game terminates this can be compared directly with the +/-1 outcome,
    # giving on-policy critic calibration telemetry without another forward.
    played_q: float | None = None
    target_prior_kl: float | None = None
    reward: float = 0.0
    return_target: float | None = None
    # Optional fixed-search completed-Q labels for unplayed legal actions.
    # The played action always retains return_target as its authoritative
    # KLENT target and is therefore excluded from these tensors.
    auxiliary_q_action_indices: Tensor | None = None
    auxiliary_q_targets: Tensor | None = None


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    winner: str | None = None
    truncated: bool = False
    spatial_truncated: bool = False
    chunk_truncated: bool = False
    bootstrap_value: float | None = None
    entropy_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    target_top1_probability_sum: float = 0.0
    prior_normalized_entropy_sum: float = 0.0
    prior_top1_probability_sum: float = 0.0
    legal_actions_sum: float = 0.0
    reverse_kl_sum: float = 0.0
    abs_q_sum: float = 0.0
    q_span_sum: float = 0.0


@dataclass(frozen=True)
class CollectionStats:
    games: int
    positions: int
    discarded_positions: int
    p1_wins: int
    p2_wins: int
    truncations: int
    horizon_truncations: int
    spatial_truncations: int
    chunk_truncations: int
    max_dense_position_cells: int
    mean_game_length: float
    mean_entropy: float
    mean_normalized_entropy: float
    mean_target_top1_probability: float
    mean_prior_normalized_entropy: float
    mean_prior_top1_probability: float
    mean_legal_actions: float
    mean_reverse_kl: float
    mean_abs_q: float
    mean_q_span: float
    mean_abs_return: float
    mean_abs_bootstrap_value: float
    worker_processes: int
    elapsed_seconds: float


InferenceFn = Callable[[list[object]], tuple[list[Tensor], list[Tensor]]]
CompletedTrajectoryFn = Callable[[list[Trajectory]], None]


# A worker may collect tens of thousands of ragged S4 positions.  Sending the
# whole shard only at the end retains one full copy in the worker while
# multiprocessing serializes another and the learner restores a third.  Keep
# that transport window bounded independently of the configured generation.
_WORKER_RESULT_CHUNK_POSITIONS = 2_048


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


def _rust_game_config(game_config):
    import hexo_rs

    return hexo_rs.GameConfig(
        win_length=game_config.win_length,
        placement_radius=game_config.placement_radius,
        # HeXO has no draw rule. KLENT applies its finite rollout horizon
        # externally so a still-live final state remains available for value
        # bootstrapping.
        max_moves=2**32 - 1,
    )


def _new_game_state(game_config, board_radius: int = 0):
    """Construct one normal or finite-board self-play state."""

    import hexo_rs

    return hexo_rs.GameState(
        _rust_game_config(game_config),
        board_radius=board_radius or None,
    )


def _attach_returns(
    trajectory: Trajectory,
    trace_decay: float,
    *,
    bootstrap_player: str | None = None,
    bootstrap_value: float | None = None,
) -> None:
    targets = lambda_returns(
        players=[step.player for step in trajectory.steps],
        rewards=[step.reward for step in trajectory.steps],
        state_values=[step.state_value for step in trajectory.steps],
        trace_decay=trace_decay,
        bootstrap_player=bootstrap_player,
        bootstrap_value=bootstrap_value,
    )
    for step, target in zip(trajectory.steps, targets, strict=True):
        if not math.isfinite(target):
            raise FloatingPointError("non-finite KLENT return target")
        step.return_target = target


def _collect_with_inference(
    infer: InferenceFn,
    *,
    game_config,
    algorithm: AlgorithmConfig,
    positions: int,
    parallel_games: int,
    dense_position_cell_limit: int,
    seed: int | None,
    worker_processes: int,
    board_radius: int = 0,
    retain_horizon_truncations: bool = False,
    completed_callback: CompletedTrajectoryFn | None = None,
    completed_chunk_positions: int = 0,
) -> tuple[list[Trajectory], CollectionStats]:
    """Collect complete games from a frozen policy/Q network.

    ``positions`` is the replacement-game budget, not a hard trajectory cut.
    Once it is reached, every live lane drains to a genuine terminal result.
    Games stopped by a safety horizon or dense spatial bound are excluded from
    the returned training trajectories in their entirety. Distillation may
    explicitly retain horizon-capped prefixes because it consumes frozen
    teacher outputs rather than outcome-derived return targets; spatial and
    chunk safety truncations are never retained.
    """

    if positions <= 0 or parallel_games <= 0:
        raise ValueError("positions and parallel_games must be positive")
    if parallel_games > positions:
        raise ValueError("parallel_games cannot exceed positions")
    if completed_callback is not None and completed_chunk_positions <= 0:
        raise ValueError(
            "completed_chunk_positions must be positive with a callback"
        )

    started_at = time.monotonic()
    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    active: list[tuple[object, Trajectory]] = [
        (_new_game_state(game_config, board_radius), Trajectory([]))
        for _ in range(parallel_games)
    ]
    completed: list[Trajectory] = []
    completed_positions = 0
    accepted_position_count = 0
    discarded_position_count = 0

    game_count = 0
    game_length_sum = 0
    p1_wins = 0
    p2_wins = 0
    truncations = 0
    horizon_truncations = 0
    spatial_truncations = 0
    chunk_truncations = 0
    abs_bootstrap_sum = 0.0
    bootstrap_count = 0
    abs_return_sum = 0.0
    return_count = 0

    entropy_sum = 0.0
    normalized_entropy_sum = 0.0
    target_top1_probability_sum = 0.0
    prior_normalized_entropy_sum = 0.0
    prior_top1_probability_sum = 0.0
    legal_actions_sum = 0.0
    reverse_kl_sum = 0.0
    abs_q_sum = 0.0
    q_span_sum = 0.0

    def finish_trajectory(
        trajectory: Trajectory, *, include_in_training: bool
    ) -> None:
        nonlocal completed, completed_positions
        nonlocal accepted_position_count, discarded_position_count
        nonlocal game_count, game_length_sum, p1_wins, p2_wins
        nonlocal truncations, horizon_truncations
        nonlocal spatial_truncations, chunk_truncations
        nonlocal abs_bootstrap_sum, bootstrap_count
        nonlocal abs_return_sum, return_count
        nonlocal entropy_sum, normalized_entropy_sum
        nonlocal target_top1_probability_sum
        nonlocal prior_normalized_entropy_sum
        nonlocal prior_top1_probability_sum, legal_actions_sum
        nonlocal reverse_kl_sum, abs_q_sum, q_span_sum

        length = len(trajectory.steps)
        game_count += 1
        game_length_sum += length
        p1_wins += int(trajectory.winner == "P1")
        p2_wins += int(trajectory.winner == "P2")
        truncations += int(trajectory.truncated)
        spatial_truncations += int(trajectory.spatial_truncated)
        chunk_truncations += int(trajectory.chunk_truncated)
        horizon_truncations += int(
            trajectory.truncated
            and not trajectory.spatial_truncated
            and not trajectory.chunk_truncated
        )
        if trajectory.bootstrap_value is not None:
            abs_bootstrap_sum += abs(float(trajectory.bootstrap_value))
            bootstrap_count += 1
        if not include_in_training:
            discarded_position_count += length
            return

        retained_horizon = (
            retain_horizon_truncations
            and trajectory.truncated
            and not trajectory.spatial_truncated
            and not trajectory.chunk_truncated
        )
        if (
            not retained_horizon
            and (
                trajectory.winner not in {"P1", "P2"}
                or trajectory.truncated
            )
        ):
            raise RuntimeError(
                "only genuinely terminal HeXO games may enter KLENT FIT"
            )
        accepted_position_count += length
        entropy_sum += trajectory.entropy_sum
        normalized_entropy_sum += trajectory.normalized_entropy_sum
        target_top1_probability_sum += (
            trajectory.target_top1_probability_sum
        )
        prior_normalized_entropy_sum += (
            trajectory.prior_normalized_entropy_sum
        )
        prior_top1_probability_sum += trajectory.prior_top1_probability_sum
        legal_actions_sum += trajectory.legal_actions_sum
        reverse_kl_sum += trajectory.reverse_kl_sum
        abs_q_sum += trajectory.abs_q_sum
        q_span_sum += trajectory.q_span_sum
        for step in trajectory.steps:
            if step.return_target is not None:
                abs_return_sum += abs(float(step.return_target))
                return_count += 1

        completed.append(trajectory)
        completed_positions += length
        if (
            completed_callback is not None
            and completed_positions >= completed_chunk_positions
        ):
            chunk = completed
            completed = []
            completed_positions = 0
            completed_callback(chunk)

    generated_position_count = 0
    max_dense_position_cells = 0
    if dense_position_cell_limit > 0:
        initial_height, initial_width = raster_shape(active[0][0])
        max_dense_position_cells = initial_height * initial_width
        if max_dense_position_cells > dense_position_cell_limit:
            raise ValueError(
                "dense_position_cell_limit is smaller than the initial "
                f"raster ({max_dense_position_cells} cells)"
            )

    with torch.no_grad():
        while generated_position_count < positions or active:
            collecting = generated_position_count < positions
            remaining = positions - generated_position_count
            step_count = min(len(active), remaining) if collecting else len(active)
            stepping = active[:step_count]
            # Lanes beyond the final partial budget batch remain live. The
            # next loop drains every one to a terminal result.
            next_active = active[step_count:]
            states = [game for game, _trajectory in stepping]
            logit_chunks, q_chunks = infer(states)
            if (
                len(logit_chunks) != len(stepping)
                or len(q_chunks) != len(stepping)
            ):
                raise RuntimeError("inference response count does not match actors")

            for (game, trajectory), logits, q_values in zip(
                stepping, logit_chunks, q_chunks, strict=True
            ):
                legal_coords = game.legal_moves()
                if logits.numel() != len(legal_coords):
                    raise RuntimeError(
                        "inference legal-action count differs from GameState"
                    )

                # KLENT's exponent is sensitive at the paper defaults, so form
                # the improved policy in fp32 even after bf16 inference.
                logits_f = logits.float()
                q_f = q_values.float()
                policy = improved_policy(
                    logits_f,
                    q_f,
                    alpha=algorithm.alpha,
                    beta=algorithm.beta,
                )
                prior = torch.softmax(logits_f, dim=0)
                policy_cpu = policy.cpu()
                action_index = int(
                    torch.multinomial(
                        policy_cpu, 1, generator=generator
                    ).item()
                )
                log_prior = prior.clamp_min(1e-12).log()
                prior_diagnostics = torch.stack(
                    (
                        (
                            policy
                            * (
                                policy.clamp_min(1e-12).log()
                                - log_prior
                            )
                        ).sum(),
                        -(prior * log_prior).sum(),
                        prior.max(),
                    )
                ).cpu().tolist()
                target_prior_kl = max(0.0, float(prior_diagnostics[0]))
                prior_entropy = float(prior_diagnostics[1])
                prior_top1_probability = float(prior_diagnostics[2])

                player = game.current_player()
                if player not in {"P1", "P2"}:
                    raise RuntimeError("self-play reached a terminal actor state")
                trajectory.steps.append(
                    TrajectoryStep(
                        state=game.clone(),
                        target_policy=policy_cpu,
                        action_index=action_index,
                        player=player,
                        state_value=float(torch.dot(policy, q_f).item()),
                        played_q=float(q_f[action_index].item()),
                        target_prior_kl=target_prior_kl,
                    )
                )

                entropy = float(
                    -(
                        policy_cpu
                        * policy_cpu.clamp_min(1e-12).log()
                    ).sum().item()
                )
                legal_action_count = len(legal_coords)
                trajectory.entropy_sum += entropy
                trajectory.normalized_entropy_sum += (
                    entropy / math.log(legal_action_count)
                    if legal_action_count > 1
                    else 0.0
                )
                trajectory.target_top1_probability_sum += float(
                    policy_cpu.max().item()
                )
                trajectory.prior_normalized_entropy_sum += (
                    min(
                        1.0,
                        max(
                            0.0,
                            prior_entropy / math.log(legal_action_count),
                        ),
                    )
                    if legal_action_count > 1
                    else 0.0
                )
                trajectory.prior_top1_probability_sum += (
                    prior_top1_probability
                )
                trajectory.legal_actions_sum += legal_action_count
                trajectory.reverse_kl_sum += target_prior_kl
                mean_abs_q, q_span = torch.stack(
                    (q_f.abs().mean(), q_f.max() - q_f.min())
                ).cpu().tolist()
                trajectory.abs_q_sum += float(mean_abs_q)
                trajectory.q_span_sum += float(q_span)
                generated_position_count += 1

                q, r = legal_coords[action_index]
                game.apply_move(q, r)
                if game.is_terminal():
                    winner = game.winner()
                    trajectory.winner = winner
                    if winner is None:
                        raise RuntimeError(
                            "unbounded KLENT engine produced a draw"
                        )
                    if winner == player:
                        trajectory.steps[-1].reward = 1.0
                    else:
                        # HeXO can award the opponent a win after an illegal
                        # double-threat shape by the actor.
                        trajectory.steps[-1].reward = -1.0
                    _attach_returns(trajectory, algorithm.trace_decay)
                    finish_trajectory(
                        trajectory, include_in_training=True
                    )
                elif game.move_count() >= game_config.rollout_horizon:
                    trajectory.truncated = True
                    finish_trajectory(
                        trajectory,
                        include_in_training=retain_horizon_truncations,
                    )
                elif dense_position_cell_limit > 0:
                    height, width = raster_shape(game)
                    successor_cells = height * width
                    if successor_cells > dense_position_cell_limit:
                        # Dense execution scales with the padded bounding box,
                        # not the number of live nodes. The entire capped game
                        # is excluded: FIT only sees terminal outcome targets.
                        trajectory.truncated = True
                        trajectory.spatial_truncated = True
                        finish_trajectory(
                            trajectory, include_in_training=False
                        )
                    else:
                        max_dense_position_cells = max(
                            max_dense_position_cells,
                            successor_cells,
                        )
                        next_active.append((game, trajectory))
                else:
                    next_active.append((game, trajectory))

            active = next_active

            if generated_position_count < positions:
                remaining = positions - generated_position_count
                new_lanes = min(
                    parallel_games - len(active),
                    max(0, remaining - len(active)),
                )
                active.extend(
                    (
                        _new_game_state(game_config, board_radius),
                        Trajectory([]),
                    )
                    for _ in range(new_lanes)
                )

    if completed_callback is not None and completed:
        chunk = completed
        completed = []
        completed_positions = 0
        completed_callback(chunk)

    denominator = max(accepted_position_count, 1)
    stats = CollectionStats(
        games=game_count,
        positions=accepted_position_count,
        discarded_positions=discarded_position_count,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        truncations=truncations,
        horizon_truncations=horizon_truncations,
        spatial_truncations=spatial_truncations,
        chunk_truncations=chunk_truncations,
        max_dense_position_cells=max_dense_position_cells,
        mean_game_length=game_length_sum / max(game_count, 1),
        mean_entropy=entropy_sum / denominator,
        mean_normalized_entropy=normalized_entropy_sum / denominator,
        mean_target_top1_probability=(
            target_top1_probability_sum / denominator
        ),
        mean_prior_normalized_entropy=(
            prior_normalized_entropy_sum / denominator
        ),
        mean_prior_top1_probability=(
            prior_top1_probability_sum / denominator
        ),
        mean_legal_actions=legal_actions_sum / denominator,
        mean_reverse_kl=max(0.0, reverse_kl_sum / denominator),
        mean_abs_q=abs_q_sum / denominator,
        mean_q_span=q_span_sum / denominator,
        mean_abs_return=(
            abs_return_sum / max(return_count, 1)
        ),
        mean_abs_bootstrap_value=(
            abs_bootstrap_sum / max(bootstrap_count, 1)
        ),
        worker_processes=worker_processes,
        elapsed_seconds=time.monotonic() - started_at,
    )
    return completed, stats


def collect_games(
    model: KlentNet,
    *,
    model_config,
    game_config,
    algorithm: AlgorithmConfig,
    positions: int,
    parallel_games: int,
    inference_batch_size: int,
    device: torch.device,
    inference_edge_budget: int = 0,
    dense_position_cell_limit: int = 0,
    board_radius: int = 0,
    precision: str = "float32",
    seed: int | None = None,
    retain_horizon_truncations: bool = False,
) -> tuple[list[Trajectory], CollectionStats]:
    """Collect a position-budgeted chunk in the learner process."""

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    was_training = model.training
    model.eval()

    def infer(states: list[object]) -> tuple[list[Tensor], list[Tensor]]:
        logits: list[Tensor] = []
        q_values: list[Tensor] = []
        for start in range(0, len(states), inference_batch_size):
            chunk = states[start : start + inference_batch_size]
            ordered, source_indices = order_states_for_batching(
                chunk, model_config
            )
            ordered_logits: list[Tensor] = []
            ordered_q: list[Tensor] = []
            for batch_cpu, _state_slice in prepare_graph_batches(
                ordered,
                model_config=model_config,
                edge_budget=inference_edge_budget,
            ):
                batch = move_batch_to_device(batch_cpu, device)
                with torch.inference_mode(), _autocast(device, precision):
                    output = model.forward_batch(batch)
                counts = [
                    int(item)
                    for item in output.legal_counts.detach().cpu().tolist()
                ]
                ordered_logits.extend(output.policy_logits.split(counts))
                ordered_q.extend(output.q_values.split(counts))
            logits.extend(restore_state_order(ordered_logits, source_indices))
            q_values.extend(restore_state_order(ordered_q, source_indices))
        return logits, q_values

    try:
        return _collect_with_inference(
            infer,
            game_config=game_config,
            algorithm=algorithm,
            positions=positions,
            parallel_games=parallel_games,
            dense_position_cell_limit=dense_position_cell_limit,
            board_radius=board_radius,
            seed=seed,
            worker_processes=1,
            retain_horizon_truncations=retain_horizon_truncations,
        )
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class _CollectTask:
    game_config: Any
    algorithm: AlgorithmConfig
    positions: int
    parallel_games: int
    dense_position_cell_limit: int
    board_radius: int
    seed: int | None
    retain_horizon_truncations: bool = False


@dataclass
class _EvalRequest:
    worker_id: int
    request_id: int
    states: list[object]


@dataclass
class _EvalResponse:
    request_id: int
    legal_counts: list[int]
    policy_logits: Any
    q_values: Any


@dataclass
class _WorkerDone:
    worker_id: int
    trajectories: list[Trajectory]
    stats: CollectionStats


@dataclass
class _WorkerTrajectoryChunk:
    worker_id: int
    chunk_id: int
    trajectories: list[Trajectory]


@dataclass(frozen=True)
class _WorkerTrajectoryAck:
    chunk_id: int


@dataclass(frozen=True)
class _WorkerFailure:
    worker_id: int
    detail: str


@dataclass(frozen=True)
class _StopWorker:
    """Wake an actor blocked on an inference response during shutdown."""


class _ActorShutdown(Exception):
    """Internal cooperative-shutdown signal within an actor process."""


def _actor_worker_main(
    worker_id: int,
    task_queue,
    request_queue,
    response_queue,
    stop_event,
    result_chunk_positions: int = _WORKER_RESULT_CHUNK_POSITIONS,
) -> None:
    """Own Rust games on CPU and request all neural evaluations from parent."""

    # Ctrl-C is delivered to the whole foreground process group. The parent
    # owns shutdown and explicitly wakes these workers; letting every child
    # raise KeyboardInterrupt produces tracebacks and can corrupt queue writes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    while True:
        task = task_queue.get()
        if task is None or stop_event.is_set():
            return
        request_id = 0

        def infer(states: list[object]) -> tuple[list[Tensor], list[Tensor]]:
            nonlocal request_id
            if stop_event.is_set():
                raise _ActorShutdown
            current_id = request_id
            request_id += 1
            request_queue.put(
                _EvalRequest(
                    worker_id=worker_id,
                    request_id=current_id,
                    states=states,
                )
            )
            response = response_queue.get()
            if isinstance(response, _StopWorker) or stop_event.is_set():
                raise _ActorShutdown
            if not isinstance(response, _EvalResponse):
                raise RuntimeError(
                    f"unexpected inference response {type(response).__name__}"
                )
            if response.request_id != current_id:
                raise RuntimeError(
                    f"inference response id {response.request_id} "
                    f"does not match request {current_id}"
                )
            logits = torch.from_numpy(response.policy_logits)
            q_values = torch.from_numpy(response.q_values)
            return (
                list(logits.split(response.legal_counts)),
                list(q_values.split(response.legal_counts)),
            )

        try:
            result_chunk_id = 0

            def emit_trajectories(
                trajectories: list[Trajectory],
            ) -> None:
                nonlocal result_chunk_id
                # Never send thousands of Torch storages through
                # multiprocessing: its reducer consumes one shared-memory file
                # descriptor per tensor.  NumPy arrays travel in the ordinary
                # pickle payload and are restored as zero-copy CPU tensors.
                for trajectory in trajectories:
                    for step in trajectory.steps:
                        step.target_policy = step.target_policy.numpy()
                chunk_id = result_chunk_id
                result_chunk_id += 1
                request_queue.put(
                    _WorkerTrajectoryChunk(
                        worker_id,
                        chunk_id,
                        trajectories,
                    )
                )
                # One acknowledged chunk per actor bounds both retained game
                # data and multiprocessing's serialization copies.  Inference
                # and result messages share this response queue, but no neural
                # request is outstanding while a completed chunk is emitted.
                response = response_queue.get()
                if isinstance(response, _StopWorker) or stop_event.is_set():
                    raise _ActorShutdown
                if not isinstance(response, _WorkerTrajectoryAck):
                    raise RuntimeError(
                        "unexpected trajectory acknowledgement "
                        f"{type(response).__name__}"
                    )
                if response.chunk_id != chunk_id:
                    raise RuntimeError(
                        f"trajectory acknowledgement {response.chunk_id} "
                        f"does not match chunk {chunk_id}"
                    )

            trajectories, stats = _collect_with_inference(
                infer,
                game_config=task.game_config,
                algorithm=task.algorithm,
                positions=task.positions,
                parallel_games=task.parallel_games,
                dense_position_cell_limit=task.dense_position_cell_limit,
                board_radius=task.board_radius,
                seed=task.seed,
                worker_processes=1,
                retain_horizon_truncations=(
                    task.retain_horizon_truncations
                ),
                completed_callback=emit_trajectories,
                completed_chunk_positions=result_chunk_positions,
            )
            if trajectories:
                raise RuntimeError(
                    "streaming actor retained completed trajectories"
                )
            request_queue.put(_WorkerDone(worker_id, trajectories, stats))
        except _ActorShutdown:
            return
        except BaseException:
            request_queue.put(
                _WorkerFailure(worker_id, traceback.format_exc())
            )
            return


def _restore_trajectory_tensors(
    trajectories: list[Trajectory],
) -> list[Trajectory]:
    for trajectory in trajectories:
        for step in trajectory.steps:
            if not isinstance(step.target_policy, Tensor):
                step.target_policy = torch.from_numpy(step.target_policy)
    return trajectories


def _restore_worker_tensors(done: _WorkerDone) -> _WorkerDone:
    _restore_trajectory_tensors(done.trajectories)
    return done


def _merge_worker_results(
    results: list[_WorkerDone],
    *,
    workers: int,
    elapsed_seconds: float,
    streamed_trajectories: list[list[Trajectory]] | None = None,
) -> tuple[list[Trajectory], CollectionStats]:
    if streamed_trajectories is None:
        trajectories = [
            trajectory
            for result in results
            for trajectory in result.trajectories
        ]
    else:
        trajectories = [
            trajectory
            for worker_trajectories in streamed_trajectories
            for trajectory in worker_trajectories
        ]
        trajectories.extend(
            trajectory
            for result in results
            for trajectory in result.trajectories
        )
    stats = [result.stats for result in results]
    games = sum(item.games for item in stats)
    positions = sum(item.positions for item in stats)
    truncations = sum(item.truncations for item in stats)

    def position_mean(field: str) -> float:
        return (
            sum(getattr(item, field) * item.positions for item in stats)
            / max(positions, 1)
        )

    return trajectories, CollectionStats(
        games=games,
        positions=positions,
        discarded_positions=sum(
            item.discarded_positions for item in stats
        ),
        p1_wins=sum(item.p1_wins for item in stats),
        p2_wins=sum(item.p2_wins for item in stats),
        truncations=truncations,
        horizon_truncations=sum(item.horizon_truncations for item in stats),
        spatial_truncations=sum(item.spatial_truncations for item in stats),
        chunk_truncations=sum(item.chunk_truncations for item in stats),
        max_dense_position_cells=max(
            (item.max_dense_position_cells for item in stats),
            default=0,
        ),
        mean_game_length=(
            sum(item.mean_game_length * item.games for item in stats)
            / max(games, 1)
        ),
        mean_entropy=position_mean("mean_entropy"),
        mean_normalized_entropy=position_mean("mean_normalized_entropy"),
        mean_target_top1_probability=position_mean(
            "mean_target_top1_probability"
        ),
        mean_prior_normalized_entropy=position_mean(
            "mean_prior_normalized_entropy"
        ),
        mean_prior_top1_probability=position_mean(
            "mean_prior_top1_probability"
        ),
        mean_legal_actions=position_mean("mean_legal_actions"),
        mean_reverse_kl=position_mean("mean_reverse_kl"),
        mean_abs_q=position_mean("mean_abs_q"),
        mean_q_span=position_mean("mean_q_span"),
        mean_abs_return=position_mean("mean_abs_return"),
        mean_abs_bootstrap_value=(
            sum(
                item.mean_abs_bootstrap_value * item.truncations
                for item in stats
            )
            / max(truncations, 1)
        ),
        worker_processes=workers,
        elapsed_seconds=elapsed_seconds,
    )


class SharedInferenceActors:
    """Persistent CPU actors served by the learner's single frozen GPU model."""

    def __init__(
        self,
        workers: int,
        *,
        result_chunk_positions: int = _WORKER_RESULT_CHUNK_POSITIONS,
    ) -> None:
        if workers <= 1:
            raise ValueError("shared actors require at least two workers")
        if result_chunk_positions <= 0:
            raise ValueError("result_chunk_positions must be positive")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(name, "1")

        self.workers = workers
        self.result_chunk_positions = result_chunk_positions
        self._ctx = mp.get_context("spawn")
        self._request_queue = self._ctx.Queue()
        self._task_queues = [self._ctx.Queue(maxsize=1) for _ in range(workers)]
        self._response_queues = [
            self._ctx.Queue(maxsize=1) for _ in range(workers)
        ]
        self._stop_event = self._ctx.Event()
        processes = [
            self._ctx.Process(
                target=_actor_worker_main,
                args=(
                    worker_id,
                    self._task_queues[worker_id],
                    self._request_queue,
                    self._response_queues[worker_id],
                    self._stop_event,
                    self.result_chunk_positions,
                ),
                daemon=True,
            )
            for worker_id in range(workers)
        ]
        self._processes = []
        self._closed = False
        try:
            for process in processes:
                process.start()
                self._processes.append(process)
        except BaseException:
            # A signal or spawn failure partway through construction must not
            # orphan the subset of actors that did start successfully.
            self.close()
            raise

    def _next_message(self, pending: deque, timeout: float = 600.0):
        if pending:
            return pending.popleft()
        try:
            return self._request_queue.get(timeout=timeout)
        except queue.Empty as error:
            dead = [
                (index, process.exitcode)
                for index, process in enumerate(self._processes)
                if not process.is_alive()
            ]
            raise RuntimeError(
                f"timed out waiting for KLENT actors; dead={dead}"
            ) from error

    @staticmethod
    def _serve_requests(
        requests: list[_EvalRequest],
        *,
        model: KlentNet,
        model_config,
        inference_batch_size: int,
        inference_edge_budget: int,
        device: torch.device,
        precision: str,
        response_queues: list,
    ) -> None:
        states = [state for request in requests for state in request.states]
        ordered_states, source_indices = order_states_for_batching(
            states, model_config
        )
        graph_batches = []
        for start in range(0, len(ordered_states), inference_batch_size):
            graph_batches.extend(
                prepare_graph_batches(
                    ordered_states[start : start + inference_batch_size],
                    model_config=model_config,
                    edge_budget=inference_edge_budget,
                )
            )

        ordered_logits: list[Tensor] = []
        ordered_q: list[Tensor] = []
        for batch_cpu, _state_slice in graph_batches:
            batch = move_batch_to_device(batch_cpu, device)
            with torch.inference_mode(), _autocast(device, precision):
                output = model.forward_batch(batch)
            counts = [
                int(item)
                for item in output.legal_counts.detach().cpu().tolist()
            ]
            ordered_logits.extend(
                output.policy_logits.detach().float().cpu().split(counts)
            )
            ordered_q.extend(
                output.q_values.detach().float().cpu().split(counts)
            )
        logits_parts = restore_state_order(ordered_logits, source_indices)
        q_parts = restore_state_order(ordered_q, source_indices)
        legal_counts = [int(part.numel()) for part in logits_parts]
        logits = torch.cat(logits_parts).numpy()
        q_values = torch.cat(q_parts).numpy()

        graph_offset = 0
        legal_offset = 0
        for request in requests:
            count = len(request.states)
            request_counts = legal_counts[graph_offset : graph_offset + count]
            request_legal = sum(request_counts)
            response_queues[request.worker_id].put(
                _EvalResponse(
                    request_id=request.request_id,
                    legal_counts=request_counts,
                    policy_logits=logits[
                        legal_offset : legal_offset + request_legal
                    ].copy(),
                    q_values=q_values[
                        legal_offset : legal_offset + request_legal
                    ].copy(),
                )
            )
            graph_offset += count
            legal_offset += request_legal

    def collect(
        self,
        model: KlentNet,
        *,
        model_config,
        game_config,
        algorithm: AlgorithmConfig,
        positions: int,
        parallel_games: int,
        inference_batch_size: int,
        inference_edge_budget: int,
        batch_timeout_ms: float,
        device: torch.device,
        precision: str,
        seed: int | None,
        dense_position_cell_limit: int = 0,
        board_radius: int = 0,
        retain_horizon_truncations: bool = False,
    ) -> tuple[list[Trajectory], CollectionStats]:
        """Collect one frozen generation while serving all actor requests."""

        if self._closed:
            raise RuntimeError("shared KLENT actors are closed")
        active_workers = min(self.workers, positions, parallel_games)
        position_quotient, position_remainder = divmod(
            positions, active_workers
        )
        position_shards = [
            position_quotient + int(index < position_remainder)
            for index in range(active_workers)
        ]
        lane_quotient, lane_remainder = divmod(
            parallel_games, active_workers
        )
        lane_shards = [
            lane_quotient + int(index < lane_remainder)
            for index in range(active_workers)
        ]
        for index, (shard_positions, shard_lanes) in enumerate(
            zip(position_shards, lane_shards, strict=True)
        ):
            self._task_queues[index].put(
                _CollectTask(
                    game_config=game_config,
                    algorithm=algorithm,
                    positions=shard_positions,
                    parallel_games=shard_lanes,
                    dense_position_cell_limit=dense_position_cell_limit,
                    board_radius=board_radius,
                    seed=None if seed is None else seed + index * 1_000_003,
                    retain_horizon_truncations=(
                        retain_horizon_truncations
                    ),
                )
            )

        started_at = time.monotonic()
        pending: deque = deque()
        completed: dict[int, _WorkerDone] = {}
        streamed_trajectories: list[list[Trajectory]] = [
            [] for _ in range(active_workers)
        ]
        streamed_positions = [0 for _ in range(active_workers)]
        next_result_chunk = [0 for _ in range(active_workers)]
        was_training = model.training
        model.eval()
        timeout_seconds = max(batch_timeout_ms, 0.0) / 1000.0

        def handle_actor_control(message: object) -> bool:
            if isinstance(message, _WorkerFailure):
                raise RuntimeError(
                    f"KLENT actor {message.worker_id} failed:\n"
                    f"{message.detail}"
                )
            if isinstance(message, _WorkerTrajectoryChunk):
                worker_id = message.worker_id
                if not 0 <= worker_id < active_workers:
                    raise RuntimeError(
                        f"trajectory chunk has invalid worker {worker_id}"
                    )
                expected_chunk = next_result_chunk[worker_id]
                if message.chunk_id != expected_chunk:
                    raise RuntimeError(
                        f"KLENT actor {worker_id} sent trajectory chunk "
                        f"{message.chunk_id}, expected {expected_chunk}"
                    )
                restored = _restore_trajectory_tensors(
                    message.trajectories
                )
                streamed_trajectories[worker_id].extend(restored)
                streamed_positions[worker_id] += sum(
                    len(trajectory.steps) for trajectory in restored
                )
                next_result_chunk[worker_id] += 1
                self._response_queues[worker_id].put(
                    _WorkerTrajectoryAck(message.chunk_id)
                )
                return True
            if isinstance(message, _WorkerDone):
                if message.worker_id in completed:
                    raise RuntimeError(
                        f"KLENT actor {message.worker_id} completed twice"
                    )
                completed[message.worker_id] = _restore_worker_tensors(
                    message
                )
                return True
            return False

        try:
            with torch.no_grad():
                while len(completed) < active_workers:
                    message = self._next_message(pending)
                    if handle_actor_control(message):
                        continue
                    if not isinstance(message, _EvalRequest):
                        raise RuntimeError(
                            f"unexpected actor message {type(message).__name__}"
                        )

                    requests = [message]
                    total_states = len(message.states)
                    deadline = time.monotonic() + timeout_seconds
                    while total_states < inference_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            candidate = self._request_queue.get(
                                timeout=remaining
                            )
                        except queue.Empty:
                            break
                        if handle_actor_control(candidate):
                            continue
                        if not isinstance(candidate, _EvalRequest):
                            raise RuntimeError(
                                "unexpected actor message "
                                f"{type(candidate).__name__}"
                            )
                        candidate_states = len(candidate.states)
                        if total_states + candidate_states > inference_batch_size:
                            pending.append(candidate)
                            break
                        requests.append(candidate)
                        total_states += candidate_states

                    self._serve_requests(
                        requests,
                        model=model,
                        model_config=model_config,
                        inference_batch_size=inference_batch_size,
                        inference_edge_budget=inference_edge_budget,
                        device=device,
                        precision=precision,
                        response_queues=self._response_queues,
                    )
        finally:
            model.train(was_training)

        results = [completed[index] for index in range(active_workers)]
        for worker_id, result in enumerate(results):
            actual_positions = (
                streamed_positions[worker_id]
                + sum(
                    len(trajectory.steps)
                    for trajectory in result.trajectories
                )
            )
            if actual_positions != result.stats.positions:
                raise RuntimeError(
                    f"KLENT actor {worker_id} returned {actual_positions} "
                    f"streamed positions, expected {result.stats.positions}"
                )
        return _merge_worker_results(
            results,
            workers=active_workers,
            elapsed_seconds=time.monotonic() - started_at,
            streamed_trajectories=streamed_trajectories,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()

        # Busy actors are normally blocked waiting for the parent to answer an
        # inference request. Wake them explicitly before asking idle actors to
        # leave through their task queues.
        for response_queue in self._response_queues:
            try:
                response_queue.put_nowait(_StopWorker())
            except queue.Full:
                pass
        for task_queue in self._task_queues:
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass

        # Use one deadline for the pool, rather than waiting N * timeout for N
        # workers. Most actors exit cooperatively; terminate/kill are bounded
        # fallbacks for a worker stuck in native code or queue serialization.
        self._join_until(self._processes, timeout=1.0)
        remaining = [
            process for process in self._processes if process.is_alive()
        ]
        for process in remaining:
            process.terminate()
        self._join_until(remaining, timeout=1.0)
        remaining = [
            process for process in remaining if process.is_alive()
        ]
        for process in remaining:
            process.kill()
        self._join_until(remaining, timeout=1.0)

        for process_queue in (
            [self._request_queue]
            + self._task_queues
            + self._response_queues
        ):
            process_queue.close()
            process_queue.cancel_join_thread()

    @staticmethod
    def _join_until(processes: list, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))


def collect_games_parallel(
    model: KlentNet,
    *,
    model_config,
    game_config,
    algorithm: AlgorithmConfig,
    positions: int,
    parallel_games: int,
    inference_batch_size: int,
    inference_edge_budget: int,
    workers: int,
    batch_timeout_ms: float,
    device: torch.device,
    dense_position_cell_limit: int = 0,
    board_radius: int = 0,
    precision: str = "float32",
    seed: int | None = None,
    actors: SharedInferenceActors | None = None,
    retain_horizon_truncations: bool = False,
) -> tuple[list[Trajectory], CollectionStats]:
    """Collect through CPU actors sharing the parent learner's model."""

    if workers <= 1:
        return collect_games(
            model,
            model_config=model_config,
            game_config=game_config,
            algorithm=algorithm,
            positions=positions,
            parallel_games=parallel_games,
            inference_batch_size=inference_batch_size,
            inference_edge_budget=inference_edge_budget,
            dense_position_cell_limit=dense_position_cell_limit,
            board_radius=board_radius,
            device=device,
            precision=precision,
            seed=seed,
            retain_horizon_truncations=retain_horizon_truncations,
        )

    owned_actors = actors is None
    actor_pool = actors or SharedInferenceActors(workers)
    try:
        return actor_pool.collect(
            model,
            model_config=model_config,
            game_config=game_config,
            algorithm=algorithm,
            positions=positions,
            parallel_games=parallel_games,
            inference_batch_size=inference_batch_size,
            inference_edge_budget=inference_edge_budget,
            dense_position_cell_limit=dense_position_cell_limit,
            batch_timeout_ms=batch_timeout_ms,
            device=device,
            precision=precision,
            seed=seed,
            board_radius=board_radius,
            retain_horizon_truncations=retain_horizon_truncations,
        )
    finally:
        if owned_actors:
            actor_pool.close()


def flatten_trajectories(
    trajectories: list[Trajectory],
) -> list[TrajectoryStep]:
    return [step for trajectory in trajectories for step in trajectory.steps]


def terminal_played_q_calibration(
    trajectories: list[Trajectory],
    *,
    opening_plies: int = 16,
) -> dict[str, float]:
    """Compare frozen played-action Q with eventual terminal outcomes.

    The result is position weighted, matching the Q loss.  A second set of
    metrics restricted to the first ``opening_plies`` placements prevents
    terminal-adjacent positions from making a critic look calibrated while its
    long-horizon opening estimates remain poor.  Truncated trajectories and
    legacy steps without ``played_q`` are excluded rather than bootstrapped.
    """

    if opening_plies < 0:
        raise ValueError("opening_plies cannot be negative")

    def summarize(max_ply: int | None) -> dict[str, float]:
        count = 0
        sum_q = 0.0
        sum_outcome = 0.0
        sum_q_squared = 0.0
        sum_outcome_squared = 0.0
        sum_q_outcome = 0.0
        sum_squared_error = 0.0
        sum_absolute_error = 0.0
        correct_sign = 0

        for trajectory in trajectories:
            if trajectory.truncated or trajectory.winner not in {"P1", "P2"}:
                continue
            for ply, step in enumerate(trajectory.steps):
                if max_ply is not None and ply >= max_ply:
                    break
                if step.played_q is None:
                    continue
                q = float(step.played_q)
                if not math.isfinite(q):
                    raise FloatingPointError("non-finite played-action Q")
                outcome = 1.0 if step.player == trajectory.winner else -1.0
                error = q - outcome
                count += 1
                sum_q += q
                sum_outcome += outcome
                sum_q_squared += q * q
                sum_outcome_squared += outcome * outcome
                sum_q_outcome += q * outcome
                sum_squared_error += error * error
                sum_absolute_error += abs(error)
                correct_sign += int(q * outcome > 0.0)

        if count == 0:
            return {}
        inverse_count = 1.0 / count
        mean_q = sum_q * inverse_count
        mean_outcome = sum_outcome * inverse_count
        q_variance = max(
            0.0,
            sum_q_squared - sum_q * sum_q * inverse_count,
        )
        outcome_variance = max(
            0.0,
            sum_outcome_squared
            - sum_outcome * sum_outcome * inverse_count,
        )
        covariance = sum_q_outcome - sum_q * sum_outcome * inverse_count
        slope = covariance / q_variance if q_variance > 0.0 else 0.0
        correlation_denominator = math.sqrt(q_variance * outcome_variance)
        return {
            "positions": float(count),
            "mse": sum_squared_error * inverse_count,
            "mae": sum_absolute_error * inverse_count,
            "bias": mean_q - mean_outcome,
            "mean_q": mean_q,
            "mean_outcome": mean_outcome,
            "sign_accuracy": correct_sign * inverse_count,
            "pearson_r": (
                covariance / correlation_denominator
                if correlation_denominator > 0.0
                else 0.0
            ),
            "calibration_slope": slope,
            "calibration_intercept": mean_outcome - slope * mean_q,
        }

    metrics = {
        f"played_q_outcome_{key}": value
        for key, value in summarize(None).items()
    }
    if opening_plies > 0:
        metrics.update(
            {
                f"opening_played_q_outcome_{key}": value
                for key, value in summarize(opening_plies).items()
            }
        )
    return metrics
