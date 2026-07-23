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

from hexo_klent.batching import move_batch_to_device, prepare_graph_batches
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
    reward: float = 0.0
    return_target: float | None = None


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]
    winner: str | None = None
    truncated: bool = False
    chunk_truncated: bool = False
    bootstrap_value: float | None = None


@dataclass(frozen=True)
class CollectionStats:
    games: int
    positions: int
    p1_wins: int
    p2_wins: int
    truncations: int
    horizon_truncations: int
    chunk_truncations: int
    mean_game_length: float
    mean_entropy: float
    mean_normalized_entropy: float
    mean_target_top1_probability: float
    mean_legal_actions: float
    mean_reverse_kl: float
    mean_abs_q: float
    mean_q_span: float
    mean_abs_return: float
    mean_abs_bootstrap_value: float
    worker_processes: int
    elapsed_seconds: float


InferenceFn = Callable[[list[object]], tuple[list[Tensor], list[Tensor]]]


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
    seed: int | None,
    worker_processes: int,
) -> tuple[list[Trajectory], CollectionStats]:
    """Collect a fixed number of transitions from a frozen policy/Q network."""

    if positions <= 0 or parallel_games <= 0:
        raise ValueError("positions and parallel_games must be positive")
    if parallel_games > positions:
        raise ValueError("parallel_games cannot exceed positions")

    import hexo_rs

    started_at = time.monotonic()
    rust_config = _rust_game_config(game_config)
    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    active: list[tuple[object, Trajectory]] = [
        (hexo_rs.GameState(rust_config), Trajectory([]))
        for _ in range(parallel_games)
    ]
    completed: list[Trajectory] = []
    pending_bootstraps: list[tuple[object, Trajectory]] = []

    entropy_sum = 0.0
    normalized_entropy_sum = 0.0
    target_top1_probability_sum = 0.0
    legal_actions_sum = 0.0
    reverse_kl_sum = 0.0
    abs_q_sum = 0.0
    q_span_sum = 0.0
    position_count = 0

    with torch.no_grad():
        while position_count < positions:
            remaining = positions - position_count
            step_count = min(len(active), remaining)
            stepping = active[:step_count]
            # Lanes beyond the final partial batch remain at their current
            # successor state and are bootstrapped below.
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
                    )
                )

                entropy = float(
                    -(
                        policy_cpu
                        * policy_cpu.clamp_min(1e-12).log()
                    ).sum().item()
                )
                legal_action_count = len(legal_coords)
                entropy_sum += entropy
                normalized_entropy_sum += (
                    entropy / math.log(legal_action_count)
                    if legal_action_count > 1
                    else 0.0
                )
                target_top1_probability_sum += float(
                    policy_cpu.max().item()
                )
                legal_actions_sum += legal_action_count
                reverse_kl_sum += max(
                    0.0,
                    float(
                        (
                            policy
                            * (
                                policy.clamp_min(1e-12).log()
                                - prior.clamp_min(1e-12).log()
                            )
                        )
                        .sum()
                        .item()
                    ),
                )
                mean_abs_q, q_span = torch.stack(
                    (q_f.abs().mean(), q_f.max() - q_f.min())
                ).cpu().tolist()
                abs_q_sum += float(mean_abs_q)
                q_span_sum += float(q_span)
                position_count += 1

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
                    completed.append(trajectory)
                elif game.move_count() >= game_config.rollout_horizon:
                    trajectory.truncated = True
                    pending_bootstraps.append((game, trajectory))
                else:
                    next_active.append((game, trajectory))

            active = next_active

            if position_count < positions:
                remaining = positions - position_count
                new_lanes = min(
                    parallel_games - len(active),
                    max(0, remaining - len(active)),
                )
                active.extend(
                    (hexo_rs.GameState(rust_config), Trajectory([]))
                    for _ in range(new_lanes)
                )

        # The collection budget is a standard TD(lambda) truncation boundary:
        # close every still-live lane against the same frozen network rather
        # than discarding its positions or pretending HeXO drew.
        for game, trajectory in active:
            if not trajectory.steps:
                raise RuntimeError("collection ended with an empty live lane")
            trajectory.truncated = True
            trajectory.chunk_truncated = True
            pending_bootstraps.append((game, trajectory))

        for start in range(0, len(pending_bootstraps), parallel_games):
            chunk = pending_bootstraps[start : start + parallel_games]
            states = [game for game, _trajectory in chunk]
            logit_chunks, q_chunks = infer(states)
            for (game, trajectory), logits, q_values in zip(
                chunk, logit_chunks, q_chunks, strict=True
            ):
                bootstrap_player = game.current_player()
                if bootstrap_player not in {"P1", "P2"}:
                    raise RuntimeError(
                        "truncated successor must remain non-terminal"
                    )
                policy = improved_policy(
                    logits.float(),
                    q_values.float(),
                    alpha=algorithm.alpha,
                    beta=algorithm.beta,
                )
                bootstrap_value = float(
                    torch.dot(policy, q_values.float()).item()
                )
                trajectory.bootstrap_value = bootstrap_value
                _attach_returns(
                    trajectory,
                    algorithm.trace_decay,
                    bootstrap_player=bootstrap_player,
                    bootstrap_value=bootstrap_value,
                )
                completed.append(trajectory)

    lengths = [len(item.steps) for item in completed]
    p1_wins = sum(item.winner == "P1" for item in completed)
    p2_wins = sum(item.winner == "P2" for item in completed)
    truncations = sum(item.truncated for item in completed)
    chunk_truncations = sum(item.chunk_truncated for item in completed)
    horizon_truncations = truncations - chunk_truncations
    bootstrap_values = [
        abs(float(trajectory.bootstrap_value))
        for trajectory in completed
        if trajectory.bootstrap_value is not None
    ]
    return_targets = [
        float(step.return_target)
        for trajectory in completed
        for step in trajectory.steps
        if step.return_target is not None
    ]
    denominator = max(position_count, 1)
    stats = CollectionStats(
        games=len(completed),
        positions=position_count,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        truncations=truncations,
        horizon_truncations=horizon_truncations,
        chunk_truncations=chunk_truncations,
        mean_game_length=sum(lengths) / max(len(lengths), 1),
        mean_entropy=entropy_sum / denominator,
        mean_normalized_entropy=normalized_entropy_sum / denominator,
        mean_target_top1_probability=(
            target_top1_probability_sum / denominator
        ),
        mean_legal_actions=legal_actions_sum / denominator,
        mean_reverse_kl=max(0.0, reverse_kl_sum / denominator),
        mean_abs_q=abs_q_sum / denominator,
        mean_q_span=q_span_sum / denominator,
        mean_abs_return=(
            sum(abs(target) for target in return_targets)
            / max(len(return_targets), 1)
        ),
        mean_abs_bootstrap_value=(
            sum(bootstrap_values) / max(len(bootstrap_values), 1)
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
    precision: str = "float32",
    seed: int | None = None,
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
            [(batch_cpu, _state_slice)] = prepare_graph_batches(
                chunk,
                model_config=model_config,
                edge_budget=0,
            )
            batch = move_batch_to_device(batch_cpu, device)
            with _autocast(device, precision):
                output = model.forward_batch(batch)
            counts = [
                int(item)
                for item in output.legal_counts.detach().cpu().tolist()
            ]
            logits.extend(output.policy_logits.split(counts))
            q_values.extend(output.q_values.split(counts))
        return logits, q_values

    try:
        return _collect_with_inference(
            infer,
            game_config=game_config,
            algorithm=algorithm,
            positions=positions,
            parallel_games=parallel_games,
            seed=seed,
            worker_processes=1,
        )
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class _CollectTask:
    game_config: Any
    algorithm: AlgorithmConfig
    positions: int
    parallel_games: int
    seed: int | None


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
            trajectories, stats = _collect_with_inference(
                infer,
                game_config=task.game_config,
                algorithm=task.algorithm,
                positions=task.positions,
                parallel_games=task.parallel_games,
                seed=task.seed,
                worker_processes=1,
            )
            # Never send thousands of Torch storages through multiprocessing:
            # its reducer consumes one shared-memory file descriptor per tensor.
            for trajectory in trajectories:
                for step in trajectory.steps:
                    step.target_policy = step.target_policy.numpy()
            request_queue.put(_WorkerDone(worker_id, trajectories, stats))
        except _ActorShutdown:
            return
        except BaseException:
            request_queue.put(
                _WorkerFailure(worker_id, traceback.format_exc())
            )
            return


def _restore_worker_tensors(done: _WorkerDone) -> _WorkerDone:
    for trajectory in done.trajectories:
        for step in trajectory.steps:
            if not isinstance(step.target_policy, Tensor):
                step.target_policy = torch.from_numpy(step.target_policy)
    return done


def _merge_worker_results(
    results: list[_WorkerDone],
    *,
    workers: int,
    elapsed_seconds: float,
) -> tuple[list[Trajectory], CollectionStats]:
    trajectories = [
        trajectory
        for result in results
        for trajectory in result.trajectories
    ]
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
        p1_wins=sum(item.p1_wins for item in stats),
        p2_wins=sum(item.p2_wins for item in stats),
        truncations=truncations,
        horizon_truncations=sum(item.horizon_truncations for item in stats),
        chunk_truncations=sum(item.chunk_truncations for item in stats),
        mean_game_length=(
            sum(item.mean_game_length * item.games for item in stats)
            / max(games, 1)
        ),
        mean_entropy=position_mean("mean_entropy"),
        mean_normalized_entropy=position_mean("mean_normalized_entropy"),
        mean_target_top1_probability=position_mean(
            "mean_target_top1_probability"
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

    def __init__(self, workers: int) -> None:
        if workers <= 1:
            raise ValueError("shared actors require at least two workers")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(name, "1")

        self.workers = workers
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
        graph_batches = []
        for start in range(0, len(states), inference_batch_size):
            graph_batches.extend(
                prepare_graph_batches(
                    states[start : start + inference_batch_size],
                    model_config=model_config,
                    edge_budget=inference_edge_budget,
                )
            )

        legal_counts: list[int] = []
        logits_parts: list[Tensor] = []
        q_parts: list[Tensor] = []
        for batch_cpu, _state_slice in graph_batches:
            batch = move_batch_to_device(batch_cpu, device)
            with _autocast(device, precision):
                output = model.forward_batch(batch)
            legal_counts.extend(
                int(item)
                for item in output.legal_counts.detach().cpu().tolist()
            )
            logits_parts.append(
                output.policy_logits.detach().float().cpu()
            )
            q_parts.append(output.q_values.detach().float().cpu())
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
                    seed=None if seed is None else seed + index * 1_000_003,
                )
            )

        started_at = time.monotonic()
        pending: deque = deque()
        completed: dict[int, _WorkerDone] = {}
        was_training = model.training
        model.eval()
        timeout_seconds = max(batch_timeout_ms, 0.0) / 1000.0

        try:
            with torch.no_grad():
                while len(completed) < active_workers:
                    message = self._next_message(pending)
                    if isinstance(message, _WorkerFailure):
                        raise RuntimeError(
                            f"KLENT actor {message.worker_id} failed:\n"
                            f"{message.detail}"
                        )
                    if isinstance(message, _WorkerDone):
                        completed[message.worker_id] = _restore_worker_tensors(
                            message
                        )
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
                        if isinstance(candidate, _WorkerFailure):
                            raise RuntimeError(
                                f"KLENT actor {candidate.worker_id} failed:\n"
                                f"{candidate.detail}"
                            )
                        if isinstance(candidate, _WorkerDone):
                            completed[candidate.worker_id] = (
                                _restore_worker_tensors(candidate)
                            )
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
        return _merge_worker_results(
            results,
            workers=active_workers,
            elapsed_seconds=time.monotonic() - started_at,
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
    precision: str = "float32",
    seed: int | None = None,
    actors: SharedInferenceActors | None = None,
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
            device=device,
            precision=precision,
            seed=seed,
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
            batch_timeout_ms=batch_timeout_ms,
            device=device,
            precision=precision,
            seed=seed,
        )
    finally:
        if owned_actors:
            actor_pool.close()


def flatten_trajectories(
    trajectories: list[Trajectory],
) -> list[TrajectoryStep]:
    return [step for trajectory in trajectories for step in trajectory.steps]
