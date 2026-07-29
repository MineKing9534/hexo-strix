"""Periodic KLENT evaluation against configurable fixed opponents."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from hexo_klent.actor import _autocast, _rust_game_config
from hexo_klent.batching import (
    move_batch_to_device,
    order_states_for_batching,
    prepare_graph_batches,
    restore_state_order,
)
from hexo_klent.config import AlgorithmConfig
from hexo_klent.model import KlentNet, is_dense_axis_config


@dataclass(frozen=True)
class EvaluationStats:
    games: int
    wins: int
    losses: int
    truncations: int
    decided_rate: float
    win_rate_decided: float
    mean_game_length: float
    mean_opponent_depth: float
    opening_pairs: int = 0
    frac_unique_opening: float = 0.0


class CheckpointOpponentCache:
    """Keep fixed evaluation checkpoints resident on CPU between rounds.

    A checkpoint is read and reconstructed at most once per trainer process.
    Its model moves onto the evaluation device only for the duration of its
    match, so adding an anchored opponent does not permanently reserve another
    model's worth of accelerator memory during collection and fitting.
    """

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("checkpoint cache max_entries must be positive")
        self.max_entries = max_entries
        self._loaded: OrderedDict[Path, Any] = OrderedDict()

    @contextmanager
    def activate(
        self,
        checkpoint: str | Path,
        device: torch.device,
    ) -> Iterator[Any]:
        from hexo_a0.head_to_head import load_checkpoint

        path = Path(checkpoint).expanduser().resolve()
        loaded = self._loaded.get(path)
        if loaded is None:
            loaded = load_checkpoint(path, torch.device("cpu"))
            self._loaded[path] = loaded
            while len(self._loaded) > self.max_entries:
                self._loaded.popitem(last=False)
        else:
            self._loaded.move_to_end(path)
        loaded.model.to(device)
        loaded.model.eval()
        try:
            yield loaded
        finally:
            if device.type != "cpu":
                loaded.model.to(torch.device("cpu"))

    def clear(self) -> None:
        """Release all cached CPU models."""

        self._loaded.clear()


def _klent_policy_chunks(
    model: KlentNet,
    states: list[object],
    *,
    model_config,
    device: torch.device,
    precision: str,
) -> list[torch.Tensor]:
    """Evaluate a KLENT model across one or more crop-compatible batches."""

    ordered_states, source_indices = order_states_for_batching(
        states, model_config
    )
    chunks: list[torch.Tensor] = []
    for batch_cpu, _state_slice in prepare_graph_batches(
        ordered_states,
        model_config=model_config,
        edge_budget=0,
    ):
        batch = move_batch_to_device(batch_cpu, device)
        with _autocast(device, precision):
            output = model.forward_batch(batch)
        counts = [
            int(item) for item in output.legal_counts.detach().cpu().tolist()
        ]
        chunks.extend(output.policy_logits.split(counts))
    return restore_state_order(chunks, source_indices)


def _compatible_outputs(
    model: torch.nn.Module,
    states: list[object],
    *,
    model_config,
    device: torch.device,
    precision: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Evaluate an A0 model or KLENT MCTS adapter over raster shape runs."""

    ordered_states, source_indices = order_states_for_batching(
        states, model_config
    )
    policy_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    for batch_cpu, _state_slice in prepare_graph_batches(
        ordered_states,
        model_config=model_config,
        edge_budget=0,
    ):
        batch = move_batch_to_device(batch_cpu, device)
        with torch.inference_mode(), _autocast(device, precision):
            policy_logits, values = model.forward_batch(batch)
        policy_chunks.extend(policy_logits)
        value_chunks.extend(values.unbind())
    return (
        restore_state_order(policy_chunks, source_indices),
        restore_state_order(value_chunks, source_indices),
    )


def _sample_paired_checkpoint_openings(
    *,
    candidate: torch.nn.Module,
    candidate_config,
    opponent: torch.nn.Module,
    opponent_config,
    rust_game_config,
    games: int,
    opening_plies: int,
    opening_temperature: float,
    opening_generator: str,
    device: torch.device,
    seed: int | None,
) -> list[list[tuple[int, int]]]:
    """Sample one replayable opening for each swapped-side game pair.

    This deliberately reuses ``hexo_a0.evaluate.sample_opening``, the opening
    path used by standalone head-to-head/SPRT.  Sampling mutates torch's global
    RNG, so preserve the trainer's CPU and accelerator RNG states around the
    whole operation.
    """

    if opening_plies <= 0:
        return []
    if games % 2 != 0:
        raise ValueError(
            "paired-opening checkpoint evaluation requires an even number "
            "of games"
        )
    if opening_generator not in {"alternate", "a", "b", "champion"}:
        raise ValueError(f"unknown opening generator: {opening_generator}")

    from hexo_a0.evaluate import sample_opening

    accelerator_devices: list[int] = []
    if device.type == "cuda":
        accelerator_devices.append(
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

    base_seed = 0 if seed is None else seed
    openings: list[list[tuple[int, int]]] = []
    with torch.random.fork_rng(devices=accelerator_devices):
        for pair_index in range(games // 2):
            if opening_generator == "a":
                generator = candidate
                generator_config = candidate_config
            elif opening_generator in {"b", "champion"}:
                generator = opponent
                generator_config = opponent_config
            elif pair_index % 2 == 0:
                # Match head-to-head's alternate protocol: B, A, B, A, ...
                generator = opponent
                generator_config = opponent_config
            else:
                generator = candidate
                generator_config = candidate_config
            openings.append(
                sample_opening(
                    generator,
                    rust_game_config,
                    device,
                    opening_plies,
                    opening_temperature,
                    seed=base_seed + pair_index,
                    model_config=generator_config,
                )
            )
    return openings


def resolve_lagged_checkpoint(
    *,
    iteration: int,
    lag_iterations: int,
    checkpoint_dirs: tuple[str | Path, ...],
) -> Path:
    """Resolve one exact prior generation across branch and resume histories."""

    if iteration <= 0:
        raise ValueError("lagged evaluation requires a positive iteration")
    if lag_iterations <= 0:
        raise ValueError("lagged evaluation requires positive lag_iterations")
    target_iteration = iteration - lag_iterations
    if target_iteration < 0:
        raise FileNotFoundError(
            f"lagged checkpoint target iteration {target_iteration} is before "
            "the start of training"
        )
    filename = f"checkpoint_{target_iteration:06d}.pt"
    searched = []
    for directory in checkpoint_dirs:
        candidate = Path(directory).expanduser().resolve() / filename
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"lagged checkpoint for iteration {target_iteration} was not found; "
        f"searched: {', '.join(searched) or '(no checkpoint directories)'}"
    )


def _mcts_move_selector(
    model: KlentNet,
    *,
    model_config,
    algorithm: AlgorithmConfig | None,
    simulations: int,
    actions: int,
    device: torch.device,
    precision: str,
    seed: int | None,
    disable_gumbel_noise: bool = False,
) -> Callable[[object], tuple[int, int]] | None:
    if simulations < 0:
        raise ValueError("MCTS simulations cannot be negative")
    if actions <= 0:
        raise ValueError("MCTS actions must be positive")
    if simulations <= 0:
        return None
    if algorithm is None:
        raise ValueError("KLENT algorithm config is required for MCTS evaluation")

    from hexo_klent.mcts_adapter import KlentMCTSAdapter

    adapter = KlentMCTSAdapter(model, algorithm).to(device).eval()
    return _compatible_mcts_move_selector(
        adapter,
        model_config=model_config,
        simulations=simulations,
        actions=actions,
        device=device,
        precision=precision,
        seed=seed,
        disable_gumbel_noise=disable_gumbel_noise,
    )


def _compatible_mcts_move_selector(
    model: torch.nn.Module,
    *,
    model_config,
    simulations: int,
    actions: int,
    device: torch.device,
    precision: str,
    seed: int | None,
    disable_gumbel_noise: bool = False,
) -> Callable[[object], tuple[int, int]] | None:
    """Build MCTS selection for an A0 model or a KLENT MCTS adapter."""

    if simulations < 0:
        raise ValueError("MCTS simulations cannot be negative")
    if actions <= 0:
        raise ValueError("MCTS actions must be positive")
    if simulations <= 0:
        return None

    import hexo_rs
    if is_dense_axis_config(model_config):
        def model_eval_fn(states):
            logits, values = _compatible_outputs(
                model,
                list(states),
                model_config=model_config,
                device=device,
                precision=precision,
            )
            return (
                [chunk.tolist() for chunk in logits],
                [float(value) for value in values],
            )
    else:
        from hexo_a0.evaluate import make_eval_fn
        from hexo_a0.graph import graph_fn_from_model_config

        graph_fn = graph_fn_from_model_config(model_config)
        model_eval_fn = make_eval_fn(
            model,
            device,
            graph_type=model_config.graph_type,
            prune_empty_edges=model_config.prune_empty_edges,
            threat_features=model_config.threat_features,
            relative_stones=model_config.relative_stone_encoding,
            graph_fn=graph_fn,
        )

    def eval_fn(states):
        # make_eval_fn predates KLENT's configurable inference precision.
        # Keep MCTS leaf evaluation on the same precision path as collection,
        # fitting, and greedy evaluation. In particular, this prevents the
        # compiled GNN core from unexpectedly tracing a float32 variant during
        # bf16 runs (and emitting TorchInductor's tensor-core warning).
        with _autocast(device, precision):
            return model_eval_fn(states)

    mcts_config = hexo_rs.MCTSConfig(
        n_simulations=simulations,
        m_actions=actions,
        c_visit=50,
        c_scale=1.0,
        disable_gumbel_noise=disable_gumbel_noise,
    )
    search_index = 0

    def choose(game) -> tuple[int, int]:
        nonlocal search_index
        search_seed = None if seed is None else seed + search_index
        search_index += 1
        _sampled_action, improved_policy = hexo_rs.gumbel_mcts(
            game,
            eval_fn,
            mcts_config,
            seed=search_seed,
        )
        best_index = max(
            range(len(improved_policy)),
            key=improved_policy.__getitem__,
        )
        return game.legal_moves()[best_index]

    return choose


def evaluate_vs_random(
    model: KlentNet,
    *,
    model_config,
    game_config,
    games: int,
    device: torch.device,
    precision: str = "float32",
    seed: int | None = None,
    depth: int = 0,
    algorithm: AlgorithmConfig | None = None,
    mcts_simulations: int = 0,
    mcts_actions: int = 16,
    checkpoint: str = "",
    checkpoint_cache: CheckpointOpponentCache | None = None,
    opponent_mcts_simulations: int = 0,
    opponent_mcts_actions: int = 16,
    iteration: int = 0,
    checkpoint_dirs: tuple[str | Path, ...] = (),
    lag_iterations: int = 0,
) -> EvaluationStats:
    """Play games against random, optionally selecting model moves by MCTS."""

    del (
        checkpoint,
        checkpoint_cache,
        opponent_mcts_simulations,
        opponent_mcts_actions,
        iteration,
        checkpoint_dirs,
        lag_iterations,
    )
    if games <= 0:
        raise ValueError("evaluation games must be positive")
    if depth != 0:
        raise ValueError("random evaluation does not use a search depth")

    import hexo_rs

    rng = random.Random(seed)
    rust_config = _rust_game_config(game_config)
    records = [
        {
            "game": hexo_rs.GameState(rust_config),
            "model_side": "P1" if index % 2 == 0 else "P2",
        }
        for index in range(games)
    ]
    active = list(range(games))
    was_training = model.training
    model.eval()
    choose_mcts = _mcts_move_selector(
        model,
        model_config=model_config,
        algorithm=algorithm,
        simulations=mcts_simulations,
        actions=mcts_actions,
        device=device,
        precision=precision,
        seed=seed,
    )

    try:
        with torch.inference_mode():
            while active:
                model_indices: list[int] = []
                for index in active:
                    game = records[index]["game"]
                    if game.current_player() == records[index]["model_side"]:
                        if choose_mcts is None:
                            model_indices.append(index)
                        else:
                            q, r = choose_mcts(game)
                            game.apply_move(q, r)
                    else:
                        q, r = rng.choice(game.legal_moves())
                        game.apply_move(q, r)

                if model_indices:
                    states = [records[index]["game"] for index in model_indices]
                    chunks = _klent_policy_chunks(
                        model,
                        states,
                        model_config=model_config,
                        device=device,
                        precision=precision,
                    )
                    for index, logits in zip(
                        model_indices, chunks, strict=True
                    ):
                        game = records[index]["game"]
                        action_index = int(logits.argmax().item())
                        q, r = game.legal_moves()[action_index]
                        game.apply_move(q, r)

                active = [
                    index
                    for index in active
                    if (
                        not records[index]["game"].is_terminal()
                        and records[index]["game"].move_count()
                        < game_config.rollout_horizon
                    )
                ]
    finally:
        model.train(was_training)

    wins = losses = truncations = 0
    lengths: list[int] = []
    for record in records:
        game = record["game"]
        winner = game.winner()
        lengths.append(int(game.move_count()))
        if winner is None:
            truncations += 1
        elif winner == record["model_side"]:
            wins += 1
        else:
            losses += 1
    decided = wins + losses
    return EvaluationStats(
        games=games,
        wins=wins,
        losses=losses,
        truncations=truncations,
        decided_rate=decided / games,
        win_rate_decided=wins / decided if decided else 0.0,
        mean_game_length=sum(lengths) / games,
        mean_opponent_depth=0.0,
    )


def evaluate_vs_checkpoint(
    model: KlentNet,
    *,
    model_config,
    game_config,
    games: int,
    checkpoint: str,
    device: torch.device,
    precision: str = "float32",
    seed: int | None = None,
    depth: int = 0,
    algorithm: AlgorithmConfig | None = None,
    mcts_simulations: int = 0,
    mcts_actions: int = 16,
    checkpoint_cache: CheckpointOpponentCache | None = None,
    opponent_mcts_simulations: int = 0,
    opponent_mcts_actions: int = 16,
    iteration: int = 0,
    checkpoint_dirs: tuple[str | Path, ...] = (),
    lag_iterations: int = 0,
    opening_plies: int = 0,
    opening_temperature: float = 0.5,
    opening_generator: str = "alternate",
) -> EvaluationStats:
    """Play against one fixed KLENT or HeXO-A0/Strix checkpoint.

    The two sides have independent MCTS budgets. Zero simulations selects that
    side's greedy raw policy. With ``opening_plies > 0``, one raw-policy
    opening is sampled per two-game pair and replayed with model sides swapped;
    in-tree Gumbel noise is then disabled on both sides, matching standalone
    head-to-head.
    """

    if games <= 0:
        raise ValueError("evaluation games must be positive")
    if depth != 0:
        raise ValueError("checkpoint evaluation does not use a search depth")
    if not checkpoint:
        raise ValueError("checkpoint evaluation requires a checkpoint path")
    if opening_plies < 0:
        raise ValueError("opening_plies cannot be negative")
    if opening_temperature <= 0:
        raise ValueError("opening_temperature must be positive")
    if opening_plies > 0 and games % 2 != 0:
        raise ValueError(
            "paired-opening checkpoint evaluation requires an even number "
            "of games"
        )
    if opening_plies >= game_config.rollout_horizon:
        raise ValueError(
            "opening_plies must be below the evaluation rollout horizon"
        )
    del iteration, checkpoint_dirs, lag_iterations

    import hexo_rs

    rust_config = _rust_game_config(game_config)
    records = [
        {
            "game": hexo_rs.GameState(rust_config),
            "model_side": "P1" if index % 2 == 0 else "P2",
        }
        for index in range(games)
    ]
    active = list(range(games))
    was_training = model.training
    model.eval()
    choose_mcts = _mcts_move_selector(
        model,
        model_config=model_config,
        algorithm=algorithm,
        simulations=mcts_simulations,
        actions=mcts_actions,
        device=device,
        precision=precision,
        seed=seed,
        disable_gumbel_noise=opening_plies > 0,
    )
    cache = checkpoint_cache or CheckpointOpponentCache()
    paired_openings: list[list[tuple[int, int]]] = []

    try:
        with cache.activate(checkpoint, device) as loaded:
            opponent = loaded.model
            opponent_config = loaded.model_config
            if opening_plies > 0:
                from hexo_klent.mcts_adapter import KlentMCTSAdapter

                candidate = KlentMCTSAdapter(
                    model,
                    algorithm or AlgorithmConfig(),
                ).to(device).eval()
                paired_openings = _sample_paired_checkpoint_openings(
                    candidate=candidate,
                    candidate_config=model_config,
                    opponent=opponent,
                    opponent_config=opponent_config,
                    rust_game_config=rust_config,
                    games=games,
                    opening_plies=opening_plies,
                    opening_temperature=opening_temperature,
                    opening_generator=opening_generator,
                    device=device,
                    seed=seed,
                )
                for pair_index, opening in enumerate(paired_openings):
                    for game_index in (2 * pair_index, 2 * pair_index + 1):
                        game = records[game_index]["game"]
                        for q, r in opening:
                            game.apply_move(q, r)
            choose_opponent_mcts = _compatible_mcts_move_selector(
                opponent,
                model_config=opponent_config,
                simulations=opponent_mcts_simulations,
                actions=opponent_mcts_actions,
                device=device,
                precision=precision,
                seed=(
                    None
                    if seed is None
                    else seed + 500_000_003
                ),
                disable_gumbel_noise=opening_plies > 0,
            )
            with torch.inference_mode():
                while active:
                    model_indices: list[int] = []
                    opponent_indices: list[int] = []
                    for index in active:
                        game = records[index]["game"]
                        if game.current_player() == records[index]["model_side"]:
                            if choose_mcts is None:
                                model_indices.append(index)
                            else:
                                q, r = choose_mcts(game)
                                game.apply_move(q, r)
                        else:
                            if choose_opponent_mcts is None:
                                opponent_indices.append(index)
                            else:
                                q, r = choose_opponent_mcts(game)
                                game.apply_move(q, r)

                    if model_indices:
                        states = [
                            records[index]["game"] for index in model_indices
                        ]
                        chunks = _klent_policy_chunks(
                            model,
                            states,
                            model_config=model_config,
                            device=device,
                            precision=precision,
                        )
                        for index, logits in zip(
                            model_indices, chunks, strict=True
                        ):
                            game = records[index]["game"]
                            action_index = int(logits.argmax().item())
                            q, r = game.legal_moves()[action_index]
                            game.apply_move(q, r)

                    if opponent_indices:
                        states = [
                            records[index]["game"]
                            for index in opponent_indices
                        ]
                        policy_logits, _values = _compatible_outputs(
                            opponent,
                            states,
                            model_config=opponent_config,
                            device=device,
                            precision=precision,
                        )
                        for index, logits in zip(
                            opponent_indices, policy_logits, strict=True
                        ):
                            game = records[index]["game"]
                            action_index = int(logits.argmax().item())
                            q, r = game.legal_moves()[action_index]
                            game.apply_move(q, r)

                    active = [
                        index
                        for index in active
                        if (
                            not records[index]["game"].is_terminal()
                            and records[index]["game"].move_count()
                            < game_config.rollout_horizon
                        )
                    ]
    finally:
        model.train(was_training)

    wins = losses = truncations = 0
    lengths: list[int] = []
    for record in records:
        game = record["game"]
        winner = game.winner()
        lengths.append(int(game.move_count()))
        if winner is None:
            truncations += 1
        elif winner == record["model_side"]:
            wins += 1
        else:
            losses += 1
    decided = wins + losses
    unique_opening_fraction = (
        len({tuple(opening) for opening in paired_openings})
        / len(paired_openings)
        if paired_openings
        else 0.0
    )
    return EvaluationStats(
        games=games,
        wins=wins,
        losses=losses,
        truncations=truncations,
        decided_rate=decided / games,
        win_rate_decided=wins / decided if decided else 0.0,
        mean_game_length=sum(lengths) / games,
        mean_opponent_depth=0.0,
        opening_pairs=len(paired_openings),
        frac_unique_opening=unique_opening_fraction,
    )


def evaluate_vs_lagged(
    model: KlentNet,
    *,
    model_config,
    game_config,
    games: int,
    device: torch.device,
    iteration: int,
    checkpoint_dirs: tuple[str | Path, ...],
    lag_iterations: int,
    precision: str = "float32",
    seed: int | None = None,
    depth: int = 0,
    algorithm: AlgorithmConfig | None = None,
    mcts_simulations: int = 0,
    mcts_actions: int = 16,
    checkpoint: str = "",
    checkpoint_cache: CheckpointOpponentCache | None = None,
    opponent_mcts_simulations: int = 0,
    opponent_mcts_actions: int = 16,
    opening_plies: int = 0,
    opening_temperature: float = 0.5,
    opening_generator: str = "alternate",
) -> EvaluationStats:
    """Evaluate against the exact checkpoint a configured generation lag ago."""

    del checkpoint, opponent_mcts_simulations, opponent_mcts_actions
    lagged_checkpoint = resolve_lagged_checkpoint(
        iteration=iteration,
        lag_iterations=lag_iterations,
        checkpoint_dirs=checkpoint_dirs,
    )
    return evaluate_vs_checkpoint(
        model,
        model_config=model_config,
        game_config=game_config,
        games=games,
        checkpoint=str(lagged_checkpoint),
        device=device,
        precision=precision,
        seed=seed,
        depth=depth,
        algorithm=algorithm,
        mcts_simulations=mcts_simulations,
        mcts_actions=mcts_actions,
        checkpoint_cache=checkpoint_cache,
        opponent_mcts_simulations=mcts_simulations,
        opponent_mcts_actions=mcts_actions,
        opening_plies=opening_plies,
        opening_temperature=opening_temperature,
        opening_generator=opening_generator,
    )


def evaluate_vs_sealbot(
    model: KlentNet,
    *,
    model_config,
    game_config,
    games: int,
    device: torch.device,
    precision: str = "float32",
    seed: int | None = None,
    depth: int,
    algorithm: AlgorithmConfig | None = None,
    mcts_simulations: int = 0,
    mcts_actions: int = 16,
    checkpoint: str = "",
    checkpoint_cache: CheckpointOpponentCache | None = None,
    opponent_mcts_simulations: int = 0,
    opponent_mcts_actions: int = 16,
    iteration: int = 0,
    checkpoint_dirs: tuple[str | Path, ...] = (),
    lag_iterations: int = 0,
) -> EvaluationStats:
    """Play against fixed-depth SealBot, optionally selecting via MCTS."""

    del (
        checkpoint,
        checkpoint_cache,
        opponent_mcts_simulations,
        opponent_mcts_actions,
        iteration,
        checkpoint_dirs,
        lag_iterations,
    )
    if games <= 0:
        raise ValueError("evaluation games must be positive")
    if depth <= 0:
        raise ValueError("SealBot depth must be positive")

    import hexo_rs
    from hexo_a0.sealbot_eval import _ensure_sealbot

    _ensure_sealbot()
    from game import HexGame, Player as SealPlayer
    from minimax_cpp import MinimaxBot

    rust_config = _rust_game_config(game_config)
    records = []
    for index in range(games):
        seal_game = HexGame(win_length=game_config.win_length)
        if not seal_game.make_move(0, 0):
            raise RuntimeError("SealBot rejected HeXO's forced opening")
        sealbot = MinimaxBot(time_limit=3600.0)
        sealbot.max_depth = depth
        records.append(
            {
                "game": hexo_rs.GameState(rust_config),
                "seal_game": seal_game,
                "sealbot": sealbot,
                "model_side": "P1" if index % 2 == 0 else "P2",
            }
        )

    active = list(range(games))
    opponent_depth_total = 0
    opponent_searches = 0
    was_training = model.training
    model.eval()
    choose_mcts = _mcts_move_selector(
        model,
        model_config=model_config,
        algorithm=algorithm,
        simulations=mcts_simulations,
        actions=mcts_actions,
        device=device,
        precision=precision,
        seed=seed,
    )

    try:
        with torch.inference_mode():
            while active:
                model_indices: list[int] = []
                for index in active:
                    record = records[index]
                    game = record["game"]
                    if game.current_player() == record["model_side"]:
                        if choose_mcts is None:
                            model_indices.append(index)
                        else:
                            q, r = choose_mcts(game)
                            if not record["seal_game"].make_move(q, r):
                                raise RuntimeError(
                                    "MCTS model move was illegal in SealBot"
                                )
                            game.apply_move(q, r)
                        continue

                    seal_game = record["seal_game"]
                    seal_player = (
                        "P1"
                        if seal_game.current_player == SealPlayer.A
                        else "P2"
                    )
                    if seal_player != game.current_player():
                        raise RuntimeError(
                            "SealBot and Rust player turns diverged"
                        )
                    moves = record["sealbot"].get_move(seal_game)
                    opponent_depth_total += int(record["sealbot"].last_depth)
                    opponent_searches += 1
                    for move in moves:
                        if (
                            seal_game.game_over
                            or game.is_terminal()
                            or game.move_count() >= game_config.rollout_horizon
                        ):
                            break
                        q, r = int(move[0]), int(move[1])
                        if not seal_game.make_move(q, r):
                            raise RuntimeError(
                                f"SealBot returned illegal move {(q, r)}"
                            )
                        game.apply_move(q, r)

                if model_indices:
                    states = [records[index]["game"] for index in model_indices]
                    chunks = _klent_policy_chunks(
                        model,
                        states,
                        model_config=model_config,
                        device=device,
                        precision=precision,
                    )
                    for index, logits in zip(
                        model_indices, chunks, strict=True
                    ):
                        record = records[index]
                        game = record["game"]
                        if (
                            game.is_terminal()
                            or game.move_count() >= game_config.rollout_horizon
                        ):
                            continue
                        action_index = int(logits.argmax().item())
                        q, r = game.legal_moves()[action_index]
                        if not record["seal_game"].make_move(q, r):
                            raise RuntimeError(
                                "greedy model move was illegal in SealBot"
                            )
                        game.apply_move(q, r)

                active = [
                    index
                    for index in active
                    if (
                        not records[index]["game"].is_terminal()
                        and records[index]["game"].move_count()
                        < game_config.rollout_horizon
                    )
                ]
    finally:
        model.train(was_training)

    wins = losses = truncations = 0
    lengths: list[int] = []
    for record in records:
        game = record["game"]
        winner = game.winner()
        lengths.append(int(game.move_count()))
        if winner is None:
            truncations += 1
        elif winner == record["model_side"]:
            wins += 1
        else:
            losses += 1
    decided = wins + losses
    return EvaluationStats(
        games=games,
        wins=wins,
        losses=losses,
        truncations=truncations,
        decided_rate=decided / games,
        win_rate_decided=wins / decided if decided else 0.0,
        mean_game_length=sum(lengths) / games,
        mean_opponent_depth=(
            opponent_depth_total / opponent_searches
            if opponent_searches
            else 0.0
        ),
    )


OpponentEvaluator = Callable[..., EvaluationStats]
OPPONENT_EVALUATORS: dict[str, OpponentEvaluator] = {
    "random": evaluate_vs_random,
    "sealbot": evaluate_vs_sealbot,
    "checkpoint": evaluate_vs_checkpoint,
    "lagged": evaluate_vs_lagged,
}


def evaluate_opponent(
    kind: str,
    model: KlentNet,
    *,
    model_config,
    game_config,
    games: int,
    depth: int,
    algorithm: AlgorithmConfig | None,
    mcts_simulations: int,
    mcts_actions: int,
    device: torch.device,
    precision: str = "float32",
    seed: int | None = None,
    checkpoint: str = "",
    checkpoint_cache: CheckpointOpponentCache | None = None,
    opponent_mcts_simulations: int = 0,
    opponent_mcts_actions: int = 16,
    iteration: int = 0,
    checkpoint_dirs: tuple[str | Path, ...] = (),
    lag_iterations: int = 0,
    opening_plies: int = 0,
    opening_temperature: float = 0.5,
    opening_generator: str = "alternate",
) -> EvaluationStats:
    """Dispatch one configured opponent through the evaluator registry."""

    try:
        evaluator = OPPONENT_EVALUATORS[kind]
    except KeyError as error:
        raise ValueError(f"unknown evaluation opponent kind: {kind}") from error
    kwargs = dict(
        model_config=model_config,
        game_config=game_config,
        games=games,
        depth=depth,
        algorithm=algorithm,
        mcts_simulations=mcts_simulations,
        mcts_actions=mcts_actions,
        device=device,
        precision=precision,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_cache=checkpoint_cache,
        opponent_mcts_simulations=opponent_mcts_simulations,
        opponent_mcts_actions=opponent_mcts_actions,
        iteration=iteration,
        checkpoint_dirs=checkpoint_dirs,
        lag_iterations=lag_iterations,
    )
    if kind in {"checkpoint", "lagged"}:
        kwargs.update(
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            opening_generator=opening_generator,
        )
    return evaluator(model, **kwargs)
