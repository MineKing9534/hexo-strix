"""Fixed-checkpoint SPRT matches using KLENT's MCTS inference path."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from hexo_a0.head_to_head import _atomic_write_json, _score_to_elo
from hexo_a0.sealbot_eval import _win_rate_stats
from hexo_a0.sprt_eval import SPRTConfig, SPRTState
from hexo_klent.actor import _rust_game_config
from hexo_klent.config import GameConfig
from hexo_klent.evaluation import _compatible_mcts_move_selector
from hexo_klent.mcts_adapter import LoadedKlentMCTS, load_checkpoint


logger = logging.getLogger(__name__)

Outcome = Literal["W", "D", "L"]


@dataclass(frozen=True)
class GameResult:
    """One game, expressed from the candidate checkpoint's perspective."""

    outcome: Outcome
    moves: int
    truncated: bool = False


@dataclass(frozen=True)
class MatchResult:
    """Terminal statistics for one fixed-checkpoint SPRT."""

    decision: str
    games: int
    wins: int
    draws: int
    losses: int
    truncations: int
    score: float
    llr: float
    elapsed_seconds: float


def _compile_network(loaded: LoadedKlentMCTS) -> None:
    """Compile the KLENT core while leaving its MCTS adapter intact."""

    loaded.model.network._forward_batch_core = torch.compile(
        loaded.model.network._forward_batch_core,
        dynamic=True,
    )


def _validate_match_settings(
    *,
    max_games: int,
    max_moves: int,
    mcts_simulations: int,
    mcts_actions: int,
    sprt_config: SPRTConfig,
) -> None:
    if max_games <= 0 or max_games % 2:
        raise ValueError("max_games must be a positive even number")
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")
    if mcts_simulations <= 0:
        raise ValueError("MCTS simulations must be positive")
    if mcts_actions <= 0:
        raise ValueError("MCTS actions must be positive")
    if not 0.0 <= sprt_config.s0 < sprt_config.s1 <= 1.0:
        raise ValueError("SPRT scores must satisfy 0 <= s0 < s1 <= 1")
    if not 0.0 < sprt_config.alpha < 1.0:
        raise ValueError("SPRT alpha must be between zero and one")
    if not 0.0 < sprt_config.beta < 1.0:
        raise ValueError("SPRT beta must be between zero and one")
    if sprt_config.pair_variance <= 0.0:
        raise ValueError("SPRT pair_variance must be positive")


def _state_payload(
    *,
    state: SPRTState,
    sprt_config: SPRTConfig,
    candidate: Path,
    opponent: Path,
    candidate_iteration: int | str,
    opponent_iteration: int | str,
    truncations: int,
    total_moves: int,
    elapsed_seconds: float,
    settings: dict,
) -> dict:
    score, ci_low, ci_high, elo = _win_rate_stats(
        state.wins,
        state.losses,
        state.draws,
    )
    lower, upper = sprt_config.bounds()
    return {
        "timestamp": time.time(),
        "candidate": str(candidate),
        "opponent": str(opponent),
        "candidate_iteration": candidate_iteration,
        "opponent_iteration": opponent_iteration,
        "games": state.games,
        "pairs": state.pairs,
        "wins": state.wins,
        "draws": state.draws,
        "losses": state.losses,
        "truncations": truncations,
        "score": score,
        "score_ci_low": ci_low,
        "score_ci_high": ci_high,
        "elo": elo,
        "elo_ci_low": _score_to_elo(ci_low),
        "elo_ci_high": _score_to_elo(ci_high),
        "llr": state.llr,
        "decision": state.decision,
        "bounds": {"lower": lower, "upper": upper},
        "pair_variance": sprt_config.pair_variance,
        "total_moves": total_moves,
        "mean_game_length": (
            total_moves / state.games if state.games else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": (
            state.games / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "outcomes": "".join(state._outcomes),
        "settings": settings,
    }


def run_sprt_games(
    play_game: Callable[[Literal["P1", "P2"]], GameResult],
    *,
    sprt_config: SPRTConfig,
    max_games: int,
    state_file: Path | None = None,
    candidate: Path = Path("candidate"),
    opponent: Path = Path("opponent"),
    candidate_iteration: int | str = "?",
    opponent_iteration: int | str = "?",
    settings: dict | None = None,
) -> MatchResult:
    """Run alternating-side game pairs until Wald decides or the cap is hit."""

    _validate_match_settings(
        max_games=max_games,
        max_moves=int((settings or {}).get("max_moves", 1)),
        mcts_simulations=int(
            (settings or {}).get("mcts_simulations", 1)
        ),
        mcts_actions=int((settings or {}).get("mcts_actions", 1)),
        sprt_config=sprt_config,
    )
    state = SPRTState()
    truncations = 0
    total_moves = 0
    started_at = time.monotonic()

    while state.games < max_games and state.decision == "continue":
        pair_results = (play_game("P1"), play_game("P2"))
        for game_result in pair_results:
            state.record(game_result.outcome, sprt_config)
            total_moves += game_result.moves
            truncations += int(game_result.truncated)

        elapsed = time.monotonic() - started_at
        payload = _state_payload(
            state=state,
            sprt_config=sprt_config,
            candidate=candidate,
            opponent=opponent,
            candidate_iteration=candidate_iteration,
            opponent_iteration=opponent_iteration,
            truncations=truncations,
            total_moves=total_moves,
            elapsed_seconds=elapsed,
            settings=settings or {},
        )
        if state_file is not None:
            _atomic_write_json(state_file, payload)
        logger.info(
            "games=%d  pair=%s%s  W-D-L=%d-%d-%d  score=%.3f  "
            "Elo=%+.0f  LLR=%+.3f  %s",
            state.games,
            pair_results[0].outcome,
            pair_results[1].outcome,
            state.wins,
            state.draws,
            state.losses,
            state.score,
            payload["elo"],
            state.llr,
            state.decision,
        )

    elapsed = time.monotonic() - started_at
    return MatchResult(
        decision=state.decision,
        games=state.games,
        wins=state.wins,
        draws=state.draws,
        losses=state.losses,
        truncations=truncations,
        score=state.score,
        llr=state.llr,
        elapsed_seconds=elapsed,
    )


def run_checkpoint_sprt(
    candidate: str | Path,
    opponent: str | Path,
    *,
    win_length: int = 6,
    radius: int = 2,
    max_moves: int = 1000,
    mcts_simulations: int = 24,
    mcts_actions: int = 8,
    device_str: str = "cuda",
    precision: str = "bf16",
    s0: float = 0.50,
    s1: float = 0.55,
    alpha: float = 0.05,
    beta: float = 0.05,
    pair_variance: float = 0.50,
    max_games: int = 1000,
    seed: int | None = 0,
    state_file: str | Path | None = None,
    compile_model: bool = True,
) -> MatchResult:
    """Compare two immutable KLENT checkpoints with identical MCTS settings."""

    candidate_path = Path(candidate).expanduser().resolve()
    opponent_path = Path(opponent).expanduser().resolve()
    state_path = (
        None
        if state_file is None
        else Path(state_file).expanduser().resolve()
    )
    for path, label in (
        (candidate_path, "candidate"),
        (opponent_path, "opponent"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but is unavailable")
    if precision not in {"float32", "bf16"}:
        raise ValueError("precision must be 'float32' or 'bf16'")

    sprt_config = SPRTConfig(
        s0=s0,
        s1=s1,
        alpha=alpha,
        beta=beta,
        pair_variance=pair_variance,
        window_size=None,
        pentanomial=True,
    )
    _validate_match_settings(
        max_games=max_games,
        max_moves=max_moves,
        mcts_simulations=mcts_simulations,
        mcts_actions=mcts_actions,
        sprt_config=sprt_config,
    )
    game_config = GameConfig(
        win_length=win_length,
        placement_radius=radius,
        rollout_horizon=max_moves,
    )

    logger.info("loading candidate %s", candidate_path)
    candidate_loaded = load_checkpoint(candidate_path, device)
    logger.info("loading opponent %s", opponent_path)
    opponent_loaded = load_checkpoint(opponent_path, device)
    if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            "/tmp/torchinductor_hexo/klent",
        )
        _compile_network(candidate_loaded)
        _compile_network(opponent_loaded)

    choose_candidate = _compatible_mcts_move_selector(
        candidate_loaded.model,
        model_config=candidate_loaded.model_config,
        simulations=mcts_simulations,
        actions=mcts_actions,
        device=device,
        precision=precision,
        seed=seed,
    )
    choose_opponent = _compatible_mcts_move_selector(
        opponent_loaded.model,
        model_config=opponent_loaded.model_config,
        simulations=mcts_simulations,
        actions=mcts_actions,
        device=device,
        precision=precision,
        seed=None if seed is None else seed + 500_000_003,
    )
    assert choose_candidate is not None
    assert choose_opponent is not None

    import hexo_rs

    rust_config = _rust_game_config(game_config)

    def play_game(candidate_side: Literal["P1", "P2"]) -> GameResult:
        game = hexo_rs.GameState(rust_config)
        with torch.inference_mode():
            while (
                not game.is_terminal()
                and game.move_count() < max_moves
            ):
                chooser = (
                    choose_candidate
                    if game.current_player() == candidate_side
                    else choose_opponent
                )
                q, r = chooser(game)
                game.apply_move(q, r)
        winner = game.winner()
        truncated = winner is None
        if truncated:
            outcome: Outcome = "D"
        elif winner == candidate_side:
            outcome = "W"
        else:
            outcome = "L"
        return GameResult(
            outcome=outcome,
            moves=int(game.move_count()),
            truncated=truncated,
        )

    settings = {
        "win_length": win_length,
        "radius": radius,
        "max_moves": max_moves,
        "mcts_simulations": mcts_simulations,
        "mcts_actions": mcts_actions,
        "device": str(device),
        "precision": precision,
        "s0": s0,
        "s1": s1,
        "alpha": alpha,
        "beta": beta,
        "pair_variance": pair_variance,
        "max_games": max_games,
        "seed": seed,
        "compile": compile_model,
    }
    lower, upper = sprt_config.bounds()
    logger.info(
        "candidate iteration=%s vs opponent iteration=%s; "
        "MCTS=%d/%d, max_moves=%d, max_games=%d",
        candidate_loaded.iteration,
        opponent_loaded.iteration,
        mcts_simulations,
        mcts_actions,
        max_moves,
        max_games,
    )
    logger.info(
        "SPRT H0=%.3f H1=%.3f, bounds=[%.3f, %.3f], "
        "pentanomial pair_variance=%.3f",
        s0,
        s1,
        lower,
        upper,
        pair_variance,
    )
    result = run_sprt_games(
        play_game,
        sprt_config=sprt_config,
        max_games=max_games,
        state_file=state_path,
        candidate=candidate_path,
        opponent=opponent_path,
        candidate_iteration=candidate_loaded.iteration,
        opponent_iteration=opponent_loaded.iteration,
        settings=settings,
    )
    logger.info(
        "SPRT complete: %s after %d games; W-D-L=%d-%d-%d, "
        "score=%.3f, LLR=%+.3f, elapsed=%.1fs",
        result.decision,
        result.games,
        result.wins,
        result.draws,
        result.losses,
        result.score,
        result.llr,
        result.elapsed_seconds,
    )
    return result
