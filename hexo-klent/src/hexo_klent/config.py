"""Configuration for the standalone HeXO KLENT experiment."""

from __future__ import annotations

import dataclasses
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from hexo_a0.config import ModelConfig


@dataclass
class KlentModelConfig(ModelConfig):
    """KLENT network plus its execution representation."""

    architecture: str = "graph"
    dense_ray_radius: int = 5
    ray_channels: int = 12
    ray_update_hidden: int = 48
    ray_branch_scale: float = 1.0
    exact_graft_init: bool = True
    ray_after_layers: list[int] = field(default_factory=list)


@dataclass
class GameConfig:
    """HeXO rules plus the finite horizon used to bound a rollout.

    HeXO has no draw outcome. ``rollout_horizon`` is therefore a truncation
    boundary: KLENT bootstraps from the final non-terminal position instead of
    training it as a zero-valued draw.
    """

    win_length: int = 6
    placement_radius: int = 8
    rollout_horizon: int = 200


@dataclass
class AlgorithmConfig:
    """KLENT policy improvement and trace parameters."""

    alpha: float = 0.03
    beta: float = 0.1
    tau: float = 8.0

    @property
    def trace_decay(self) -> float:
        """Paper's lambda = exp(-1 / tau)."""

        return math.exp(-1.0 / self.tau)


@dataclass
class CollectionConfig:
    """Fresh on-policy self-play batch."""

    positions_per_iteration: int = 32_768
    parallel_games: int = 64
    inference_batch_size: int = 64
    inference_edge_budget: int = 250_000
    # Dense rasters scale with the spatial bounding box rather than the number
    # of active cells. End a wandering lane at this exact, bootstrapped
    # truncation boundary before it can create an unsafe fit example. Zero
    # disables the boundary (and remains the graph-backend default).
    dense_position_cell_limit: int = 0
    workers: int = 1
    batch_timeout_ms: float = 2.0


@dataclass
class TrainingConfig:
    """The single epoch fitted to each fresh self-play batch."""

    batch_size: int = 256
    edge_budget: int = 250_000
    policy_diagnostic_samples: int = 2_048
    grad_accumulation: bool = True
    prefetch_batches: bool = True
    fit_max_autotune: bool = False
    fit_compile_seed_nodes: int = 0
    learning_rate: float = 1e-3
    learning_rate_warmup_iterations: int = 0
    learning_rate_warmup_start_factor: float = 0.1
    weight_decay: float = 1e-4
    q_loss_weight: float = 1.0
    max_grad_norm: float = 1.0


@dataclass
class EvaluationOpponentConfig:
    """One named opponent in the periodic evaluation suite."""

    name: str = ""
    kind: str = "random"
    checkpoint: str = ""
    lag_iterations: int = 0
    best_promotion_win_rate: float = 0.55
    games: int = 64
    depth: int = 0
    placement_radius: int = 0
    mcts_simulations: int = 0
    mcts_actions: int = 16
    opponent_mcts_simulations: int = 0
    opponent_mcts_actions: int = 16


def _default_evaluation_opponents() -> list[EvaluationOpponentConfig]:
    return [EvaluationOpponentConfig()]


@dataclass
class EvaluationConfig:
    """Periodic evaluation against configured opponents."""

    interval: int = 10
    # Fixed/lagged checkpoint matches use the same paired-opening protocol as
    # head-to-head: sample one opening per pair, replay it with sides swapped,
    # then disable in-tree Gumbel noise.  Random and SealBot probes retain
    # their existing protocols.
    opening_plies: int = 8
    opening_temperature: float = 0.5
    opening_generator: str = "alternate"
    opponents: list[EvaluationOpponentConfig] = field(
        default_factory=_default_evaluation_opponents
    )


@dataclass
class RunConfig:
    """Runtime and artifact settings."""

    iterations: int = 1_000
    device: str = "cuda"
    precision: str = "bf16"
    compile: bool = True
    output_dir: str = "runs/klent/reference-w1"
    checkpoint_interval: int = 10
    seed: int | None = None


@dataclass
class Config:
    """Complete reference experiment configuration."""

    model: KlentModelConfig = field(default_factory=KlentModelConfig)
    game: GameConfig = field(default_factory=GameConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run: RunConfig = field(default_factory=RunConfig)


_T = TypeVar("_T")
_SECTIONS: dict[str, type[Any]] = {
    "model": KlentModelConfig,
    "game": GameConfig,
    "algorithm": AlgorithmConfig,
    "collection": CollectionConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "run": RunConfig,
}


def _load_section(cls: type[_T], raw: Any, name: str) -> _T:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown [{name}] keys: {', '.join(unknown)}")
    return cls(**raw)


def _load_evaluation_section(raw: Any) -> EvaluationConfig:
    if raw is None:
        return EvaluationConfig()
    if not isinstance(raw, dict):
        raise ValueError("[evaluation] must be a TOML table")
    known = {item.name for item in dataclasses.fields(EvaluationConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"unknown [evaluation] keys: {', '.join(unknown)}"
        )
    raw_opponents = raw.get("opponents")
    if raw_opponents is None:
        opponents = _default_evaluation_opponents()
    else:
        if not isinstance(raw_opponents, list):
            raise ValueError("[evaluation].opponents must be an array of tables")
        opponents = [
            _load_section(
                EvaluationOpponentConfig,
                item,
                "evaluation.opponents",
            )
            for item in raw_opponents
        ]
    return EvaluationConfig(
        interval=raw.get("interval", 10),
        opening_plies=raw.get("opening_plies", 8),
        opening_temperature=raw.get("opening_temperature", 0.5),
        opening_generator=raw.get("opening_generator", "alternate"),
        opponents=opponents,
    )


def _validate(cfg: Config) -> None:
    if cfg.algorithm.alpha < 0 or cfg.algorithm.beta < 0:
        raise ValueError("algorithm alpha and beta must be non-negative")
    if cfg.algorithm.alpha + cfg.algorithm.beta <= 0:
        raise ValueError("algorithm alpha + beta must be positive")
    if cfg.algorithm.tau <= 0:
        raise ValueError("algorithm tau must be positive")
    if cfg.game.win_length < 2:
        raise ValueError("game.win_length must be at least 2")
    if cfg.game.placement_radius < 1:
        raise ValueError("game.placement_radius must be positive")
    if cfg.game.rollout_horizon <= 0:
        raise ValueError("game.rollout_horizon must be positive")
    if cfg.collection.positions_per_iteration <= 0:
        raise ValueError("collection.positions_per_iteration must be positive")
    if cfg.collection.parallel_games <= 0:
        raise ValueError("collection.parallel_games must be positive")
    if (
        cfg.collection.parallel_games
        > cfg.collection.positions_per_iteration
    ):
        raise ValueError(
            "collection.parallel_games cannot exceed "
            "collection.positions_per_iteration"
        )
    if cfg.collection.inference_batch_size <= 0:
        raise ValueError("collection.inference_batch_size must be positive")
    if cfg.collection.inference_edge_budget < 0:
        raise ValueError("collection.inference_edge_budget cannot be negative")
    if cfg.collection.dense_position_cell_limit < 0:
        raise ValueError(
            "collection.dense_position_cell_limit cannot be negative"
        )
    if (
        cfg.collection.dense_position_cell_limit > 0
        and cfg.model.architecture
        not in {"dense_axis", "persistent_ray_axis"}
    ):
        raise ValueError(
            "collection.dense_position_cell_limit is only valid for "
            "dense raster model architectures"
        )
    if cfg.collection.workers <= 0:
        raise ValueError("collection.workers must be positive")
    if cfg.collection.batch_timeout_ms < 0:
        raise ValueError("collection.batch_timeout_ms cannot be negative")
    if cfg.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if cfg.training.edge_budget < 0:
        raise ValueError("training.edge_budget cannot be negative")
    if cfg.training.policy_diagnostic_samples < 0:
        raise ValueError(
            "training.policy_diagnostic_samples cannot be negative"
        )
    if cfg.training.fit_compile_seed_nodes < 0:
        raise ValueError("training.fit_compile_seed_nodes cannot be negative")
    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if cfg.training.learning_rate_warmup_iterations < 0:
        raise ValueError(
            "training.learning_rate_warmup_iterations cannot be negative"
        )
    if not 0 <= cfg.training.learning_rate_warmup_start_factor <= 1:
        raise ValueError(
            "training.learning_rate_warmup_start_factor must be in [0, 1]"
        )
    if cfg.training.q_loss_weight < 0:
        raise ValueError("training.q_loss_weight cannot be negative")
    if cfg.run.iterations <= 0:
        raise ValueError("run.iterations must be positive")
    if cfg.run.checkpoint_interval < 0:
        raise ValueError("run.checkpoint_interval cannot be negative")
    if cfg.run.precision not in {"float32", "bf16"}:
        raise ValueError("run.precision must be 'float32' or 'bf16'")
    if cfg.evaluation.interval < 0:
        raise ValueError("evaluation interval cannot be negative")
    if cfg.evaluation.opening_plies < 0:
        raise ValueError("evaluation opening_plies cannot be negative")
    if cfg.evaluation.opening_temperature <= 0:
        raise ValueError("evaluation opening_temperature must be positive")
    if cfg.evaluation.opening_generator not in {
        "alternate",
        "a",
        "b",
        "champion",
    }:
        raise ValueError(
            "evaluation opening_generator must be 'alternate', 'a', 'b', "
            "or 'champion'"
        )
    opponent_names: set[str] = set()
    for opponent in cfg.evaluation.opponents:
        if opponent.kind not in {
            "random",
            "sealbot",
            "checkpoint",
            "lagged",
            "best_so_far",
        }:
            raise ValueError(
                f"unknown evaluation opponent kind: {opponent.kind}"
            )
        opponent_name = opponent.name or opponent.kind
        if not re.fullmatch(r"[A-Za-z0-9_/-]+", opponent_name):
            raise ValueError(
                "evaluation opponent name may contain only letters, "
                "digits, underscores, hyphens, and slashes"
            )
        if opponent_name in opponent_names:
            raise ValueError(
                f"duplicate evaluation opponent name: {opponent_name}"
            )
        opponent_names.add(opponent_name)
        if opponent.games <= 0:
            raise ValueError("evaluation opponent games must be positive")
        if (
            cfg.evaluation.opening_plies > 0
            and opponent.kind in {
                "checkpoint",
                "lagged",
                "best_so_far",
            }
            and opponent.games % 2 != 0
        ):
            raise ValueError(
                "paired-opening checkpoint evaluation requires an even "
                "number of games"
            )
        if (
            cfg.evaluation.opening_plies > 0
            and opponent.kind in {
                "checkpoint",
                "lagged",
                "best_so_far",
            }
            and cfg.evaluation.opening_plies >= cfg.game.rollout_horizon
        ):
            raise ValueError(
                "evaluation opening_plies must be below the rollout horizon"
            )
        if opponent.kind == "sealbot" and opponent.depth <= 0:
            raise ValueError("SealBot evaluation depth must be positive")
        if opponent.kind != "sealbot" and opponent.depth != 0:
            raise ValueError(
                "evaluation depth is only supported for SealBot"
            )
        if (
            opponent.kind in {"checkpoint", "best_so_far"}
            and not opponent.checkpoint
        ):
            raise ValueError(
                f"{opponent.kind} evaluation opponent requires a "
                "checkpoint path"
            )
        if (
            opponent.kind not in {"checkpoint", "best_so_far"}
            and opponent.checkpoint
        ):
            raise ValueError(
                "evaluation checkpoint is only supported for checkpoint "
                "and best_so_far opponents"
            )
        if opponent.kind == "lagged" and opponent.lag_iterations <= 0:
            raise ValueError(
                "lagged evaluation opponent requires positive lag_iterations"
            )
        if opponent.kind != "lagged" and opponent.lag_iterations != 0:
            raise ValueError(
                "evaluation lag_iterations is only supported for lagged "
                "opponents"
            )
        if (
            opponent.kind == "best_so_far"
            and not 0.5 <= opponent.best_promotion_win_rate <= 1.0
        ):
            raise ValueError(
                "best_so_far promotion win rate must be in [0.5, 1]"
            )
        if opponent.placement_radius < 0:
            raise ValueError(
                "evaluation opponent placement_radius cannot be negative"
            )
        if opponent.mcts_simulations < 0:
            raise ValueError(
                "evaluation opponent mcts_simulations cannot be negative"
            )
        if opponent.mcts_actions <= 0:
            raise ValueError(
                "evaluation opponent mcts_actions must be positive"
            )
        if opponent.opponent_mcts_simulations < 0:
            raise ValueError(
                "evaluation opponent opponent_mcts_simulations cannot be "
                "negative"
            )
        if opponent.opponent_mcts_actions <= 0:
            raise ValueError(
                "evaluation opponent opponent_mcts_actions must be positive"
            )
        if opponent.kind not in {"checkpoint", "best_so_far"} and (
            opponent.opponent_mcts_simulations != 0
            or opponent.opponent_mcts_actions != 16
        ):
            raise ValueError(
                "opponent-side MCTS settings are only supported for "
                "checkpoint and best_so_far opponents"
            )
    if cfg.model.num_layers <= 0:
        raise ValueError("model.num_layers must be positive")
    if cfg.model.axis_window < cfg.game.win_length - 1:
        raise ValueError(
            "model.axis_window must be at least game.win_length - 1"
        )
    if cfg.model.architecture not in {
        "graph",
        "dense_axis",
        "persistent_ray_axis",
    }:
        raise ValueError(
            "model.architecture must be 'graph', 'dense_axis', or "
            "'persistent_ray_axis'"
        )
    if not 1 <= cfg.model.dense_ray_radius <= 5:
        raise ValueError("model.dense_ray_radius must be between 1 and 5")
    if cfg.model.architecture in {
        "dense_axis",
        "persistent_ray_axis",
    }:
        architecture = cfg.model.architecture
        if cfg.game.win_length - 1 > cfg.model.dense_ray_radius:
            raise ValueError(
                f"{architecture} requires model.dense_ray_radius to cover "
                "game.win_length - 1"
            )
        required = {
            "graph_type": "axis",
            "axis_relational": True,
            "threat_features": True,
            "relative_stone_encoding": True,
            "compact_stone_onehot": True,
            "node_coords": False,
            "moves_scope": "node",
            "pre_norm": True,
            "dropout": 0.0,
        }
        for name, expected in required.items():
            actual = getattr(cfg.model, name)
            if actual != expected:
                raise ValueError(
                    f"{architecture} requires model.{name}={expected!r}, "
                    f"got {actual!r}"
                )
        if cfg.model.use_jk and cfg.model.jk_mode not in {"sum", "cat"}:
            raise ValueError(
                f"{architecture} supports JK modes 'sum' and 'cat'"
            )
    if cfg.model.architecture == "persistent_ray_axis":
        if cfg.model.ray_channels <= 0:
            raise ValueError("model.ray_channels must be positive")
        if cfg.model.ray_update_hidden <= 0:
            raise ValueError("model.ray_update_hidden must be positive")
        if not math.isfinite(cfg.model.ray_branch_scale):
            raise ValueError("model.ray_branch_scale must be finite")
        if len(set(cfg.model.ray_after_layers)) != len(
            cfg.model.ray_after_layers
        ):
            raise ValueError("model.ray_after_layers must not contain duplicates")
        invalid_layers = [
            layer
            for layer in cfg.model.ray_after_layers
            if layer < 0 or layer >= cfg.model.num_layers
        ]
        if invalid_layers:
            raise ValueError(
                "model.ray_after_layers entries must be valid layer indices"
            )


def load_config(path: str | Path) -> Config:
    """Load a strict TOML configuration."""

    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"unknown top-level sections: {', '.join(unknown)}")
    sections = {
        name: (
            _load_evaluation_section(raw.get(name))
            if name == "evaluation"
            else _load_section(cls, raw.get(name), name)
        )
        for name, cls in _SECTIONS.items()
    }
    cfg = Config(**sections)
    _validate(cfg)
    return cfg
