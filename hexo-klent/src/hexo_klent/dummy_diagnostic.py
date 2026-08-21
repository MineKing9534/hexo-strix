"""Reproducible diagnostics for KLENT's axis-graph dummy/global path.

This is deliberately a read-only checkpoint probe.  It reconstructs the
axis-relational forward pass explicitly so the spatial and global branches can
be inspected before their outputs are added.  Runtime interventions measure
checkpoint reliance, not the strength of a retrained architecture.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data

import hexo_rs
from hexo_a0.config import FullConfig, GameConfig, legacy_lean_columns
from hexo_a0.graph import graph_fn_from_model_config
from hexo_a0.head_to_head import load_checkpoint
from hexo_a0.self_play import TrainingExample, batched_self_play_games


STAR_INTERVENTIONS = ("mean_in", "no_star", "no_global_branch")
MOVE_INTERVENTIONS = (
    "zero_real_moves",
    "zero_dummy_moves",
    "zero_both_moves",
    "flip_real_moves",
    "flip_dummy_moves",
    "flip_both_moves",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as raw:
            # mtime=0 makes the compressed artifact byte-reproducible.
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                zipped.write(
                    (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
                )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_json_or_gz(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def cluster_summary(
    records: Sequence[tuple[int, float]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Summarize positions with a game-cluster bootstrap confidence interval."""

    if not records:
        return {}
    grouped: dict[int, list[float]] = defaultdict(list)
    for game_id, value in records:
        if math.isfinite(float(value)):
            grouped[int(game_id)].append(float(value))
    clusters = sorted(grouped)
    values = np.asarray(
        [value for cluster in clusters for value in grouped[cluster]],
        dtype=np.float64,
    )
    if values.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    medians = np.empty(bootstrap_samples, dtype=np.float64)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = np.concatenate(
            [np.asarray(grouped[int(cluster)], dtype=np.float64) for cluster in chosen]
        )
        medians[index] = np.median(sample)
        means[index] = np.mean(sample)

    return {
        "games": len(clusters),
        "positions": int(values.size),
        "mean": float(values.mean()),
        "mean_ci95_game_cluster": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "median": float(np.median(values)),
        "median_ci95_game_cluster": [
            float(np.quantile(medians, 0.025)),
            float(np.quantile(medians, 0.975)),
        ],
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _legacy_feature_names(model_config: object) -> list[str]:
    if bool(getattr(model_config, "relative_stone_encoding", False)):
        names = [
            "own_stone",
            "opp_stone",
            "empty",
            "moves_remaining",
            "norm_q",
            "norm_r",
            "inv_stone_dist",
        ]
    else:
        names = [
            "p1_stone",
            "p2_stone",
            "empty",
            "to_move",
            "moves_remaining",
            "norm_q",
            "norm_r",
            "inv_stone_dist",
        ]
    if bool(getattr(model_config, "threat_features", False)):
        names += [
            "own_max_line",
            "opp_max_line",
            "own_threat_axes",
            "opp_threat_axes",
        ]
    return names


def move_feature_index(model_config: object) -> int:
    """Return the moves-remaining column in the actual lean node schema."""

    names = _legacy_feature_names(model_config)
    columns = legacy_lean_columns(model_config)
    if columns is not None:
        names = [names[index] for index in columns]
    try:
        return names.index("moves_remaining")
    except ValueError as error:
        raise ValueError(
            "dummy diagnostic requires moves_scope='node'; no moves_remaining "
            "node feature was found"
        ) from error


def _modified_features(
    features: Tensor,
    *,
    move_index: int,
    intervention: str,
) -> Tensor:
    if intervention not in MOVE_INTERVENTIONS:
        return features
    result = features.clone()
    dummy = result.shape[0] - 1
    real = slice(0, dummy)
    if intervention.startswith("zero_"):
        transform = torch.zeros_like
    else:
        transform = lambda value: 1.5 - value
    if "real" in intervention or "both" in intervention:
        result[real, move_index] = transform(result[real, move_index])
    if "dummy" in intervention or "both" in intervention:
        result[dummy, move_index] = transform(result[dummy, move_index])
    return result


def _entropy_effective_rank(messages: Tensor) -> float:
    centered = messages - messages.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    variances = singular_values.square()
    probabilities = variances / variances.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def relational_forward(
    network: torch.nn.Module,
    data: Data,
    *,
    model_config: object,
    intervention: str = "normal",
    collect: bool = False,
    skip_dead_final: bool = False,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, float]]]:
    """Explicit single-graph forward for the current relational JK-cat model."""

    representation = network.representation
    if not bool(getattr(representation, "axis_relational", False)):
        raise ValueError("dummy diagnostic requires axis_relational=True")
    if not bool(getattr(representation, "pre_norm", False)):
        raise ValueError("dummy diagnostic currently requires pre_norm=True")
    if not (
        bool(getattr(representation, "use_jk", False))
        and str(getattr(representation, "jk_mode", "")) == "cat"
    ):
        raise ValueError("dummy diagnostic currently requires JK-cat")
    if skip_dead_final and intervention != "normal":
        raise ValueError("dead-final prototype only supports the normal forward")

    dummy = data.x.shape[0] - 1
    move_index = move_feature_index(model_config)
    features = _modified_features(
        data.x,
        move_index=move_index,
        intervention=intervention,
    )
    x = representation.input_proj(features)
    source, destination = data.edge_index
    global_source, global_destination = data.global_edge_index
    states: list[Tensor] = []
    rows: list[dict[str, float]] = []

    for layer_index, (convolution, normalization) in enumerate(
        zip(representation.convs, representation.norms, strict=True)
    ):
        residual = x
        hidden = normalization(x)
        final_fast = skip_dead_final and layer_index == len(representation.convs) - 1
        target_nodes = dummy if final_fast else hidden.shape[0]

        distance_table = convolution.axis_conv.lin(
            convolution.dist_embed.weight
        )
        projected_edges = distance_table.index_select(
            0, data.edge_dist.long() - 1
        )
        messages = F.relu(hidden.index_select(0, source) + projected_edges)
        buckets = hidden.new_zeros((target_nodes, 3, hidden.shape[1]))
        flat_bucket = destination * 3 + data.edge_type
        buckets.view(-1, hidden.shape[1]).index_add_(
            0, flat_bucket, messages
        )
        axis_inputs = (
            (1.0 + convolution.axis_conv.eps)
            * hidden[:target_nodes].unsqueeze(1)
            + buckets
        )
        axis_first = convolution.axis_conv.nn[0]
        axis_second = convolution.axis_conv.nn[2]
        axis_hidden = F.relu(
            axis_first(axis_inputs.reshape(-1, hidden.shape[1]))
        )
        axis_hidden_sum = axis_hidden.view(target_nodes, 3, -1).sum(dim=1)
        axis_bias = (
            None
            if axis_second.bias is None
            else axis_second.bias * 3
        )
        axis_output = F.linear(
            axis_hidden_sum, axis_second.weight, axis_bias
        )

        global_edge = convolution.global_conv.lin(
            convolution.global_edge_embed
        )
        if final_fast:
            # The dummy at the start of the final layer is still required for
            # broadcast.  Its newly gathered/updated output is not consumed.
            broadcast = F.relu(hidden[dummy] + global_edge).expand(
                dummy, -1
            )
            global_inputs = (
                (1.0 + convolution.global_conv.eps) * hidden[:dummy]
                + broadcast
            )
        else:
            dummy_sources = global_source[0::2]
            real_destinations = global_destination[0::2]
            real_sources = global_source[1::2]
            dummy_destinations = global_destination[1::2]
            broadcast = F.relu(
                hidden.index_select(0, dummy_sources) + global_edge
            )
            incoming = F.relu(
                hidden.index_select(0, real_sources) + global_edge
            )
            global_bucket = hidden.new_zeros(hidden.shape)
            global_bucket.index_copy_(0, real_destinations, broadcast)
            global_bucket.index_add_(0, dummy_destinations, incoming)
            if intervention == "mean_in":
                global_bucket[dummy] /= max(len(real_sources), 1)
            elif intervention == "no_star":
                global_bucket.zero_()
            global_inputs = (
                (1.0 + convolution.global_conv.eps) * hidden
                + global_bucket
            )

        global_output = convolution.global_conv.nn(global_inputs)
        if intervention == "no_global_branch":
            global_output = torch.zeros_like(global_output)

        if collect:
            if final_fast:
                raise ValueError("collection is unavailable in dead-final mode")
            real_axis = axis_output[:dummy]
            real_global = global_output[:dummy]
            axis_norm = real_axis.norm(dim=1)
            global_norm = real_global.norm(dim=1)
            denominator = (axis_norm * global_norm).clamp_min(1e-12)
            cosine = (real_axis * real_global).sum(dim=1) / denominator
            cancellation = (real_axis + real_global).norm(dim=1) / (
                axis_norm + global_norm
            ).clamp_min(1e-12)
            incoming = F.relu(
                hidden.index_select(0, global_source[1::2]) + global_edge
            )
            rows.append(
                {
                    "layer": float(layer_index),
                    "n_real": float(dummy),
                    "move_count": float(data.stone_mask.sum()),
                    "axis_global_cosine_median": float(cosine.median()),
                    "axis_global_negative_cosine_fraction": float(
                        (cosine < 0.0).float().mean()
                    ),
                    "axis_global_cancellation_ratio_median": float(
                        cancellation.median()
                    ),
                    "axis_global_strong_cancellation_fraction": float(
                        (cancellation < 0.75).float().mean()
                    ),
                    "axis_global_severe_cancellation_fraction": float(
                        (cancellation < 0.50).float().mean()
                    ),
                    "global_out_over_axis_out_median": float(
                        (global_norm / axis_norm.clamp_min(1e-12)).median()
                    ),
                    "global_out_dominates_axis_fraction": float(
                        (global_norm > axis_norm).float().mean()
                    ),
                    "incoming_effective_rank": _entropy_effective_rank(
                        incoming
                    ),
                    "incoming_sum_alignment": float(
                        incoming.sum(dim=0).norm()
                        / incoming.norm(dim=1).sum().clamp_min(1e-12)
                    ),
                    "dummy_global_input_over_ordinary": float(
                        global_inputs[dummy].norm()
                        / global_inputs[:dummy]
                        .norm(dim=1)
                        .median()
                        .clamp_min(1e-12)
                    ),
                    "dummy_global_input_norm": float(
                        global_inputs[dummy].norm()
                    ),
                }
            )

        update = convolution.node_update(
            torch.cat(
                [hidden[:target_nodes], axis_output + global_output],
                dim=-1,
            )
        )
        updated = representation.activation(update + residual[:target_nodes])
        updated = representation.dropout(updated)
        if final_fast:
            # Preserve the public N-row representation contract even though the
            # trailing row is dead to every current KLENT head.
            x = torch.cat([updated, residual[dummy:]], dim=0)
        else:
            x = updated
        states.append(x)

    embeddings = torch.cat(
        [representation.final_norm(state) for state in states], dim=-1
    )
    legal_embeddings = embeddings[data.legal_mask]
    policy_logits = network.policy_head.mlp(legal_embeddings).squeeze(-1)
    q_values = network.q_head.mlp(legal_embeddings).squeeze(-1)
    state_value = torch.dot(
        policy_logits.softmax(dim=0).float(), q_values.float()
    )
    return policy_logits, q_values, state_value, rows


def _output_effects(
    base: tuple[Tensor, Tensor, Tensor],
    variant: tuple[Tensor, Tensor, Tensor],
) -> dict[str, float]:
    base_logits, base_q, base_value = base
    logits, q_values, value = variant
    base_log_policy = base_logits.float().log_softmax(dim=0)
    log_policy = logits.float().log_softmax(dim=0)
    base_policy = base_log_policy.exp()
    centered_delta = (logits - base_logits).float()
    centered_delta -= centered_delta.mean()
    return {
        "policy_kl": float(
            (base_policy * (base_log_policy - log_policy)).sum()
        ),
        "policy_centered_logit_rms_delta": float(
            centered_delta.square().mean().sqrt()
        ),
        "top1_flip": float(base_logits.argmax() != logits.argmax()),
        "q_mean_abs_delta": float((base_q - q_values).abs().mean()),
        "state_value_abs_delta": float((base_value - value).abs()),
    }


def _gradient_parameter_pairs(
    network: torch.nn.Module,
) -> list[list[tuple[Tensor, Tensor]]]:
    layers: list[list[tuple[Tensor, Tensor]]] = []
    for convolution in network.representation.convs:
        axis = convolution.axis_conv
        global_branch = convolution.global_conv
        pairs = [
            (axis.eps, global_branch.eps),
            (axis.lin.weight, global_branch.lin.weight),
            (axis.lin.bias, global_branch.lin.bias),
            (axis.nn[0].weight, global_branch.nn[0].weight),
            (axis.nn[0].bias, global_branch.nn[0].bias),
            (axis.nn[2].weight, global_branch.nn[2].weight),
            (axis.nn[2].bias, global_branch.nn[2].bias),
        ]
        layers.append(pairs)
    return layers


def policy_gradient_alignment(
    network: torch.nn.Module,
    examples: Sequence[TrainingExample],
    graphs: Sequence[Data],
) -> list[dict[str, float]]:
    """Compare axis/global parameter-gradient direction for policy correction."""

    batch = Batch.from_data_list(list(graphs))
    output = network.forward_batch(batch)
    targets = torch.cat(
        [example.policy_target.float() for example in examples]
    )
    if targets.shape != output.policy_logits.shape:
        raise ValueError("policy targets do not align with legal logits")
    counts = output.legal_counts.tolist()
    offset = 0
    losses: list[Tensor] = []
    for count in counts:
        count = int(count)
        target = targets[offset : offset + count]
        logits = output.policy_logits[offset : offset + count].float()
        losses.append(-(target * logits.log_softmax(dim=0)).sum())
        offset += count
    loss = torch.stack(losses).mean()

    pairs_by_layer = _gradient_parameter_pairs(network)
    parameters = [
        parameter
        for layer in pairs_by_layer
        for pair in layer
        for parameter in pair
    ]
    gradients = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False
    )
    iterator = iter(gradients)
    result: list[dict[str, float]] = []
    for layer_index, layer in enumerate(pairs_by_layer):
        axis_parts: list[Tensor] = []
        global_parts: list[Tensor] = []
        for axis_parameter, global_parameter in layer:
            axis_gradient = next(iterator)
            global_gradient = next(iterator)
            axis_parts.append(
                torch.zeros_like(axis_parameter).flatten()
                if axis_gradient is None
                else axis_gradient.detach().flatten()
            )
            global_parts.append(
                torch.zeros_like(global_parameter).flatten()
                if global_gradient is None
                else global_gradient.detach().flatten()
            )
        axis_vector = torch.cat(axis_parts).float()
        global_vector = torch.cat(global_parts).float()
        axis_norm = axis_vector.norm()
        global_norm = global_vector.norm()
        cosine = torch.dot(axis_vector, global_vector) / (
            axis_norm * global_norm
        ).clamp_min(1e-12)
        nonzero = (axis_vector != 0.0) | (global_vector != 0.0)
        sign_agreement = (
            (axis_vector[nonzero].sign() == global_vector[nonzero].sign())
            .float()
            .mean()
            if bool(nonzero.any())
            else axis_vector.new_tensor(1.0)
        )
        result.append(
            {
                "layer": float(layer_index),
                "policy_cross_entropy": float(loss.detach()),
                "axis_global_parameter_gradient_cosine": float(cosine),
                "global_over_axis_parameter_gradient_norm": float(
                    global_norm / axis_norm.clamp_min(1e-12)
                ),
                "axis_global_parameter_gradient_sign_agreement": float(
                    sign_agreement
                ),
            }
        )
    return result


def _serialize_example(
    example: TrainingExample,
    *,
    game_id: int,
    ply: int,
    trajectory_length: int,
) -> dict[str, Any]:
    state = example.game_state
    if state is None:
        raise ValueError("self-play example has no game_state")
    return {
        "game_id": game_id,
        "ply": ply,
        "trajectory_length": trajectory_length,
        "stones": [
            [int(coord[0]), int(coord[1]), str(player)]
            for coord, player in state.placed_stones()
        ],
        "current_player": str(state.current_player()),
        "moves_remaining": int(state.moves_remaining_this_turn()),
        "policy_target": [float(value) for value in example.policy_target],
        "value_target": float(example.value_target),
    }


def _deserialize_example(
    record: dict[str, Any], game_config: object
) -> TrainingExample:
    stones = [
        ((int(q), int(r)), str(player))
        for q, r, player in record["stones"]
    ]
    state = hexo_rs.GameState.from_state(
        stones,
        str(record["current_player"]),
        int(record["moves_remaining"]),
        game_config,
    )
    return TrainingExample(
        policy_target=torch.tensor(record["policy_target"], dtype=torch.float32),
        value_target=float(record["value_target"]),
        game_state=state,
        trajectory_id=int(record["game_id"]),
    )


def _selected_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    # Use quantile-bin midpoints.  Including both endpoints makes every game
    # contribute the same forced opening and overweights the final decision.
    return [
        min(length - 1, int((index + 0.5) * length / count))
        for index in range(count)
    ]


def _checkpoint_game_config(raw: dict[str, Any]) -> tuple[GameConfig, object, dict[str, int]]:
    raw_game = raw.get("config", {}).get("game", raw.get("game_config", {}))
    win_length = int(raw_game["win_length"])
    placement_radius = int(raw_game["placement_radius"])
    max_moves = int(raw_game.get("rollout_horizon", raw_game.get("max_moves", 500)))
    python_config = GameConfig(
        win_length=win_length,
        placement_radius=placement_radius,
        max_moves=max_moves,
    )
    rust_config = hexo_rs.GameConfig(
        win_length, placement_radius, max_moves
    )
    serializable = {
        "win_length": win_length,
        "placement_radius": placement_radius,
        "max_moves": max_moves,
    }
    return python_config, rust_config, serializable


def collect_position_bank(
    checkpoint: Path,
    *,
    games: int,
    positions_per_game: int,
    simulations: int,
    actions: int,
    seed: int,
) -> tuple[dict[str, Any], list[TrainingExample]]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded = load_checkpoint(checkpoint, torch.device("cpu"))
    network = loaded.model.network
    model_config = loaded.model_config
    if move_feature_index(model_config) < 0:
        raise AssertionError("unreachable")
    python_game, rust_game, serializable_game = _checkpoint_game_config(raw)

    config = FullConfig()
    config.model = model_config
    config.game = python_game
    config.mcts.n_simulations = simulations
    config.mcts.m_actions = actions
    config.mcts.exploration_moves = 30
    config.run.seed = seed
    config.self_play.precision = "fp32"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(
        f"collecting checkpoint={checkpoint.name} iteration={loaded.train_steps} "
        f"games={games} seed={seed}",
        flush=True,
    )
    started = time.perf_counter()
    trajectories = batched_self_play_games(
        loaded.model,
        config,
        rust_game,
        torch.device("cpu"),
        games,
        n_simulations_override=simulations,
    )
    selected: list[TrainingExample] = []
    records: list[dict[str, Any]] = []
    for game_id, trajectory in enumerate(trajectories):
        for ply in _selected_indices(len(trajectory), positions_per_game):
            example = trajectory[ply]
            selected.append(example)
            records.append(
                _serialize_example(
                    example,
                    game_id=game_id,
                    ply=ply,
                    trajectory_length=len(trajectory),
                )
            )

    digest = _sha256(checkpoint)
    bank = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "iteration": loaded.train_steps,
        "model_config": raw["model_config"],
        "game_config": serializable_game,
        "collection": {
            "games": games,
            "positions_per_game": positions_per_game,
            "positions_total": sum(len(item) for item in trajectories),
            "positions_selected": len(selected),
            "game_lengths": [len(item) for item in trajectories],
            "simulations": simulations,
            "actions": actions,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "positions": records,
    }
    print(
        f"collected checkpoint={checkpoint.name} positions={len(selected)} "
        f"seconds={bank['collection']['elapsed_seconds']:.1f}",
        flush=True,
    )
    del network
    return bank, selected


def _examples_from_bank(bank: dict[str, Any]) -> list[TrainingExample]:
    game = bank["game_config"]
    rust_game = hexo_rs.GameConfig(
        int(game["win_length"]),
        int(game["placement_radius"]),
        int(game["max_moves"]),
    )
    return [
        _deserialize_example(record, rust_game)
        for record in bank["positions"]
    ]


def _summarize_metric_rows(
    rows: Sequence[dict[str, float]],
    *,
    game_ids: Sequence[int],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    excluded = {"layer", "n_real", "move_count"}
    result: dict[str, Any] = {}
    for key in rows[0]:
        if key in excluded:
            continue
        result[key] = cluster_summary(
            [
                (int(game_id), float(row[key]))
                for game_id, row in zip(game_ids, rows, strict=True)
            ],
            seed=seed + sum(ord(character) for character in key),
            bootstrap_samples=bootstrap_samples,
        )
    return result


def _representative_graphs(graphs: Sequence[Data]) -> list[tuple[str, Data]]:
    ordered = sorted(graphs, key=lambda graph: int(graph.x.shape[0]))
    return [
        ("p10", ordered[round(0.10 * (len(ordered) - 1))]),
        ("p50", ordered[round(0.50 * (len(ordered) - 1))]),
        ("p90", ordered[round(0.90 * (len(ordered) - 1))]),
    ]


def benchmark_dead_final(
    network: torch.nn.Module,
    graphs: Sequence[Data],
    *,
    model_config: object,
    warmup: int,
    repeats: int,
    threads: int,
) -> list[dict[str, Any]]:
    """Benchmark an exact prototype that omits the last new dummy state."""

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    results: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for label, graph in _representative_graphs(graphs):
                baseline = relational_forward(
                    network, graph, model_config=model_config
                )[:3]
                optimized = relational_forward(
                    network,
                    graph,
                    model_config=model_config,
                    skip_dead_final=True,
                )[:3]
                max_error = max(
                    float((baseline[0] - optimized[0]).abs().max()),
                    float((baseline[1] - optimized[1]).abs().max()),
                    float((baseline[2] - optimized[2]).abs()),
                )
                for _ in range(warmup):
                    relational_forward(
                        network, graph, model_config=model_config
                    )
                    relational_forward(
                        network,
                        graph,
                        model_config=model_config,
                        skip_dead_final=True,
                    )

                normal_times: list[float] = []
                optimized_times: list[float] = []
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for index in range(repeats):
                        if index % 2 == 0:
                            order = (False, True)
                        else:
                            order = (True, False)
                        measured: dict[bool, float] = {}
                        for skip in order:
                            started = time.perf_counter_ns()
                            relational_forward(
                                network,
                                graph,
                                model_config=model_config,
                                skip_dead_final=skip,
                            )
                            measured[skip] = float(
                                time.perf_counter_ns() - started
                            )
                        normal_times.append(measured[False])
                        optimized_times.append(measured[True])
                finally:
                    if gc_was_enabled:
                        gc.enable()

                normal_median = float(np.median(normal_times))
                optimized_median = float(np.median(optimized_times))
                paired_reductions = 1.0 - (
                    np.asarray(optimized_times, dtype=np.float64)
                    / np.asarray(normal_times, dtype=np.float64)
                )
                n_real = int(graph.x.shape[0] - 1)
                axis_edges = int(graph.edge_index.shape[1])
                layers = len(network.representation.convs)
                message_fraction = n_real / (
                    layers * (axis_edges + 2 * n_real)
                )
                results.append(
                    {
                        "size_quantile": label,
                        "nodes_real": n_real,
                        "axis_edges": axis_edges,
                        "normal_median_ms": normal_median / 1e6,
                        "skip_dead_final_median_ms": optimized_median / 1e6,
                        "speedup": normal_median / optimized_median,
                        "time_reduction_fraction": 1.0
                        - optimized_median / normal_median,
                        "paired_time_reduction_median": float(
                            np.median(paired_reductions)
                        ),
                        "paired_time_reduction_p10": float(
                            np.quantile(paired_reductions, 0.10)
                        ),
                        "paired_time_reduction_p90": float(
                            np.quantile(paired_reductions, 0.90)
                        ),
                        "removed_message_fraction_of_four_layer_forward": message_fraction,
                        "max_abs_output_error": max_error,
                        "threads": threads,
                        "warmup": warmup,
                        "repeats": repeats,
                    }
                )
    finally:
        torch.set_num_threads(previous_threads)
    return results


def analyze_bank(
    checkpoint: Path,
    bank: dict[str, Any],
    *,
    seed: int,
    bootstrap_samples: int,
    gradient_positions_per_game: int,
    include_gradients: bool,
    include_benchmark: bool,
    benchmark_warmup: int,
    benchmark_repeats: int,
    benchmark_threads: int,
) -> dict[str, Any]:
    loaded = load_checkpoint(checkpoint, torch.device("cpu"))
    network = loaded.model.network
    network.eval()
    examples = _examples_from_bank(bank)
    graph_function = graph_fn_from_model_config(loaded.model_config)
    graphs = [graph_function(example.game_state) for example in examples]
    game_ids = [int(record["game_id"]) for record in bank["positions"]]

    print(
        f"analyzing model={checkpoint.name} iteration={loaded.train_steps} "
        f"bank_source={bank['iteration']} positions={len(graphs)}",
        flush=True,
    )
    layer_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    layer_game_ids: dict[int, list[int]] = defaultdict(list)
    effects: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    maximum_forward_error = 0.0

    with torch.inference_mode():
        for graph, game_id in zip(graphs, game_ids, strict=True):
            logits, q_values, value, rows = relational_forward(
                network,
                graph,
                model_config=loaded.model_config,
                collect=True,
            )
            output = network.forward_batch(Batch.from_data_list([graph]))
            true_value = torch.dot(
                output.policy_logits.softmax(dim=0).float(),
                output.q_values.float(),
            )
            maximum_forward_error = max(
                maximum_forward_error,
                float((logits - output.policy_logits).abs().max()),
                float((q_values - output.q_values).abs().max()),
                float((value - true_value).abs()),
            )
            for row in rows:
                layer = int(row["layer"])
                layer_rows[layer].append(row)
                layer_game_ids[layer].append(game_id)

            base = (logits, q_values, value)
            for intervention in STAR_INTERVENTIONS + MOVE_INTERVENTIONS:
                variant = relational_forward(
                    network,
                    graph,
                    model_config=loaded.model_config,
                    intervention=intervention,
                )[:3]
                for metric, effect in _output_effects(base, variant).items():
                    effects[intervention][metric].append((game_id, effect))

    layers: dict[str, Any] = {}
    for layer, rows in sorted(layer_rows.items()):
        layer_summary = _summarize_metric_rows(
            rows,
            game_ids=layer_game_ids[layer],
            seed=seed + 1009 * layer,
            bootstrap_samples=bootstrap_samples,
        )
        n_real = [row["n_real"] for row in rows]
        dummy_input = [row["dummy_global_input_norm"] for row in rows]
        layer_summary["corr_n_real_dummy_global_input_norm"] = _finite(
            np.corrcoef(n_real, dummy_input)[0, 1]
        )
        layers[str(layer)] = layer_summary

    intervention_summary = {
        intervention: {
            metric: cluster_summary(
                values,
                seed=seed
                + sum(ord(character) for character in intervention + metric),
                bootstrap_samples=bootstrap_samples,
            )
            for metric, values in metrics.items()
        }
        for intervention, metrics in effects.items()
    }

    gradient_summary: dict[str, Any] = {}
    if include_gradients:
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index, game_id in enumerate(game_ids):
            grouped_indices[game_id].append(index)
        gradient_rows: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
        for game_id, indices in sorted(grouped_indices.items()):
            selected_local = _selected_indices(
                len(indices), gradient_positions_per_game
            )
            chosen = [indices[index] for index in selected_local]
            rows = policy_gradient_alignment(
                network,
                [examples[index] for index in chosen],
                [graphs[index] for index in chosen],
            )
            for row in rows:
                gradient_rows[int(row["layer"])].append((game_id, row))
        for layer, rows in sorted(gradient_rows.items()):
            gradient_summary[str(layer)] = {
                key: cluster_summary(
                    [(game_id, float(row[key])) for game_id, row in rows],
                    seed=seed + layer * 2027 + sum(map(ord, key)),
                    bootstrap_samples=bootstrap_samples,
                )
                for key in rows[0][1]
                if key != "layer"
            }

    benchmark = (
        benchmark_dead_final(
            network,
            graphs,
            model_config=loaded.model_config,
            warmup=benchmark_warmup,
            repeats=benchmark_repeats,
            threads=benchmark_threads,
        )
        if include_benchmark
        else []
    )
    node_records = list(zip(game_ids, [float(graph.x.shape[0]) for graph in graphs]))
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "iteration": loaded.train_steps,
        "bank_source_iteration": bank["iteration"],
        "bank_source_checkpoint_sha256": bank["checkpoint_sha256"],
        "positions": len(graphs),
        "games": len(set(game_ids)),
        "node_count": cluster_summary(
            node_records,
            seed=seed + 17,
            bootstrap_samples=bootstrap_samples,
        ),
        "manual_forward_max_abs_error": maximum_forward_error,
        "layers": layers,
        "interventions": intervention_summary,
        "policy_gradient_alignment": gradient_summary,
        "dead_final_benchmark": benchmark,
        "interpretation_warning": (
            "Runtime interventions are out-of-distribution probes of a "
            "co-adapted checkpoint, not estimates of retrained strength."
        ),
    }
    del loaded, network, graphs, examples
    gc.collect()
    return result


def _bank_output_path(summary_path: Path) -> Path:
    suffix = "".join(summary_path.suffixes)
    stem = summary_path.name[: -len(suffix)] if suffix else summary_path.name
    return summary_path.with_name(stem + ".positions.json.gz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect KLENT dummy/global behavior across checkpoints"
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--games-per-checkpoint", type=int, default=12)
    parser.add_argument("--positions-per-game", type=int, default=12)
    parser.add_argument("--gradient-positions-per-game", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=24)
    parser.add_argument("--actions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--shared-bank", choices=("none", "first", "last"), default="last")
    parser.add_argument("--position-bank-input", type=Path)
    parser.add_argument("--position-bank-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-gradients", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--benchmark-threads", type=int, default=1)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    positive = (
        "games_per_checkpoint",
        "positions_per_game",
        "gradient_positions_per_game",
        "simulations",
        "actions",
        "bootstrap_samples",
        "benchmark_warmup",
        "benchmark_repeats",
        "benchmark_threads",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    checkpoints = [path.resolve() for path in args.checkpoints]

    if args.position_bank_input is not None:
        bank_artifact = _read_json_or_gz(args.position_bank_input)
        banks = list(bank_artifact["banks"])
        by_hash = {bank["checkpoint_sha256"]: bank for bank in banks}
        for checkpoint in checkpoints:
            digest = _sha256(checkpoint)
            if digest not in by_hash:
                raise ValueError(
                    f"position bank has no own-policy entry for {checkpoint}"
                )
    else:
        banks = []
        for index, checkpoint in enumerate(checkpoints):
            bank, _examples = collect_position_bank(
                checkpoint,
                games=args.games_per_checkpoint,
                positions_per_game=args.positions_per_game,
                simulations=args.simulations,
                actions=args.actions,
                seed=args.seed + index * 1_000_003,
            )
            banks.append(bank)
            del _examples
        bank_artifact = {
            "schema": "hexo-klent-dummy-position-bank-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "banks": banks,
        }
        bank_output = args.position_bank_output or _bank_output_path(args.output)
        _atomic_json_gz(bank_output, bank_artifact)
        print(f"wrote exact position bank: {bank_output}", flush=True)

    by_hash = {bank["checkpoint_sha256"]: bank for bank in banks}
    shared_bank = None
    if args.shared_bank != "none":
        shared_checkpoint = (
            checkpoints[0] if args.shared_bank == "first" else checkpoints[-1]
        )
        shared_bank = by_hash[_sha256(shared_checkpoint)]

    results: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        own_bank = by_hash[_sha256(checkpoint)]
        own = analyze_bank(
            checkpoint,
            own_bank,
            seed=args.seed + index * 10_007,
            bootstrap_samples=args.bootstrap_samples,
            gradient_positions_per_game=args.gradient_positions_per_game,
            include_gradients=not args.no_gradients,
            include_benchmark=not args.no_benchmark,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_repeats=args.benchmark_repeats,
            benchmark_threads=args.benchmark_threads,
        )
        shared = None
        if shared_bank is not None and shared_bank is not own_bank:
            shared = analyze_bank(
                checkpoint,
                shared_bank,
                seed=args.seed + index * 10_007 + 503,
                bootstrap_samples=args.bootstrap_samples,
                gradient_positions_per_game=args.gradient_positions_per_game,
                include_gradients=False,
                include_benchmark=False,
                benchmark_warmup=args.benchmark_warmup,
                benchmark_repeats=args.benchmark_repeats,
                benchmark_threads=args.benchmark_threads,
            )
        results.append({"own_policy_bank": own, "shared_bank": shared})

    output = {
        "schema": "hexo-klent-dummy-diagnostic-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in {"checkpoints", "position_bank_input", "position_bank_output", "output"}
        },
        "checkpoints": [str(path) for path in checkpoints],
        "position_bank": str(
            (args.position_bank_input or args.position_bank_output or _bank_output_path(args.output)).resolve()
        ),
        "results": results,
    }
    _atomic_json(args.output, output)
    print(f"wrote diagnostic summary: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
