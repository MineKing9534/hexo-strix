"""Sparse fixed-checkpoint search-Q supervision for KLENT.

KLENT's ordinary critic target covers only the action actually played.  This
module optionally queries a fixed, previously trained checkpoint with batched
Gumbel MCTS and returns completed-Q labels for the root actions that search
visited.  The fixed teacher is deliberately deterministic: root Gumbel noise
and the forcing solver are both disabled, so changes in the labels reflect the
sampled on-policy states rather than a second source of exploration noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class SearchQLabels:
    """Visited root-action indices and completed-Q targets for one state."""

    action_indices: Tensor
    targets: Tensor
    legal_actions: int


def _autocast(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


class FixedCheckpointSearchQTeacher:
    """Keep one fixed checkpoint resident and label roots in lockstep batches."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: torch.device,
        precision: str,
        simulations: int,
        actions: int,
        root_batch_size: int,
    ) -> None:
        if simulations <= 0:
            raise ValueError("search-Q teacher simulations must be positive")
        if actions <= 0:
            raise ValueError("search-Q teacher actions must be positive")
        if root_batch_size <= 0:
            raise ValueError("search-Q teacher root batch size must be positive")

        import hexo_rs
        from hexo_a0.evaluate import make_eval_fn
        from hexo_a0.graph import graph_fn_from_model_config
        from hexo_a0.head_to_head import load_checkpoint

        if not hasattr(hexo_rs, "batched_gumbel_mcts_with_diagnostics"):
            raise RuntimeError(
                "hexo_rs lacks batched_gumbel_mcts_with_diagnostics; "
                "rebuild the local Rust extension"
            )

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"search-Q teacher checkpoint not found: {self.checkpoint}"
            )
        self.device = device
        self.precision = precision
        self.root_batch_size = root_batch_size
        self.loaded = load_checkpoint(self.checkpoint, device)
        model_config = self.loaded.model_config
        graph_fn = graph_fn_from_model_config(model_config)
        base_eval_fn = make_eval_fn(
            self.loaded.model,
            device,
            graph_type=model_config.graph_type,
            prune_empty_edges=model_config.prune_empty_edges,
            threat_features=model_config.threat_features,
            relative_stones=model_config.relative_stone_encoding,
            graph_fn=graph_fn,
            model_config=model_config,
        )

        def eval_fn(states):
            with _autocast(device, precision):
                return base_eval_fn(states)

        self.eval_fn = eval_fn
        self.search_config = hexo_rs.MCTSConfig(
            n_simulations=simulations,
            m_actions=actions,
            c_visit=50,
            c_scale=1.0,
            disable_gumbel_noise=True,
            disable_forcing_solver=True,
        )

    def label(
        self,
        states: list[object],
        *,
        seed: int | None,
    ) -> list[SearchQLabels]:
        """Return visited-action labels aligned with each state's legal moves."""

        import hexo_rs

        labels: list[SearchQLabels] = []
        for start in range(0, len(states), self.root_batch_size):
            chunk = states[start : start + self.root_batch_size]
            chunk_seed = None if seed is None else seed + start
            results = hexo_rs.batched_gumbel_mcts_with_diagnostics(
                chunk,
                self.eval_fn,
                self.search_config,
                seed=chunk_seed,
            )
            if len(results) != len(chunk):
                raise RuntimeError(
                    "search-Q teacher returned the wrong number of roots"
                )
            for state, result in zip(chunk, results, strict=True):
                _action, _policy, visits, q_values, _priors, _candidates = result
                legal_actions = len(state.legal_moves())
                if len(visits) != legal_actions or len(q_values) != legal_actions:
                    raise RuntimeError(
                        "search-Q diagnostics are not legal-move aligned"
                    )
                visits_tensor = torch.as_tensor(visits, dtype=torch.long)
                action_indices = visits_tensor.nonzero(as_tuple=False).squeeze(1)
                targets = torch.as_tensor(q_values, dtype=torch.float32).index_select(
                    0, action_indices
                )
                if action_indices.numel() == 0:
                    raise RuntimeError("search-Q teacher visited no root actions")
                if not bool(torch.isfinite(targets).all()):
                    raise RuntimeError("search-Q teacher produced non-finite targets")
                labels.append(
                    SearchQLabels(
                        action_indices=action_indices,
                        targets=targets,
                        legal_actions=legal_actions,
                    )
                )
        return labels

