"""Adapter from a KLENT policy/Q checkpoint to the existing MCTS interface."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from hexo_a0.config import ModelConfig
from hexo_klent.config import AlgorithmConfig, KlentModelConfig
from hexo_klent.model import (
    BatchOutput,
    KlentNet,
    make_klent_net,
)


class KlentMCTSAdapter(nn.Module):
    """Present KLENT as the policy-plus-state-value model MCTS expects.

    KLENT does not train a separate scalar value head. For test-time MCTS,
    Appendix M of the paper reconstructs its state value from the learned
    policy and action-value estimates:

        V(s) = sum_a pi_theta(a | s) Q_theta(s, a)

    This intentionally differs from self-play's TD(lambda) bootstrap, which
    follows Algorithm 1 and weights Q with the freshly improved policy pi'.
    Policy logits remain the learned policy head's raw logits, so the existing
    Rust Gumbel MCTS can consume this wrapper without modification.
    """

    def __init__(
        self,
        network: KlentNet,
        algorithm: AlgorithmConfig,
    ) -> None:
        super().__init__()
        self.network = network
        self.algorithm = algorithm

    def _state_values(self, output: BatchOutput) -> Tensor:
        counts = [int(count) for count in output.legal_counts.detach().cpu()]
        logits_chunks = output.policy_logits.split(counts)
        q_chunks = output.q_values.split(counts)
        values = []
        for logits, q_values in zip(
            logits_chunks, q_chunks, strict=True
        ):
            logits_f = logits.float()
            q_f = q_values.float()
            policy = torch.softmax(logits_f, dim=0)
            values.append(torch.dot(policy, q_f))
        return torch.stack(values)

    def _forward_batch_core(
        self,
        batch,
        *,
        legal_idx: Tensor | None = None,
        stone_idx: Tensor | None = None,
        stone_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Match ``HeXONet._forward_batch_core`` for the axis MCTS fast path."""

        del stone_idx, stone_batch
        output = self.network._forward_batch_core(
            batch, legal_idx=legal_idx
        )
        return (
            output.policy_logits,
            output.legal_counts,
            self._state_values(output),
        )

    def forward_batch(self, batch) -> tuple[list[Tensor], Tensor]:
        """Match ``HeXONet.forward_batch`` for evaluation helpers."""

        output = self.network.forward_batch(batch)
        counts = [int(count) for count in output.legal_counts.detach().cpu()]
        return (
            list(output.policy_logits.split(counts)),
            self._state_values(output),
        )


@dataclass(frozen=True)
class LoadedKlentMCTS:
    """A reconstructed KLENT checkpoint ready for existing evaluators."""

    model: KlentMCTSAdapter
    model_config: ModelConfig
    algorithm: AlgorithmConfig
    iteration: int | str


def _model_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> KlentModelConfig:
    raw = checkpoint.get("model_config", {})
    if not isinstance(raw, dict):
        raw = {}
    known = {field.name for field in dataclasses.fields(KlentModelConfig)}
    return KlentModelConfig(
        **{key: value for key, value in raw.items() if key in known}
    )


def _algorithm_from_checkpoint(checkpoint: dict[str, Any]) -> AlgorithmConfig:
    config = checkpoint.get("config")
    raw = config.get("algorithm", {}) if isinstance(config, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    known = {field.name for field in dataclasses.fields(AlgorithmConfig)}
    return AlgorithmConfig(**{key: value for key, value in raw.items() if key in known})


def adapt_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device | str,
) -> LoadedKlentMCTS:
    """Reconstruct a KLENT checkpoint and wrap it for MCTS evaluation."""

    if checkpoint.get("format") != "hexo-klent-v1":
        raise ValueError("checkpoint is not in hexo-klent-v1 format")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("KLENT checkpoint has no model_state_dict")

    model_config = _model_config_from_checkpoint(checkpoint)
    algorithm = _algorithm_from_checkpoint(checkpoint)
    network = make_klent_net(model_config).to(device)
    clean_state = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    network.load_state_dict(clean_state, strict=True)
    model = KlentMCTSAdapter(network, algorithm).to(device)
    model.eval()
    return LoadedKlentMCTS(
        model=model,
        model_config=model_config,
        algorithm=algorithm,
        iteration=checkpoint.get("iteration", "?"),
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> LoadedKlentMCTS:
    """Load a KLENT checkpoint from disk as an MCTS-compatible model."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint {path} is not a dict-style state file")
    return adapt_checkpoint(checkpoint, device)
