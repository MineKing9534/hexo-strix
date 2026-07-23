"""HeXO-aware TD(lambda) returns for KLENT."""

from __future__ import annotations

from collections.abc import Sequence


def transition_sign(player: str, next_player: str) -> float:
    """Convert a next-state value into the current player's perspective."""

    if player not in {"P1", "P2"} or next_player not in {"P1", "P2"}:
        raise ValueError("players must be 'P1' or 'P2'")
    return 1.0 if player == next_player else -1.0


def lambda_returns(
    *,
    players: Sequence[str],
    rewards: Sequence[float],
    state_values: Sequence[float],
    trace_decay: float,
    bootstrap_player: str | None = None,
    bootstrap_value: float | None = None,
) -> list[float]:
    """Compute action targets for a complete HeXO trajectory.

    ``state_values[t]`` is E[Q(s_t, a)] under the improved policy. For a
    non-terminal transition, the bootstrap uses ``state_values[t + 1]`` and
    the sign implied by the actual players at those two positions. This is
    important in HeXO because a player normally makes two consecutive
    placements; negating after every action would be wrong.

    A true terminal trajectory omits ``bootstrap_player`` and
    ``bootstrap_value`` and ends at its final reward. A truncated trajectory
    supplies both values. Its final action target then bootstraps from the
    frozen network's value of the non-terminal successor state rather than
    injecting a fictitious draw.
    """

    size = len(players)
    if size == 0:
        return []
    if len(rewards) != size or len(state_values) != size:
        raise ValueError("players, rewards, and state_values must align")
    if not 0.0 <= trace_decay <= 1.0:
        raise ValueError("trace_decay must be in [0, 1]")
    if (bootstrap_player is None) != (bootstrap_value is None):
        raise ValueError(
            "bootstrap_player and bootstrap_value must be supplied together"
        )

    returns = [0.0] * size
    if bootstrap_player is None:
        returns[-1] = float(rewards[-1])
    else:
        sign = transition_sign(players[-1], bootstrap_player)
        returns[-1] = float(rewards[-1]) + sign * float(bootstrap_value)
    for index in range(size - 2, -1, -1):
        sign = transition_sign(players[index], players[index + 1])
        continuation = (
            (1.0 - trace_decay) * float(state_values[index + 1])
            + trace_decay * returns[index + 1]
        )
        returns[index] = float(rewards[index]) + sign * continuation
    return returns
