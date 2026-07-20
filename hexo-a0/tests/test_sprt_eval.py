"""Tests for SPRT decision math, window sizing, and fixed-round restore logic.

These lock in the 2026-06-04 sizing decision: the sliding window must exceed
the natural decision horizon for the H1 effect size (~412 games at s1=0.55),
otherwise the windowed LLR plateaus below the accept bound and a genuinely
stronger trainee can never be promoted (the "dead band"). See
docs/research/ for the derivation.
"""
from __future__ import annotations

from hexo_a0.sprt_eval import (
    SPRTConfig,
    SPRTState,
    checkpoint_step,
    restore_decision,
)


def _sustained_stream(n_games: int, win_rate: float) -> list[str]:
    """Deterministic W/L stream with W's distributed evenly (Bresenham-style).

    Keeps the running mean close to `win_rate` throughout so there are no
    transient LLR excursions — the decision reflects the steady state, not
    an ordering artifact.
    """
    out: list[str] = []
    num = round(win_rate * 100)  # e.g. 0.55 -> 55 per 100
    den = 100
    for i in range(n_games):
        out.append("W" if (i * num) % den < num else "L")
    return out


def test_default_window_exceeds_s1_decision_horizon():
    """Default window must clear the ~412-game accept horizon for s1=0.55."""
    cfg = SPRTConfig()
    # Decision horizon = U / per-pair drift at s1, in games.
    assert cfg.window_size is not None
    assert cfg.window_size >= 412
    assert cfg.window_size == 1000


def test_sustained_s1_stream_accepts_with_default_window():
    """A sustained s1=0.55 trainee reaches accept_h1 under the default window."""
    cfg = SPRTConfig()  # window 1000
    state = SPRTState()
    for o in _sustained_stream(1000, 0.55):
        state.record(o, cfg)
    assert state.decision == "accept_h1"


def test_sustained_s1_stream_stalls_under_narrow_window():
    """The old window=400 plateaus an s1 trainee below accept — the dead band."""
    cfg = SPRTConfig(window_size=400)
    state = SPRTState()
    for o in _sustained_stream(1000, 0.55):
        state.record(o, cfg)
    _, upper = cfg.bounds()
    assert state.decision == "continue"
    assert state.llr < upper


def test_from_dict_drops_trailing_odd_outcome_to_keep_pairs_aligned():
    """A daemon killed mid-pair can persist an odd-length outcome deque.

    With an unbounded window (no eviction ever re-aligns it) a single orphaned
    outcome would mis-phase every subsequent pentanomial pair for the rest of
    the round. Restore must drop a trailing odd outcome so the deque is always
    an even number of (P1, P2) units.
    """
    cfg = SPRTConfig(window_size=None, pentanomial=True)
    state = SPRTState()
    for o in "WLWLW":  # odd length: 5 outcomes
        state.record(o, cfg)
    restored = SPRTState.from_dict(state.to_dict())
    assert len(restored._outcomes) == 4
    assert "".join(restored._outcomes) == "WLWL"


def test_checkpoint_step_parses_step_and_rejects_other_names():
    assert checkpoint_step("runs/x/checkpoints/checkpoint_00024900.pt") == 24900
    assert checkpoint_step("checkpoint_0.pt") == 0
    assert checkpoint_step("runs/x/checkpoints/self_play/champion.pt") is None


def test_restore_decision_allows_same_champion_fresh_trainee():
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=26198,
    )
    assert ok is True
    assert reasons == []


def test_restore_decision_blocks_changed_champion():
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=False,
        prior_trainee_step=24900, latest_trainee_step=24900,
    )
    assert ok is False
    assert any("champion" in r for r in reasons)


def test_restore_decision_blocks_terminal_decision():
    ok, reasons = restore_decision(
        prior_decision="accept_h1", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=24900,
    )
    assert ok is False
    assert any("accept_h1" in r for r in reasons)


def test_restore_decision_blocks_obsolete_trainee_evidence():
    """The staleness gate is TRAINING PROGRESS, not wall-clock: a round whose
    evidence was earned >max_trainee_lag steps behind the current latest
    checkpoint describes an obsolete network. Wall-clock was the wrong proxy
    (2026-06-11: a wedged-then-restarted round was silently discarded at
    4742s > 3600s while champion and trainee were both still current)."""
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=20000,
    )
    assert ok is False
    assert any("step" in r for r in reasons)
    # exactly at the limit is still fine
    ok, _ = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=15000,
    )
    assert ok is True


def test_fixed_candidate_restore_ignores_newer_training_checkpoints():
    """A leased candidate remains the tested population even if training races ahead."""
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=50000,
        fixed_candidate=True, same_test_spec=True, candidate_exists=True,
    )
    assert ok is True
    assert reasons == []


def test_fixed_candidate_restore_requires_same_spec_and_candidate_file():
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=10000, latest_trainee_step=10000,
        fixed_candidate=True, same_test_spec=False, candidate_exists=False,
    )
    assert ok is False
    assert any("specification" in r for r in reasons)
    assert any("candidate" in r for r in reasons)


def test_restore_decision_unknown_steps_do_not_block():
    """Champion identity is the load-bearing gate; unparseable checkpoint
    names must not discard an otherwise-valid round."""
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=None, latest_trainee_step=26198,
    )
    assert ok is True and reasons == []
    ok, reasons = restore_decision(
        prior_decision="continue", same_champion=True,
        prior_trainee_step=24900, latest_trainee_step=None,
    )
    assert ok is True and reasons == []


def test_restore_decision_collects_all_failed_gates():
    ok, reasons = restore_decision(
        prior_decision="reject_h1", same_champion=False,
        prior_trainee_step=0, latest_trainee_step=99999,
    )
    assert ok is False
    assert len(reasons) == 3
