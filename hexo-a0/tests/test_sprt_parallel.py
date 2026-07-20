"""Tests for parallel SPRT pair-eval bookkeeping (sprt_parallel.PairCoordinator).

The coordinator owns the pure dispatch/drain/inflight/epoch logic so the hard
concurrency invariants can be locked down without spinning real subprocesses:

  - B4: a worker that dies mid-pair must not permanently leak an inflight slot;
    its unreturned pair_ids are reclaimed and re-dispatched.
  - B3: a pair played against a since-swapped champion (stale epoch) must be
    discarded, never recorded into the fresh round.
  - dispatch gating: no new pairs once a decision is reached (drain-only).
  - per-pair timeout: a silently-stuck pair is reissued.
"""
from __future__ import annotations

from hexo_a0.sprt_parallel import PairCoordinator


def test_dead_worker_pairs_are_reclaimed_and_inflight_corrected():
    coord = PairCoordinator(n_workers=2, target_inflight=4)
    coord.on_dispatch(0, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    coord.on_dispatch(1, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    coord.on_dispatch(2, "B", champion_epoch=1, candidate_epoch=10, now=0.0)
    coord.on_dispatch(3, "B", champion_epoch=1, candidate_epoch=10, now=0.0)
    assert coord.inflight_count() == 4

    reissue = coord.reclaim_dead({"A"})

    assert sorted(reissue) == [0, 1]
    # A's pairs left inflight; only B's two remain until they are re-dispatched.
    assert coord.inflight_count() == 2


def test_result_for_current_epoch_is_recorded_stale_is_discarded():
    coord = PairCoordinator(n_workers=2, target_inflight=4)
    coord.on_dispatch(0, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    coord.on_dispatch(1, "A", champion_epoch=1, candidate_epoch=10, now=0.0)

    # pair 0 was played under epoch 1; champion is still epoch 1 -> record it.
    assert coord.on_result(0, 1, 10, 1, 10) is True
    # pair 1 was played under epoch 1 but champion has since swapped to epoch 2
    # -> discard, it is evidence against the wrong opponent.
    assert coord.on_result(1, 1, 10, 2, 10) is False

    # both leave inflight regardless of record/discard.
    assert coord.inflight_count() == 0


def test_result_for_already_retired_pair_is_discarded():
    """A reissued pair (worker died, or timed out) can have BOTH the original
    and the reissue complete. The second result to arrive carries a pair_id no
    longer in flight and must be discarded — otherwise it double-counts into
    the SPRT stream and mis-phases the pentanomial pairing."""
    coord = PairCoordinator(n_workers=2, target_inflight=4)
    coord.on_dispatch(7, "A", champion_epoch=1, candidate_epoch=10, now=0.0)

    # first completion records it and retires the pair_id
    assert coord.on_result(7, 1, 10, 1, 10) is True
    # a duplicate completion (same epoch, same id) is no longer in flight
    assert coord.on_result(7, 1, 10, 1, 10) is False


def test_result_for_completed_candidate_round_is_discarded():
    """A pair from candidate A must never become evidence for candidate B."""
    coord = PairCoordinator(n_workers=2, target_inflight=4)
    coord.on_dispatch(8, "A", champion_epoch=1, candidate_epoch=10, now=0.0)

    assert coord.on_result(8, 1, 10, 1, 11) is False
    assert coord.inflight_count() == 0


def test_reclaimed_pair_late_result_is_discarded():
    """reclaim_dead/timed_out retires the pair_id for re-dispatch; if the
    original worker's lost-then-found result shows up later it must not count."""
    coord = PairCoordinator(n_workers=2, target_inflight=4, pair_timeout=30.0)
    coord.on_dispatch(3, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    assert coord.reclaim_timed_out(now=100.0) == [3]
    # late result for the timed-out original — already retired, discard.
    assert coord.on_result(3, 1, 10, 1, 10) is False


def test_dispatch_fills_to_target_only_while_continuing():
    """Dispatch runs only while the fixed candidate's decision is continuing."""
    coord = PairCoordinator(n_workers=2, target_inflight=4)
    # empty pool, continuing -> top up to target.
    assert coord.dispatch_quota(decision="continue") == 4

    coord.on_dispatch(0, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    assert coord.dispatch_quota(decision="continue") == 3

    # a terminal decision halts new dispatch (drain-only).
    assert coord.dispatch_quota(decision="accept_h1") == 0
    assert coord.dispatch_quota(decision="reject_h1") == 0


def test_timed_out_pairs_are_reclaimed_for_reissue():
    coord = PairCoordinator(n_workers=2, target_inflight=4, pair_timeout=30.0)
    coord.on_dispatch(0, "A", champion_epoch=1, candidate_epoch=10, now=0.0)
    coord.on_dispatch(1, "A", champion_epoch=1, candidate_epoch=10, now=100.0)

    # at t=40, pair 0 (dispatched t=0) is >30s old; pair 1 (t=100) is not.
    stuck = coord.reclaim_timed_out(now=40.0)

    assert stuck == [0]
    assert coord.inflight_count() == 1
