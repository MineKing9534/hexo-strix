"""Trainer-side guards for fixed-candidate SPRT state consumption."""

from __future__ import annotations

import json

from hexo_a0.sprt_watcher import SPRTWatcher


def _watcher(tmp_path) -> SPRTWatcher:
    return SPRTWatcher(
        config_path=tmp_path / "config.toml",
        trainee_dir=tmp_path / "checkpoints",
        champion_path=tmp_path / "champion.pt",
        state_file=tmp_path / "sprt_state.json",
        stage_index=0,
        s0=0.5,
        s1=0.525,
        alpha=0.05,
        beta=0.05,
        window_size=0,
        mcts_sims=64,
        mcts_m_actions=16,
        device="cpu",
        poll_interval=0.1,
    )


def _payload(timestamp: float) -> dict:
    return {
        "timestamp": timestamp,
        "trainee_path": "checkpoint_00002000.pt",
        "champion_mtime": 1.0,
        "score": 0.55,
        "reject_count": 0,
        "peak_history": [],
        "round_id": 7,
        "round_status": "accept_h1",
        "candidate_step": 2000,
        "latest_available_step": 3000,
        "candidate_lag_steps": 1000,
        "state": {
            "decision": "accept_h1",
            "games": 500,
            "wins": 275,
            "draws": 0,
            "losses": 225,
            "llr": 3.0,
            "pairs": 250,
        },
    }


def test_check_ignores_terminal_state_older_than_current_daemon(tmp_path):
    watcher = _watcher(tmp_path)
    watcher._started_at = 200.0
    watcher.state_file.write_text(json.dumps(_payload(timestamp=100.0)))

    assert watcher.check() is None


def test_check_reports_heartbeated_exact_candidate(tmp_path):
    watcher = _watcher(tmp_path)
    watcher._started_at = 200.0
    watcher.state_file.write_text(json.dumps(_payload(timestamp=300.0)))

    decision = watcher.check()
    assert decision is not None
    assert decision.decision == "accept_h1"
    assert decision.trainee_path == "checkpoint_00002000.pt"
    assert decision.round_id == 7
    assert decision.candidate_step == 2000
    assert decision.latest_available_step == 3000
    assert decision.candidate_lag_steps == 1000
