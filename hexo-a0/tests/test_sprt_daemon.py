"""Tests for fixed-candidate daemon selection and noise-off openings."""

import json

from hexo_a0.sprt_daemon import (
    _candidate_is_newer,
    _latest_checkpoint,
    _opening_generator_for_pair,
    _publish_waiting_state,
    _test_spec_id,
)
from hexo_a0.sprt_eval import SPRTState


T, C = ("trainee_model", "tmc"), ("champion_model", "cmc")


class TestOpeningGeneratorForPair:
    def test_champion_fixed(self):
        assert _opening_generator_for_pair("champion", 0, T, C) == C
        assert _opening_generator_for_pair("champion", 1, T, C) == C

    def test_trainee_fixed(self):
        assert _opening_generator_for_pair("trainee", 0, T, C) == T
        assert _opening_generator_for_pair("trainee", 7, T, C) == T

    def test_alternate_even_trainee_odd_champion(self):
        assert _opening_generator_for_pair("alternate", 0, T, C) == T
        assert _opening_generator_for_pair("alternate", 1, T, C) == C
        assert _opening_generator_for_pair("alternate", 2, T, C) == T
        assert _opening_generator_for_pair("alternate", 3, T, C) == C


def test_latest_checkpoint_uses_training_step_not_mtime(tmp_path):
    older_step = tmp_path / "checkpoint_00001000.pt"
    newer_step = tmp_path / "checkpoint_00002000.pt"
    newer_step.touch()
    older_step.touch()  # deliberately newer mtime but lower learner step

    assert _latest_checkpoint(tmp_path) == newer_step


def test_candidate_must_be_strictly_newer_after_completed_round(tmp_path):
    same = tmp_path / "checkpoint_00002000.pt"
    newer = tmp_path / "checkpoint_00003000.pt"

    assert _candidate_is_newer(same, minimum_step=2001) is False
    assert _candidate_is_newer(newer, minimum_step=2001) is True
    assert _candidate_is_newer(same, minimum_step=None) is True


def test_test_spec_fingerprint_is_order_stable_and_change_sensitive():
    assert _test_spec_id({"a": 1, "b": 2}) == _test_spec_id({"b": 2, "a": 1})
    assert _test_spec_id({"a": 1}) != _test_spec_id({"a": 2})


def test_publish_waiting_state_clears_terminal_candidate(tmp_path):
    state_file = tmp_path / "sprt_state.json"
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"new champion")
    checkpoint = tmp_path / "checkpoint_00002000.pt"
    checkpoint.touch()
    state_file.write_text(json.dumps({
        "test_spec_id": "preserved",
        "trainee_path": "checkpoint_00001000.pt",
        "candidate_path": "checkpoint_00001000.pt",
        "candidate_step": 1000,
        "round_status": "accept_h1",
        "last_game": {"outcome": "W"},
        "state": {"decision": "accept_h1", "games": 120},
    }))
    st = champion.stat()

    _publish_waiting_state(
        state_file,
        state=SPRTState(),
        trainee_dir=tmp_path,
        champion_path=champion,
        champion_mtime=st.st_mtime,
        champion_identity={
            "device": st.st_dev,
            "inode": st.st_ino,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        },
        round_id=2,
        candidate_epoch=3,
        last_completed_candidate_step=1000,
        reject_count=0,
        peak_history=[0],
    )

    payload = json.loads(state_file.read_text())
    assert payload["test_spec_id"] == "preserved"
    assert payload["round_status"] == "waiting_candidate"
    assert payload["trainee_path"] is None
    assert payload["candidate_path"] is None
    assert payload["candidate_step"] is None
    assert payload["latest_available_step"] == 2000
    assert payload["last_completed_candidate_step"] == 1000
    assert payload["state"]["decision"] == "continue"
    assert payload["state"]["games"] == 0
    assert "last_game" not in payload
