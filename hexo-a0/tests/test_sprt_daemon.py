"""Tests for fixed-candidate daemon selection and noise-off openings."""

from hexo_a0.sprt_daemon import (
    _candidate_is_newer,
    _latest_checkpoint,
    _opening_generator_for_pair,
    _test_spec_id,
)


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
