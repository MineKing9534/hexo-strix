"""Tests for the bounded checkpoint search script."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "find_strongest_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "find_strongest_checkpoint_test_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/find_strongest_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


search = _load_script()


def _make_checkpoints(root: Path, steps: range | list[int]) -> dict[int, Path]:
    checkpoints = {}
    for step in steps:
        path = root / f"checkpoint_{step:06d}.pt"
        path.write_bytes(f"checkpoint {step}".encode())
        checkpoints[step] = path
    return checkpoints


def _fake_result(
    elo: float,
    *,
    winner: str = "inconclusive",
    decision: str = "continue",
) -> dict:
    opening_a = [[0, 0], [1, 0]]
    opening_b = [[0, 0], [0, 1]]
    return {
        "winner": winner,
        "decision": decision,
        "elo_diff": elo,
        "elo_ci_lo": elo - 10.0,
        "elo_ci_hi": elo + 10.0,
        "games": 4,
        "wins": 2,
        "draws": 0,
        "losses": 2,
        "per_game_opening": [opening_a, opening_a, opening_b, opening_b],
    }


def test_discovery_uses_training_step_not_mtime(tmp_path):
    checkpoints = _make_checkpoints(tmp_path, [5, 10, 20])
    (tmp_path / "final.pt").write_bytes(b"ignored final")
    os.utime(checkpoints[10], (200, 200))
    os.utime(checkpoints[20], (100, 100))
    os.utime(checkpoints[5], (300, 300))

    _, baseline_step, candidates = search.discover_checkpoints(
        tmp_path, checkpoints[10]
    )

    assert baseline_step == 10
    assert [candidate.step for candidate in candidates] == [20]


def test_evenly_spaced_indices_include_both_endpoints_without_duplicates():
    assert search.evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert search.evenly_spaced_indices(3, 20) == [0, 1, 2]
    assert search.evenly_spaced_indices(10, 1) == [9]


def test_search_respects_budget_and_never_repeats_a_screen_or_self_compares(
    tmp_path, monkeypatch
):
    checkpoints = _make_checkpoints(tmp_path, list(range(21)))
    calls = []

    def fake_head_to_head(checkpoint_a, checkpoint_b, config):
        step_a = search.checkpoint_step_from_name(checkpoint_a)
        step_b = search.checkpoint_step_from_name(checkpoint_b)
        assert step_a is not None and step_b is not None
        calls.append((step_a, step_b, config.seed))
        # A deliberately non-monotone strength curve peaking at step 13.
        strength_a = -abs(step_a - 13)
        strength_b = -abs(step_b - 13)
        elo = float(strength_a - strength_b)
        if elo >= 2.0:
            return _fake_result(elo, winner="A", decision="accept_h1")
        if elo <= -2.0:
            return _fake_result(elo, winner="B", decision="reject_h1")
        return _fake_result(elo)

    monkeypatch.setattr(search, "run_head_to_head", fake_head_to_head)
    result_path = tmp_path / "result.json"
    ladder_path = tmp_path / "ladder.txt"
    result = search.find_strongest_checkpoint(
        tmp_path,
        checkpoints[0],
        search.Config(max_games=20),
        search.SearchConfig(max_evaluations=8, coarse_count=4, finalists=2),
        result_path=result_path,
        ladder_path=ladder_path,
    )

    assert result["status"] == "complete"
    assert len(calls) == 8
    assert all(step_a != step_b for step_a, step_b, _ in calls)
    screen_calls = calls[:7]
    assert all(step_b == 0 for _, step_b, _ in screen_calls)
    assert len({step_a for step_a, _, _ in screen_calls}) == len(screen_calls)
    assert calls[-1][2] == 1_000_000
    assert len([c for c in result["comparisons"] if c["stage"] == "screen"]) == 7
    assert len([c for c in result["comparisons"] if c["stage"] == "playoff"]) == 1
    assert result["screening_opening_bank"] == [
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
    ]
    assert json.loads(result_path.read_text())["status"] == "complete"
    assert "Best checkpoint found under" in ladder_path.read_text()


def test_inconclusive_playoff_is_preserved_and_keeps_baseline_incumbent(
    tmp_path, monkeypatch
):
    checkpoints = _make_checkpoints(tmp_path, [0, 1, 2])

    def inconclusive(checkpoint_a, _checkpoint_b, _config):
        step = search.checkpoint_step_from_name(checkpoint_a)
        assert step is not None
        return _fake_result(float(step))

    monkeypatch.setattr(search, "run_head_to_head", inconclusive)
    result = search.find_strongest_checkpoint(
        tmp_path,
        checkpoints[0],
        search.Config(max_games=20),
        search.SearchConfig(max_evaluations=3, coarse_count=2, finalists=2),
        result_path=tmp_path / "result.json",
        ladder_path=tmp_path / "ladder.txt",
    )

    assert result["status"] == "complete"
    assert result["comparisons"][-1]["stage"] == "playoff"
    assert result["comparisons"][-1]["result"]["winner"] == "inconclusive"
    assert result["selected_checkpoint"]["step"] == 0


def test_non_numbered_newer_pt_is_ignored_without_index_error(tmp_path, monkeypatch):
    baseline = _make_checkpoints(tmp_path, [10])[10]
    (tmp_path / "final.pt").write_bytes(b"not a numbered candidate")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("head-to-head should not run")

    monkeypatch.setattr(search, "run_head_to_head", must_not_run)
    result = search.find_strongest_checkpoint(
        tmp_path,
        baseline,
        search.Config(max_games=20),
        result_path=tmp_path / "result.json",
        ladder_path=tmp_path / "ladder.txt",
    )

    assert result["status"] == "no_candidates"
    assert result["selected_checkpoint"]["step"] == 10


def test_different_screening_opening_bank_fails_and_persists_state(
    tmp_path, monkeypatch
):
    checkpoints = _make_checkpoints(tmp_path, [0, 1, 2])
    call_count = 0

    def mismatched_openings(*_args):
        nonlocal call_count
        result = _fake_result(5.0, winner="A", decision="accept_h1")
        if call_count:
            result["per_game_opening"][0][0][0] = 99
        call_count += 1
        return result

    monkeypatch.setattr(search, "run_head_to_head", mismatched_openings)
    result_path = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="opening banks differ"):
        search.find_strongest_checkpoint(
            tmp_path,
            checkpoints[0],
            search.Config(max_games=20),
            search.SearchConfig(max_evaluations=2, coarse_count=2, finalists=1),
            result_path=result_path,
            ladder_path=tmp_path / "ladder.txt",
        )

    persisted = json.loads(result_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["error"]["type"] == "RuntimeError"
    assert len(persisted["comparisons"]) == 1


def test_bare_baseline_filename_uses_current_directory(tmp_path, monkeypatch):
    checkpoints = _make_checkpoints(tmp_path, [0, 1])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        search,
        "run_head_to_head",
        lambda *_args: _fake_result(5.0, winner="A", decision="accept_h1"),
    )

    result = search.find_strongest_checkpoint(
        None,
        checkpoints[0].name,
        search.Config(max_games=20),
        search.SearchConfig(max_evaluations=1, coarse_count=1, finalists=1),
    )

    assert result["status"] == "complete"
    assert result["selected_checkpoint"]["step"] == 1
    assert (tmp_path / "strongest_checkpoint_search.json").is_file()


@pytest.mark.parametrize("max_games", [0, 1, 99999])
def test_odd_or_non_positive_max_games_are_rejected(max_games):
    with pytest.raises(ValueError, match="positive even"):
        search.validate_config(
            search.Config(max_games=max_games),
            search.SearchConfig(),
        )


def test_inconsistent_sprt_outcome_is_rejected():
    with pytest.raises(ValueError, match="Inconsistent"):
        search._normalise_result(
            _fake_result(5.0, winner="B", decision="accept_h1")
        )


def test_output_path_cannot_overwrite_a_checkpoint(tmp_path):
    checkpoints = _make_checkpoints(tmp_path, [0, 1])
    with pytest.raises(ValueError, match="checkpoint path as result_file"):
        search.find_strongest_checkpoint(
            tmp_path,
            checkpoints[0],
            search.Config(max_games=20),
            search.SearchConfig(max_evaluations=1, coarse_count=1, finalists=1),
            result_path=checkpoints[1],
            ladder_path=tmp_path / "ladder.txt",
        )


def test_numbered_baseline_rejects_conflicting_step_override(tmp_path):
    baseline = _make_checkpoints(tmp_path, [10])[10]
    with pytest.raises(ValueError, match="conflicts"):
        search.discover_checkpoints(tmp_path, baseline, baseline_step=11)
