import logging
import os
from pathlib import Path

import pytest

from hexo_klent.cli import _parser, _resolve_resume, main


FIXTURE = Path(__file__).parent / "fixtures" / "config.toml"


def test_tui_can_be_forced_or_disabled():
    enabled = _parser().parse_args(
        ["train", "--config", str(FIXTURE), "--tui"]
    )
    disabled = _parser().parse_args(
        ["train", "--config", str(FIXTURE), "--no-tui"]
    )

    assert enabled.tui is True
    assert disabled.tui is False


def test_sprt_defaults_match_klent_ablation_evaluation():
    args = _parser().parse_args(
        ["sprt", "--candidate", "candidate.pt", "--opponent", "opponent.pt"]
    )

    assert args.max_moves == 1000
    assert args.max_games == 1000
    assert args.mcts_simulations == 24
    assert args.mcts_actions == 8
    assert args.precision == "bf16"
    assert args.compile is True


def test_resume_latest_uses_most_recent_complete_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    stale_future = checkpoint_dir / "checkpoint_000999.pt"
    final = checkpoint_dir / "final.pt"
    newest_branch = checkpoint_dir / "checkpoint_000050.pt"
    for checkpoint in (stale_future, final, newest_branch):
        checkpoint.touch()
    os.utime(stale_future, ns=(1, 1))
    os.utime(final, ns=(2, 2))
    os.utime(newest_branch, ns=(3, 3))

    assert _resolve_resume("latest", tmp_path) == newest_branch
    assert _resolve_resume(str(stale_future), tmp_path) == str(stale_future)


def test_resume_latest_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(SystemExit, match="found no checkpoints"):
        _resolve_resume("latest", tmp_path)


def test_keyboard_interrupt_exits_cleanly_with_status_130(
    monkeypatch,
    caplog,
):
    class InterruptingTrainer:
        def __init__(self, *_args, **_kwargs):
            self.iteration = 7

        @staticmethod
        def run(_iterations):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "hexo_klent.trainer.Trainer",
        InterruptingTrainer,
    )

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(SystemExit) as raised,
    ):
        main(
            [
                "train",
                "--config",
                str(FIXTURE),
                "--iterations",
                "1",
                "--no-tensorboard",
            ]
        )

    assert raised.value.code == 130
    assert (
        "training interrupted after completed iteration 7; workers stopped"
        in caplog.text
    )


def test_tui_is_halted_and_closed_on_keyboard_interrupt(monkeypatch):
    events = []

    class FakeDashboard:
        def __init__(self, _config):
            events.append("create")
            self.logging_handler = logging.NullHandler()

        def open(self):
            events.append("open")

        def interrupt(self, iteration):
            events.append(("interrupt", iteration))

        def close(self):
            events.append("close")

    class InterruptingTrainer:
        def __init__(self, *_args, display=None, **_kwargs):
            assert isinstance(display, FakeDashboard)
            self.iteration = 9

        @staticmethod
        def run(_iterations):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "hexo_klent.tui.TrainingDashboard",
        FakeDashboard,
    )
    monkeypatch.setattr(
        "hexo_klent.trainer.Trainer",
        InterruptingTrainer,
    )
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    try:
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "train",
                    "--config",
                    str(FIXTURE),
                    "--iterations",
                    "1",
                    "--no-tensorboard",
                    "--tui",
                ]
            )
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)

    assert raised.value.code == 130
    assert events == ["create", "open", ("interrupt", 9), "close"]


def test_tui_waits_after_successful_completion(monkeypatch):
    events = []

    class FakeDashboard:
        def __init__(self, _config):
            events.append("create")
            self.logging_handler = logging.NullHandler()

        def open(self):
            events.append("open")

        def wait_for_exit(self):
            events.append("wait")

        def close(self):
            events.append("close")

    class CompletingTrainer:
        def __init__(self, *_args, display=None, **_kwargs):
            assert isinstance(display, FakeDashboard)
            self.iteration = 1

        @staticmethod
        def run(iterations):
            assert iterations == 1
            events.append("run")

    monkeypatch.setattr(
        "hexo_klent.tui.TrainingDashboard",
        FakeDashboard,
    )
    monkeypatch.setattr(
        "hexo_klent.trainer.Trainer",
        CompletingTrainer,
    )
    monkeypatch.setattr(
        "hexo_klent.cli._release_accelerator_cache",
        lambda: events.append("release"),
    )
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    try:
        main(
            [
                "train",
                "--config",
                str(FIXTURE),
                "--iterations",
                "1",
                "--no-tensorboard",
                "--tui",
            ]
        )
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)

    assert events == [
        "create",
        "open",
        "run",
        "release",
        "wait",
        "close",
    ]
