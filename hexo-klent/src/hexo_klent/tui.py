"""Full-screen terminal telemetry for KLENT training."""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, TextIO

from rich import box
from rich.align import Align
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_CYAN = "#5cecff"
_BLUE = "#6d8cff"
_MAGENTA = "#ff4fd8"
_VIOLET = "#a879ff"
_MINT = "#53f6c6"
_ACID = "#b8ff5c"
_AMBER = "#ffc857"
_RED = "#ff5f70"
_WHITE = "#e8f7ff"
_MUTED = "#73859a"
_DIM = "#445366"
_GRID = "#263957"
_PANEL = "#111a2d"
_SPARKS = "▁▂▃▄▅▆▇█"
_SCANNER = ("◇", "◈", "◆", "◈")
_PHASES = ("COLLECT", "FIT", "EVAL", "COMMIT")
_PULSE = (_CYAN, _BLUE, _VIOLET, _MAGENTA, _VIOLET, _BLUE)
_GRADIENT_PERIOD_SECONDS = 18.0
_SCANNER_STEP_SECONDS = 1.5


def sparkline(values: list[float], width: int = 24) -> str:
    """Render a finite numeric series as a fixed-width Unicode sparkline."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite or width <= 0:
        return "·"
    finite = finite[-width:]
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        return _SPARKS[3] * len(finite)
    scale = (len(_SPARKS) - 1) / (high - low)
    return "".join(
        _SPARKS[min(len(_SPARKS) - 1, int((value - low) * scale))]
        for value in finite
    )


def _number(
    value: float | None,
    *,
    decimals: int = 3,
    suffix: str = "",
) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000:
        rendered = f"{value / 1_000_000:.1f}M"
    elif absolute >= 10_000:
        rendered = f"{value / 1_000:.1f}k"
    elif absolute >= 100:
        rendered = f"{value:.0f}"
    else:
        rendered = f"{value:.{decimals}f}"
    return rendered + suffix


def _integer(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{int(value):,}"


def _percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def _digital_duration(seconds: float | None) -> str:
    """Format a duration as an unambiguous cyber-console clock."""

    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "--:--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _interpolate_hex(left: str, right: str, amount: float) -> str:
    """Blend two ``#rrggbb`` colours for a smooth terminal gradient."""

    amount = max(0.0, min(1.0, amount))
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start + (end - start) * amount)
        for start, end in zip(left_rgb, right_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _add_derived_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Populate display-only metrics absent from older persisted records."""

    if "training/policy_excess_kl" not in metrics:
        policy_loss = metrics.get("training/policy_loss")
        target_entropy = metrics.get("collection/mean_entropy")
        if policy_loss is not None and target_entropy is not None:
            metrics["training/policy_excess_kl"] = max(
                0.0,
                policy_loss - target_entropy,
            )
    return metrics


@dataclass(frozen=True)
class EtaEstimate:
    """Run ETA derived from completed generation telemetry."""

    remaining_seconds: float
    mean_iteration_seconds: float
    samples: int
    scheduled_evaluations: int
    confidence: str


class _DynamicDashboard:
    def __init__(self, dashboard: TrainingDashboard) -> None:
        self.dashboard = dashboard

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        yield self.dashboard.render(
            width=options.max_width,
            height=options.max_height,
        )


class DashboardLogHandler(logging.Handler):
    """Route logs into the dashboard event strip without disturbing Live."""

    def __init__(self, dashboard: TrainingDashboard) -> None:
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.name == "hexo_klent.trainer" and message.startswith(
                ("iteration=", "evaluation=")
            ):
                return
            self.dashboard.add_event(record.levelname, message)
        except Exception:
            self.handleError(record)


class TrainingDashboard:
    """A full-screen KLENT telemetry cockpit backed by completed metrics."""

    history_limit = 96

    def __init__(
        self,
        config,
        *,
        stream: TextIO | None = None,
        input_stream: TextIO | None = None,
        force_terminal: bool | None = None,
        screen: bool = True,
    ) -> None:
        self.config = config
        self.output_dir = Path(config.run.output_dir)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.console = Console(
            file=stream or sys.stdout,
            force_terminal=force_terminal,
            color_system="truecolor",
            highlight=False,
        )
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self._screen = screen and self.console.is_terminal
        self._lock = threading.RLock()
        self._history: deque[dict[str, float]] = deque(
            maxlen=self.history_limit
        )
        self._events: deque[tuple[str, str, float]] = deque(maxlen=6)
        self._started_at = time.monotonic()
        self._completed_at: float | None = None
        self._iteration_started_at = self._started_at
        self._phase_started_at = self._started_at
        self._run_start_iteration = 0
        self._current_iteration = 0
        self._active_iteration = 1
        self._stop_at = config.run.iterations
        self._phase = "INITIALIZE"
        self._phase_detail = "allocating model + actor fabric"
        self._state = "BOOT"
        self._final_checkpoint: str | None = None
        self._live: Live | None = None

    @property
    def logging_handler(self) -> DashboardLogHandler:
        handler = DashboardLogHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def _load_history(self, up_to_iteration: int) -> None:
        history_by_iteration: dict[int, dict[str, float]] = {}
        if not self.metrics_path.exists():
            with self._lock:
                self._history.clear()
            return
        try:
            with self.metrics_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        parsed = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(parsed, dict):
                        snapshot = _add_derived_metrics(
                            {
                                str(key): float(value)
                                for key, value in parsed.items()
                                if isinstance(value, (int, float))
                            }
                        )
                        iteration_value = snapshot.get("iteration")
                        if iteration_value is None:
                            continue
                        iteration = int(iteration_value)
                        if iteration > up_to_iteration:
                            continue
                        # Metrics are append-only. The last record for a
                        # generation belongs to the newest resumed branch and
                        # replaces older duplicates with the same number.
                        history_by_iteration[iteration] = snapshot
                        if len(history_by_iteration) > self.history_limit:
                            del history_by_iteration[
                                min(history_by_iteration)
                            ]
        except OSError:
            # Dashboard history is advisory; authoritative metric writing must
            # never depend on an old display record being readable.
            history_by_iteration.clear()
        with self._lock:
            self._history = deque(
                (
                    history_by_iteration[iteration]
                    for iteration in sorted(history_by_iteration)
                ),
                maxlen=self.history_limit,
            )

    def open(self) -> None:
        with self._lock:
            if self._live is not None:
                return
            self._live = Live(
                _DynamicDashboard(self),
                console=self.console,
                screen=self._screen,
                auto_refresh=True,
                refresh_per_second=6,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            )
            self._live.start(refresh=True)

    def close(self) -> None:
        with self._lock:
            live, self._live = self._live, None
        if live is not None:
            live.stop()

    def begin_run(self, current_iteration: int, stop_at: int) -> None:
        self._load_history(current_iteration)
        now = time.monotonic()
        with self._lock:
            self._completed_at = None
            self._iteration_started_at = now
            self._phase_started_at = now
            self._run_start_iteration = current_iteration
            self._current_iteration = current_iteration
            self._active_iteration = min(current_iteration + 1, stop_at)
            self._stop_at = stop_at
            self._state = "RUN"
            self._phase = "COLLECT"
            self._phase_detail = "waiting for actor telemetry"
        self._refresh()

    def set_phase(
        self,
        phase: str,
        iteration: int,
        detail: str = "",
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._phase = phase.upper()
            if iteration != self._active_iteration:
                self._iteration_started_at = now
            self._active_iteration = iteration
            self._phase_started_at = now
            self._phase_detail = detail
        self._refresh()

    def update_metrics(self, metrics: dict[str, float]) -> None:
        now = time.monotonic()
        with self._lock:
            snapshot = _add_derived_metrics(
                {
                    str(key): float(value)
                    for key, value in metrics.items()
                }
            )
            iteration = int(snapshot.get("iteration", 0.0))
            # Replacing generation N starts a new visible branch at N: discard
            # an older N and any stale future generations from a prior resume.
            retained = [
                item
                for item in self._history
                if int(item.get("iteration", 0.0)) < iteration
            ]
            retained.append(snapshot)
            self._history = deque(
                retained[-self.history_limit :],
                maxlen=self.history_limit,
            )
            self._current_iteration = iteration
            self._active_iteration = min(
                self._current_iteration + 1,
                self._stop_at,
            )
            self._iteration_started_at = now
            self._phase_started_at = now
            if self._current_iteration >= self._stop_at:
                self._phase = "COMMIT"
                self._phase_detail = "generation sealed"
            else:
                self._phase = "COLLECT"
                self._phase_detail = "next generation queued"
        self._refresh()

    def complete(self, checkpoint: str | Path) -> None:
        with self._lock:
            self._completed_at = time.monotonic()
            self._state = "COMPLETE"
            self._phase = "COMPLETE"
            self._phase_detail = "run target reached"
            self._final_checkpoint = str(checkpoint)
        self._refresh()

    def _elapsed_seconds(self) -> float:
        with self._lock:
            stopped_at = self._completed_at
            started_at = self._started_at
        now = time.monotonic() if stopped_at is None else stopped_at
        return max(0.0, now - started_at)

    def wait_for_exit(self) -> bool:
        """Keep a completed interactive cockpit visible until Enter."""

        with self._lock:
            should_wait = (
                self._state == "COMPLETE"
                and self.console.is_terminal
                and self.input_stream.isatty()
            )
        if not should_wait:
            return False
        self._refresh()
        try:
            self.input_stream.readline()
        except (KeyboardInterrupt, EOFError, OSError):
            # At this point the checkpoint and metrics are already durable.
            pass
        return True

    def interrupt(self, iteration: int) -> None:
        with self._lock:
            self._completed_at = time.monotonic()
            self._state = "INTERRUPTED"
            self._phase = "HALT"
            self._phase_detail = f"last committed generation {iteration}"
            self._current_iteration = iteration
        self._refresh()

    def add_event(self, level: str, message: str) -> None:
        with self._lock:
            self._events.append(
                (level.upper(), message[:180], time.monotonic())
            )
        self._refresh()

    def _refresh(self) -> None:
        with self._lock:
            live = self._live
        if live is not None:
            live.refresh()

    def _latest(self) -> dict[str, float]:
        return self._history[-1] if self._history else {}

    def _series(self, key: str) -> list[float]:
        return [
            metrics[key]
            for metrics in self._history
            if key in metrics and math.isfinite(metrics[key])
        ]

    def _animation_clock(self) -> float:
        """Return an animation clock that freezes at terminal states."""
        stopped_at = self._completed_at
        return time.monotonic() if stopped_at is None else stopped_at

    def _animation_step(self) -> int:
        """Return a slow scanner frame that freezes at terminal states."""

        return int(self._animation_clock() / _SCANNER_STEP_SECONDS)

    def _pulse_color(self, offset: int = 0) -> str:
        position = (
            self._animation_clock() / _GRADIENT_PERIOD_SECONDS
            + offset / len(_PULSE)
        ) % 1.0 * len(_PULSE)
        index = int(position)
        amount = position - index
        return _interpolate_hex(
            _PULSE[index],
            _PULSE[(index + 1) % len(_PULSE)],
            amount,
        )

    def _run_progress(self) -> tuple[int, int]:
        """Return completed/total generations for this invocation."""

        total = max(0, self._stop_at - self._run_start_iteration)
        completed = max(
            0,
            min(self._current_iteration, self._stop_at)
            - self._run_start_iteration,
        )
        return min(completed, total), total

    def _is_scheduled_evaluation(self, iteration: int) -> bool:
        evaluation = self.config.evaluation
        return bool(
            evaluation.interval > 0
            and evaluation.opponents
            and iteration % evaluation.interval == 0
        )

    @staticmethod
    def _contains_evaluation(metrics: dict[str, float]) -> bool:
        return any(key.startswith("evaluation/") for key in metrics)

    def _configured_evaluation_names(self) -> set[str]:
        return {
            opponent.name or opponent.kind
            for opponent in self.config.evaluation.opponents
        }

    @staticmethod
    def _metric_evaluation_names(metrics: dict[str, float]) -> set[str]:
        return {
            key[len("evaluation/") : -len("/games")]
            for key in metrics
            if key.startswith("evaluation/") and key.endswith("/games")
        }

    @staticmethod
    def _metric_evaluation_games(metrics: dict[str, float]) -> float:
        return sum(
            value
            for key, value in metrics.items()
            if key.startswith("evaluation/") and key.endswith("/games")
        )

    def _eta_estimate(self) -> EtaEstimate | None:
        """Estimate remaining wall time from recent regular/eval generations.

        Evaluation generations are modelled separately because fixed-checkpoint
        MCTS can dominate their wall time. The active generation's elapsed time
        is subtracted from its prediction, while later generations retain their
        complete estimates.
        """

        with self._lock:
            history = list(self._history)
            current_iteration = self._current_iteration
            stop_at = self._stop_at
            state = self._state
            iteration_started_at = self._iteration_started_at
            stopped_at = self._completed_at

        durations: list[tuple[float, bool]] = []
        for metrics in history:
            duration = metrics.get("iteration_seconds")
            if (
                duration is None
                or not math.isfinite(duration)
                or duration <= 0.0
            ):
                continue
            durations.append(
                (float(duration), self._contains_evaluation(metrics))
            )
        if not durations:
            return None

        recent = durations[-48:]
        fallback = float(median(value for value, _is_eval in recent))
        regular_values = [
            value for value, is_eval in recent if not is_eval
        ][-32:]
        evaluation_values = [
            value for value, is_eval in recent if is_eval
        ][-16:]
        regular_duration = (
            float(median(regular_values)) if regular_values else fallback
        )
        evaluation_duration = (
            float(median(evaluation_values))
            if evaluation_values
            else regular_duration
        )
        evaluation_metrics = [
            metrics
            for metrics in history[-48:]
            if self._contains_evaluation(metrics)
        ]
        historical_game_totals = [
            self._metric_evaluation_games(metrics)
            for metrics in evaluation_metrics
        ]
        historical_game_totals = [
            total for total in historical_game_totals if total > 0.0
        ]
        configured_games = sum(
            opponent.games
            for opponent in self.config.evaluation.opponents
        )
        if historical_game_totals and configured_games > 0:
            historical_games = float(median(historical_game_totals))
            overhead = max(0.0, evaluation_duration - regular_duration)
            evaluation_duration = regular_duration + overhead * (
                configured_games / historical_games
            )
        latest_evaluation_names = (
            self._metric_evaluation_names(evaluation_metrics[-1])
            if evaluation_metrics
            else set()
        )
        evaluation_suite_matches = (
            latest_evaluation_names == self._configured_evaluation_names()
        )

        pending = list(range(current_iteration + 1, stop_at + 1))
        scheduled_evaluations = sum(
            self._is_scheduled_evaluation(iteration)
            for iteration in pending
        )
        predicted = [
            (
                evaluation_duration
                if self._is_scheduled_evaluation(iteration)
                else regular_duration
            )
            for iteration in pending
        ]
        if not predicted or state == "COMPLETE":
            remaining = 0.0
        else:
            now = (
                stopped_at
                if stopped_at is not None
                else time.monotonic()
            )
            active_elapsed = max(0.0, now - iteration_started_at)
            remaining = max(0.0, predicted[0] - active_elapsed)
            remaining += sum(predicted[1:])

        confidence = (
            "LOCKED"
            if len(durations) >= 8 and (
                not scheduled_evaluations
                or (bool(evaluation_values) and evaluation_suite_matches)
            )
            else "ACQUIRING"
        )
        return EtaEstimate(
            remaining_seconds=remaining,
            mean_iteration_seconds=sum(predicted) / len(predicted)
            if predicted
            else fallback,
            samples=len(durations),
            scheduled_evaluations=scheduled_evaluations,
            confidence=confidence,
        )

    def _landing_time(self, eta: EtaEstimate | None) -> str:
        if eta is None:
            return "--:--"
        arrival = time.time() + eta.remaining_seconds
        template = "%a %H:%M" if eta.remaining_seconds >= 86_400 else "%H:%M"
        return time.strftime(template, time.localtime(arrival))

    def _compact_meter(self, width: int = 12) -> Text:
        completed, total = self._run_progress()
        fraction = completed / total if total else 1.0
        filled = min(width, int(fraction * width))
        meter = Text()
        meter.append("▰" * filled, style=f"bold {self._pulse_color()}")
        if filled < width:
            meter.append("◆", style=f"bold {self._pulse_color(2)}")
            meter.append("▱" * max(0, width - filled - 1), style=_GRID)
        return meter

    def _wide_meter(self, width: int = 48) -> Text:
        completed, total = self._run_progress()
        fraction = completed / total if total else 1.0
        meter = self._compact_meter(width)
        meter.append(f"  {100.0 * fraction:5.1f}%", style=f"bold {_WHITE}")
        return meter

    def _health(self) -> tuple[str, str, list[tuple[str, str, str]]]:
        latest = self._latest()
        if not latest:
            return (
                "CALIBRATING",
                _BLUE,
                [
                    ("NUMERICS", "WAIT", _MUTED),
                    ("DATA PLANE", "WAIT", _MUTED),
                    ("GRADIENTS", "WAIT", _MUTED),
                    ("Q RANGE", "WAIT", _MUTED),
                ],
            )

        # History remains visible immediately after a resume, but it may have
        # been produced with a different collection target (for example when
        # testing a new positions-per-iteration cadence). Do not judge the new
        # invocation's data plane from an old generation. The first completed
        # generation after ``begin_run`` will replace this waiting state with
        # an exact, current-config check.
        latest_iteration = int(latest.get("iteration", 0.0))
        if (
            self._state == "RUN"
            and latest_iteration <= self._run_start_iteration
        ):
            return (
                "CALIBRATING",
                _BLUE,
                [
                    ("NUMERICS", "WAIT", _MUTED),
                    ("DATA PLANE", "WAIT", _MUTED),
                    ("GRAD CLIP", "WAIT", _MUTED),
                    ("Q RANGE", "WAIT", _MUTED),
                ],
            )

        relevant = [
            value
            for key, value in latest.items()
            if key.startswith(("collection/", "training/"))
        ]
        finite = bool(relevant) and all(math.isfinite(value) for value in relevant)
        positions = latest.get("collection/positions")
        examples = latest.get("training/examples")
        target = float(self.config.collection.positions_per_iteration)
        data_exact = positions == target and examples == positions
        grad_limit = float(self.config.training.max_grad_norm)
        clip_fraction = latest.get("training/clip_fraction")
        clip_scale = latest.get("training/mean_clip_scale")
        grad_p95 = latest.get("training/grad_norm_p95")
        clip_measured = (
            clip_fraction is not None
            and clip_scale is not None
            and grad_p95 is not None
            and all(
                math.isfinite(value)
                for value in (clip_fraction, clip_scale, grad_p95)
            )
        )
        if grad_limit <= 0:
            clip_status = "DISABLED"
            clip_style = _MUTED
            clip_watch = False
        elif not clip_measured:
            clip_status = "UNMEASURED"
            clip_style = _MUTED
            clip_watch = False
        elif clip_fraction == 0:
            clip_status = "CLEAR"
            clip_style = _MINT
            clip_watch = False
        else:
            clip_status = f"{_percent(clip_fraction)} ×{clip_scale:.2f}"
            clip_severe = (
                grad_p95 >= 4.0 * grad_limit
                or (clip_fraction >= 0.90 and clip_scale < 0.50)
            )
            clip_watch = clip_severe or (
                clip_fraction >= 0.50 and clip_scale < 0.80
            )
            clip_style = _RED if clip_severe else (
                _AMBER if clip_watch else _BLUE
            )
        mean_abs_q = latest.get("collection/mean_abs_q")
        q_hot = (
            mean_abs_q is not None
            and math.isfinite(mean_abs_q)
            and mean_abs_q >= 0.90
        )

        signals = [
            (
                "NUMERICS",
                "FINITE" if finite else "FAULT",
                _MINT if finite else _RED,
            ),
            (
                "DATA PLANE",
                "EXACT" if data_exact else "MISMATCH",
                _MINT if data_exact else _RED,
            ),
            (
                "GRAD CLIP",
                clip_status,
                clip_style,
            ),
            (
                "Q RANGE",
                "HOT" if q_hot else "NOMINAL",
                _AMBER if q_hot else _MINT,
            ),
        ]
        if not finite or not data_exact:
            return "FAULT", _RED, signals
        if clip_watch or q_hot:
            return "WATCH", _AMBER, signals
        return "NOMINAL", _MINT, signals

    def _header(self, width: int = 140) -> Panel:
        health, health_style, _signals = self._health()
        if self._state == "COMPLETE":
            health, health_style = "COMPLETE", _MINT
        elif self._state == "INTERRUPTED":
            health, health_style = "HALTED", _AMBER
        elif self._state == "BOOT":
            health, health_style = "BOOT", _BLUE
        scanner = _SCANNER[self._animation_step() % len(_SCANNER)]
        elapsed = self._elapsed_seconds()
        eta = self._eta_estimate()
        identity = Text()
        identity.append(f" {scanner} ", style=f"bold {self._pulse_color(3)}")
        identity.append("HΞXO", style=f"bold {_WHITE}")
        identity.append(" // ", style=_DIM)
        identity.append(
            "K L E N T" if width >= 100 else "KLENT",
            style=f"bold {self._pulse_color()}",
        )
        if width >= 100:
            identity.append("  ANALYTIC SELF-PLAY CORE", style=_MUTED)

        chrono = Text(justify="right")
        chrono.append(f" {health} ", style=f"bold black on {health_style}")
        if self._state == "COMPLETE":
            chrono.append("  ETA ARRIVED", style=f"bold {_MINT}")
        elif self._state == "INTERRUPTED":
            chrono.append("  ETA HALTED", style=f"bold {_AMBER}")
        else:
            chrono.append(
                f"  ETA {_digital_duration(
                    None if eta is None else eta.remaining_seconds
                )}",
                style=f"bold {self._pulse_color(1)}",
            )
        chrono.append(f"  T+{elapsed:06.1f}s ", style=_MUTED)

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1, no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(identity, chrono)
        return Panel(
            grid,
            box=box.DOUBLE,
            border_style=self._pulse_color(),
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _pipeline(self) -> Text:
        active = self._phase
        if active.startswith("EVAL"):
            active = "EVAL"
        active_index = _PHASES.index(active) if active in _PHASES else -1
        line = Text()
        for index, phase in enumerate(_PHASES):
            if index:
                line.append(" ━━━▶ ", style=_GRID)
            if self._state == "COMPLETE":
                line.append(f" ◆ {phase} ", style=f"bold black on {_MINT}")
            elif active == phase:
                line.append(
                    f" ◉ {phase} ",
                    style=f"bold black on {self._pulse_color()}",
                )
            elif active_index >= 0 and index < active_index:
                line.append(f" ◆ {phase} ", style=f"bold {_MINT}")
            else:
                line.append(f" ◇ {phase} ", style=f"{_MUTED} on #1a253b")
        return line

    def _run_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style=_MUTED, width=11)
        table.add_column(style=_WHITE, ratio=1)
        completed, total = self._run_progress()
        eta = self._eta_estimate()
        table.add_row(
            "GENERATION",
            Text(
                f"LIVE {self._active_iteration:06d}  //  "
                f"RUN {completed:03d}/{total:03d}  //  "
                f"TARGET {self._stop_at:06d}",
                style=f"bold {_WHITE}",
            ),
        )
        table.add_row(
            "",
            self._wide_meter(),
        )
        table.add_row("PHASE", self._pipeline())
        chrono = Text()
        chrono.append("ETA ", style=_MUTED)
        chrono.append(
            _digital_duration(
                None if eta is None else eta.remaining_seconds
            ),
            style=f"bold {self._pulse_color(1)}",
        )
        chrono.append("  ◆  LAND ", style=_DIM)
        chrono.append(self._landing_time(eta), style=f"bold {_ACID}")
        chrono.append("  ◆  μGEN ", style=_DIM)
        chrono.append(
            _digital_duration(
                None if eta is None else eta.mean_iteration_seconds
            ),
            style=_CYAN,
        )
        chrono.append("  ◆  ", style=_DIM)
        chrono.append(
            (
                "CALIBRATING"
                if eta is None
                else f"{eta.confidence} N={eta.samples}"
            ),
            style=_AMBER if eta is None or eta.confidence != "LOCKED" else _MINT,
        )
        table.add_row("CHRONO", chrono)
        table.add_row(
            "VECTOR",
            Text(
                f"{self.config.run.device.upper()} / "
                f"{self.config.run.precision.upper()}   "
                f"{self.config.collection.workers} ACTORS × "
                f"{self.config.collection.parallel_games} LANES",
                style=_BLUE,
            ),
        )
        table.add_row(
            "RULESPACE",
            Text(
                f"W{self.config.game.win_length}  "
                f"R{self.config.game.placement_radius}  "
                f"H{self.config.game.rollout_horizon}",
                style=_VIOLET,
            ),
        )
        detail = self._phase_detail or "telemetry link active"
        table.add_row("CHANNEL", Text(detail, style=_MUTED))
        return Panel(
            table,
            title=Text(
                " ◈ RUN VECTOR // CHRONO-LINK ",
                style=f"bold {self._pulse_color()}",
            ),
            title_align="left",
            border_style=self._pulse_color(1),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _health_panel(self) -> Panel:
        health, health_style, signals = self._health()
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", width=15)
        for name, value, style in signals:
            table.add_row(
                Text(name, style=_MUTED),
                Text(f"● {value}", style=f"bold {style}"),
            )
        decided = [
            values["decided_rate"]
            for _iteration, values in self._latest_evaluations().values()
            if "decided_rate" in values
        ]
        coverage = min(decided) if decided else None
        table.add_row(
            Text("EVAL COVERAGE", style=_MUTED),
            Text(
                _percent(coverage),
                style=_MINT if coverage is not None and coverage >= 0.8 else _AMBER,
            ),
        )
        return Panel(
            table,
            title=Text(
                f" ◈ HEALTH // {health} ",
                style=f"bold {health_style}",
            ),
            title_align="left",
            border_style=health_style,
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _collection_panel(self) -> Panel:
        latest = self._latest()
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        rows = (
            (
                "POSITIONS",
                _integer(latest.get("collection/positions")),
                "POS / SEC",
                _number(
                    latest.get("collection/positions_per_second"),
                    decimals=1,
                ),
            ),
            (
                "FRAGMENTS",
                _integer(latest.get("collection/games")),
                "MEAN LENGTH",
                _number(latest.get("collection/mean_game_length"), decimals=1),
            ),
            (
                "P1 / P2 WINS",
                (
                    f"{_integer(latest.get('collection/p1_wins'))} / "
                    f"{_integer(latest.get('collection/p2_wins'))}"
                ),
                "TRUNC H / C",
                (
                    f"{_integer(latest.get('collection/horizon_truncations'))}"
                    " / "
                    f"{_integer(latest.get('collection/chunk_truncations'))}"
                ),
            ),
            (
                "COLLECT",
                _number(
                    latest.get("collection/elapsed_seconds"),
                    decimals=1,
                    suffix="s",
                ),
                "FIT",
                _number(
                    latest.get("training/elapsed_seconds"),
                    decimals=1,
                    suffix="s",
                ),
            ),
        )
        for row in rows:
            table.add_row(*row)
        return Panel(
            table,
            title=Text(
                " ◈ SELF-PLAY FABRIC // ACTOR MESH ",
                style=f"bold {self._pulse_color()}",
            ),
            title_align="left",
            border_style=self._pulse_color(),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _optimization_panel(self) -> Panel:
        latest = self._latest()
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        rows = (
            (
                "POLICY LOSS",
                _number(latest.get("training/policy_loss")),
                "EXCESS KL",
                _number(
                    latest.get("training/policy_excess_kl"),
                    decimals=4,
                ),
            ),
            (
                "Q LOSS",
                _number(latest.get("training/q_loss")),
                "REVERSE KL",
                _number(latest.get("collection/mean_reverse_kl"), decimals=4),
            ),
            (
                "ENTROPY H / Hₙ",
                (
                    _number(latest.get("collection/mean_entropy"))
                    + " / "
                    + _number(
                        latest.get("collection/mean_normalized_entropy")
                    )
                ),
                "TOP-1 MASS",
                _percent(
                    latest.get("collection/mean_target_top1_probability")
                ),
            ),
            (
                "LEGAL MOVES",
                _number(
                    latest.get("collection/mean_legal_actions"),
                    decimals=1,
                ),
                "Q SPAN",
                _number(latest.get("collection/mean_q_span")),
            ),
            (
                "GRAD MEAN",
                _number(latest.get("training/mean_grad_norm")),
                "GRAD P50",
                _number(latest.get("training/grad_norm_p50")),
            ),
            (
                "GRAD P95",
                _number(latest.get("training/grad_norm_p95")),
                "GRAD MAX",
                _number(latest.get("training/grad_norm_max")),
            ),
            (
                "CLIP RATE",
                _percent(latest.get("training/clip_fraction")),
                "CLIP SCALE",
                _number(latest.get("training/mean_clip_scale"), decimals=3),
            ),
            (
                "BATCH / MICRO",
                (
                    _number(
                        latest.get("training/mean_optimizer_batch_size"),
                        decimals=1,
                    )
                    + " / "
                    + _number(
                        latest.get("training/mean_microbatches_per_step"),
                        decimals=2,
                    )
                ),
                "EX / SEC",
                _number(
                    latest.get("training/examples_per_second"),
                    decimals=1,
                ),
            ),
        )
        for row in rows:
            table.add_row(*row)
        return Panel(
            table,
            title=Text(
                " ◈ POLICY / Q DYNAMICS // GRADIENT CORE ",
                style=f"bold {self._pulse_color(3)}",
            ),
            title_align="left",
            border_style=self._pulse_color(3),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _trend_panel(self, width: int) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style=_MUTED, width=13)
        table.add_column(ratio=1)
        table.add_column(justify="right", width=15)
        trace_width = max(8, min(34, width - 35))
        trends = (
            ("EXCESS KL", "training/policy_excess_kl", _VIOLET, 4),
            ("REVERSE KL", "collection/mean_reverse_kl", _CYAN, 4),
            (
                "ENTROPY N",
                "collection/mean_normalized_entropy",
                _BLUE,
                3,
            ),
            ("LEGAL MOVES", "collection/mean_legal_actions", _BLUE, 1),
            ("Q SPAN", "collection/mean_q_span", _MAGENTA, 3),
            ("GRAD P95", "training/grad_norm_p95", _AMBER, 3),
            ("CLIP RATE", "training/clip_fraction", _AMBER, 3),
            ("ITERATION", "iteration_seconds", _AMBER, 1),
        )
        for label, key, style, decimals in trends:
            values = self._series(key)
            latest = values[-1] if values else None
            delta = ""
            if len(values) >= 2 and values[-2] != 0:
                change = 100.0 * (values[-1] - values[-2]) / abs(values[-2])
                delta = f" {change:+.1f}%"
            table.add_row(
                label,
                Text(sparkline(values, trace_width), style=f"bold {style}"),
                Text(
                    (
                        _percent(latest)
                        if key == "training/clip_fraction"
                        else _number(latest, decimals=decimals)
                    )
                    + delta,
                    style=_WHITE,
                ),
            )
        return Panel(
            table,
            title=Text(
                " ◈ TEMPORAL TRACE // LAST 96 GENERATIONS ",
                style=f"bold {self._pulse_color(2)}",
            ),
            title_align="left",
            border_style=self._pulse_color(2),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _latest_evaluations(
        self,
    ) -> dict[str, tuple[int, dict[str, float]]]:
        evaluations: dict[str, tuple[int, dict[str, float]]] = {}
        for metrics in reversed(self._history):
            iteration = int(metrics.get("iteration", 0.0))
            names = {
                key[len("evaluation/") : -len("/wins")]
                for key in metrics
                if key.startswith("evaluation/") and key.endswith("/wins")
            }
            for name in sorted(names):
                if name in evaluations:
                    continue
                prefix = f"evaluation/{name}/"
                evaluations[name] = (
                    iteration,
                    {
                        key[len(prefix) :]: value
                        for key, value in metrics.items()
                        if key.startswith(prefix)
                    },
                )
        return evaluations

    def _evaluation_panel(self) -> Panel:
        table = Table(
            expand=True,
            box=None,
            show_header=True,
            header_style=f"bold {_MUTED}",
            padding=(0, 1),
        )
        table.add_column("OPPONENT", style=_WHITE, ratio=1)
        table.add_column("GEN", justify="right", style=_MUTED)
        table.add_column("W / L / T", justify="right")
        table.add_column("DECIDED", justify="right")
        table.add_column("WIN•DEC", justify="right")
        table.add_column("MCTS  MODEL ↔ OPP", justify="right", style=_BLUE)
        evaluations = self._latest_evaluations()
        if not evaluations:
            table.add_row(
                "awaiting evaluation window",
                "—",
                "—",
                "—",
                "—",
                "—",
            )
        for name, (iteration, values) in evaluations.items():
            simulations = values.get("mcts_simulations", 0.0)
            actions = values.get("mcts_actions", 0.0)
            mcts = (
                "RAW"
                if simulations <= 0
                else f"{int(simulations)}/{int(actions)}"
            )
            if "opponent_mcts_simulations" in values:
                opponent_simulations = values["opponent_mcts_simulations"]
                opponent_actions = values.get("opponent_mcts_actions", 0.0)
                opponent_mcts = (
                    "RAW"
                    if opponent_simulations <= 0
                    else (
                        f"{int(opponent_simulations)}/"
                        f"{int(opponent_actions)}"
                    )
                )
                mcts = f"{mcts} ↔ {opponent_mcts}"
            table.add_row(
                name,
                str(iteration),
                (
                    f"{_integer(values.get('wins'))} / "
                    f"{_integer(values.get('losses'))} / "
                    f"{_integer(values.get('truncations'))}"
                ),
                _percent(values.get("decided_rate")),
                _percent(values.get("win_rate_decided")),
                mcts,
            )
        return Panel(
            table,
            title=Text(
                " ◈ OPPONENT ARRAY // THREAT GRID ",
                style=f"bold {self._pulse_color(1)}",
            ),
            title_align="left",
            border_style=self._pulse_color(1),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 1),
        )

    def _footer(self) -> Panel:
        event = None
        with self._lock:
            if self._events:
                event = self._events[-1]
        line = Text()
        if self._state == "COMPLETE":
            line.append(" ENTER ", style=f"bold black on {_MINT}")
            line.append(" RELEASE TERMINAL", style=_MUTED)
        else:
            line.append(" CTRL-C ", style=f"bold black on {_MAGENTA}")
            line.append(" SAFE SHUTDOWN", style=_MUTED)
        line.append("   ◆ NEURAL LINK ", style=_DIM)
        line.append("SECURE", style=f"bold {self._pulse_color()}")
        line.append(" ◆   ", style=_DIM)
        if self._final_checkpoint is None:
            line.append(str(self.metrics_path), style=_CYAN)
        else:
            line.append(self._final_checkpoint, style=_MINT)
        if event is not None:
            level, message, _created = event
            style = _RED if level in {"ERROR", "CRITICAL"} else (
                _AMBER if level == "WARNING" else _MUTED
            )
            line.append("   ◆   ", style=_DIM)
            line.append(f"{level} ", style=f"bold {style}")
            line.append(message, style=style)
        return Panel(
            line,
            box=box.DOUBLE,
            border_style=self._pulse_color(4),
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _compact_run_panel(self) -> Panel:
        health, health_style, signals = self._health()
        completed, total = self._run_progress()
        eta = self._eta_estimate()
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style=_MUTED, width=8)
        table.add_column(ratio=1)
        generation = Text()
        generation.append(
            f"{self._active_iteration:06d}  ",
            style=f"bold {_WHITE}",
        )
        generation.append_text(self._compact_meter())
        generation.append(
            f"  {completed}/{total} → {self._stop_at:06d}",
            style=_MUTED,
        )
        table.add_row(
            "GEN",
            generation,
        )
        table.add_row("PHASE", self._pipeline())
        chrono = Text()
        chrono.append(
            _digital_duration(
                None if eta is None else eta.remaining_seconds
            ),
            style=f"bold {self._pulse_color(1)}",
        )
        chrono.append("  LAND ", style=_DIM)
        chrono.append(self._landing_time(eta), style=_ACID)
        chrono.append(
            (
                "  CAL"
                if eta is None
                else f"  {eta.confidence}·{eta.samples}"
            ),
            style=_AMBER if eta is None or eta.confidence != "LOCKED" else _MINT,
        )
        table.add_row("ETA", chrono)
        signal_text = Text()
        for index, (name, value, style) in enumerate(signals):
            if index:
                signal_text.append("  ")
            signal_text.append(f"{name.split()[0]} ", style=_MUTED)
            signal_text.append(value, style=f"bold {style}")
        table.add_row("HEALTH", signal_text)
        return Panel(
            table,
            title=Text(
                f" ◈ RUN VECTOR // {health} ",
                style=f"bold {health_style}",
            ),
            title_align="left",
            border_style=health_style,
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _compact_metrics_panel(self) -> Panel:
        latest = self._latest()
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        table.add_column(style=_MUTED, ratio=1)
        table.add_column(justify="right", style=_WHITE)
        rows = (
            (
                "EXCESS KL",
                _number(
                    latest.get("training/policy_excess_kl"),
                    decimals=4,
                ),
                "REVERSE KL",
                _number(latest.get("collection/mean_reverse_kl"), decimals=4),
            ),
            (
                "ENTROPY H / Hₙ",
                (
                    _number(latest.get("collection/mean_entropy"))
                    + " / "
                    + _number(
                        latest.get("collection/mean_normalized_entropy")
                    )
                ),
                "TOP-1 MASS",
                _percent(
                    latest.get("collection/mean_target_top1_probability")
                ),
            ),
            (
                "LEGAL MOVES",
                _number(
                    latest.get("collection/mean_legal_actions"),
                    decimals=1,
                ),
                "Q SPAN",
                _number(latest.get("collection/mean_q_span")),
            ),
            (
                "COLLECT / S",
                _number(
                    latest.get("collection/positions_per_second"),
                    decimals=1,
                ),
                "FIT / S",
                _number(
                    latest.get("training/examples_per_second"),
                    decimals=1,
                ),
            ),
        )
        for row in rows:
            table.add_row(*row)
        return Panel(
            table,
            title=Text(
                " ◈ LIVE TELEMETRY // NEURAL CORE ",
                style=f"bold {self._pulse_color(3)}",
            ),
            title_align="left",
            border_style=self._pulse_color(3),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _compact_trace_panel(self, width: int) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style=_MUTED, width=12)
        table.add_column(ratio=1)
        table.add_column(justify="right", width=9)
        trace_width = max(8, width - 28)
        for label, key, style, decimals in (
            ("EXCESS KL", "training/policy_excess_kl", _VIOLET, 4),
            ("REVERSE KL", "collection/mean_reverse_kl", _CYAN, 4),
            (
                "ENTROPY N",
                "collection/mean_normalized_entropy",
                _BLUE,
                3,
            ),
        ):
            values = self._series(key)
            table.add_row(
                label,
                Text(sparkline(values, trace_width), style=f"bold {style}"),
                _number(values[-1] if values else None, decimals=decimals),
            )

        evaluation_parts = []
        for name, (iteration, values) in list(
            self._latest_evaluations().items()
        )[:2]:
            evaluation_parts.append(
                f"{name} {_percent(values.get('win_rate_decided'))}"
                f" @{iteration}"
            )
        table.add_row(
            "EVAL",
            Text("  ◆  ".join(evaluation_parts) or "awaiting window", style=_BLUE),
            "",
        )
        return Panel(
            table,
            title=Text(
                " ◈ TEMPORAL TRACE // 96 GEN ",
                style=f"bold {self._pulse_color(2)}",
            ),
            title_align="left",
            border_style=self._pulse_color(2),
            box=box.ROUNDED,
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _compact_footer(self) -> Panel:
        line = Text()
        if self._state == "COMPLETE":
            line.append(" ENTER ", style=f"bold black on {_MINT}")
            line.append(" RELEASE TERMINAL", style=_MUTED)
        else:
            line.append(" CTRL-C ", style=f"bold black on {_MAGENTA}")
            line.append(" SAFE HALT", style=_MUTED)
        line.append("   ◆ LINK ", style=_DIM)
        line.append("SECURE", style=f"bold {self._pulse_color()}")
        line.append(" ◆   ", style=_DIM)
        if self._final_checkpoint is None:
            line.append(
                "metrics.jsonl + TensorBoard remain authoritative",
                style=_CYAN,
            )
        else:
            line.append("FINAL CHECKPOINT SEALED", style=_MINT)
        return Panel(
            Align.center(line),
            box=box.DOUBLE,
            border_style=self._pulse_color(4),
            style=f"on {_PANEL}",
            padding=(0, 0),
        )

    def _compact(self, width: int) -> Layout:
        root = Layout(name="compact")
        root.split_column(
            Layout(self._header(width), size=3),
            Layout(self._compact_run_panel(), size=6),
            Layout(self._compact_metrics_panel(), size=6),
            Layout(self._compact_trace_panel(width), ratio=1),
            Layout(self._compact_footer(), size=3),
        )
        return root

    def render(self, *, width: int, height: int) -> Any:
        with self._lock:
            if width < 100 or height < 34:
                return self._compact(width)

            root = Layout(name="root")
            root.split_column(
                Layout(self._header(width), name="header", size=3),
                Layout(name="overview", size=10),
                Layout(name="metrics", size=10),
                Layout(name="history", ratio=1),
                Layout(self._footer(), name="footer", size=3),
            )
            root["overview"].split_row(
                Layout(self._run_panel(), name="run", ratio=2),
                Layout(self._health_panel(), name="health", ratio=1),
            )
            root["metrics"].split_row(
                Layout(self._collection_panel(), name="collection"),
                Layout(self._optimization_panel(), name="optimization"),
            )
            root["history"].split_row(
                Layout(
                    self._trend_panel(max(50, width // 2)),
                    name="trends",
                    ratio=1,
                ),
                Layout(self._evaluation_panel(), name="evaluation", ratio=1),
            )
            return root
