import io
import json
import math
import threading
from pathlib import Path

import pytest
from rich.console import Console

from hexo_klent.config import EvaluationOpponentConfig, load_config
from hexo_klent.tui import TrainingDashboard, sparkline


FIXTURE = Path(__file__).parent / "fixtures" / "config.toml"


def _metrics(
    *,
    iteration=1.0,
    iteration_seconds=9.5,
    grad=0.5,
    grad_p50=0.45,
    grad_p95=0.75,
    grad_max=0.9,
    clip_fraction=0.0,
    clip_scale=1.0,
    mean_abs_q=0.2,
    include_evaluation=True,
):
    metrics = {
        "iteration": iteration,
        "iteration_seconds": iteration_seconds,
        "collection/positions": 16.0,
        "collection/games": 4.0,
        "collection/p1_wins": 1.0,
        "collection/p2_wins": 2.0,
        "collection/discarded_positions": 0.0,
        "collection/horizon_truncations": 0.0,
        "collection/chunk_truncations": 0.0,
        "collection/mean_game_length": 4.0,
        "collection/mean_entropy": 2.4,
        "collection/mean_normalized_entropy": 0.68,
        "collection/mean_target_top1_probability": 0.32,
        "collection/mean_prior_normalized_entropy": 0.74,
        "collection/mean_prior_top1_probability": 0.26,
        "collection/mean_legal_actions": 36.0,
        "collection/mean_reverse_kl": 0.08,
        "collection/mean_abs_q": mean_abs_q,
        "collection/mean_q_span": 0.45,
        "collection/mean_abs_return": 0.35,
        "collection/elapsed_seconds": 6.0,
        "collection/positions_per_second": 2.67,
        "training/examples": 16.0,
        "training/microbatches": 12.0,
        "training/mean_microbatch_size": 4.0 / 3.0,
        "training/mean_optimizer_batch_size": 2.0,
        "training/mean_microbatches_per_step": 1.5,
        "training/policy_loss": 2.41,
        "training/policy_excess_kl": 0.01,
        "training/policy_diagnostic_examples": 16.0,
        "training/policy_diagnostic_seconds": 0.1,
        "training/policy_target_kl_before": 0.08,
        "training/policy_target_kl_after": 0.02,
        "training/policy_target_progress": 0.75,
        "training/policy_target_top1_agreement_before": 0.72,
        "training/policy_target_top1_agreement_after": 0.81,
        "training/policy_target_top1_agreement_delta": 0.09,
        "training/q_loss": 0.18,
        "training/total_loss": 2.59,
        "training/trunk_gradient_diagnostic_examples": 4.0,
        "training/trunk_gradient_diagnostic_seconds": 0.05,
        "training/policy_trunk_grad_norm": 0.7,
        "training/q_trunk_grad_norm": 1.4,
        "training/policy_q_trunk_grad_cosine": -0.2,
        "training/mean_grad_norm": grad,
        "training/grad_norm_p50": grad_p50,
        "training/grad_norm_p95": grad_p95,
        "training/grad_norm_max": grad_max,
        "training/optimizer_steps": 8.0,
        "training/clipped_optimizer_steps": 8.0 * clip_fraction,
        "training/clip_fraction": clip_fraction,
        "training/mean_clip_scale": clip_scale,
        "training/mean_parameter_update_norm": 0.0042,
        "training/parameter_update_norm_p95": 0.0061,
        "training/mean_update_to_weight_ratio": 0.000021,
        "training/update_to_weight_ratio_p95": 0.000030,
        "training/played_action_target_top1": 0.42,
        "training/elapsed_seconds": 3.0,
        "training/examples_per_second": 5.33,
        "evaluation/random/wins": 3.0,
        "evaluation/random/losses": 1.0,
        "evaluation/random/truncations": 0.0,
        "evaluation/random/decided_rate": 1.0,
        "evaluation/random/win_rate_decided": 0.75,
        "evaluation/random/mcts_simulations": 0.0,
        "evaluation/random/mcts_actions": 16.0,
        "evaluation/sealbot_mcts24/8/wins": 2.0,
        "evaluation/sealbot_mcts24/8/losses": 1.0,
        "evaluation/sealbot_mcts24/8/truncations": 1.0,
        "evaluation/sealbot_mcts24/8/decided_rate": 0.75,
        "evaluation/sealbot_mcts24/8/win_rate_decided": 2.0 / 3.0,
        "evaluation/sealbot_mcts24/8/mcts_simulations": 24.0,
        "evaluation/sealbot_mcts24/8/mcts_actions": 8.0,
        "evaluation/s1_anchor/wins": 2.0,
        "evaluation/s1_anchor/losses": 2.0,
        "evaluation/s1_anchor/truncations": 0.0,
        "evaluation/s1_anchor/decided_rate": 1.0,
        "evaluation/s1_anchor/win_rate_decided": 0.5,
        "evaluation/s1_anchor/mcts_simulations": 24.0,
        "evaluation/s1_anchor/mcts_actions": 8.0,
        "evaluation/s1_anchor/opponent_mcts_simulations": 48.0,
        "evaluation/s1_anchor/opponent_mcts_actions": 12.0,
    }
    if not include_evaluation:
        metrics = {
            key: value
            for key, value in metrics.items()
            if not key.startswith("evaluation/")
        }
    return metrics


def _dashboard(tmp_path):
    config = load_config(FIXTURE)
    config.run.output_dir = str(tmp_path)
    return TrainingDashboard(
        config,
        stream=io.StringIO(),
        force_terminal=True,
        screen=False,
    )


class _TTYInput(io.StringIO):
    def isatty(self):
        return True


def test_sparkline_filters_nonfinite_values_and_tracks_shape():
    rendered = sparkline([1.0, math.nan, 2.0, 4.0], width=8)

    assert len(rendered) == 3
    assert rendered[0] == "▁"
    assert rendered[-1] == "█"
    assert sparkline([2.0, 2.0, 2.0]) == "▄▄▄"


def test_completed_dashboard_waits_for_enter_and_updates_prompt(tmp_path):
    config = load_config(FIXTURE)
    config.run.output_dir = str(tmp_path)
    input_stream = _TTYInput("\n")
    dashboard = TrainingDashboard(
        config,
        stream=io.StringIO(),
        input_stream=input_stream,
        force_terminal=True,
        screen=False,
    )

    dashboard.complete(tmp_path / "checkpoints" / "final.pt")

    assert dashboard.wait_for_exit() is True
    assert input_stream.tell() == 1
    assert "RELEASE TERMINAL" in dashboard._footer().renderable.plain
    assert (
        "RELEASE TERMINAL"
        in dashboard._compact_footer().renderable.renderable.plain
    )


def test_completed_dashboard_freezes_elapsed_timer(tmp_path, monkeypatch):
    dashboard = _dashboard(tmp_path)
    dashboard._started_at = 100.0
    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 125.0)

    dashboard.complete(tmp_path / "checkpoints" / "final.pt")

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 999.0)
    assert dashboard._elapsed_seconds() == pytest.approx(25.0)
    stream = io.StringIO()
    Console(
        file=stream,
        force_terminal=False,
        width=140,
    ).print(dashboard._header())
    assert "T+0025.0s" in stream.getvalue()
    assert "ETA ARRIVED" in stream.getvalue()


def test_dashboard_uses_slow_gradient_and_freezes_animation(
    tmp_path,
    monkeypatch,
):
    dashboard = _dashboard(tmp_path)

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 0.0)
    start = dashboard._pulse_color()
    assert dashboard._animation_step() == 0

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 0.1)
    nearby = dashboard._pulse_color()
    assert dashboard._animation_step() == 0

    def rgb(color):
        return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))

    assert max(
        abs(start_channel - nearby_channel)
        for start_channel, nearby_channel in zip(
            rgb(start),
            rgb(nearby),
            strict=True,
        )
    ) <= 4

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 5.0)
    dashboard.complete(tmp_path / "checkpoints" / "final.pt")
    frozen_color = dashboard._pulse_color()
    frozen_scanner = dashboard._animation_step()

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 999.0)
    assert dashboard._pulse_color() == frozen_color
    assert dashboard._animation_step() == frozen_scanner


def test_completed_dashboard_does_not_wait_without_interactive_input(tmp_path):
    config = load_config(FIXTURE)
    config.run.output_dir = str(tmp_path)
    input_stream = io.StringIO("\n")
    dashboard = TrainingDashboard(
        config,
        stream=io.StringIO(),
        input_stream=input_stream,
        force_terminal=True,
        screen=False,
    )
    dashboard.complete(tmp_path / "checkpoints" / "final.pt")

    assert dashboard.wait_for_exit() is False
    assert input_stream.tell() == 0


def test_pause_key_arms_boundary_wait_and_resumes_with_frozen_clock(
    tmp_path,
    monkeypatch,
):
    dashboard = _dashboard(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(
        "hexo_klent.tui.time.monotonic",
        lambda: clock[0],
    )
    dashboard._started_at = 90.0
    dashboard.begin_run(2, 5)

    dashboard._handle_key("p")

    assert dashboard.pause_requested is True
    assert "CANCEL PAUSE" in dashboard._footer().renderable.plain

    entered = threading.Event()
    finished = threading.Event()

    def wait_at_boundary():
        dashboard.wait_if_paused(3, on_pause=entered.set)
        finished.set()

    thread = threading.Thread(target=wait_at_boundary)
    thread.start()
    assert entered.wait(timeout=1.0)
    assert dashboard._state == "PAUSED"
    assert dashboard._phase == "PAUSED"
    assert "RESUME" in dashboard._footer().renderable.plain

    clock[0] = 110.0
    assert dashboard._elapsed_seconds() == pytest.approx(10.0)
    frozen_color = dashboard._pulse_color()
    clock[0] = 120.0
    assert dashboard._elapsed_seconds() == pytest.approx(10.0)
    assert dashboard._pulse_color() == frozen_color

    dashboard._handle_key("P")
    thread.join(timeout=1.0)

    assert finished.is_set()
    assert dashboard._state == "RUN"
    assert dashboard.pause_requested is False
    assert dashboard._phase == "COLLECT"
    assert dashboard._elapsed_seconds() == pytest.approx(10.0)
    assert "PAUSE @ GEN END" in dashboard._footer().renderable.plain


def test_pause_key_can_cancel_an_armed_boundary_pause(tmp_path):
    dashboard = _dashboard(tmp_path)
    dashboard.begin_run(1, 3)

    assert dashboard.toggle_pause() is True
    assert dashboard.toggle_pause() is False

    assert dashboard.pause_requested is False
    assert dashboard.wait_if_paused(2) is False


def test_health_is_factual_and_surfaces_watch_states(tmp_path):
    dashboard = _dashboard(tmp_path)

    dashboard.update_metrics(_metrics())
    assert dashboard._health()[0] == "NOMINAL"

    dashboard.update_metrics(
        _metrics(
            iteration=2.0,
            grad=1.5,
            grad_p95=2.0,
            grad_max=2.5,
            clip_fraction=0.75,
            clip_scale=0.70,
        )
    )
    health, _style, signals = dashboard._health()
    assert health == "WATCH"
    assert ("GRAD CLIP", "75.0% ×0.70") in [
        (name, value) for name, value, _signal_style in signals
    ]

    dashboard.update_metrics(
        _metrics(
            iteration=3.0,
            grad=1.1,
            grad_p95=1.2,
            grad_max=1.3,
            clip_fraction=0.90,
            clip_scale=0.95,
        )
    )
    assert dashboard._health()[0] == "NOMINAL"

    dashboard.update_metrics(_metrics(iteration=4.0, mean_abs_q=0.95))
    assert dashboard._health()[0] == "WATCH"

    cache_pressure = _metrics(iteration=5.0)
    cache_pressure["memory/training_reserved_after_gib"] = 0.1
    cache_pressure["memory/training_peak_reserved_gib"] = 93.5
    dashboard.update_metrics(cache_pressure)
    health, _style, signals = dashboard._health()
    assert health == "WATCH"
    assert ("GPU CACHE", "0.1G/94G PEAK") in [
        (name, value) for name, value, _signal_style in signals
    ]

    policy_regression = _metrics(iteration=6.0)
    policy_regression["training/policy_target_kl_after"] = 0.10
    policy_regression["training/policy_target_progress"] = -0.25
    dashboard.update_metrics(policy_regression)
    health, _style, signals = dashboard._health()
    assert health == "NOMINAL"
    assert ("FROZEN TARGET", "FARTHER -25.0%") in [
        (name, value) for name, value, _signal_style in signals
    ]


def test_resume_history_deduplicates_and_discards_stale_future(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    records = [
        _metrics(iteration=1.0, grad=0.1),
        _metrics(iteration=2.0, grad=0.2),
        _metrics(iteration=3.0, grad=0.3),
        _metrics(iteration=2.0, grad=0.8),
        _metrics(iteration=3.0, grad=0.9),
    ]
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    dashboard = _dashboard(tmp_path)

    dashboard.begin_run(2, 5)

    assert [int(item["iteration"]) for item in dashboard._history] == [1, 2]
    assert dashboard._series("training/mean_grad_norm") == [0.1, 0.8]
    health, _style, signals = dashboard._health()
    assert health == "CALIBRATING"
    assert ("DATA PLANE", "WAIT") in [
        (name, value) for name, value, _signal_style in signals
    ]

    dashboard.update_metrics(_metrics(iteration=3.0, grad=0.7))
    assert [int(item["iteration"]) for item in dashboard._history] == [1, 2, 3]
    assert dashboard._series("training/mean_grad_norm") == [0.1, 0.8, 0.7]
    assert dashboard._health()[0] == "NOMINAL"

    dashboard.update_metrics(_metrics(iteration=2.0, grad=0.6))
    assert [int(item["iteration"]) for item in dashboard._history] == [1, 2]
    assert dashboard._series("training/mean_grad_norm") == [0.1, 0.6]


def test_resume_history_derives_policy_excess_kl_for_old_metrics(tmp_path):
    old_metrics = _metrics()
    del old_metrics["training/policy_excess_kl"]
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(old_metrics) + "\n",
        encoding="utf-8",
    )

    dashboard = _dashboard(tmp_path)
    dashboard.begin_run(1, 2)

    assert dashboard._series("training/policy_excess_kl") == pytest.approx(
        [0.01]
    )


def test_eta_models_regular_and_scheduled_evaluation_generations_separately(
    tmp_path,
    monkeypatch,
):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "".join(
            (
                json.dumps(
                    _metrics(
                        iteration=1.0,
                        iteration_seconds=10.0,
                        include_evaluation=False,
                    )
                )
                + "\n",
                json.dumps(
                    _metrics(iteration=2.0, iteration_seconds=30.0)
                )
                + "\n",
            )
        ),
        encoding="utf-8",
    )
    dashboard = _dashboard(tmp_path)
    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 100.0)
    dashboard.begin_run(2, 6)

    estimate = dashboard._eta_estimate()
    assert estimate is not None
    assert estimate.remaining_seconds == pytest.approx(80.0)
    assert estimate.mean_iteration_seconds == pytest.approx(20.0)
    assert estimate.scheduled_evaluations == 2
    assert estimate.samples == 2
    assert estimate.confidence == "ACQUIRING"

    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 104.0)
    assert dashboard._eta_estimate().remaining_seconds == pytest.approx(76.0)


def test_eta_scales_changed_evaluation_workload_and_reacquires_suite(
    tmp_path,
    monkeypatch,
):
    records = []
    for iteration in range(1, 9):
        is_evaluation = iteration % 2 == 0
        metrics = _metrics(
            iteration=float(iteration),
            iteration_seconds=30.0 if is_evaluation else 10.0,
            include_evaluation=False,
        )
        if is_evaluation:
            metrics["evaluation/random/games"] = 4.0
        records.append(metrics)
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    dashboard = _dashboard(tmp_path)
    random_opponent = dashboard.config.evaluation.opponents[0]
    random_opponent.games = 8
    dashboard.config.evaluation.opponents = [random_opponent]
    monkeypatch.setattr("hexo_klent.tui.time.monotonic", lambda: 100.0)
    dashboard.begin_run(8, 10)

    estimate = dashboard._eta_estimate()
    assert estimate is not None
    # Historical eval overhead is 20s for four games. The configured suite
    # doubles that workload, so iteration 10 predicts 10s base + 40s eval.
    assert estimate.remaining_seconds == pytest.approx(60.0)
    assert estimate.confidence == "LOCKED"

    dashboard.config.evaluation.opponents.append(
        EvaluationOpponentConfig(name="new_anchor", games=1)
    )
    assert dashboard._eta_estimate().confidence == "ACQUIRING"


def test_resumed_run_progress_is_relative_to_this_invocation(tmp_path):
    dashboard = _dashboard(tmp_path)
    dashboard.begin_run(1025, 1250)

    assert dashboard._run_progress() == (0, 225)

    dashboard.update_metrics(_metrics(iteration=1026.0))
    assert dashboard._run_progress() == (1, 225)


def test_dashboard_renders_cockpit_and_loads_metric_history(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(_metrics()) + "\n",
        encoding="utf-8",
    )
    dashboard = _dashboard(tmp_path)
    dashboard.begin_run(1, 4)
    dashboard.set_phase("EVAL", 2, "sealbot_mcts24/8 // 16 games")

    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="truecolor",
        width=140,
        height=42,
    )
    console.print(dashboard.render(width=140, height=42))
    output = stream.getvalue()

    assert "HΞXO" in output
    assert "ANALYTIC SELF-PLAY CORE" in output
    assert "RUN VECTOR" in output
    assert "CHRONO-LINK" in output
    assert "ETA" in output
    assert "NEURAL LINK" in output
    assert "SELF-PLAY FABRIC" in output
    assert "POLICY / Q DYNAMICS" in output
    assert "TEMPORAL TRACE" in output
    assert "TARGET / PRIOR Hₙ" in output
    assert "T / P TOP-1" in output
    assert "LEGAL MOVES" in output
    assert "Q SPAN" in output
    assert "EXCESS KL" in output
    assert "FROZEN KL B / A" in output
    assert "TARGET RETENTION" in output
    assert "ARGMAX MATCH B / A" in output
    assert "MATCH DELTA" in output
    assert "P / Q TRUNK L2" in output
    assert "TRUNK COS" in output
    assert "OPPONENT ARRAY" in output
    assert "random" in output
    assert "sealbot_mcts24/8" in output
    assert "s1_anchor" in output
    assert "24/8 ↔ 48/12" in output

    compact_stream = io.StringIO()
    compact_console = Console(
        file=compact_stream,
        force_terminal=True,
        color_system="truecolor",
        width=80,
        height=24,
    )
    compact_console.print(dashboard.render(width=80, height=24))
    compact_output = compact_stream.getvalue()

    assert "LIVE TELEMETRY" in compact_output
    assert "TEMPORAL TRACE" in compact_output
    assert "EXCESS KL" in compact_output
    assert "TARGET RETENTION" in compact_output
    assert "T / P TOP-1" in compact_output
    assert "Q SPAN" in compact_output
    assert "SAFE HALT" in compact_output
