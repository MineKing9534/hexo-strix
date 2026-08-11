"""Command-line entry point for the standalone KLENT experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _release_accelerator_cache() -> None:
    """Return cached accelerator memory after model references are gone."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        # Completion is already durable; cache release is best-effort.
        pass


def _resolve_resume(
    resume: str | None,
    output_dir: str | Path,
) -> str | Path | None:
    """Resolve the explicit ``latest`` alias within one run directory."""

    if resume != "latest":
        return resume

    checkpoint_dir = Path(output_dir) / "checkpoints"
    candidates = list(checkpoint_dir.glob("checkpoint_*.pt"))
    final_checkpoint = checkpoint_dir / "final.pt"
    if final_checkpoint.is_file():
        candidates.append(final_checkpoint)

    existing: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            if candidate.is_file():
                existing.append((candidate.stat().st_mtime_ns, candidate))
        except OSError:
            continue
    if not existing:
        raise SystemExit(
            f"--resume latest found no checkpoints in {checkpoint_dir}"
        )

    _modified_ns, latest = max(
        existing,
        key=lambda item: (item[0], item[1].name),
    )
    logger.info("resolved --resume latest to %s", latest)
    return latest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexo-klent",
        description="Search-free KLENT self-play training for HeXO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run synchronous KLENT training")
    train.add_argument("--config", required=True, help="TOML configuration")
    checkpoint_source = train.add_mutually_exclusive_group()
    checkpoint_source.add_argument(
        "--resume",
        help="checkpoint path to resume, or 'latest' in run.output_dir",
    )
    train.add_argument(
        "--resume-configured-lr",
        action="store_true",
        help=(
            "when resuming, preserve optimizer state but replace its restored "
            "learning rate with training.learning_rate from the config"
        ),
    )
    checkpoint_source.add_argument(
        "--init-from",
        help=(
            "compatible production Axis-GINE Q-head or dense KLENT "
            "checkpoint used to initialize a new KLENT run"
        ),
    )
    train.add_argument("--device", help="override run.device")
    train.add_argument("--output-dir", help="override run.output_dir")
    train.add_argument(
        "--workers",
        type=int,
        help="override collection.workers (spawned self-play processes)",
    )
    train.add_argument(
        "--iterations",
        type=int,
        help="iterations to run in this invocation (default: config total)",
    )
    train.add_argument(
        "--no-tensorboard", action="store_true", help="disable TensorBoard"
    )
    train.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable the KLENT cockpit (default: enabled on an interactive TTY)",
    )
    train.add_argument("-v", "--verbose", action="store_true")

    distill = subparsers.add_parser(
        "distill",
        help="distill a graph KLENT checkpoint into a native hex CNN",
    )
    distill.add_argument("--config", required=True, help="target TOML configuration")
    distill.add_argument("--teacher", required=True, help="graph KLENT checkpoint")
    distill.add_argument(
        "--init-student",
        help=(
            "initialize the target CNN weights from an earlier distilled "
            "checkpoint before fitting fresh teacher positions"
        ),
    )
    distill.add_argument(
        "--positions",
        type=int,
        help="teacher positions to collect (default: config collection size)",
    )
    distill.add_argument(
        "--parallel-games",
        type=int,
        help=(
            "distillation-only live lane count; does not alter the saved "
            "KLENT training config"
        ),
    )
    distill.add_argument(
        "--teacher-horizon",
        type=int,
        help=(
            "distillation-only teacher prefix length in placements; "
            "horizon-capped prefixes are retained because distillation uses "
            "teacher policy/Q labels rather than outcome returns"
        ),
    )
    distill.add_argument("--epochs", type=int, default=4)
    distill.add_argument(
        "--augment-symmetries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train on a balanced schedule of all 12 D6 rotations/reflections; "
            "validation remains in the original orientation"
        ),
    )
    distill.add_argument(
        "--batch-size",
        type=int,
        help="logical distillation batch (default: config training batch)",
    )
    distill.add_argument("--validation-positions", type=int, default=2_048)
    distill.add_argument("--device", help="override run.device")
    distill.add_argument(
        "--precision",
        choices=("float32", "bf16"),
        help="override run.precision",
    )
    distill.add_argument(
        "--workers",
        type=int,
        help="override collection.workers",
    )
    distill.add_argument(
        "--output",
        help=(
            "checkpoint path (default: run.output_dir/checkpoints/"
            "checkpoint_000000.pt)"
        ),
    )
    distill.add_argument("--temperature", type=float, default=1.0)
    distill.add_argument("--policy-weight", type=float, default=1.0)
    distill.add_argument("--q-weight", type=float, default=1.0)
    distill.add_argument("--learning-rate", type=float, default=1e-3)
    distill.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "stop after this many epochs without a meaningful held-out "
            "objective improvement (0 disables plateau stopping)"
        ),
    )
    distill.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="absolute held-out objective improvement required to reset patience",
    )
    distill.add_argument(
        "--restore-best-fit",
        action="store_true",
        help=(
            "save the epoch with the lowest held-out imitation objective "
            "instead of the final fitted epoch"
        ),
    )
    distill.add_argument(
        "--strength-eval-interval",
        type=int,
        default=0,
        help=(
            "run paired-opening matches against the graph teacher and an "
            "equally lagged student every N epochs (0 disables)"
        ),
    )
    distill.add_argument(
        "--strength-eval-games",
        type=int,
        default=32,
        help="games per periodic distillation strength opponent",
    )
    distill.add_argument(
        "--strength-eval-mcts-simulations",
        type=int,
        default=24,
    )
    distill.add_argument(
        "--strength-eval-mcts-actions",
        type=int,
        default=8,
    )
    distill.add_argument(
        "--target-policy-kl",
        type=float,
        help="stop once held-out policy KL is at or below this value",
    )
    distill.add_argument(
        "--target-q-mse",
        type=float,
        help="stop once held-out Q MSE is at or below this value",
    )
    distill.add_argument(
        "--target-top1",
        type=float,
        help=(
            "stop once held-out policy top-1 agreement reaches this fraction; "
            "all configured targets must be met"
        ),
    )
    distill.add_argument(
        "--student-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "compile the student FIT core (default: run.compile); "
            "--no-student-compile does not alter the saved training config"
        ),
    )
    distill.add_argument("--seed", type=int)
    distill.add_argument("-v", "--verbose", action="store_true")

    sprt = subparsers.add_parser(
        "sprt",
        help="run a fixed-checkpoint pentanomial SPRT match",
    )
    sprt.add_argument("--candidate", required=True, help="checkpoint under test")
    sprt.add_argument("--opponent", required=True, help="reference checkpoint")
    sprt.add_argument("--win-length", type=int, default=6)
    sprt.add_argument("--radius", type=int, default=2)
    sprt.add_argument("--max-moves", type=int, default=1000)
    sprt.add_argument("--mcts-simulations", type=int, default=24)
    sprt.add_argument("--mcts-actions", type=int, default=8)
    sprt.add_argument("--device", default="cuda")
    sprt.add_argument(
        "--precision",
        choices=("float32", "bf16"),
        default="bf16",
    )
    sprt.add_argument("--s0", type=float, default=0.50)
    sprt.add_argument("--s1", type=float, default=0.55)
    sprt.add_argument("--alpha", type=float, default=0.05)
    sprt.add_argument("--beta", type=float, default=0.05)
    sprt.add_argument("--pair-variance", type=float, default=0.50)
    sprt.add_argument("--max-games", type=int, default=1000)
    sprt.add_argument("--seed", type=int, default=0)
    sprt.add_argument("--state-file")
    sprt.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile both KLENT model cores (default: enabled)",
    )
    sprt.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "train":
        # Configure the allocator before importing torch through Trainer.
        from hexo_a0.gpu_memory import configure_cuda_alloc

        configure_cuda_alloc()
        from hexo_klent.config import load_config
        from hexo_klent.trainer import Trainer

        config = load_config(args.config)
        if args.device is not None:
            config.run.device = args.device
        if args.output_dir is not None:
            config.run.output_dir = args.output_dir
        if args.workers is not None:
            if args.workers <= 0:
                raise SystemExit("--workers must be positive")
            config.collection.workers = args.workers
        if args.iterations is not None and args.iterations <= 0:
            raise SystemExit("--iterations must be positive")
        resume = _resolve_resume(args.resume, config.run.output_dir)
        if args.resume_configured_lr and resume is None:
            raise SystemExit("--resume-configured-lr requires --resume")

        use_tui = sys.stdout.isatty() if args.tui is None else args.tui
        dashboard = None
        if use_tui:
            from hexo_klent.tui import TrainingDashboard

            dashboard = TrainingDashboard(config)
            dashboard.open()
            logging.basicConfig(
                level=logging.DEBUG if args.verbose else logging.INFO,
                handlers=[dashboard.logging_handler],
                force=True,
            )

        trainer = None
        try:
            trainer = Trainer(
                config,
                tensorboard=not args.no_tensorboard,
                resume=resume,
                resume_configured_lr=args.resume_configured_lr,
                init_from=args.init_from,
                display=dashboard,
            )
            trainer.run(args.iterations)
            if dashboard is not None:
                trainer = None
                _release_accelerator_cache()
                dashboard.wait_for_exit()
        except KeyboardInterrupt:
            iteration = 0 if trainer is None else trainer.iteration
            if dashboard is not None:
                dashboard.interrupt(iteration)
            logger.info(
                "training interrupted after completed iteration %d; "
                "workers stopped",
                iteration,
            )
            raise SystemExit(130) from None
        finally:
            if dashboard is not None:
                dashboard.close()
    elif args.command == "distill":
        from hexo_a0.gpu_memory import configure_cuda_alloc

        configure_cuda_alloc()
        from hexo_klent.config import load_config
        from hexo_klent.distill import distill_checkpoint

        config = load_config(args.config)
        if args.workers is not None:
            if args.workers <= 0:
                raise SystemExit("--workers must be positive")
            config.collection.workers = args.workers
        positions = (
            config.collection.positions_per_iteration
            if args.positions is None
            else args.positions
        )
        batch_size = (
            config.training.batch_size
            if args.batch_size is None
            else args.batch_size
        )
        try:
            distill_checkpoint(
                config,
                args.teacher,
                positions=positions,
                epochs=args.epochs,
                batch_size=batch_size,
                validation_positions=args.validation_positions,
                device_str=args.device,
                precision=args.precision,
                output=args.output,
                temperature=args.temperature,
                policy_weight=args.policy_weight,
                q_weight=args.q_weight,
                learning_rate=args.learning_rate,
                parallel_games=args.parallel_games,
                teacher_horizon=args.teacher_horizon,
                augment_symmetries=args.augment_symmetries,
                student_compile=args.student_compile,
                student_checkpoint=args.init_student,
                seed=args.seed,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                restore_best_fit=args.restore_best_fit,
                strength_eval_interval=args.strength_eval_interval,
                strength_eval_games=args.strength_eval_games,
                strength_eval_mcts_simulations=(
                    args.strength_eval_mcts_simulations
                ),
                strength_eval_mcts_actions=args.strength_eval_mcts_actions,
                target_policy_kl=args.target_policy_kl,
                target_q_mse=args.target_q_mse,
                target_top1=args.target_top1,
            )
        except KeyboardInterrupt:
            logger.info("distillation interrupted before checkpoint commit")
            raise SystemExit(130) from None
    elif args.command == "sprt":
        from hexo_a0.gpu_memory import configure_cuda_alloc

        configure_cuda_alloc()
        from hexo_klent.sprt import run_checkpoint_sprt

        try:
            run_checkpoint_sprt(
                args.candidate,
                args.opponent,
                win_length=args.win_length,
                radius=args.radius,
                max_moves=args.max_moves,
                mcts_simulations=args.mcts_simulations,
                mcts_actions=args.mcts_actions,
                device_str=args.device,
                precision=args.precision,
                s0=args.s0,
                s1=args.s1,
                alpha=args.alpha,
                beta=args.beta,
                pair_variance=args.pair_variance,
                max_games=args.max_games,
                seed=args.seed,
                state_file=args.state_file,
                compile_model=args.compile,
            )
        except KeyboardInterrupt:
            logger.info("SPRT interrupted; complete pairs are in the state file")
            raise SystemExit(130) from None
