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
