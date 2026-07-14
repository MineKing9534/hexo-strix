#!/usr/bin/env python3
"""Solve every standalone position fixture under scripts/fixtures/forcing_puzzles/
with the Rust deep prover (`prove --driver race`) and emit one verified-corpus
JSONL record per puzzle.

Each fixture is raced across idtt/dfpn/pdspn (first definitive verdict wins,
losers cancelled); the record captures which driver actually closed it, so a
puzzle that times out under one algorithm but falls to another is visible, not
silently hidden behind a single verdict.

Most `*_line.json` fixtures are real full-game move logs, not disposable
companions — specific prefixes are pinned, documented positions with proven
forced wins (see SPECIAL_POSITIONS below), and are solved standalone alongside
the regular puzzle fixtures. Only `d9ci11d_line.json` and `7zoidcw_line.json`
are true companions with no independent position of their own: composer-proposed
*continuations* of `0l4291i_live`, verified via the `hybrid` driver rather than
solved from scratch — those two are skipped here.

Usage:
    uv run --no-sync python scripts/build_forcing_corpus.py
    uv run --no-sync python scripts/build_forcing_corpus.py --only xsnfyll 0hz3hty
    uv run --no-sync python scripts/build_forcing_corpus.py --time-limit 3600 --node-budget 80m

Requires the `prove` binary built into a scoped target dir (never the default
`target/`, which a live self-play run may hold open):
    cd hexo-rs && CARGO_TARGET_DIR=target-forcing cargo build --release -p hexo-mcts --bin prove
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Make sibling `forcing_search_prototype` importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "scripts" / "fixtures" / "forcing_puzzles"
ANALYSIS_BASE_URL = "https://hexo.tyto.cc/analysis#c="

# WIN/NO are sound, definitive verdicts; UNVERIFIED/BUDGET_EXCEEDED are not -
# never relabel them as NO-WIN, that would corrupt the "verified" claim.
OUTCOME_MAP = {"WIN": "WIN", "NO": "NO-WIN"}

# The live-serve config these fixtures were recorded under (hexo-solver/src/
# forcing.rs's `CFG` in the strongloss/hayes test modules) - max_moves differs
# from the prover's own PosConfig::default() (300), so it's pinned explicitly
# rather than left to the default.
LIVE_CONFIG = {"win_length": 6, "placement_radius": 8, "max_moves": 400}

# Prefix cuts of full-game *_line.json logs that are themselves documented,
# pinned positions with a proven forced win - verified against the exact
# hardcoded stone arrays and current_player()/moves_remaining_this_turn() the
# cited tests assert, not independently re-derived. `attacker`/`placements_remaining`
# match what the cited solve actually proves, which is NOT always "the mover at
# that prefix": the strongloss positions are threat checks (`solve_threat` in
# forcing.rs flips to `GameState::from_state(stones, mover.opponent(), 2, cfg)`),
# so `attacker` there is the prefix mover's OPPONENT with a fresh 2 placements.
SPECIAL_POSITIONS = [
    {
        "id": "strongloss_a_prefix6",
        "name": "strongloss_a_line.json prefix 6 - P2's missed VCF "
                "(defense_finds_killer_pairs_strongloss_a, hexo-solver/src/forcing.rs)",
        "source": "strongloss_a_line.json",
        "prefix": 6,
        "attacker": "P2",
        "placements_remaining": 2,
    },
    {
        "id": "strongloss_b_prefix8",
        "name": "strongloss_b_line.json prefix 8 - P1's missed VCF "
                "(defense_finds_killer_pairs_strongloss_b, hexo-solver/src/forcing.rs)",
        "source": "strongloss_b_line.json",
        "prefix": 8,
        "attacker": "P1",
        "placements_remaining": 2,
    },
    {
        "id": "hayes_20260712_turn16",
        "name": "hayes_20260712_line.json placement 30 - proven win missed at the old "
                "20k live budget (test_live_budget_finds_hayes_turn16_win, "
                "hexo-a0/tests/test_serving_game.py)",
        "source": "hayes_20260712_line.json",
        "prefix": 30,
        "attacker": "P1",
        "placements_remaining": 2,
    },
    {
        "id": "hayes_20260712_placement31",
        "name": "hayes_20260712_line.json placement 31 - the same missed win one "
                "placement later (test_live_budget_finds_hayes_placement31_win, "
                "hexo-a0/tests/test_serving_game.py)",
        "source": "hayes_20260712_line.json",
        "prefix": 31,
        "attacker": "P1",
        "placements_remaining": 1,
    },
]

# True companions: composer-proposed continuations of 0l4291i_live, verified
# via the hybrid driver rather than solved as independent positions.
COMPANION_LINES = {"d9ci11d_line.json", "7zoidcw_line.json"}

# Fixtures solved TWICE (tight, then wide) because the interesting fact about
# them is the contrast: the tight (hot-cell-only) generator is exhaustive but
# incomplete (a documented blind spot for quiet-building moves - see
# docs/research/2026-07-10-wide-solver-analysis-default.md), so it can return a
# sound-looking NO that the wide generator (hot + quiet-build partners)
# overturns. The record's top-level verdict/driver/width/depth/line always
# describe the WIDE solve (the more complete answer); `tight_result` preserves
# the narrower solve for comparison.
WIDTH_CONTRAST_IDS = {"1o3nm0m"}


def load_special_position(spec: dict, fixtures_dir: Path):
    """A prefix cut of a full-game line fixture -> (position_dict, move_log),
    same shape as load_position."""
    moves = json.loads((fixtures_dir / spec["source"]).read_text())["moves"]
    stones = moves[: spec["prefix"] + 1]  # +1: index 0 is the seed
    position = {
        "stones": stones,
        "attacker": spec["attacker"],
        "placements_remaining": spec["placements_remaining"],
        "config": LIVE_CONFIG,
    }
    move_log = [(q, r, p) for (q, r, p) in stones]
    return position, move_log


def default_prove_bin() -> Path:
    return Path(os.environ.get(
        "PROVE_BIN", ROOT / "hexo-rs" / "target-forcing" / "release" / "prove"))


def load_position(path: Path):
    """Return (position_dict, move_log_or_None) for a fixture file, or
    (None, None) if the fixture isn't a standalone position (a true *_line.json
    companion - see COMPANION_LINES).

    position_dict is the prover's own JSON shape: stones/attacker/placements_remaining.
    move_log is the ordered [(q, r, player)] list (needed to render the analysis
    URL). By convention every native-format fixture in this repo (written by
    `prove.py fetch` or `normalize_forcing_fixtures.py`) lists `stones` in
    chronological play order, so it doubles as the move log.
    """
    if path.name in COMPANION_LINES:
        return None, None
    data = json.loads(path.read_text())
    if "gamePosition" in data:
        from forcing_search_prototype import load_puzzle
        _board, attacker, placements, move_log = load_puzzle(str(path))
        position = {
            "stones": [[q, r, p] for (q, r, p) in move_log],
            "attacker": attacker,
            "placements_remaining": placements,
        }
        return position, move_log
    if "stones" in data:
        position = {
            "stones": data["stones"],
            "attacker": data["attacker"],
            "placements_remaining": data["placements_remaining"],
        }
        move_log = [(q, r, p) for (q, r, p) in data["stones"]]
        return position, move_log
    return None, None


def render_analysis_url(move_log, pv):
    if not move_log:
        return None
    from forcing_search_prototype import encode_moves_compact
    prefix = [(q, r) for (q, r, _p) in move_log]
    pv_coords = [(c[0], c[1]) for c in pv]
    return ANALYSIS_BASE_URL + encode_moves_compact(prefix + pv_coords)


def solve_one(prove_bin: Path, position: dict, cfg_args: list[str], workdir: Path) -> dict:
    pos_path = workdir / "position.json"
    report_path = workdir / "report.json"
    pos_path.write_text(json.dumps(position))
    cmd = [str(prove_bin), "--position", str(pos_path), "--driver", "race",
           "--out", str(report_path), *cfg_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"prove failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return json.loads(report_path.read_text())


def build_record(fixture_id: str, name: str, position: dict, move_log, report: dict) -> dict:
    verdict = report["verdict"]
    pv = report.get("pv", [])
    return {
        "id": fixture_id,
        "name": name,
        "position": position,
        "outcome": OUTCOME_MAP.get(verdict, "UNRESOLVED"),
        "verdict": verdict,
        "driver": report["driver"],
        "width": report["width"],
        "depth": report.get("depth"),
        "line": pv,
        "analysis_url": render_analysis_url(move_log, pv),
        "stats": report["stats"],
        "race": report.get("race", []),
        "unverified": report.get("unverified", []),
    }


def build_contrast_record(fixture_id: str, name: str, position: dict, move_log,
                           tight_report: dict, wide_report: dict) -> dict:
    rec = build_record(fixture_id, name, position, move_log, wide_report)
    rec["tight_result"] = {
        "outcome": OUTCOME_MAP.get(tight_report["verdict"], "UNRESOLVED"),
        "verdict": tight_report["verdict"],
        "driver": tight_report["driver"],
        "stats": tight_report["stats"],
        "race": tight_report.get("race", []),
    }
    rec["note"] = ("tight generator: " + rec["tight_result"]["outcome"] + "; wide generator: "
                   + rec["outcome"] + " - demonstrates the tight-generator quiet-builder blind spot")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "forcing_solver_corpus.jsonl"))
    ap.add_argument("--prove-bin", default=None)
    ap.add_argument("--time-limit", type=float, default=1800.0,
                     help="wall-clock seconds per puzzle (prove's --time-limit); 0 = none")
    ap.add_argument("--node-budget", default=None,
                     help="override prove's --node-budget (accepts k/m suffixes)")
    ap.add_argument("--only", nargs="*",
                     help="only solve these ids (fixture basenames without .json, "
                          "or a SPECIAL_POSITIONS id like strongloss_a_prefix6)")
    ap.add_argument("--append", action="store_true",
                     help="append to an existing --out JSONL (replacing any record with "
                          "the same id) instead of rebuilding the whole corpus")
    args = ap.parse_args()

    prove_bin = Path(args.prove_bin) if args.prove_bin else default_prove_bin()
    if not prove_bin.exists():
        raise SystemExit(
            f"prove binary not found at {prove_bin}\n"
            "build it first (scoped target dir - never the default target/, a live "
            "self-play run may hold it open):\n"
            "  cd hexo-rs && CARGO_TARGET_DIR=target-forcing "
            "cargo build --release -p hexo-mcts --bin prove")

    cfg_args = ["--time-limit", str(args.time_limit)]
    if args.node_budget:
        cfg_args += ["--node-budget", str(args.node_budget)]

    fixtures_dir = Path(args.fixtures_dir)
    fixtures = sorted(fixtures_dir.glob("*.json"))
    special_positions = SPECIAL_POSITIONS
    if args.only:
        wanted = set(args.only)
        fixtures = [f for f in fixtures if f.stem in wanted]
        special_positions = [s for s in special_positions if s["id"] in wanted]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    special_sources = {spec["source"] for spec in SPECIAL_POSITIONS}
    skipped_companions = []
    skipped_as_whole_file = []  # raw sources only solved via their SPECIAL_POSITIONS prefixes
    records = []
    with tempfile.TemporaryDirectory(prefix="hexo-corpus-") as td:
        workdir = Path(td)
        for f in fixtures:
            data = json.loads(f.read_text())
            position, move_log = load_position(f)
            if position is None:
                (skipped_as_whole_file if f.name in special_sources else skipped_companions).append(f.name)
                continue
            name = data.get("name") or data.get("id") or f.stem
            if f.stem in WIDTH_CONTRAST_IDS:
                print(f"solving {f.stem} (tight) ...", file=sys.stderr)
                tight_report = solve_one(prove_bin, position, cfg_args, workdir)
                print(f"solving {f.stem} (wide) ...", file=sys.stderr)
                wide_report = solve_one(prove_bin, position, cfg_args + ["--width", "wide"], workdir)
                rec = build_contrast_record(f.stem, name, position, move_log, tight_report, wide_report)
                records.append(rec)
                print(f"  -> tight={rec['tight_result']['outcome']} wide={rec['outcome']} "
                      f"via {rec['driver']} (depth={rec['depth']})", file=sys.stderr)
                continue
            print(f"solving {f.stem} ...", file=sys.stderr)
            report = solve_one(prove_bin, position, cfg_args, workdir)
            rec = build_record(f.stem, name, position, move_log, report)
            records.append(rec)
            print(f"  -> {rec['outcome']} via {rec['driver']} "
                  f"(depth={rec['depth']}, {rec['stats']['elapsed_s']:.2f}s)", file=sys.stderr)

        for spec in special_positions:
            position, move_log = load_special_position(spec, fixtures_dir)
            print(f"solving {spec['id']} ...", file=sys.stderr)
            report = solve_one(prove_bin, position, cfg_args, workdir)
            rec = build_record(spec["id"], spec["name"], position, move_log, report)
            records.append(rec)
            print(f"  -> {rec['outcome']} via {rec['driver']} "
                  f"(depth={rec['depth']}, {rec['stats']['elapsed_s']:.2f}s)", file=sys.stderr)

    if args.append and out_path.exists():
        existing = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        new_ids = {rec["id"] for rec in records}
        existing = [rec for rec in existing if rec["id"] not in new_ids]
        records = existing + records

    with out_path.open("w") as out:
        for rec in records:
            out.write(json.dumps(rec) + "\n")

    print(f"wrote {len(records)} records to {out_path}", file=sys.stderr)
    if skipped_companions:
        print(f"skipped {len(skipped_companions)} true composer-line companions "
              f"(continuations of another position, not independent): {', '.join(skipped_companions)}",
              file=sys.stderr)
    if skipped_as_whole_file:
        print(f"skipped {len(skipped_as_whole_file)} raw multi-position source files as a whole "
              f"(solved via their SPECIAL_POSITIONS prefixes instead): {', '.join(skipped_as_whole_file)}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
