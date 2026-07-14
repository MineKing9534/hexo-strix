#!/usr/bin/env python3
"""Normalize raw HDS-sandbox fixtures under scripts/fixtures/forcing_puzzles/ to the
prover's native position format.

The directory mixes two position shapes: raw HDS sandbox exports
(`{"id", "name", "gamePosition": {"cells": [...], ...}}`, fetched straight off
hexo.did.science) and the prover's own native format
(`{"stones": [[q, r, "P1"|"P2"], ...], "attacker": ..., "placements_remaining": ...}`,
e.g. `0l4291i_live.json`). Rust's fixtures.rs loads files directly via
`Position::load`/`Line::load` with no HDS parser (by design - no network/HDS-codec
code in Rust), so anything Rust needs must already be native; anything Python-only
can be either, and this script converts the raw ones to match. `*_line.json`
companions (`{"moves": [...]}`) are a different, already-consistent fixture kind
and are left untouched.

Idempotent: files already in native format (no `gamePosition` key) are skipped.

Usage:
    uv run --no-sync python scripts/normalize_forcing_fixtures.py
    uv run --no-sync python scripts/normalize_forcing_fixtures.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "scripts" / "fixtures" / "forcing_puzzles"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    from forcing_search_prototype import load_puzzle

    converted, skipped = [], []
    for path in sorted(Path(args.fixtures_dir).glob("*.json")):
        data = json.loads(path.read_text())
        if "gamePosition" not in data:
            skipped.append(path.name)
            continue
        _board, attacker, placements, move_log = load_puzzle(str(path))
        native = {
            "id": data.get("id", path.stem),
            "name": data.get("name"),
            "stones": [[q, r, p] for (q, r, p) in move_log],
            "attacker": attacker,
            "placements_remaining": placements,
        }
        converted.append(path.name)
        if not args.dry_run:
            path.write_text(json.dumps(native, indent=None) + "\n")

    verb = "would convert" if args.dry_run else "converted"
    print(f"{verb} {len(converted)}: {', '.join(converted)}", file=sys.stderr)
    print(f"already native, skipped {len(skipped)}: {', '.join(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
