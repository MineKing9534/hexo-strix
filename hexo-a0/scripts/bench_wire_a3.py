"""A3 Task 3b GO/NO-GO bench: MSG_FORWARD_STATES reader-thread cost vs MSG_FORWARD.

Quantifies what states mode ADDS to the inference server's reader-thread
critical path (parse + graph rebuild) compared to today's graph-mode body
parse, at production-like shapes, CPU-only.

Measured per batch of {16, 64, 256} real positions (random legal games at
production config: win_length 6, placement_radius 8, prune+threat+relative,
mixed mid-game depths):

1. Graph-mode body parse: ``_read_forward_body`` on a prebuilt in-memory body
   (the today-baseline reader-thread cost).
2. States-mode parse+rebuild: ``_read_forward_states_body`` on a prebuilt
   states body (the new reader-thread cost). Components (struct parse,
   from_state loop, axis_states_to_batch, dict/hash assembly) are timed via a
   mirrored copy of the ``_states_to_forward_body`` internals — the
   production function itself is NOT edited; the mirrored sum is
   cross-checked against the real function's total.
3. Wire body sizes for both modes.
4. Client-side savings estimate: ``hexo_rs.game_to_axis_graph_raw`` per
   position (what the Rust client skips building per leaf in states mode;
   timed through PyO3, so includes binding overhead the Rust client would
   not pay).

Usage:
    uv run --no-sync python hexo-a0/scripts/bench_wire_a3.py \
        [--batch-sizes 16,64,256] [--reps 30] [--warmup 3]
"""

from __future__ import annotations

import argparse
import io
import random
import statistics
import struct
import time

import numpy as np
import torch  # noqa: F401  (inference_server imports it; fail early if missing)

import hexo_rs
from hexo_a0.graph import axis_states_to_batch
from hexo_a0.inference_server import (
    _fnv1a64,
    _read_exact,
    _read_forward_body,
    _read_forward_states_body,
)

# --- Production config -------------------------------------------------------
WIN_LENGTH = 6
PLACEMENT_RADIUS = 8
MAX_MOVES = 300
BUILDER_KWARGS = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)
BUILDER_FLAGS = 0x01 | 0x02 | 0x04  # prune | threat | relative
NODE_DIM = 11  # lean relative (7) + threat (4)

# Mixed mid-game snapshot depths (plies == placements; 2 per turn), covering
# both mr parities and a spread of board sizes.
DEPTHS = (8, 14, 21, 27, 34, 40, 47, 53)


# --- Position generation (pattern from tests/test_wire_states_parity.py) -----

def _cfg() -> "hexo_rs.GameConfig":
    return hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)


def _snapshot(game, cfg):
    """Immutable copy of a live (mutated-in-place) game via from_state."""
    return hexo_rs.GameState.from_state(
        game.placed_stones(),
        game.current_player(),
        game.moves_remaining_this_turn(),
        cfg,
    )


def _play_to_depth(seed: int, depth: int):
    """Random legal play to ``depth`` plies; return a non-terminal snapshot
    at that depth (or the last non-terminal state if the game ends early)."""
    cfg = _cfg()
    rng = random.Random(seed)
    game = hexo_rs.GameState(cfg)
    last = None
    for _ in range(depth):
        if game.is_terminal():
            break
        game.apply_move(*rng.choice(game.legal_moves()))
        if not game.is_terminal():
            last = _snapshot(game, cfg)
    assert last is not None, f"seed {seed}: no non-terminal state reached"
    return last


def build_positions(n: int) -> list:
    return [_play_to_depth(seed=1000 + i, depth=DEPTHS[i % len(DEPTHS)]) for i in range(n)]


# --- Wire-body builders (mirroring tests/test_wire_states_parity.py) ---------

def collate_graph_body(games) -> bytes:
    """MSG_FORWARD BODY collated the way the Rust client encoder lays it out."""
    raws = [hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS) for g in games]
    total_nodes = sum(r["num_nodes"] for r in raws)
    total_edges = sum(len(r["edge_src"]) for r in raws)

    buf = bytearray()
    buf.extend(struct.pack("<III", total_nodes, total_edges, len(raws)))
    buf.extend(struct.pack("<BB", 1, NODE_DIM))
    for r in raws:
        buf.extend(np.asarray(r["features"], dtype=np.float32).tobytes())
    offset = 0
    for r in raws:
        buf.extend((np.asarray(r["edge_src"], dtype=np.int64) + offset).tobytes())
        offset += r["num_nodes"]
    offset = 0
    for r in raws:
        buf.extend((np.asarray(r["edge_dst"], dtype=np.int64) + offset).tobytes())
        offset += r["num_nodes"]
    for r in raws:
        buf.extend(np.asarray(r["edge_attr"], dtype=np.float32).tobytes())
    for r in raws:
        buf.extend(np.asarray(r["legal_mask"], dtype=np.uint8).tobytes())
    for r in raws:
        buf.extend(np.asarray(r["stone_mask"], dtype=np.uint8).tobytes())
    for i, r in enumerate(raws):
        buf.extend(np.full(r["num_nodes"], i, dtype=np.int32).tobytes())
    return bytes(buf)


def states_body(games) -> bytes:
    """MSG_FORWARD_STATES BODY for the same positions (outer header omitted)."""
    buf = bytearray()
    buf.extend(struct.pack(
        "<IBBIBB", len(games), WIN_LENGTH, PLACEMENT_RADIUS,
        MAX_MOVES, BUILDER_FLAGS, NODE_DIM,
    ))
    for g in games:
        stones = g.placed_stones()
        cur = 0 if g.current_player() == "P1" else 1
        buf.extend(struct.pack(
            "<HBBH", len(stones), cur,
            g.moves_remaining_this_turn(), g.legal_move_count(),
        ))
        for (q, r), p in stones:
            buf.extend(struct.pack("<hhB", q, r, 0 if p == "P1" else 1))
    return bytes(buf)


# --- Mirrored states parse+rebuild with component timers ----------------------
# Copy of _read_forward_states_body / _states_to_forward_body internals
# (inference_server.py, Task 1) with perf_counter section timers inserted.
# Production code is untouched; the sum is cross-checked against the real
# function's total below.

def states_components(body: bytes) -> dict:
    stdin = io.BytesIO(body)
    t0 = time.perf_counter()

    # -- struct parse (mirror of _read_forward_states_body) --
    body_header = _read_exact(stdin, 12)
    (num_graphs, win_length, placement_radius, max_moves,
     _builder_flags, _node_dim) = struct.unpack("<IBBIBB", body_header)
    graphs = []
    for _ in range(num_graphs):
        gh = _read_exact(stdin, 6)
        num_stones, cur, mr, num_legal = struct.unpack("<HBBH", gh)
        stone_bytes = _read_exact(stdin, num_stones * 5)
        stones = list(struct.iter_unpack("<hhB", stone_bytes))
        graphs.append((stones, cur, mr, num_legal))
    t1 = time.perf_counter()

    # -- from_state loop (mirror of _states_to_forward_body validation) --
    config = hexo_rs.GameConfig(win_length, placement_radius, max_moves)
    games = []
    for stones, cur, mr, _num_legal in graphs:
        assert mr in (1, 2) and cur in (0, 1)
        stone_list = []
        for q, r, p in stones:  # faithful to the per-stone validation loop
            assert p in (0, 1)
            stone_list.append(((q, r), "P1" if p == 0 else "P2"))
        games.append(hexo_rs.GameState.from_state(
            stone_list, "P1" if cur == 0 else "P2", mr, config
        ))
    t2 = time.perf_counter()

    # -- batch rebuild --
    batch, aux = axis_states_to_batch(games, device="cpu", **BUILDER_KWARGS)
    t3 = time.perf_counter()

    # -- dict assembly + legal-count check + stone_mask scatter + hashes --
    legal_counts = aux.legal_counts.tolist()
    for i, (_stones, _cur, _mr, num_legal) in enumerate(graphs):
        assert legal_counts[i] == num_legal
    total_nodes = batch.x.shape[0]
    stone_mask = torch.zeros(total_nodes, dtype=torch.uint8)
    stone_mask[aux.stone_idx] = 1
    _body = {
        "total_nodes": total_nodes, "total_edges": batch.edge_index.shape[1],
        "num_graphs": batch.num_graphs, "has_edge_attr": 1,
        "node_dim": batch.x.shape[1],
        "features": bytearray(batch.x.numpy().tobytes()),
        "edge_src": bytearray(batch.edge_index[0].contiguous().numpy().tobytes()),
        "edge_dst": bytearray(batch.edge_index[1].contiguous().numpy().tobytes()),
        "edge_attr": bytearray(batch.edge_attr.numpy().tobytes()),
        "legal_mask": bytearray(batch.legal_mask.to(torch.uint8).numpy().tobytes()),
        "stone_mask": bytearray(stone_mask.numpy().tobytes()),
        "batch": bytearray(batch.batch.to(torch.int32).numpy().tobytes()),
    }
    t4 = time.perf_counter()
    legal_coords = aux.coords[aux.legal_idx].numpy().astype("<i4", copy=False)
    hashes = []
    offset = 0
    for count in legal_counts:
        hashes.append(_fnv1a64(legal_coords[offset:offset + count].tobytes()))
        offset += count
    t5 = time.perf_counter()

    return {
        "parse_ms": (t1 - t0) * 1e3,
        "from_state_ms": (t2 - t1) * 1e3,
        "batch_build_ms": (t3 - t2) * 1e3,
        "assemble_ms": (t4 - t3) * 1e3,
        "fnv_hash_ms": (t5 - t4) * 1e3,
        "mirror_total_ms": (t5 - t0) * 1e3,
    }


# --- Timing harness -----------------------------------------------------------

def med(fn, warmup: int, reps: int) -> tuple[float, float]:
    """(median, min) wall time in ms of fn over reps after warmup.

    min is reported alongside median because this bench may run on a
    machine with live self-play/training load — median is the contended
    (production-representative) figure, min approximates the uncontended
    floor.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), min(times)


def med_components(body: bytes, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        states_components(body)
    runs = [states_components(body) for _ in range(reps)]
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="16,64,256")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]

    print(f"# A3 wire-states bench (Task 3b) — CPU, reps={args.reps} (median), "
          f"warmup={args.warmup}")
    print(f"# config: win_length={WIN_LENGTH} placement_radius={PLACEMENT_RADIUS} "
          f"max_moves={MAX_MOVES} flags=prune+threat+relative node_dim={NODE_DIM}")
    print(f"# depths cycled: {DEPTHS}")

    t0 = time.perf_counter()
    all_games = build_positions(max(batch_sizes))
    print(f"# generated {len(all_games)} positions in {time.perf_counter() - t0:.1f}s; "
          f"stones per position: min={min(len(g.placed_stones()) for g in all_games)} "
          f"max={max(len(g.placed_stones()) for g in all_games)}")

    rows = []
    for bs in batch_sizes:
        games = all_games[:bs]
        graph_body = collate_graph_body(games)
        st_body = states_body(games)

        # batch shape facts
        hdr = struct.unpack_from("<III", graph_body, 0)
        total_nodes, total_edges, _ = hdr

        graph_ms, graph_min = med(
            lambda: _read_forward_body(io.BytesIO(graph_body), NODE_DIM),
            args.warmup, args.reps,
        )
        states_ms, states_min = med(
            lambda: _read_forward_states_body(
                io.BytesIO(st_body), NODE_DIM, **BUILDER_KWARGS),
            args.warmup, args.reps,
        )
        comp = med_components(st_body, args.warmup, args.reps)
        client_build_ms, client_build_min = med(
            lambda: [hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS) for g in games],
            args.warmup, args.reps,
        )

        rows.append(dict(
            bs=bs, nodes=total_nodes, edges=total_edges,
            graph_bytes=len(graph_body), states_bytes=len(st_body),
            graph_ms=graph_ms, graph_min=graph_min,
            states_ms=states_ms, states_min=states_min, comp=comp,
            client_build_ms=client_build_ms, client_build_min=client_build_min,
        ))

    print()
    print("## Reader-thread cost per batch (ms, median | min)")
    print(f"{'batch':>5} {'nodes':>8} {'edges':>9} | "
          f"{'graph-parse':>17} {'states-total':>17} {'added(med)':>10} {'added(min)':>10}")
    for r in rows:
        print(f"{r['bs']:>5} {r['nodes']:>8} {r['edges']:>9} | "
              f"{r['graph_ms']:>8.3f}|{r['graph_min']:>8.3f} "
              f"{r['states_ms']:>8.3f}|{r['states_min']:>8.3f} "
              f"{r['states_ms'] - r['graph_ms']:>10.3f} "
              f"{r['states_min'] - r['graph_min']:>10.3f}")

    print()
    print("## States-mode components per batch (ms, median; mirrored internals)")
    print(f"{'batch':>5} | {'parse':>7} {'from_state':>10} {'batch_build':>11} "
          f"{'assemble':>9} {'fnv_hash':>9} {'mirror-sum':>10}")
    for r in rows:
        c = r["comp"]
        print(f"{r['bs']:>5} | {c['parse_ms']:>7.3f} {c['from_state_ms']:>10.3f} "
              f"{c['batch_build_ms']:>11.3f} {c['assemble_ms']:>9.3f} "
              f"{c['fnv_hash_ms']:>9.3f} {c['mirror_total_ms']:>10.3f}")

    print()
    print("## Per-graph reader-thread cost (ms/graph, median | min)")
    print(f"{'batch':>5} | {'graph-parse':>17} {'states-total':>17} {'added(med)':>10} "
          f"| {'client game_to_axis_graph_raw (skipped per leaf in states mode)'}")
    for r in rows:
        print(f"{r['bs']:>5} | "
              f"{r['graph_ms'] / r['bs']:>8.4f}|{r['graph_min'] / r['bs']:>8.4f} "
              f"{r['states_ms'] / r['bs']:>8.4f}|{r['states_min'] / r['bs']:>8.4f} "
              f"{(r['states_ms'] - r['graph_ms']) / r['bs']:>10.4f} | "
              f"{r['client_build_ms'] / r['bs']:>7.4f}|{r['client_build_min'] / r['bs']:>7.4f} ms/graph "
              f"({r['client_build_ms']:>8.3f}|{r['client_build_min']:>8.3f} ms/batch)")

    print()
    print("## Wire body bytes")
    print(f"{'batch':>5} {'graph-mode':>12} {'states-mode':>12} {'ratio':>8} "
          f"{'graph B/graph':>13} {'states B/graph':>14}")
    for r in rows:
        print(f"{r['bs']:>5} {r['graph_bytes']:>12,} {r['states_bytes']:>12,} "
              f"{r['graph_bytes'] / r['states_bytes']:>7.1f}x "
              f"{r['graph_bytes'] // r['bs']:>13,} {r['states_bytes'] // r['bs']:>14,}")


if __name__ == "__main__":
    main()
