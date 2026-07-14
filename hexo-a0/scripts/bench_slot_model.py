"""A2 acceptance benchmark: slot-based static-shape model vs the legacy PyG path.

Forward-pass throughput at several batch sizes on identical positions, with
graphs prebuilt for BOTH paths (this isolates the NN forward, the A2 target;
graph building was A1). The slot model is additionally measured under
``torch.compile(mode="reduce-overhead")`` when ``--compile`` is passed.

Usage:
    uv run --no-sync python hexo-a0/scripts/bench_slot_model.py \
        --device cpu --batch-sizes 1,16,64,256
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import torch

from torch_geometric.data import Batch

from hexo_a0 import slot_graph as sg
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_axis_graph
from hexo_a0.model import HeXONet
from hexo_a0.model_slots import collate_slot_graphs, slot_model_from_legacy

WIN_LENGTH = 6


def build_positions(n: int, n_moves: int, radius: int):
    import hexo_rs

    games = []
    for seed in range(n):
        cfg = hexo_rs.GameConfig(win_length=WIN_LENGTH, placement_radius=radius, max_moves=600)
        g = hexo_rs.GameState(cfg)
        rng = random.Random(seed)
        for _ in range(n_moves):
            if g.is_terminal():
                break
            moves = g.legal_moves()
            q, r = moves[rng.randrange(len(moves))]
            g.apply_move(q, r)
        if not g.is_terminal():
            games.append(g)
    return games


def timeit(fn, device, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-sizes", default="1,16,64,256")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--moves", type=int, default=120, help="placements per position")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--pad-to", type=int, default=0, help="0 = batch max")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    device = torch.device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    # Production-shaped config (lean-d6-qhead run: gine, jk cat, rel+threat).
    cfg = ModelConfig(
        hidden_dim=args.hidden, num_layers=args.layers, num_heads=8,
        conv_type="gine", policy_hidden=128, value_hidden=32,
        graph_type="axis", prune_empty_edges=True,
        threat_features=True, relative_stone_encoding=True,
        use_jk=True, jk_mode="cat",
    )
    flags = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)

    torch.manual_seed(0)
    legacy = HeXONet(cfg).to(device).eval()
    slot = slot_model_from_legacy(legacy, cfg, WIN_LENGTH).to(device).eval()

    games = build_positions(max(batch_sizes), args.moves, args.radius)
    if not games:
        raise SystemExit("all sampled games were terminal — lower --moves")
    print(f"{len(games)} positions, ~{args.moves} placements, radius {args.radius}, device {device}")

    datas = [game_to_axis_graph(g, **flags) for g in games]
    slots = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        for g in games
    ]
    n_nodes = [d.num_nodes for d in datas]
    print(f"nodes/graph: min {min(n_nodes)} max {max(n_nodes)}")

    slot_fwd = slot.forward_padded
    warmup = 5
    if args.compile:
        slot_fwd = torch.compile(slot.forward_padded, mode="reduce-overhead")
        warmup = 20  # CUDA-graph capture needs several steady-state iterations

    header = f"{'batch':>6} {'legacy ms':>10} {'slot ms':>10} {'speedup':>8} {'legacy g/s':>11} {'slot g/s':>10}"
    print(header)
    for b in batch_sizes:
        reps = -(-b // len(games))  # ceil — the batch must actually hold b graphs
        chosen = (datas * reps)[:b], (slots * reps)[:b]
        assert len(chosen[0]) == b
        pyg_batch = Batch.from_data_list(chosen[0]).to(device)
        pad_to = args.pad_to or None
        slot_batch = collate_slot_graphs(chosen[1], pad_to=pad_to).to(device)

        with torch.no_grad():
            # _forward_batch_core (not forward_batch) so the legacy side isn't
            # charged for the per-graph .split()/.tolist() host sync — this
            # compares the GPU-resident forward on both sides.
            t_legacy = timeit(lambda: legacy._forward_batch_core(pyg_batch), device,
                              warmup=warmup, iters=args.iters)
            t_slot = timeit(lambda: slot_fwd(slot_batch), device,
                            warmup=warmup, iters=args.iters)
        print(
            f"{b:>6} {t_legacy * 1e3:>10.2f} {t_slot * 1e3:>10.2f} "
            f"{t_legacy / t_slot:>8.2f} {b / t_legacy:>11.0f} {b / t_slot:>10.0f}"
        )


if __name__ == "__main__":
    main()
