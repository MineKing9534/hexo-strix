"""Slot serving-path decomposition bench: WHERE do the ~27 ms/batch go?

Context: ``bench_wire_a3_gpu.py`` measured e2e server round-trips and found the
slot MSG_FORWARD_STATES path 3.5-5.1x SLOWER than the legacy MSG_FORWARD path
(~27 ms vs ~5.2 ms at b=8), yet A2's ``bench_slot_model.py`` showed the compiled
slot forward 1.45-1.81x FASTER than the legacy forward. This script isolates the
slot path's cost centres, in-process on CUDA bf16, at the SAME production shapes
as ``bench_wire_a3_gpu.py``, so the e2e gap can be attributed.

Components measured (all median, >=20 warmup, >=50 reps, proper CUDA sync):
  1. FORWARD-ONLY (compiled fullgraph, uncontended) on PREBUILT GPU inputs:
       - legacy ScriptableHeXONet (server's _forward_batch_core-equivalent),
         compiled fullgraph — the model the wire bench's legacy server runs.
       - legacy HeXONet._forward_batch_core EAGER (bench_slot_model's exact
         legacy side) — to reproduce A2's original claim apples-to-apples.
       - slot SlotHeXONet.forward_padded, compiled fullgraph, on a prebuilt
         padded SlotBatch at the same pow2 (B, N) bucket the server uses.
  2. BUILD-ONLY on device: build_slot_batch_from_keys(device=cuda, aux) + sync.
  3. SYNC/GLUE: (a) aux.legal_counts.cpu().tolist(), (b) _slot_legal_hashes(aux),
     (c) _pad_slot_batch, (d) slot flat_logits = logits[legal_mask] gather.
  4. LEGACY server-side H2D: _prepare_tensors (frombuffer + .to(cuda) + bf16) of
     the prebuilt CPU graph body — the legacy path's per-request upload cost.
  5. Reconciliation: sum the slot components and compare to the given e2e ~27 ms
     at b=8; report the residual.

Random weights (throughput only — identical kernels/shapes to trained weights).

Usage:
    uv run --no-sync python hexo-a0/scripts/bench_slot_decomposition.py \
        [--device cuda] [--reps 50] [--warmup 20]
"""

from __future__ import annotations

import argparse
import random
import statistics
import struct
import time

import numpy as np
import torch

import hexo_rs
from hexo_a0.config import ModelConfig
from hexo_a0.graph import game_to_axis_graph
from hexo_a0.model import HeXONet
from hexo_a0.model_slots import slot_model_from_legacy
from hexo_a0.scriptable_model import ScriptableHeXONet, load_from_hexonet
from hexo_a0.slot_graph import SlotBuilderConfig, build_slot_batch_from_keys
from hexo_a0.inference_server import (
    _pad_slot_batch,
    _prepare_tensors,
    _slot_bucket_shape,
    _slot_legal_hashes,
)
from torch_geometric.data import Batch

# --- Production serving config (identical to bench_wire_a3_gpu.py) -----------
WIN_LENGTH = 6
PLACEMENT_RADIUS = 8
MAX_MOVES = 300
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 8
POLICY_HIDDEN = 128
VALUE_HIDDEN = 32
CONV_TYPE = "gine"
GRAPH_TYPE = "axis"
JK_MODE = "cat"
NODE_DIM = 11  # relative (7) + threat (4)
BUILDER_KWARGS = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)
DEPTHS = (20, 24, 27, 31, 34, 37, 40, 23, 29, 36)
BATCH_SIZES = (1, 8, 16)


# --- Positions: real random-legal radius-8 games (mirror bench_wire) ---------

def _cfg() -> "hexo_rs.GameConfig":
    return hexo_rs.GameConfig(WIN_LENGTH, PLACEMENT_RADIUS, MAX_MOVES)


def _snapshot(game, cfg):
    return hexo_rs.GameState.from_state(
        game.placed_stones(), game.current_player(),
        game.moves_remaining_this_turn(), cfg,
    )


def _play_to_depth(seed: int, depth: int):
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


def build_pool(n: int) -> list:
    return [_play_to_depth(seed=3000 + i, depth=DEPTHS[i % len(DEPTHS)]) for i in range(n)]


# --- Slot keys input: (p1_keys, p2_keys, current_player, moves_remaining) -----

def _pack_hexkey(q: int, r: int) -> int:
    key = ((q & 0xFFFF) << 16) | ((r ^ 0x8000) & 0xFFFF)
    return key - 0x100000000 if key >= 0x80000000 else key


def games_to_keys(games) -> list:
    out = []
    for g in games:
        stones = g.placed_stones()
        p1 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P1"]
        p2 = [_pack_hexkey(q, r) for (q, r), p in stones if p == "P2"]
        cur = 0 if g.current_player() == "P1" else 1
        out.append((p1, p2, cur, g.moves_remaining_this_turn()))
    return out


# --- Legacy graph body dict (exact _read_forward_body shape) ------------------

def _buf(arr) -> bytearray:
    return bytearray(np.ascontiguousarray(arr).tobytes())


def build_forward_body(games) -> dict:
    raws = [hexo_rs.game_to_axis_graph_raw(g, **BUILDER_KWARGS) for g in games]
    total_nodes = sum(r["num_nodes"] for r in raws)
    total_edges = sum(len(r["edge_src"]) for r in raws)

    features = np.concatenate([np.asarray(r["features"], dtype=np.float32).reshape(-1) for r in raws])
    esrc, edst, off = [], [], 0
    for r in raws:
        esrc.append(np.asarray(r["edge_src"], dtype=np.int64) + off)
        edst.append(np.asarray(r["edge_dst"], dtype=np.int64) + off)
        off += r["num_nodes"]
    edge_src = np.concatenate(esrc)
    edge_dst = np.concatenate(edst)
    edge_attr = np.concatenate([np.asarray(r["edge_attr"], dtype=np.float32).reshape(-1) for r in raws])
    legal_mask = np.concatenate([np.asarray(r["legal_mask"], dtype=np.uint8) for r in raws])
    stone_mask = np.concatenate([np.asarray(r["stone_mask"], dtype=np.uint8) for r in raws])
    batch = np.concatenate([np.full(r["num_nodes"], i, dtype=np.int32) for i, r in enumerate(raws)])

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "num_graphs": len(raws),
        "node_dim": NODE_DIM,
        "features": _buf(features),
        "edge_src": _buf(edge_src),
        "edge_dst": _buf(edge_dst),
        "edge_attr": _buf(edge_attr),
        "has_edge_attr": True,
        "legal_mask": _buf(legal_mask),
        "stone_mask": _buf(stone_mask),
        "batch": _buf(batch),
    }


# --- Model construction (bench_slot_model / test-fixture pattern) -------------

def build_models(device):
    cfg = ModelConfig(
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        conv_type=CONV_TYPE, policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN,
        graph_type=GRAPH_TYPE, use_jk=True, jk_mode=JK_MODE,
        prune_empty_edges=True, threat_features=True, relative_stone_encoding=True,
    )
    torch.manual_seed(1234)
    legacy = HeXONet(cfg).to(device).eval()  # fp32 — A2 eager legacy side

    # fp32 slot (A2-repro: bench_slot_model compiled the slot, ran fp32).
    slot_fp32 = slot_model_from_legacy(legacy, cfg, WIN_LENGTH).to(device).eval()
    slot_fwd_c_fp32 = torch.compile(slot_fp32.forward_padded, fullgraph=True)

    # bf16 production slot, exactly as the server builds it (--slot-inference).
    slot = slot_model_from_legacy(legacy, cfg, WIN_LENGTH).to(device).eval()

    # Legacy ScriptableHeXONet — the server's actual legacy forward, loaded from
    # the same weights (load_from_hexonet), compiled fullgraph like _load_model.
    scriptable = ScriptableHeXONet(
        node_features=NODE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, policy_hidden=POLICY_HIDDEN, value_hidden=VALUE_HIDDEN,
        graph_type=GRAPH_TYPE, conv_type=CONV_TYPE, use_jk=True, jk_mode=JK_MODE,
        relative_stone_encoding=True, threat_features=True,
    ).eval()
    load_from_hexonet(scriptable, legacy.state_dict())
    scriptable = scriptable.to(device)

    if device.type == "cuda":
        scriptable = scriptable.to(torch.bfloat16)
        slot = slot.to(torch.bfloat16)

    scriptable_c = torch.compile(scriptable, fullgraph=True)
    slot_fwd_c = torch.compile(slot.forward_padded, fullgraph=True)
    return legacy, scriptable_c, slot_fwd_c, slot_fwd_c_fp32


# --- Timing ------------------------------------------------------------------

def timed(fn, device, warmup, reps):
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--e2e-slot-b8", type=float, default=27.0,
                    help="e2e slot ms at b=8 from bench_wire_a3_gpu.py (for reconciliation)")
    ap.add_argument("--e2e-legacy-b8", type=float, default=5.2,
                    help="e2e legacy ms at b=8 from bench_wire_a3_gpu.py")
    ap.add_argument("--wire-parse-per-graph", type=float, default=0.3,
                    help="CPU wire/parse ms per graph (bench 3b) excluded from GPU sum")
    args = ap.parse_args()

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    hip = getattr(torch.version, "hip", None)

    print("# slot serving-path DECOMPOSITION — in-process, production shapes")
    print(f"# torch={torch.__version__} device={args.device} gpu={gpu!r} hip={hip} "
          f"dtype={'bf16' if is_cuda else 'f32'}")
    print(f"# model: GINE hidden={HIDDEN_DIM} layers={NUM_LAYERS} heads={NUM_HEADS} "
          f"jk={JK_MODE} axis prune+threat+relative node_dim={NODE_DIM} wl={WIN_LENGTH}")
    print(f"# compile: fullgraph=True dynamic=False (legacy scriptable + slot forward_padded)")
    print(f"# reps={args.reps} warmup={args.warmup} — RANDOM weights (throughput only)")

    legacy_fp32, scriptable_c, slot_fwd_c, slot_fwd_c_fp32 = build_models(device)

    pool = build_pool(max(BATCH_SIZES))
    builder_cfg = SlotBuilderConfig(
        win_length=WIN_LENGTH, placement_radius=PLACEMENT_RADIUS,
        prune_empty_edges=True, threat_features=True, relative_stones=True,
    )

    rows = {}  # b -> dict of component -> ms
    shapes = {}
    for b in BATCH_SIZES:
        games = pool[:b]
        keys = games_to_keys(games)

        # ---- prebuilt legacy inputs (GPU tensor tuple + CPU body for H2D) ----
        body = build_forward_body(games)
        legacy_tensors, _real_n = _prepare_tensors(body, device, None, padded=False)
        # fp32 PyG batch — bench_slot_model's exact eager legacy side (fp32).
        pyg_fp32 = Batch.from_data_list(
            [game_to_axis_graph(g, **BUILDER_KWARGS) for g in games]).to(device)

        # ---- prebuilt slot inputs (unpadded batch+aux, padded bf16 + fp32) ---
        unpadded, aux = build_slot_batch_from_keys(keys, builder_cfg, device=device, return_aux=True)
        bucket_b, bucket_n = _slot_bucket_shape(unpadded)
        padded_fp32 = _pad_slot_batch(unpadded, bucket_b, bucket_n)
        padded = _pad_slot_batch(unpadded, bucket_b, bucket_n)
        if is_cuda:
            padded.x = padded.x.to(torch.bfloat16)
            padded.dummy_x = padded.dummy_x.to(torch.bfloat16)
        with torch.no_grad():
            logits_pre, _v = slot_fwd_c(padded)
            slot_fwd_c_fp32(padded_fp32)  # warm fp32 trace
        legal_mask_pre = padded.legal_mask

        shapes[b] = dict(
            nodes=body["total_nodes"], edges=body["total_edges"],
            slot_real=(unpadded.num_graphs, unpadded.node_mask.shape[1]),
            slot_bucket=(bucket_b, bucket_n),
        )

        r = {}
        with torch.no_grad():
            # 1. FORWARD-ONLY
            r["legacy_fwd_scriptable_compiled"] = timed(
                lambda: scriptable_c(*legacy_tensors), device, args.warmup, args.reps)
            r["legacy_fwd_core_eager_fp32"] = timed(
                lambda: legacy_fp32._forward_batch_core(pyg_fp32), device, args.warmup, args.reps)
            r["slot_fwd_compiled"] = timed(
                lambda: slot_fwd_c(padded), device, args.warmup, args.reps)
            r["slot_fwd_compiled_fp32"] = timed(
                lambda: slot_fwd_c_fp32(padded_fp32), device, args.warmup, args.reps)

            # 2. BUILD-ONLY on device
            r["slot_build_from_keys"] = timed(
                lambda: build_slot_batch_from_keys(keys, builder_cfg, device=device, return_aux=True),
                device, args.warmup, args.reps)

            # 3. SYNC / GLUE
            r["glue_legal_counts_cpu_tolist"] = timed(
                lambda: aux.legal_counts.cpu().tolist(), device, args.warmup, args.reps)
            r["glue_slot_legal_hashes"] = timed(
                lambda: _slot_legal_hashes(aux), device, args.warmup, args.reps)
            r["glue_pad_slot_batch"] = timed(
                lambda: _pad_slot_batch(unpadded, bucket_b, bucket_n), device, args.warmup, args.reps)
            r["glue_flat_logits_gather"] = timed(
                lambda: logits_pre[legal_mask_pre], device, args.warmup, args.reps)

            # 4. LEGACY H2D upload (frombuffer + .to(cuda) + bf16)
            r["legacy_prepare_tensors_h2d"] = timed(
                lambda: _prepare_tensors(body, device, None, padded=False),
                device, args.warmup, args.reps)

        rows[b] = r

    # ---------- results table ----------
    comps = [
        ("legacy_fwd_scriptable_compiled", "legacy fwd (ScriptableHeXONet, compiled, bf16)"),
        ("slot_fwd_compiled", "slot fwd_padded (compiled, bf16)"),
        ("legacy_fwd_core_eager_fp32", "legacy fwd (_forward_batch_core, eager, fp32)"),
        ("slot_fwd_compiled_fp32", "slot fwd_padded (compiled, fp32)"),
        ("slot_build_from_keys", "slot build_from_keys (+sync)"),
        ("glue_pad_slot_batch", "glue: _pad_slot_batch"),
        ("glue_legal_counts_cpu_tolist", "glue: legal_counts.cpu().tolist()"),
        ("glue_slot_legal_hashes", "glue: _slot_legal_hashes"),
        ("glue_flat_logits_gather", "glue: logits[legal_mask] gather"),
        ("legacy_prepare_tensors_h2d", "legacy _prepare_tensors H2D"),
    ]
    print("\n" + "=" * 90)
    print("## Component medians (ms/batch)\n")
    hdr = f"| {'component':<44} |" + "".join(f" {'b='+str(b):>9} |" for b in BATCH_SIZES)
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for key, label in comps:
        line = f"| {label:<44} |"
        for b in BATCH_SIZES:
            line += f" {rows[b][key]:>9.3f} |"
        print(line)

    print("\n### shapes")
    for b in BATCH_SIZES:
        s = shapes[b]
        print(f"#  b={b:<2} legacy nodes={s['nodes']} edges={s['edges']}; "
              f"slot real(B,N)={s['slot_real']} -> bucket(B,N)={s['slot_bucket']}")

    # ---------- A2 claim ----------
    print("\n### A2 claim — does compiled slot forward beat legacy forward, uncontended?")
    print("#  A2-repro (fp32, bench_slot_model dtype): compiled slot vs eager legacy _forward_batch_core")
    print("#  production  (bf16, server dtype):        compiled slot vs compiled legacy ScriptableHeXONet")
    for b in BATCH_SIZES:
        sf32 = rows[b]["slot_fwd_compiled_fp32"]
        le32 = rows[b]["legacy_fwd_core_eager_fp32"]
        sf = rows[b]["slot_fwd_compiled"]
        lc = rows[b]["legacy_fwd_scriptable_compiled"]
        print(f"#  b={b:<2}: fp32 slot={sf32:.3f} vs legacy_eager={le32:.3f} "
              f"({le32/sf32:.2f}x) | bf16 slot={sf:.3f} vs legacy_compiled={lc:.3f} "
              f"({lc/sf:.2f}x)")

    # ---------- reconciliation at b=8 ----------
    B = 8
    r = rows[B]
    slot_sum_keys = [
        "slot_build_from_keys", "glue_pad_slot_batch",
        "glue_legal_counts_cpu_tolist", "glue_slot_legal_hashes",
        "slot_fwd_compiled", "glue_flat_logits_gather",
    ]
    slot_sum = sum(r[k] for k in slot_sum_keys)
    e2e = args.e2e_slot_b8
    wire = args.wire_parse_per_graph * B
    residual = e2e - slot_sum - wire
    print(f"\n### Reconciliation at b={B} (GPU components vs given e2e)")
    print(f"#  sum of slot components (build+pad+counts+hash+fwd+gather) = {slot_sum:.3f} ms")
    for k in slot_sum_keys:
        print(f"#      {k:<34} {r[k]:>8.3f} ms")
    print(f"#  + wire/parse (CPU, ~{args.wire_parse_per_graph}ms/graph x {B})      = {wire:.3f} ms")
    print(f"#  given e2e slot @ b={B}                              = {e2e:.3f} ms")
    print(f"#  UNEXPLAINED RESIDUAL                             = {residual:.3f} ms")
    print(f"#  legacy sum: fwd(compiled) {r['legacy_fwd_scriptable_compiled']:.3f} + "
          f"H2D {r['legacy_prepare_tensors_h2d']:.3f} = "
          f"{r['legacy_fwd_scriptable_compiled']+r['legacy_prepare_tensors_h2d']:.3f} ms "
          f"(given e2e legacy @ b={B} = {args.e2e_legacy_b8} ms)")


if __name__ == "__main__":
    main()
