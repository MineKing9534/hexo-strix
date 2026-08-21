"""Generate numerical parity fixtures for the hexo-infer Rust crate.

Writes fixture JSONs (+ the tiny models' safetensors) into
``hexo-rs/hexo-infer/tests/fixtures/``. The tiny models (jk-cat AND no-jk) are
COMMITTED (they're random-init, no private data) and are the load-bearing parity
gate for hexo-infer. Real-checkpoint fixtures carry the private model's exact logits
for 4 positions, so they're gitignored (local-only, like the weights in exports/);
the real-parity Rust test is env-gated on ``HEXO_REAL_WEIGHTS``.

Run:
    uv run --no-sync python -m hexo_a0.gen_parity_fixtures                 # tiny only
    uv run --no-sync python -m hexo_a0.gen_parity_fixtures \
        --checkpoint "$(cat exports/PINNED_CHECKPOINT)"                     # tiny + real
"""

import dataclasses
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from hexo_a0.config import ModelConfig, model_config_from_checkpoint

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "hexo-rs" / "hexo-infer" / "tests" / "fixtures"
SEED = 42
# Per-position seeds (SEED + n_plies) -> four DISTINCT game trajectories, not one
# game sampled at four depths. midturn_9 (odd ply count) lands moves_remaining==1.
PLY_COUNTS = {"initial": 0, "early_5": 5, "midturn_9": 9, "mid_20": 20}


def tiny_model_and_config():
    """jk-cat + relative + threat + prune — the rel2 config shape (node dim 11)."""
    torch.manual_seed(0)
    from hexo_a0.model import HeXONet
    mc = ModelConfig(
        hidden_dim=16, num_layers=2, num_heads=1, conv_type="gine",
        pre_norm=True, dropout=0.0, use_jk=True, jk_mode="cat",
        policy_hidden=16, value_hidden=8, graph_type="axis",
        prune_empty_edges=True, threat_features=True,
        relative_stone_encoding=True,
    )
    return HeXONet(mc).eval(), mc


def tiny_nojk_model_and_config():
    """no-JK + absolute + no-threat + no-prune — covers the otherwise-dead forward branches
    (rep = LN(hs[last]), 8-dim nodes, unpruned graphs)."""
    torch.manual_seed(1)
    from hexo_a0.model import HeXONet
    mc = ModelConfig(
        hidden_dim=16, num_layers=2, num_heads=1, conv_type="gine",
        pre_norm=True, dropout=0.0, use_jk=False,
        policy_hidden=16, value_hidden=8, graph_type="axis",
        prune_empty_edges=False, threat_features=False,
        relative_stone_encoding=False,
    )
    return HeXONet(mc).eval(), mc


def tiny_klent_model_and_config():
    """KLENT lean axis schema: axis_relational + Q-head + compact/coordless nodes
    (node dim 8 = relative 7 - compact 1 - coords 2 + threat 4)."""
    torch.manual_seed(2)
    from hexo_a0.model import PolicyHead, QHead, RepresentationNetwork
    mc = ModelConfig(
        hidden_dim=16, num_layers=2, num_heads=1, conv_type="gine",
        pre_norm=True, dropout=0.0, use_jk=True, jk_mode="cat",
        policy_hidden=16, q_hidden=8, graph_type="axis",
        prune_empty_edges=True, threat_features=True,
        relative_stone_encoding=True,
        axis_relational=True, axis_window=8,
        compact_stone_onehot=True, node_coords=False, moves_scope="node",
    )

    class TinyKlent(nn.Module):
        def __init__(self):
            super().__init__()
            self.representation = RepresentationNetwork(mc, graph_type="axis")
            head_dim = self.representation.output_dim
            self.policy_head = PolicyHead(head_dim, mc.policy_hidden)
            self.q_head = QHead(head_dim, mc.q_hidden)

        def forward(self, x, edge_index, legal_mask, edge_type, edge_dist,
                    global_edge_index):
            emb = self.representation(
                x, edge_index, None,
                edge_type=edge_type, edge_dist=edge_dist,
                global_edge_index=global_edge_index,
            )
            legal_idx = legal_mask.nonzero(as_tuple=False).squeeze(1)
            legal_emb = torch.index_select(emb, 0, legal_idx)
            logits = self.policy_head.mlp(legal_emb).squeeze(-1)
            q = self.q_head.mlp(legal_emb).squeeze(-1)
            return logits, q

    return TinyKlent().eval(), mc


def _play_moves(game_config: dict, n_plies: int):
    import hexo_rs
    gc = hexo_rs.GameConfig(game_config["win_length"], game_config["placement_radius"],
                            game_config["max_moves"])
    state = hexo_rs.GameState(gc)
    rng = random.Random(SEED + n_plies)   # per-position seed -> distinct trajectories
    moves = []
    for _ in range(n_plies):
        legal = state.legal_moves()
        assert legal and not state.is_terminal(), (
            f"seed {SEED + n_plies} hit terminal/no-legal at ply {len(moves)}; bump SEED")
        q, r = rng.choice(legal)
        state.apply_move(q, r)          # PyO3 binding takes two scalars
        moves.append([q, r])
    # The FINAL state is what gets evaluated — guard it (a winning nth ply would
    # otherwise produce an empty-logits fixture downstream).
    assert not state.is_terminal() and state.legal_moves(), (
        f"seed {SEED + n_plies} produced terminal/no-legal final state at {n_plies} plies; bump SEED")
    if n_plies == 9:  # the midturn fixture pins the mr==1 feature encoding — verify it
        assert state.moves_remaining_this_turn() == 1, "midturn fixture must be mid-turn"
    return state, moves


def _eval_position(model, mc: ModelConfig, state):
    import hexo_rs
    g = hexo_rs.game_to_axis_graph_raw(   # returns a DICT
        state, mc.prune_empty_edges, mc.threat_features, mc.relative_stone_encoding)
    from hexo_a0.config import node_feature_dim
    fdim = node_feature_dim(mc)
    x = torch.tensor(g["features"], dtype=torch.float32).view(-1, fdim)
    edge_index = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.long)
    edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32).view(-1, 5)
    legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool)
    stone_mask = torch.tensor(g["stone_mask"], dtype=torch.bool)
    with torch.no_grad():
        logits, value = model(x, edge_index, legal_mask, stone_mask, edge_attr=edge_attr)
    return logits.tolist(), float(value)


def _eval_klent_position(model, mc: ModelConfig, state):
    """KLENT lean-graph forward: policy logits + Q-values, then the state value
    V = Σ_a softmax(logits)_a · Q_a (KlentMCTSAdapter._state_values)."""
    import hexo_rs
    g = hexo_rs.game_to_axis_graph_raw(
        state, mc.prune_empty_edges, mc.threat_features, mc.relative_stone_encoding,
        axis_relational=True, compact_stone_onehot=mc.compact_stone_onehot,
        node_coords=mc.node_coords, moves_scope=mc.moves_scope)
    from hexo_a0.config import node_feature_dim
    fdim = node_feature_dim(mc)
    x = torch.tensor(g["features"], dtype=torch.float32).view(-1, fdim)
    edge_index = torch.tensor([g["edge_src"], g["edge_dst"]], dtype=torch.long)
    edge_type = torch.tensor(g["edge_type"], dtype=torch.long)
    edge_dist = torch.tensor(g["edge_dist"], dtype=torch.long)
    global_edge_index = torch.tensor(
        [g["global_edge_src"], g["global_edge_dst"]], dtype=torch.long)
    legal_mask = torch.tensor(g["legal_mask"], dtype=torch.bool)
    with torch.no_grad():
        logits, q = model(x, edge_index, legal_mask, edge_type, edge_dist,
                          global_edge_index)
    value = float(torch.dot(torch.softmax(logits, dim=0), q))
    return logits.tolist(), value


def generate_fixtures(model, mc: ModelConfig, prefix: str, out_dir: Path,
                      game_config: dict, checkpoint_name: str | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_opts = {"prune_empty_edges": mc.prune_empty_edges,
                  "threat_features": mc.threat_features,
                  "relative_stones": mc.relative_stone_encoding}
    paths = []
    for name, plies in PLY_COUNTS.items():
        state, moves = _play_moves(game_config, plies)
        logits, value = _eval_position(model, mc, state)
        legal = state.legal_moves()
        assert len(legal) == len(logits), f"{name}: legal {len(legal)} != logits {len(logits)}"
        fx = {
            "name": f"{prefix}_{name}",
            "game_config": game_config,
            "graph_opts": graph_opts,
            **({"checkpoint": checkpoint_name} if checkpoint_name else {}),
            "moves": moves,
            "expected": {
                "value": round(value, 8),
                "logits": [[list(c), round(l, 8)] for c, l in zip(legal, logits)],
            },
        }
        p = out_dir / f"{prefix}_{name}.json"
        p.write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
        paths.append(p)
    return paths


def generate_klent_fixtures(model, mc: ModelConfig, prefix: str, out_dir: Path,
                            game_config: dict) -> list[Path]:
    """KLENT lean-schema fixtures: graph_opts carries the lean flags so the Rust
    evaluator builds the same lean graph (edge_type/edge_dist/global edges)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_opts = {"prune_empty_edges": mc.prune_empty_edges,
                  "threat_features": mc.threat_features,
                  "relative_stones": mc.relative_stone_encoding,
                  "axis_relational": True,
                  "compact_stone_onehot": mc.compact_stone_onehot,
                  "node_coords": mc.node_coords,
                  "moves_scope": mc.moves_scope}
    paths = []
    for name, plies in PLY_COUNTS.items():
        state, moves = _play_moves(game_config, plies)
        logits, value = _eval_klent_position(model, mc, state)
        legal = state.legal_moves()
        assert len(legal) == len(logits), f"{name}: legal {len(legal)} != logits {len(logits)}"
        fx = {
            "name": f"{prefix}_{name}",
            "game_config": game_config,
            "graph_opts": graph_opts,
            "moves": moves,
            "expected": {
                "value": round(value, 8),
                "logits": [[list(c), round(l, 8)] for c, l in zip(legal, logits)],
            },
        }
        p = out_dir / f"{prefix}_{name}.json"
        p.write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
        paths.append(p)
    return paths


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="EXACT pinned checkpoint path for real fixtures (no globbing; "
                             "pin once via exports/PINNED_CHECKPOINT, reuse verbatim). "
                             "Omit to regenerate only the tiny fixtures.")
    args = parser.parse_args(argv)

    from hexo_a0.export import save_safetensors
    # --- tiny models (jk-cat AND no-jk): weights + fixtures, committed ---
    for model, mc, prefix in [(*tiny_model_and_config(), "tiny"),
                               (*tiny_nojk_model_and_config(), "tiny_nojk")]:
        save_safetensors(model.state_dict(), dataclasses.asdict(mc), "0", prefix,
                         FIXTURES_DIR / f"{prefix}.safetensors")
        generate_fixtures(model, mc, prefix, FIXTURES_DIR,
                          {"win_length": 6, "placement_radius": 4, "max_moves": 300})
    # --- tiny KLENT model (axis-relational + Q-head): weights + fixtures, committed ---
    klent_model, klent_mc = tiny_klent_model_and_config()
    save_safetensors(klent_model.state_dict(), dataclasses.asdict(klent_mc), "0",
                     "tiny_klent", FIXTURES_DIR / "tiny_klent.safetensors")
    generate_klent_fixtures(klent_model, klent_mc, "tiny_klent", FIXTURES_DIR,
                            {"win_length": 6, "placement_radius": 4, "max_moves": 300})
    # --- real checkpoint: fixtures only (gitignored — private model behavior) ---
    if args.checkpoint is not None:
        if not args.checkpoint.strip():
            raise SystemExit("--checkpoint was passed but is empty (unset shell var?)")
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f"pinned checkpoint missing: {ckpt_path}")
        from hexo_a0.model import HeXONet
        try:
            ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        mc_real = model_config_from_checkpoint(ck)
        real = HeXONet(mc_real)
        sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model_state_dict"].items()}
        real.load_state_dict(sd)
        real.eval()
        # Use the checkpoint's OWN embedded game_config (authoritative + STAGE-SPECIFIC:
        # a curriculum checkpoint carries its stage's radius, e.g. S3=6, not [game]'s 8).
        emb = ck.get("game_config") or {}
        gc = {k: emb.get(k, d) for k, d in
              {"win_length": 6, "placement_radius": 8, "max_moves": 300}.items()}
        generate_fixtures(real, mc_real, "real", FIXTURES_DIR, gc,
                          checkpoint_name=ckpt_path.name)
        print(f"real fixtures from pinned {ckpt_path}")
    else:
        print("no --checkpoint; generated tiny fixtures only")
    print(f"fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()