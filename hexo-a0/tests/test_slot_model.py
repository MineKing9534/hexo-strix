"""Equivalence: the slot-based static-shape model (model_slots.py, Workstream A2)
vs the legacy PyG HeXONet on identical positions.

The slot model's per-layer edge embedding decomposes the legacy affine chain
``conv.lin(edge_proj(attr))`` into ``slot_table[axis,sign,dist] +
src_player * src_vec`` (+ a constant ``dummy_emb`` for the all-zero-attr
dummy edges), so a converted model must reproduce the legacy forward to float
tolerance — not approximately, exactly up to op-order.

Runs CPU-only.
"""

from __future__ import annotations

import random

import pytest
import torch

hexo_rs = pytest.importorskip("hexo_rs")

from torch_geometric.data import Batch  # noqa: E402

from hexo_a0 import slot_graph as sg  # noqa: E402
from hexo_a0.config import ModelConfig  # noqa: E402
from hexo_a0.graph import game_to_axis_graph  # noqa: E402
from hexo_a0.model import HeXONet  # noqa: E402
from hexo_a0.model_slots import (  # noqa: E402
    SlotHeXONet,
    collate_slot_graphs,
    slot_model_from_legacy,
)

WIN_LENGTH = 6


def _random_game(seed: int, n_moves: int):
    cfg = hexo_rs.GameConfig(win_length=WIN_LENGTH, placement_radius=6, max_moves=300)
    g = hexo_rs.GameState(cfg)
    rng = random.Random(seed)
    for _ in range(n_moves):
        if g.is_terminal():
            break
        moves = g.legal_moves()
        if not moves:
            break
        q, r = moves[rng.randrange(len(moves))]
        g.apply_move(q, r)
    assert not g.is_terminal()
    return g


def _graph_flags(cfg: ModelConfig) -> dict:
    return dict(
        prune_empty_edges=cfg.prune_empty_edges,
        threat_features=cfg.threat_features,
        relative_stones=cfg.relative_stone_encoding,
    )


def _legacy_outputs(model: HeXONet, games, flags):
    datas = [game_to_axis_graph(g, **flags) for g in games]
    batch = Batch.from_data_list(datas)
    with torch.no_grad():
        policy_list, values = model.forward_batch(batch)
    return policy_list, values


def _slot_outputs(slot_model: SlotHeXONet, games, flags, pad_to=None):
    outs = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        for g in games
    ]
    batch = collate_slot_graphs(outs, pad_to=pad_to)
    with torch.no_grad():
        policy_list, values = slot_model.forward_batch(batch)
    return policy_list, values


def _make_pair(cfg: ModelConfig, seed: int = 0) -> tuple[HeXONet, SlotHeXONet]:
    torch.manual_seed(seed)
    legacy = HeXONet(cfg).eval()
    slot = slot_model_from_legacy(legacy, cfg, WIN_LENGTH).eval()
    return legacy, slot


def _assert_outputs_match(legacy_out, slot_out, *, atol=1e-5, rtol=1e-4):
    # Tolerance is op-order-only (same weights, same math, different reduction
    # order) — a systematic conversion error is orders of magnitude above this.
    lp, lv = legacy_out
    sp, sv = slot_out
    assert len(lp) == len(sp)
    for i, (a, b) in enumerate(zip(lp, sp)):
        assert a.shape == b.shape, f"graph {i}: legal-count mismatch"
        torch.testing.assert_close(b, a, atol=atol, rtol=rtol)
    torch.testing.assert_close(sv, lv, atol=atol, rtol=rtol)


def _cfg(**over) -> ModelConfig:
    base = dict(
        hidden_dim=32, num_layers=3, num_heads=4, conv_type="gine",
        policy_hidden=16, value_hidden=16, graph_type="axis",
        prune_empty_edges=True,
    )
    base.update(over)
    return ModelConfig(**base)


CONFIGS = {
    "base-prenorm": _cfg(),
    "postnorm": _cfg(pre_norm=False),
    "jk-sum": _cfg(use_jk=True, jk_mode="sum"),
    "jk-cat": _cfg(use_jk=True, jk_mode="cat"),
    "jk-max": _cfg(use_jk=True, jk_mode="max"),
    "layerscale": _cfg(use_layer_scale=True),
    # Production-shaped: relative + threat (11-dim nodes), jk cat.
    "production": _cfg(
        threat_features=True, relative_stone_encoding=True,
        use_jk=True, jk_mode="cat",
    ),
    "value-bins": _cfg(value_bins=17),
}

# Positions: early, mid-turn (odd placements), and deep.
POSITIONS = [(0, 2), (1, 7), (2, 12), (3, 25)]


@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_slot_matches_legacy_single_graphs(cfg_name):
    cfg = CONFIGS[cfg_name]
    legacy, slot = _make_pair(cfg)
    flags = _graph_flags(cfg)
    for seed, nmoves in POSITIONS:
        games = [_random_game(seed, nmoves)]
        _assert_outputs_match(
            _legacy_outputs(legacy, games, flags),
            _slot_outputs(slot, games, flags),
        )


def test_slot_matches_legacy_mixed_size_batch():
    """Two graphs of very different sizes in one padded batch (plan §7.4)."""
    cfg = CONFIGS["production"]
    legacy, slot = _make_pair(cfg)
    flags = _graph_flags(cfg)
    games = [_random_game(0, 2), _random_game(1, 12), _random_game(2, 30)]
    _assert_outputs_match(
        _legacy_outputs(legacy, games, flags),
        _slot_outputs(slot, games, flags),
    )


def test_padding_invariance():
    """Padding a graph to a larger static shape must not change its outputs."""
    cfg = CONFIGS["base-prenorm"]
    _, slot = _make_pair(cfg)
    flags = _graph_flags(cfg)
    games = [_random_game(4, 12)]
    tight_p, tight_v = _slot_outputs(slot, games, flags)
    padded_p, padded_v = _slot_outputs(slot, games, flags, pad_to=2048)
    torch.testing.assert_close(padded_p[0], tight_p[0])
    torch.testing.assert_close(padded_v, tight_v)


def test_pad_to_smaller_than_graph_raises():
    flags = _graph_flags(CONFIGS["base-prenorm"])
    g = _random_game(0, 12)
    out = sg.build_slot_graph(
        g.placed_stones(), g.legal_moves(), g.current_player(),
        g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
    )
    with pytest.raises(ValueError):
        collate_slot_graphs([out], pad_to=4)


def test_gatv2_unsupported():
    cfg = _cfg(conv_type="gatv2")
    with pytest.raises(ValueError, match="gine"):
        legacy = HeXONet(cfg)
        slot_model_from_legacy(legacy, cfg, WIN_LENGTH)


@pytest.mark.parametrize("bad", [
    dict(graph_type="hex"),
    dict(axis_relational=True),
    dict(use_jk=True, jk_mode="lstm"),
    dict(moves_scope="graph"),
])
def test_unsupported_configs_raise(bad):
    cfg = _cfg(**bad)
    with pytest.raises(ValueError):
        SlotHeXONet(cfg, WIN_LENGTH)


def test_conversion_rejects_mismatched_config():
    """The target config must describe the checkpoint's architecture — a
    mismatch must raise, never silently produce a diverging model."""
    legacy = HeXONet(_cfg())
    with pytest.raises(ValueError, match="num_layers"):
        slot_model_from_legacy(legacy, _cfg(num_layers=5), WIN_LENGTH)
    with pytest.raises(ValueError, match="pre_norm"):
        slot_model_from_legacy(legacy, _cfg(pre_norm=False), WIN_LENGTH)
    with pytest.raises(ValueError, match="jk"):
        slot_model_from_legacy(legacy, _cfg(use_jk=True, jk_mode="sum"), WIN_LENGTH)
    with pytest.raises(ValueError, match="layer_scale"):
        slot_model_from_legacy(legacy, _cfg(use_layer_scale=True), WIN_LENGTH)
    with pytest.raises(ValueError, match="hidden"):
        slot_model_from_legacy(legacy, _cfg(hidden_dim=64, num_heads=4), WIN_LENGTH)


def test_terminal_position_raises():
    """Terminal states are outside the builder's contract, same as the legacy
    game_to_axis_graph (which raises) — never a silently mislabeled graph.
    Uses a genuinely finished game (short win_length ends random play fast)."""
    cfg = hexo_rs.GameConfig(win_length=4, placement_radius=3, max_moves=300)
    rng = random.Random(0)
    g = hexo_rs.GameState(cfg)
    while not g.is_terminal():
        moves = g.legal_moves()
        q, r = moves[rng.randrange(len(moves))]
        g.apply_move(q, r)
    assert g.current_player() is None  # the terminal signal the builder keys on
    with pytest.raises(ValueError, match="[Tt]erminal"):
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), 4,
        )


def test_forward_padded_masks_illegal_and_padding():
    """The static-shape output contract: -inf exactly off the legal set, so a
    softmax over dim 1 is the move distribution directly."""
    cfg = CONFIGS["base-prenorm"]
    legacy, slot = _make_pair(cfg)
    flags = _graph_flags(cfg)
    games = [_random_game(0, 2), _random_game(1, 12)]
    outs = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        for g in games
    ]
    batch = collate_slot_graphs(outs, pad_to=1024)
    with torch.no_grad():
        logits, _ = slot.forward_padded(batch)
    assert torch.isinf(logits[~batch.legal_mask]).all()
    assert (logits[~batch.legal_mask] < 0).all()
    assert torch.isfinite(logits[batch.legal_mask]).all()
    probs = torch.softmax(logits, dim=1)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(len(games)))
    assert (probs[~batch.legal_mask] == 0).all()
    # And the distribution matches the legacy softmax on each graph.
    legacy_logits, _ = _legacy_outputs(legacy, games, flags)
    for i in range(len(games)):
        torch.testing.assert_close(
            probs[i][batch.legal_mask[i]],
            torch.softmax(legacy_logits[i], dim=0),
            atol=1e-5, rtol=1e-4,
        )


def test_fixtures_cover_all_slots_and_both_players():
    """The equivalence tests only constrain slots that are actually filled —
    assert the fixture set exercises every (axis, sign, dist) slot and both
    stone colours as message sources."""
    flags = _graph_flags(CONFIGS["production"])
    union = None
    src_kinds: set[int] = set()
    for seed, nmoves in POSITIONS:
        g = _random_game(seed, nmoves)
        out = sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        filled = out["filled"]  # [N, 3, 2, W]
        any_per_slot = filled.any(dim=0)
        union = any_per_slot if union is None else (union | any_per_slot)
        src = out["partner"][filled]
        src_kinds |= set(out["kinds"][src].tolist())
    assert union is not None and union.all(), (
        f"unexercised slots: {(~union).nonzero().tolist()}"
    )
    assert {0, 1} <= src_kinds, "both stone colours must appear as edge sources"


def test_slot_conversion_on_gpu_matches_legacy():
    """Conversion from a CUDA-resident legacy model must work (the slot-attr
    tables are built on the model's device) and reproduce the legacy outputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = CONFIGS["production"]
    torch.manual_seed(0)
    legacy_cpu = HeXONet(cfg).eval()
    flags = _graph_flags(cfg)
    games = [_random_game(1, 12)]
    expected = _legacy_outputs(legacy_cpu, games, flags)

    legacy_gpu = HeXONet(cfg).eval()
    legacy_gpu.load_state_dict(legacy_cpu.state_dict())
    legacy_gpu = legacy_gpu.cuda()
    slot = slot_model_from_legacy(legacy_gpu, cfg, WIN_LENGTH).eval()

    outs = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        for g in games
    ]
    batch = collate_slot_graphs(outs).to(torch.device("cuda"))
    with torch.no_grad():
        policy_list, values = slot.forward_batch(batch)
    torch.testing.assert_close(
        policy_list[0].cpu(), expected[0][0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(values.cpu(), expected[1], atol=1e-3, rtol=1e-3)


def test_stoneless_board_inv_dist_matches_rust_semantics():
    """Rust's builder gives every legal node inv_dist = 1.0 when there are no
    stones (min_d.unwrap_or(1)); the torch port must match. Unreachable from a
    real GameState (the engine seeds (0,0)), so pinned directly."""
    out = sg.build_slot_graph([], [(0, 0), (1, 0)], "P1", 2, WIN_LENGTH)
    inv_dist_col = out["features"].shape[1] - 1  # legacy absolute layout, no threat
    legal_rows = out["features"][:-1][out["legal_mask"][:-1]]
    assert (legal_rows[:, inv_dist_col] == 1.0).all()


def test_slot_count_mismatch_raises():
    """A win_length mismatch between builder and model must fail loudly."""
    cfg = CONFIGS["base-prenorm"]
    _, slot = _make_pair(cfg)  # model built for WIN_LENGTH
    flags = _graph_flags(cfg)
    g = _random_game(0, 6)
    out = sg.build_slot_graph(
        g.placed_stones(), g.legal_moves(), g.current_player(),
        g.moves_remaining_this_turn(), WIN_LENGTH + 2, **flags,
    )
    batch = collate_slot_graphs([out])
    with pytest.raises(ValueError, match="slot"):
        slot.forward_padded(batch)


def test_slot_batch_reports_num_legal():
    """Legal counts in the collated batch match the games (head masking input)."""
    flags = _graph_flags(CONFIGS["base-prenorm"])
    games = [_random_game(0, 2), _random_game(1, 12)]
    outs = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), WIN_LENGTH, **flags,
        )
        for g in games
    ]
    batch = collate_slot_graphs(outs)
    counts = batch.legal_mask.sum(dim=1).tolist()
    assert counts == [len(g.legal_moves()) for g in games]
