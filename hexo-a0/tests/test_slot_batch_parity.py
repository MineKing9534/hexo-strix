"""Parity: the batched state->SlotBatch builder (``slot_graph.build_slot_batch``,
plan-A3 task 4) vs the existing per-graph path (``build_slot_graph`` per state +
``collate_slot_graphs``) on real random-legal games.

The batched builder receives exactly the fields the MSG_FORWARD_STATES wire
carries — stones + current_player + moves_remaining per game — and must derive
the legal region (placement window) itself, matching the Rust engine's
semantics. Integer/bool tensors must be EXACTLY equal; float features to the
same tolerance ``test_slot_graph_parity.py`` uses (atol=1e-5, rtol=1e-4).

CPU by default; ``test_slot_batch_gpu_matches_cpu`` asserts CUDA == CPU and is
excluded from the default run via the repo-wide ``-k 'not gpu'`` addopts.
"""

from __future__ import annotations

import random

import pytest
import torch

hexo_rs = pytest.importorskip("hexo_rs")

from hexo_a0 import slot_graph as sg  # noqa: E402

WIN_LENGTH = 6


# --------------------------------------------------------------------------
# Fixtures: real random-legal games
# --------------------------------------------------------------------------
def _random_game(seed: int, n_moves: int, radius: int = 6, win_length: int = WIN_LENGTH):
    cfg = hexo_rs.GameConfig(win_length=win_length, placement_radius=radius, max_moves=300)
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
    return g


def _near_terminal_game(seed: int):
    """One placement before a decided game (win_length 4 ends random play fast)."""
    cfg = hexo_rs.GameConfig(win_length=4, placement_radius=3, max_moves=300)
    rng = random.Random(seed)
    g = hexo_rs.GameState(cfg)
    moves = []
    while not g.is_terminal():
        ms = g.legal_moves()
        q, r = ms[rng.randrange(len(ms))]
        moves.append((q, r))
        g.apply_move(q, r)
    g2 = hexo_rs.GameState(cfg)
    for q, r in moves[:-1]:
        g2.apply_move(q, r)
    assert not g2.is_terminal()
    return g2


def _state_of(g):
    """The exact fields MSG_FORWARD_STATES carries for one graph."""
    return (
        [(q, r, p) for (q, r), p in g.placed_stones()],
        g.current_player(),
        g.moves_remaining_this_turn(),
    )


def _game_batch(radius: int, win_length: int = WIN_LENGTH, depths=(2, 7, 12, 25)):
    games = [_random_game(seed, d, radius, win_length) for seed, d in enumerate(depths)]
    games = [g for g in games if not g.is_terminal()]
    assert len(games) >= 2, "fixture degenerated to fewer than 2 live games"
    return games


def _reference_batch(games, win_length, flags, pad_to=None):
    outs = [
        sg.build_slot_graph(
            g.placed_stones(), g.legal_moves(), g.current_player(),
            g.moves_remaining_this_turn(), win_length, **flags,
        )
        for g in games
    ]
    return sg.collate_slot_graphs(outs, pad_to=pad_to)


def _batched(games, win_length, radius, flags, device="cpu", pad_to=None, return_aux=False):
    cfg = sg.SlotBuilderConfig(win_length=win_length, placement_radius=radius, **flags)
    return sg.build_slot_batch(
        [_state_of(g) for g in games], cfg, device=device, pad_to=pad_to,
        return_aux=return_aux,
    )


INT_FIELDS = ("partner", "filled", "node_mask", "stone_mask", "legal_mask", "src_player")


def _assert_batches_equal(got, ref, *, atol=1e-5, rtol=1e-4, aux=None, games=None):
    for name in INT_FIELDS:
        a, b = getattr(got, name).cpu(), getattr(ref, name).cpu()
        assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} vs {b.dtype}"
        assert torch.equal(a, b), f"field {name} differs"
    torch.testing.assert_close(got.x.cpu(), ref.x.cpu(), atol=atol, rtol=rtol)
    torch.testing.assert_close(got.dummy_x.cpu(), ref.dummy_x.cpu(), atol=atol, rtol=rtol)
    # The response contract rides on the aux legal ordering (legal_keys /
    # legal_counts) as much as on the SlotBatch tensors — check it against the
    # engine oracle (legal_moves() is (q, r)-lex == ascending-HexKey ordered,
    # which is exactly the batch's legal-column order).
    if aux is not None:
        assert games is not None, "pass games= to compare aux against the engine"
        assert aux.legal_counts.cpu().tolist() == [g.legal_move_count() for g in games]
        moves = [(q, r) for g in games for q, r in g.legal_moves()]
        expected_keys = sg.pack(
            torch.tensor([q for q, _ in moves]), torch.tensor([r for _, r in moves])
        )
        assert torch.equal(aux.legal_keys.cpu(), expected_keys), "aux legal_keys order"


FLAG_COMBOS = [
    dict(prune_empty_edges=p, threat_features=t, relative_stones=r)
    for p in (False, True)
    for t in (False, True)
    for r in (False, True)
]
FLAG_IDS = [
    f"p{int(f['prune_empty_edges'])}t{int(f['threat_features'])}r{int(f['relative_stones'])}"
    for f in FLAG_COMBOS
]


# --------------------------------------------------------------------------
# Main parity sweep: flags x radius, mixed-size batches, mid-turn mr in {1,2}
# --------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [2, 8])
@pytest.mark.parametrize("flags", FLAG_COMBOS, ids=FLAG_IDS)
def test_slot_batch_matches_collate(radius, flags):
    games = _game_batch(radius)
    # The fixture must genuinely cover both mid-turn values.
    assert {g.moves_remaining_this_turn() for g in games} == {1, 2}
    ref = _reference_batch(games, WIN_LENGTH, flags)
    got, aux = _batched(games, WIN_LENGTH, radius, flags, return_aux=True)
    _assert_batches_equal(got, ref, aux=aux, games=games)


def test_slot_batch_near_terminal():
    games = [_near_terminal_game(seed) for seed in range(3)]
    for flags in (FLAG_COMBOS[0], FLAG_COMBOS[-1]):  # plain + prune/threat/relative
        ref = _reference_batch(games, 4, flags)
        got, aux = _batched(games, 4, 3, flags, return_aux=True)
        _assert_batches_equal(got, ref, aux=aux, games=games)


def test_slot_batch_lean_layout_kwargs():
    """The extra layout kwargs the single-graph builder supports must batch too."""
    games = _game_batch(6, depths=(2, 7, 12))
    flags = dict(
        prune_empty_edges=True, threat_features=True, relative_stones=True,
        node_coords=False, moves_scope="graph", compact_stone_onehot=True,
    )
    ref = _reference_batch(games, WIN_LENGTH, flags)
    got = _batched(games, WIN_LENGTH, 6, flags)
    _assert_batches_equal(got, ref)


# --------------------------------------------------------------------------
# Legal-region derivation (the builder computes it; the engine is the oracle)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [2, 8])
def test_slot_batch_legal_counts_match_engine(radius):
    games = _game_batch(radius)
    got = _batched(games, WIN_LENGTH, radius, FLAG_COMBOS[0])
    counts = got.legal_mask.sum(dim=1).tolist()
    assert counts == [len(g.legal_moves()) for g in games]
    stone_counts = got.stone_mask.sum(dim=1).tolist()
    assert stone_counts == [len(g.placed_stones()) for g in games]


# --------------------------------------------------------------------------
# Padding: pad_to parity + invariance
# --------------------------------------------------------------------------
def test_slot_batch_pad_to_matches_collate():
    games = _game_batch(8)
    flags = FLAG_COMBOS[-1]
    ref = _reference_batch(games, WIN_LENGTH, flags, pad_to=2048)
    got = _batched(games, WIN_LENGTH, 8, flags, pad_to=2048)
    _assert_batches_equal(got, ref)


def test_slot_batch_padding_invariance():
    games = _game_batch(6, depths=(2, 7, 12))
    flags = FLAG_COMBOS[-1]
    tight = _batched(games, WIN_LENGTH, 6, flags)
    n = tight.x.shape[1]
    padded = _batched(games, WIN_LENGTH, 6, flags, pad_to=n + 173)
    # Real region identical...
    assert torch.equal(padded.partner[:, :n], tight.partner)
    assert torch.equal(padded.filled[:, :n], tight.filled)
    assert torch.equal(padded.node_mask[:, :n], tight.node_mask)
    assert torch.equal(padded.stone_mask[:, :n], tight.stone_mask)
    assert torch.equal(padded.legal_mask[:, :n], tight.legal_mask)
    assert torch.equal(padded.src_player[:, :n], tight.src_player)
    assert torch.equal(padded.x[:, :n], tight.x)
    assert torch.equal(padded.dummy_x, tight.dummy_x)
    # ...and the pad region inert.
    assert not padded.node_mask[:, n:].any()
    assert not padded.filled[:, n:].any()
    assert (padded.partner[:, n:] == 0).all()
    assert (padded.x[:, n:] == 0).all()
    assert (padded.src_player[:, n:] == 0).all()


def test_slot_batch_composition_invariance():
    """A graph's rows must not depend on what else is in the batch."""
    games = _game_batch(6, depths=(2, 7, 12))
    flags = FLAG_COMBOS[-1]
    mixed = _batched(games, WIN_LENGTH, 6, flags)
    n = mixed.x.shape[1]
    solo = _batched(games[1:2], WIN_LENGTH, 6, flags, pad_to=n)
    assert torch.equal(solo.partner[0], mixed.partner[1])
    assert torch.equal(solo.filled[0], mixed.filled[1])
    assert torch.equal(solo.node_mask[0], mixed.node_mask[1])
    torch.testing.assert_close(solo.x[0], mixed.x[1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(solo.dummy_x[0], mixed.dummy_x[1], atol=1e-5, rtol=1e-4)


def test_slot_batch_pad_to_smaller_than_batch_raises():
    games = _game_batch(6, depths=(2, 12))
    with pytest.raises(ValueError, match="pad_to"):
        _batched(games, WIN_LENGTH, 6, FLAG_COMBOS[0], pad_to=4)


# --------------------------------------------------------------------------
# Contract errors: loud, never silent garbage
# --------------------------------------------------------------------------
def _cfg(**over):
    base = dict(win_length=WIN_LENGTH, placement_radius=6)
    base.update(over)
    return sg.SlotBuilderConfig(**base)


def test_slot_batch_empty_states_raises():
    with pytest.raises(ValueError, match="empty"):
        sg.build_slot_batch([], _cfg())


def test_slot_batch_terminal_state_raises():
    with pytest.raises(ValueError, match="[Tt]erminal"):
        sg.build_slot_batch([([(0, 0, "P1")], None, 1)], _cfg())


def test_slot_batch_bad_moves_remaining_raises():
    with pytest.raises(ValueError, match="moves_remaining"):
        sg.build_slot_batch([([(0, 0, "P1")], "P2", 3)], _cfg())


def test_slot_batch_stoneless_state_raises():
    # The wire always carries the engine's (0,0) seed stone; a stoneless state
    # has no legal region and must be rejected, not built silently.
    with pytest.raises(ValueError, match="stone"):
        sg.build_slot_batch([([], "P1", 2)], _cfg())


def test_slot_batch_duplicate_stone_raises():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        sg.build_slot_batch([([(0, 0, "P1"), (0, 0, "P2")], "P2", 2)], _cfg())


def test_slot_batch_out_of_range_coord_raises():
    with pytest.raises(ValueError, match="32000"):
        sg.build_slot_batch([([(0, 0, "P1"), (32500, 0, "P2")], "P2", 2)], _cfg())


def test_slot_batch_bad_player_raises():
    with pytest.raises(ValueError, match="player"):
        sg.build_slot_batch([([(0, 0, "P3")], "P1", 2)], _cfg())


def test_slot_batch_accepts_int_players():
    """The wire encodes players as u8 0/1; string and int forms must agree."""
    g = _random_game(0, 7)
    stones_str, cur, mr = _state_of(g)
    stones_int = [(q, r, 0 if p == "P1" else 1) for q, r, p in stones_str]
    cur_int = 0 if cur == "P1" else 1
    cfg = _cfg()
    a = sg.build_slot_batch([(stones_str, cur, mr)], cfg)
    b = sg.build_slot_batch([(stones_int, cur_int, mr)], cfg)
    _assert_batches_equal(a, b)


def test_slot_batch_accepts_nested_stone_tuples():
    """placed_stones()-style ((q, r), player) entries are accepted too."""
    g = _random_game(1, 7)
    cfg = _cfg()
    a = sg.build_slot_batch([_state_of(g)], cfg)
    b = sg.build_slot_batch(
        [(list(g.placed_stones()), g.current_player(), g.moves_remaining_this_turn())], cfg
    )
    _assert_batches_equal(a, b)


# --------------------------------------------------------------------------
# CUDA == CPU (tiny — the GPU is contended by live training)
# --------------------------------------------------------------------------
def test_slot_batch_gpu_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    games = [_random_game(0, 2, radius=2), _random_game(1, 5, radius=2)]
    assert all(not g.is_terminal() for g in games)
    flags = dict(prune_empty_edges=True, threat_features=True, relative_stones=True)
    cpu = _batched(games, WIN_LENGTH, 2, flags, device="cpu")
    gpu = _batched(games, WIN_LENGTH, 2, flags, device="cuda")
    assert gpu.x.device.type == "cuda"
    for name in INT_FIELDS:
        assert torch.equal(getattr(gpu, name).cpu(), getattr(cpu, name))
    torch.testing.assert_close(gpu.x.cpu(), cpu.x)
    torch.testing.assert_close(gpu.dummy_x.cpu(), cpu.dummy_x)
