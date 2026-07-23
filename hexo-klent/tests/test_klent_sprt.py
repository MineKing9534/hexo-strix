import json

import pytest

from hexo_a0.sprt_eval import SPRTConfig
from hexo_klent.sprt import GameResult, run_sprt_games


def test_sprt_alternates_sides_and_stops_only_after_a_pair(tmp_path):
    sides = []

    def play(side):
        sides.append(side)
        return GameResult("W", moves=11)

    result = run_sprt_games(
        play,
        sprt_config=SPRTConfig(
            s0=0.5,
            s1=0.75,
            alpha=0.25,
            beta=0.25,
            pair_variance=0.5,
            window_size=None,
            pentanomial=True,
        ),
        max_games=1000,
        state_file=tmp_path / "state.json",
    )

    assert result.decision == "accept_h1"
    assert result.games % 2 == 0
    assert sides == ["P1", "P2"] * (result.games // 2)
    payload = json.loads((tmp_path / "state.json").read_text())
    assert payload["games"] == result.games
    assert payload["decision"] == "accept_h1"
    assert payload["outcomes"] == "W" * result.games


def test_sprt_reports_truncations_as_draws_at_max_games():
    def play(_side):
        return GameResult("D", moves=1000, truncated=True)

    result = run_sprt_games(
        play,
        sprt_config=SPRTConfig(
            s0=0.5,
            s1=0.55,
            alpha=0.05,
            beta=0.05,
            pair_variance=0.5,
            window_size=None,
            pentanomial=True,
        ),
        max_games=4,
    )

    assert result.decision == "continue"
    assert result.games == 4
    assert result.draws == 4
    assert result.truncations == 4


@pytest.mark.parametrize("max_games", [0, 1, 3])
def test_sprt_requires_positive_complete_pair_cap(max_games):
    with pytest.raises(ValueError, match="positive even"):
        run_sprt_games(
            lambda _side: GameResult("W", moves=1),
            sprt_config=SPRTConfig(window_size=None),
            max_games=max_games,
        )
