import pytest

from hexo_klent.returns import lambda_returns, transition_sign


def test_transition_sign_tracks_hexo_turn_owner():
    assert transition_sign("P2", "P2") == 1.0
    assert transition_sign("P2", "P1") == -1.0


def test_lambda_returns_preserve_same_player_then_negate_on_turn_pass():
    decay = 0.75
    # P2 makes two placements, then P1 makes the terminal winning placement.
    result = lambda_returns(
        players=["P2", "P2", "P1"],
        rewards=[0.0, 0.0, 1.0],
        state_values=[0.2, 0.4, 0.6],
        trace_decay=decay,
    )

    expected_second = -((1 - decay) * 0.6 + decay * 1.0)
    expected_first = (1 - decay) * 0.4 + decay * expected_second
    assert result == pytest.approx([expected_first, expected_second, 1.0])


def test_zero_trace_is_one_step_bootstrap():
    result = lambda_returns(
        players=["P2", "P2", "P1"],
        rewards=[0.0, 0.0, -1.0],
        state_values=[0.0, 0.25, -0.5],
        trace_decay=0.0,
    )

    assert result == pytest.approx([0.25, 0.5, -1.0])


def test_unit_trace_is_full_player_perspective_return():
    result = lambda_returns(
        players=["P2", "P2", "P1", "P1"],
        rewards=[0.0, 0.0, 0.0, 1.0],
        state_values=[0.9, -0.8, 0.7, -0.6],
        trace_decay=1.0,
    )

    assert result == pytest.approx([-1.0, -1.0, 1.0, 1.0])


def test_gamma_discounts_each_transition_but_not_terminal_reward():
    result = lambda_returns(
        players=["P2", "P2", "P1", "P1"],
        rewards=[0.0, 0.0, 0.0, 1.0],
        state_values=[0.0, 0.0, 0.0, 0.0],
        trace_decay=1.0,
        gamma=0.5,
    )

    assert result == pytest.approx([-0.125, -0.25, 0.5, 1.0])


def test_gamma_discounts_truncated_successor_bootstrap():
    result = lambda_returns(
        players=["P1"],
        rewards=[0.0],
        state_values=[0.0],
        trace_decay=1.0,
        gamma=0.5,
        bootstrap_player="P2",
        bootstrap_value=0.8,
    )

    assert result == pytest.approx([-0.4])


def test_truncated_return_bootstraps_from_live_successor_perspective():
    result = lambda_returns(
        players=["P2", "P2"],
        rewards=[0.0, 0.0],
        state_values=[0.1, 0.4],
        trace_decay=0.75,
        bootstrap_player="P1",
        bootstrap_value=0.6,
    )

    expected_last = -0.6
    expected_first = (1 - 0.75) * 0.4 + 0.75 * expected_last
    assert result == pytest.approx([expected_first, expected_last])


def test_truncated_return_requires_complete_bootstrap_pair():
    with pytest.raises(ValueError, match="must be supplied together"):
        lambda_returns(
            players=["P2"],
            rewards=[0.0],
            state_values=[0.0],
            trace_decay=0.5,
            bootstrap_player="P1",
        )
