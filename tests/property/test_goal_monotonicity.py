"""Property: Home goal → P(home_win) increases.

When the home team scores (all else equal), the probability of a home
win must increase (or stay the same if already certain). Conversely,
P(away_win) must decrease. Same logic applies for away goals.

Tested for both analytical_pricing and aggregate_markets.

Reference: docs/phase3.md §Market Probability Estimation
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.engine.mc_pricing import aggregate_markets, analytical_pricing

# ---------------------------------------------------------------------------
# Analytical pricing: home goal monotonicity
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.1, max_value=4.0),
    mu_A=st.floats(min_value=0.1, max_value=4.0),
    S_H=st.integers(min_value=0, max_value=6),
    S_A=st.integers(min_value=0, max_value=6),
)
def test_analytical_home_goal_increases_home_win(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Home goal → P(home_win) weakly increases in analytical pricing."""
    P_before = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    P_after = analytical_pricing(mu_H, mu_A, (S_H + 1, S_A))
    assert P_after["home_win"] >= P_before["home_win"] - 1e-10


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.1, max_value=4.0),
    mu_A=st.floats(min_value=0.1, max_value=4.0),
    S_H=st.integers(min_value=0, max_value=6),
    S_A=st.integers(min_value=0, max_value=6),
)
def test_analytical_home_goal_decreases_away_win(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Home goal → P(away_win) weakly decreases in analytical pricing."""
    P_before = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    P_after = analytical_pricing(mu_H, mu_A, (S_H + 1, S_A))
    assert P_after["away_win"] <= P_before["away_win"] + 1e-10


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.1, max_value=4.0),
    mu_A=st.floats(min_value=0.1, max_value=4.0),
    S_H=st.integers(min_value=0, max_value=6),
    S_A=st.integers(min_value=0, max_value=6),
)
def test_analytical_away_goal_increases_away_win(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Away goal → P(away_win) weakly increases in analytical pricing."""
    P_before = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    P_after = analytical_pricing(mu_H, mu_A, (S_H, S_A + 1))
    assert P_after["away_win"] >= P_before["away_win"] - 1e-10


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.1, max_value=4.0),
    mu_A=st.floats(min_value=0.1, max_value=4.0),
    S_H=st.integers(min_value=0, max_value=6),
    S_A=st.integers(min_value=0, max_value=6),
)
def test_analytical_away_goal_decreases_home_win(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Away goal → P(home_win) weakly decreases in analytical pricing."""
    P_before = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    P_after = analytical_pricing(mu_H, mu_A, (S_H, S_A + 1))
    assert P_after["home_win"] <= P_before["home_win"] + 1e-10


# ---------------------------------------------------------------------------
# Analytical pricing: goal → over thresholds increase
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.1, max_value=4.0),
    mu_A=st.floats(min_value=0.1, max_value=4.0),
    S_H=st.integers(min_value=0, max_value=6),
    S_A=st.integers(min_value=0, max_value=6),
)
def test_analytical_goal_increases_over_thresholds(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Any goal → all over thresholds weakly increase."""
    P_before = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    P_home = analytical_pricing(mu_H, mu_A, (S_H + 1, S_A))
    P_away = analytical_pricing(mu_H, mu_A, (S_H, S_A + 1))

    for threshold in ["over_15", "over_25", "over_35"]:
        assert P_home[threshold] >= P_before[threshold] - 1e-10
        assert P_away[threshold] >= P_before[threshold] - 1e-10


# ---------------------------------------------------------------------------
# aggregate_markets: goal monotonicity with large N
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    S_H=st.integers(min_value=0, max_value=4),
    S_A=st.integers(min_value=0, max_value=4),
)
def test_aggregate_home_goal_increases_home_win(
    seed: int, S_H: int, S_A: int
) -> None:
    """Home goal → P(home_win) weakly increases in MC aggregation."""
    N = 5000
    rng = np.random.default_rng(seed)
    # Generate random remaining goals
    rem_H = rng.poisson(1.0, size=N)
    rem_A = rng.poisson(1.0, size=N)

    scores_before = np.column_stack([
        S_H + rem_H, S_A + rem_A,
    ]).astype(np.int32)
    scores_after = np.column_stack([
        S_H + 1 + rem_H, S_A + rem_A,
    ]).astype(np.int32)

    P_before = aggregate_markets(scores_before, (S_H, S_A))
    P_after = aggregate_markets(scores_after, (S_H + 1, S_A))

    assert P_after["home_win"] >= P_before["home_win"] - 1e-10
    assert P_after["away_win"] <= P_before["away_win"] + 1e-10
