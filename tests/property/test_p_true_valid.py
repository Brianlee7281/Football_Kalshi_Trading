"""Property: P_true ∈ [0, 1] for any valid match state.

Both analytical_pricing and aggregate_markets must return probabilities
in the closed interval [0, 1] for all seven markets, regardless of input.

Reference: docs/phase4.md Step 4.2, docs/phase3.md §Market Probability Estimation
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.engine.mc_pricing import (
    aggregate_markets,
    analytical_pricing,
    compute_mc_stderr,
)

MARKETS = ["home_win", "draw", "away_win", "over_15", "over_25", "over_35", "btts_yes"]


# ---------------------------------------------------------------------------
# analytical_pricing
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.01, max_value=5.0),
    mu_A=st.floats(min_value=0.01, max_value=5.0),
    S_H=st.integers(min_value=0, max_value=8),
    S_A=st.integers(min_value=0, max_value=8),
)
def test_analytical_pricing_in_unit_interval(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """Every market probability from analytical_pricing is in [0, 1]."""
    P = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    for market in MARKETS:
        # Allow tiny floating-point overshoot from Poisson convolution sums
        assert -1e-12 <= P[market] <= 1.0 + 1e-12, f"{market}={P[market]} out of [0,1]"


@settings(max_examples=1000)
@given(
    mu_H=st.floats(min_value=0.01, max_value=5.0),
    mu_A=st.floats(min_value=0.01, max_value=5.0),
    S_H=st.integers(min_value=0, max_value=8),
    S_A=st.integers(min_value=0, max_value=8),
)
def test_analytical_result_probabilities_sum(
    mu_H: float, mu_A: float, S_H: int, S_A: int
) -> None:
    """home_win + draw + away_win ≈ 1.0 (partition of outcomes).

    Tolerance is 1e-3 because Poisson truncation at max_goals=15 loses
    mass when mu is large (e.g. mu_A=4.0 → P(remaining>15) > 0).
    """
    P = analytical_pricing(mu_H, mu_A, (S_H, S_A))
    total = P["home_win"] + P["draw"] + P["away_win"]
    assert abs(total - 1.0) < 1e-3, f"result partition = {total}"


# ---------------------------------------------------------------------------
# aggregate_markets (MC path)
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    N=st.integers(min_value=10, max_value=500),
    max_home=st.integers(min_value=0, max_value=6),
    max_away=st.integers(min_value=0, max_value=6),
)
def test_aggregate_markets_in_unit_interval(
    N: int, max_home: int, max_away: int
) -> None:
    """Every market probability from aggregate_markets is in [0, 1]."""
    rng = np.random.default_rng(42)
    final_scores = np.column_stack([
        rng.integers(0, max_home + 1, size=N),
        rng.integers(0, max_away + 1, size=N),
    ]).astype(np.int32)
    P = aggregate_markets(final_scores, (0, 0))
    for market in MARKETS:
        assert 0.0 <= P[market] <= 1.0, f"{market}={P[market]} out of [0,1]"


@settings(max_examples=1000)
@given(
    N=st.integers(min_value=10, max_value=500),
    max_home=st.integers(min_value=0, max_value=6),
    max_away=st.integers(min_value=0, max_value=6),
)
def test_aggregate_result_probabilities_sum(
    N: int, max_home: int, max_away: int
) -> None:
    """home_win + draw + away_win ≈ 1.0 from MC aggregation."""
    rng = np.random.default_rng(42)
    final_scores = np.column_stack([
        rng.integers(0, max_home + 1, size=N),
        rng.integers(0, max_away + 1, size=N),
    ]).astype(np.int32)
    P = aggregate_markets(final_scores, (0, 0))
    total = P["home_win"] + P["draw"] + P["away_win"]
    assert abs(total - 1.0) < 1e-10, f"result partition = {total}"


# ---------------------------------------------------------------------------
# compute_mc_stderr
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    p=st.floats(min_value=0.0, max_value=1.0),
    N=st.integers(min_value=100, max_value=100_000),
    analytical=st.booleans(),
)
def test_mc_stderr_non_negative(p: float, N: int, analytical: bool) -> None:
    """σ_MC is always ≥ 0 for any valid probability and sample size."""
    P_true = {"m": p}
    sigma = compute_mc_stderr(P_true, N, analytical=analytical)
    assert sigma["m"] >= 0.0
