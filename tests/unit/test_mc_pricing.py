"""Unit tests for MC pricing — aggregate_markets, compute_mc_stderr, step_3_4_async."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engine.mc_pricing import (
    N_MC,
    aggregate_markets,
    compute_mc_stderr,
)


# ---------------------------------------------------------------------------
# aggregate_markets — known final_scores
# ---------------------------------------------------------------------------


def test_aggregate_markets_known_scores() -> None:
    """aggregate_markets with [[2,1],[1,0],[0,0],[3,2]] matches expected probs."""
    final_scores = np.array([[2, 1], [1, 0], [0, 0], [3, 2]], dtype=np.int32)
    result = aggregate_markets(final_scores, (0, 0))

    assert result["home_win"] == pytest.approx(0.75, abs=1e-9)   # 3/4
    assert result["draw"] == pytest.approx(0.25, abs=1e-9)        # 1/4 (0-0)
    assert result["away_win"] == pytest.approx(0.0, abs=1e-9)     # 0/4

    assert result["over_25"] == pytest.approx(0.5, abs=1e-9)   # 2-1 (3) and 3-2 (5)
    assert result["btts_yes"] == pytest.approx(0.5, abs=1e-9)  # 2-1 and 3-2


def test_aggregate_markets_over_15() -> None:
    final_scores = np.array([[2, 1], [1, 0], [0, 0], [3, 2]], dtype=np.int32)
    result = aggregate_markets(final_scores, (0, 0))
    # 2+1=3>1, 1+0=1 not >1, 0+0=0 not >1, 3+2=5>1 → 2/4 = 0.5
    assert result["over_15"] == pytest.approx(0.5, abs=1e-9)


def test_aggregate_markets_partition_sum() -> None:
    """home_win + draw + away_win = 1.0 exactly."""
    final_scores = np.array([[2, 1], [1, 0], [0, 0], [3, 2]], dtype=np.int32)
    result = aggregate_markets(final_scores, (0, 0))
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_aggregate_markets_all_probs_in_01() -> None:
    rng = np.random.default_rng(42)
    scores = rng.integers(0, 6, size=(1000, 2))
    result = aggregate_markets(scores.astype(np.int32), (0, 0))
    for market, p in result.items():
        assert 0.0 <= p <= 1.0, f"P({market}) = {p} out of [0,1]"


# ---------------------------------------------------------------------------
# compute_mc_stderr
# ---------------------------------------------------------------------------


def test_compute_mc_stderr_home_win_05() -> None:
    """home_win: sqrt(0.5*0.5/50000) ≈ 0.00224."""
    sigma = compute_mc_stderr({"home_win": 0.5}, N_MC)
    expected = math.sqrt(0.5 * 0.5 / N_MC)
    assert sigma["home_win"] == pytest.approx(expected, rel=1e-6)


def test_compute_mc_stderr_over_25_09() -> None:
    """over_25: sqrt(0.9*0.1/50000) ≈ 0.00134."""
    sigma = compute_mc_stderr({"over_25": 0.9}, N_MC)
    expected = math.sqrt(0.9 * 0.1 / N_MC)
    assert sigma["over_25"] == pytest.approx(expected, rel=1e-6)


def test_compute_mc_stderr_zero_at_boundaries() -> None:
    """σ = 0 for p=0 or p=1."""
    sigma = compute_mc_stderr({"market_a": 0.0, "market_b": 1.0}, N_MC)
    assert sigma["market_a"] == 0.0
    assert sigma["market_b"] == 0.0


def test_compute_mc_stderr_analytical_floor() -> None:
    """Analytical mode applies σ_floor = 0.005."""
    sigma = compute_mc_stderr({"home_win": 0.5, "over_25": 0.9}, N_MC, analytical=True)
    for market, s in sigma.items():
        assert s >= 0.005, f"σ_MC[{market}]={s} below 0.005 floor"


def test_compute_mc_stderr_analytical_floor_edge() -> None:
    """Even when natural σ < 0.005, floor applies in analytical mode."""
    # p very close to 0: natural σ would be tiny
    sigma = compute_mc_stderr({"market": 0.0001}, N_MC, analytical=True)
    assert sigma["market"] >= 0.005


# ---------------------------------------------------------------------------
# aggregate_markets + compute_mc_stderr integration
# ---------------------------------------------------------------------------


def test_aggregate_then_stderr_partition() -> None:
    """aggregate → stderr pipeline: match odds sum ≈ 1.0."""
    rng = np.random.default_rng(99)
    scores = rng.integers(0, 5, size=(10_000, 2)).astype(np.int32)
    P_true = aggregate_markets(scores, (0, 0))

    odds_sum = P_true["home_win"] + P_true["draw"] + P_true["away_win"]
    assert odds_sum == pytest.approx(1.0, abs=0.001)

    sigma = compute_mc_stderr(P_true, 10_000)
    for market, s in sigma.items():
        assert s >= 0.0
