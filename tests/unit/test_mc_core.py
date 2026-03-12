"""Unit tests for mc_core.mc_simulate_remaining — Numba JIT MC simulation.

Reference: docs/phase3.md §Logic B: Monte Carlo Pricing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.mc_core import mc_simulate_remaining


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_B = np.zeros(6, dtype=np.float64)
_GAMMA_H = np.zeros(4, dtype=np.float64)
_GAMMA_A = np.zeros(4, dtype=np.float64)
_DELTA_H = np.zeros(5, dtype=np.float64)
_DELTA_A = np.zeros(5, dtype=np.float64)
_Q_DIAG = np.array([-0.01, -0.01, -0.01, -0.01], dtype=np.float64)
_Q_OFF = np.full((4, 4), 1 / 3, dtype=np.float64)
np.fill_diagonal(_Q_OFF, 0.0)
_BASIS = np.array([0, 15, 30, 45, 60, 75, 90], dtype=np.float64)


def _run(
    *,
    t_now: float = 0.0,
    T_end: float = 90.0,
    S_H: int = 0,
    S_A: int = 0,
    state: int = 0,
    score_diff: int = 0,
    a_H: float = 0.0,
    a_A: float = 0.0,
    N: int = 100,
    seed: int = 42,
) -> np.ndarray:
    return mc_simulate_remaining(
        t_now, T_end, S_H, S_A, state, score_diff,
        a_H, a_A, _B, _GAMMA_H, _GAMMA_A, _DELTA_H, _DELTA_A,
        _Q_DIAG, _Q_OFF, _BASIS, N, seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shape() -> None:
    result = _run(N=50)
    assert result.shape == (50, 2)


def test_deterministic_same_seed() -> None:
    r1 = _run(seed=123, N=200)
    r2 = _run(seed=123, N=200)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_differ() -> None:
    r1 = _run(seed=1, N=200)
    r2 = _run(seed=2, N=200)
    # Extremely unlikely to be identical
    assert not np.array_equal(r1, r2)


def test_scores_gte_initial() -> None:
    """Final scores should always be ≥ initial scores."""
    result = _run(S_H=1, S_A=2, N=500, seed=99)
    assert np.all(result[:, 0] >= 1)
    assert np.all(result[:, 1] >= 2)


def test_t_now_gte_t_end_returns_initial() -> None:
    """No time remaining → all paths have initial scores."""
    result = _run(t_now=90.0, T_end=90.0, S_H=2, S_A=1, N=100)
    assert np.all(result[:, 0] == 2)
    assert np.all(result[:, 1] == 1)


def test_int32_dtype() -> None:
    """Output dtype is int32."""
    result = _run(N=10)
    assert result.dtype == np.int32


def test_delta_S_clamping() -> None:
    """Extreme score_diff values are clamped to [0, 4] index."""
    # Should not crash
    r1 = _run(score_diff=-10, N=10)
    r2 = _run(score_diff=10, N=10)
    assert r1.shape == (10, 2)
    assert r2.shape == (10, 2)


def test_goals_possible_over_full_match() -> None:
    """Over 90 minutes with non-zero intensity, some goals should occur."""
    result = _run(a_H=0.3, a_A=0.2, N=1000, seed=42)
    total_goals = result.sum()
    # With positive intensity over 90 min, some goals are expected
    assert total_goals > 0


def test_single_simulation() -> None:
    """N=1 should still work correctly."""
    result = _run(N=1)
    assert result.shape == (1, 2)
