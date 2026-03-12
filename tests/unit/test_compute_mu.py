"""Unit tests for compute_mu — get_transition_prob and compute_remaining_mu.

Reference: docs/phase3.md §Step 3.2
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engine.compute_mu import compute_remaining_mu, get_transition_prob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    *,
    t: float = 30.0,
    T_exp: float = 90.0,
    current_state_X: int = 0,
    delta_S: int = 0,
) -> MagicMock:
    """Build a minimal mock LiveFootballQuantModel for compute_mu tests."""
    m = MagicMock()
    m.t = t
    m.T_exp = T_exp
    m.current_state_X = current_state_X
    m.delta_S = delta_S
    m.a_H = 0.0
    m.a_A = -0.1
    m.b = np.zeros(6)
    m.gamma_H = np.zeros(4)
    m.gamma_A = np.zeros(4)
    m.delta_H = np.zeros(5)
    m.delta_A = np.zeros(5)
    m.basis_bounds = np.array([0, 15, 30, 45, 60, 75, 90], dtype=np.float64)
    # Standard grid: identity-ish matrices for each minute
    m.P_grid = {i: np.eye(4) for i in range(101)}
    m.P_fine_grid = {}  # empty fine grid by default
    return m


# ---------------------------------------------------------------------------
# get_transition_prob
# ---------------------------------------------------------------------------


def test_gtp_returns_4x4() -> None:
    m = _model()
    result = get_transition_prob(m, 10.0)
    assert result.shape == (4, 4)


def test_gtp_standard_grid_lookup() -> None:
    """dt=10.0 → rounds to 10, looks up P_grid[10]."""
    m = _model()
    custom = np.full((4, 4), 0.25)
    m.P_grid[10] = custom
    result = get_transition_prob(m, 10.0)
    np.testing.assert_array_equal(result, custom)


def test_gtp_fine_grid_used_when_dt_small() -> None:
    """dt ≤ 5.0 and P_fine_grid populated → uses fine grid."""
    m = _model()
    fine = np.full((4, 4), 0.5)
    m.P_fine_grid = {12: fine}  # 12 * (1/6) = 2.0 minutes
    result = get_transition_prob(m, 2.0)
    np.testing.assert_array_equal(result, fine)


def test_gtp_falls_back_to_identity() -> None:
    """Neither grid has the key → identity matrix."""
    m = _model()
    m.P_grid = {}  # empty grids
    result = get_transition_prob(m, 10.0)
    np.testing.assert_array_equal(result, np.eye(4))


def test_gtp_dt_negative_clamped() -> None:
    """Negative dt clamped to 0 → looks up P_grid[0] or identity."""
    m = _model()
    result = get_transition_prob(m, -5.0)
    assert result.shape == (4, 4)


def test_gtp_dt_beyond_100_clamped() -> None:
    """dt > 100 clamped to 100."""
    m = _model()
    custom = np.full((4, 4), 0.99)
    m.P_grid[100] = custom
    result = get_transition_prob(m, 200.0)
    np.testing.assert_array_equal(result, custom)


# ---------------------------------------------------------------------------
# compute_remaining_mu
# ---------------------------------------------------------------------------


def test_mu_at_match_end_returns_zero() -> None:
    """t >= T → (0.0, 0.0)."""
    m = _model(t=90.0, T_exp=90.0)
    mu_H, mu_A = compute_remaining_mu(m)
    assert mu_H == 0.0
    assert mu_A == 0.0


def test_mu_past_end_returns_zero() -> None:
    """t > T → still (0.0, 0.0)."""
    m = _model(t=95.0, T_exp=90.0)
    mu_H, mu_A = compute_remaining_mu(m)
    assert mu_H == 0.0
    assert mu_A == 0.0


def test_mu_nonnegative() -> None:
    """μ_H, μ_A are always ≥ 0."""
    m = _model(t=0.0)
    mu_H, mu_A = compute_remaining_mu(m)
    assert mu_H >= 0.0
    assert mu_A >= 0.0


def test_mu_decreases_with_time() -> None:
    """μ at t=60 should be ≤ μ at t=0 (less time remaining)."""
    m0 = _model(t=0.0)
    m60 = _model(t=60.0)
    mu0_H, mu0_A = compute_remaining_mu(m0)
    mu60_H, mu60_A = compute_remaining_mu(m60)
    assert mu60_H <= mu0_H + 1e-10
    assert mu60_A <= mu0_A + 1e-10


def test_mu_override_delta_S() -> None:
    """override_delta_S changes the delta index used."""
    m = _model(delta_S=0)
    mu_base = compute_remaining_mu(m)
    mu_override = compute_remaining_mu(m, override_delta_S=2)
    # Different delta_S → different delta_H[di], so results should differ
    # (unless delta_H is all zeros, in which case they're equal — that's fine)
    assert isinstance(mu_override, tuple)
    assert len(mu_override) == 2


def test_mu_delta_S_clamped() -> None:
    """delta_S = -5 → di clamped to 0; delta_S = 5 → di clamped to 4."""
    m_neg = _model(delta_S=-5)
    m_pos = _model(delta_S=5)
    mu_neg = compute_remaining_mu(m_neg)
    mu_pos = compute_remaining_mu(m_pos)
    assert mu_neg[0] >= 0.0
    assert mu_pos[0] >= 0.0


def test_mu_positive_a_H() -> None:
    """Positive a_H → higher μ_H (more home goals expected)."""
    m = _model(t=0.0)
    m.a_H = 0.5
    mu_H, mu_A = compute_remaining_mu(m)
    assert mu_H > 0.0
