"""Unit tests for compute_conservative_P — directional conservative adjustment.

Test values come directly from docs/implementation_roadmap.md Sprint 5 spec.

Reference: docs/phase4.md Step 4.2
"""

from __future__ import annotations

import pytest

from src.execution.edge_detection import compute_conservative_P


# ---------------------------------------------------------------------------
# Exact values from roadmap spec
# ---------------------------------------------------------------------------


def test_buy_yes_exact_value() -> None:
    """BUY_YES: P_true=0.55, σ=0.01, z=1.645 → 0.55 - 1.645*0.01 = 0.5336."""
    result = compute_conservative_P(0.55, 0.01, "BUY_YES", z=1.645)
    assert result == pytest.approx(0.55 - 1.645 * 0.01)


def test_buy_no_exact_value() -> None:
    """BUY_NO: P_true=0.55, σ=0.01, z=1.645 → 0.55 + 1.645*0.01 = 0.5664."""
    result = compute_conservative_P(0.55, 0.01, "BUY_NO", z=1.645)
    assert result == pytest.approx(0.55 + 1.645 * 0.01)


# ---------------------------------------------------------------------------
# Directional invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("P_true,sigma", [
    (0.50, 0.005),
    (0.70, 0.010),
    (0.30, 0.020),
    (0.80, 0.001),
])
def test_buy_yes_result_le_p_true(P_true: float, sigma: float) -> None:
    """BUY_YES conservative P is always ≤ P_true (we reduce to be safe)."""
    result = compute_conservative_P(P_true, sigma, "BUY_YES")
    assert result <= P_true


@pytest.mark.parametrize("P_true,sigma", [
    (0.50, 0.005),
    (0.70, 0.010),
    (0.30, 0.020),
    (0.80, 0.001),
])
def test_buy_no_result_ge_p_true(P_true: float, sigma: float) -> None:
    """BUY_NO conservative P is always ≥ P_true (we increase to be safe)."""
    result = compute_conservative_P(P_true, sigma, "BUY_NO")
    assert result >= P_true


def test_buy_yes_and_buy_no_straddle_p_true() -> None:
    """BUY_YES result < P_true < BUY_NO result (they bound P_true from both sides)."""
    p_yes = compute_conservative_P(0.60, 0.015, "BUY_YES")
    p_no = compute_conservative_P(0.60, 0.015, "BUY_NO")
    assert p_yes < 0.60 < p_no


def test_zero_sigma_returns_p_true_for_both_directions() -> None:
    """When sigma_MC=0, conservative P equals P_true for both directions."""
    assert compute_conservative_P(0.55, 0.0, "BUY_YES") == pytest.approx(0.55)
    assert compute_conservative_P(0.55, 0.0, "BUY_NO") == pytest.approx(0.55)


def test_unknown_direction_returns_p_true() -> None:
    """Unrecognised direction falls back to returning P_true unchanged."""
    assert compute_conservative_P(0.55, 0.01, "HOLD") == pytest.approx(0.55)
    assert compute_conservative_P(0.55, 0.01, "") == pytest.approx(0.55)


def test_larger_z_widens_gap() -> None:
    """Higher z increases the distance between BUY_YES and BUY_NO results."""
    gap_1 = compute_conservative_P(0.55, 0.01, "BUY_NO", z=1.0) - \
            compute_conservative_P(0.55, 0.01, "BUY_YES", z=1.0)
    gap_2 = compute_conservative_P(0.55, 0.01, "BUY_NO", z=2.0) - \
            compute_conservative_P(0.55, 0.01, "BUY_YES", z=2.0)
    assert gap_2 > gap_1
