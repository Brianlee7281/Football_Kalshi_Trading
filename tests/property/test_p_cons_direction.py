"""Property: P_cons_yes ≤ P_true ≤ P_cons_no.

The conservative adjustment always pushes against the bettor:
  - BUY_YES uses a lower bound (conservative = less favorable = lower P).
  - BUY_NO uses an upper bound (conservative = less favorable = higher P).

This guarantees P_cons_yes ≤ P_true ≤ P_cons_no for any σ_MC ≥ 0.

Reference: docs/phase4.md Step 4.2
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.execution.edge_detection import compute_conservative_P

# ---------------------------------------------------------------------------
# Directional ordering
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    P_true=st.floats(min_value=0.0, max_value=1.0),
    sigma=st.floats(min_value=0.0, max_value=0.1),
    z=st.floats(min_value=0.0, max_value=5.0),
)
def test_p_cons_yes_le_p_true(P_true: float, sigma: float, z: float) -> None:
    """BUY_YES conservative ≤ P_true (lower confidence bound)."""
    p_cons = compute_conservative_P(P_true, sigma, "BUY_YES", z)
    assert p_cons <= P_true + 1e-12


@settings(max_examples=1000)
@given(
    P_true=st.floats(min_value=0.0, max_value=1.0),
    sigma=st.floats(min_value=0.0, max_value=0.1),
    z=st.floats(min_value=0.0, max_value=5.0),
)
def test_p_cons_no_ge_p_true(P_true: float, sigma: float, z: float) -> None:
    """BUY_NO conservative ≥ P_true (upper confidence bound)."""
    p_cons = compute_conservative_P(P_true, sigma, "BUY_NO", z)
    assert p_cons >= P_true - 1e-12


@settings(max_examples=1000)
@given(
    P_true=st.floats(min_value=0.0, max_value=1.0),
    sigma=st.floats(min_value=0.0, max_value=0.1),
    z=st.floats(min_value=0.0, max_value=5.0),
)
def test_p_cons_yes_le_p_cons_no(P_true: float, sigma: float, z: float) -> None:
    """P_cons_yes ≤ P_cons_no (full sandwich inequality)."""
    p_yes = compute_conservative_P(P_true, sigma, "BUY_YES", z)
    p_no = compute_conservative_P(P_true, sigma, "BUY_NO", z)
    assert p_yes <= p_no + 1e-12


# ---------------------------------------------------------------------------
# σ_MC = 0 → P_cons = P_true
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    P_true=st.floats(min_value=0.0, max_value=1.0),
    direction=st.sampled_from(["BUY_YES", "BUY_NO", "HOLD"]),
)
def test_sigma_zero_identity(P_true: float, direction: str) -> None:
    """When σ_MC = 0, conservative adjustment is identity."""
    p_cons = compute_conservative_P(P_true, 0.0, direction)
    assert abs(p_cons - P_true) < 1e-12


# ---------------------------------------------------------------------------
# Symmetry at P_true = 0.5
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    sigma=st.floats(min_value=0.0, max_value=0.1),
    z=st.floats(min_value=0.0, max_value=5.0),
)
def test_symmetry_at_half(sigma: float, z: float) -> None:
    """BUY_YES and BUY_NO adjustments are equidistant from 0.5."""
    p_yes = compute_conservative_P(0.5, sigma, "BUY_YES", z)
    p_no = compute_conservative_P(0.5, sigma, "BUY_NO", z)
    assert abs((0.5 - p_yes) - (p_no - 0.5)) < 1e-12


# ---------------------------------------------------------------------------
# Monotonicity in sigma
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    P_true=st.floats(min_value=0.0, max_value=1.0),
    sigma_small=st.floats(min_value=0.0, max_value=0.05),
    sigma_extra=st.floats(min_value=0.0, max_value=0.05),
    z=st.floats(min_value=0.0, max_value=5.0),
)
def test_larger_sigma_wider_interval(
    P_true: float, sigma_small: float, sigma_extra: float, z: float
) -> None:
    """Larger σ_MC → wider confidence interval (more conservative)."""
    sigma_large = sigma_small + sigma_extra

    p_yes_small = compute_conservative_P(P_true, sigma_small, "BUY_YES", z)
    p_yes_large = compute_conservative_P(P_true, sigma_large, "BUY_YES", z)
    assert p_yes_large <= p_yes_small + 1e-12

    p_no_small = compute_conservative_P(P_true, sigma_small, "BUY_NO", z)
    p_no_large = compute_conservative_P(P_true, sigma_large, "BUY_NO", z)
    assert p_no_large >= p_no_small - 1e-12
