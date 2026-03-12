"""Gap-filling tests for edge_detection — σ_MC=0, EV=0, BUY_NO edge cases.

Supplements test_edge_detection.py with directional correctness and boundary tests.

Reference: docs/phase4.md Step 4.2
"""

from __future__ import annotations

import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.execution.edge_detection import (
    THETA_ENTRY,
    check_market_alignment,
    compute_conservative_P,
    compute_signal_with_vwap,
)
from src.execution.order_book_sync import OrderBookSync

_C = 0.07
_Z = 1.645
_K = 0.25
_BANKROLL = 10_000.0
_DEEP = 100_000


def _make_ob(*, ask_cents: int, bid_cents: int) -> OrderBookSync:
    ob = OrderBookSync("T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T", is_snapshot=True,
        yes=[(ask_cents, _DEEP)],
        no=[(100 - bid_cents, _DEEP)],
        timestamp=time.time(),
    ))
    return ob


# ---------------------------------------------------------------------------
# compute_conservative_P — edge cases
# ---------------------------------------------------------------------------


def test_p_cons_sigma_zero_equals_p_true() -> None:
    """σ_MC=0 → P_cons = P_true for both directions."""
    assert compute_conservative_P(0.6, 0.0, "BUY_YES") == pytest.approx(0.6)
    assert compute_conservative_P(0.6, 0.0, "BUY_NO") == pytest.approx(0.6)


def test_p_cons_buy_yes_lt_p_true() -> None:
    """BUY_YES: P_cons < P_true (lower confidence bound)."""
    p = compute_conservative_P(0.6, 0.01, "BUY_YES")
    assert p < 0.6


def test_p_cons_buy_no_gt_p_true() -> None:
    """BUY_NO: P_cons > P_true (upper confidence bound)."""
    p = compute_conservative_P(0.6, 0.01, "BUY_NO")
    assert p > 0.6


def test_p_cons_boundary_p_true_zero() -> None:
    """P_true=0.0 edge case."""
    p_yes = compute_conservative_P(0.0, 0.01, "BUY_YES")
    p_no = compute_conservative_P(0.0, 0.01, "BUY_NO")
    # BUY_YES: 0 - z*0.01 = negative → unclamped (caller responsible)
    assert p_yes < 0.0
    # BUY_NO: 0 + z*0.01 = positive
    assert p_no > 0.0


def test_p_cons_boundary_p_true_one() -> None:
    """P_true=1.0 edge case."""
    p_yes = compute_conservative_P(1.0, 0.01, "BUY_YES")
    p_no = compute_conservative_P(1.0, 0.01, "BUY_NO")
    assert p_yes < 1.0
    # BUY_NO: 1.0 + z*0.01 > 1.0 → unclamped
    assert p_no > 1.0


def test_p_cons_hold_direction_unchanged() -> None:
    """HOLD direction returns P_true unchanged."""
    assert compute_conservative_P(0.55, 0.01, "HOLD") == pytest.approx(0.55)


def test_p_cons_symmetry_at_half() -> None:
    """At P_true=0.5, BUY_YES and BUY_NO adjustments are equidistant."""
    p_yes = compute_conservative_P(0.5, 0.01, "BUY_YES")
    p_no = compute_conservative_P(0.5, 0.01, "BUY_NO")
    assert pytest.approx(0.5 - p_yes) == p_no - 0.5


# ---------------------------------------------------------------------------
# compute_signal_with_vwap — EV=0 parity and BUY_NO threshold
# ---------------------------------------------------------------------------


def test_ev_zero_at_midmarket_hold() -> None:
    """P_true = midpoint(ask, bid) → both EV near 0 → HOLD."""
    ob = _make_ob(ask_cents=50, bid_cents=48)
    result = compute_signal_with_vwap(0.49, 0.001, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "HOLD"


def test_buy_no_sigma_zero_strong_edge() -> None:
    """BUY_NO with σ_MC=0 should still fire when P_true is low."""
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.20, 0.0, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "BUY_NO"


def test_buy_no_ev_exceeds_theta() -> None:
    """BUY_NO signal has EV > THETA_ENTRY."""
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.20, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_NO":
        assert result.EV > THETA_ENTRY


def test_buy_no_rough_qty_positive() -> None:
    """BUY_NO signal has rough_qty >= 1."""
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.20, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_NO":
        assert result.rough_qty >= 1


# ---------------------------------------------------------------------------
# check_market_alignment — empty order book (P_bet365 edge)
# ---------------------------------------------------------------------------


def test_alignment_equal_prices_buy_yes() -> None:
    """P_cons == P_kalshi → model_says_high is False → DIVERGENT."""
    result = check_market_alignment(0.50, 0.50, 0.52, "BUY_YES")
    assert result.status == "DIVERGENT"


def test_alignment_equal_prices_buy_no() -> None:
    """P_cons == P_kalshi → model_says_low is False → DIVERGENT."""
    result = check_market_alignment(0.50, 0.50, 0.48, "BUY_NO")
    assert result.status == "DIVERGENT"
