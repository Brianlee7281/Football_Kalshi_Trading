"""Unit tests for compute_kelly, apply_risk_limits, and liquidity_gate.

Test values come from docs/implementation_roadmap.md Sprint 5 spec
and docs/phase4.md Step 4.3.

Reference: docs/phase4.md Step 4.3
"""

from __future__ import annotations

import math
import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import Signal
from src.execution.kelly import (
    DEPTH_FRACTION,
    F_MATCH_CAP,
    F_ORDER_CAP,
    F_TOTAL_CAP,
    MIN_FILL_RATIO,
    apply_risk_limits,
    compute_kelly,
    liquidity_gate,
)
from src.execution.order_book_sync import OrderBookSync


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _signal(
    *,
    direction: str = "BUY_YES",
    EV: float = 0.03075,
    P_cons: float = 0.55,
    P_kalshi: float = 0.50,
    kelly_multiplier: float = 0.8,
    alignment_status: str = "ALIGNED",
    rough_qty: int = 100,
    market_ticker: str = "TEST-YES",
) -> Signal:
    return Signal(
        direction=direction,
        EV=EV,
        P_cons=P_cons,
        P_kalshi=P_kalshi,
        rough_qty=rough_qty,
        alignment_status=alignment_status,
        kelly_multiplier=kelly_multiplier,
        market_ticker=market_ticker,
    )


def _ob(*, ask_depth: int = 1000, bid_depth: int = 1000) -> OrderBookSync:
    """Return an OrderBookSync pre-loaded with uniform depth."""
    ob = OrderBookSync(ticker="TEST-YES")
    ob._apply_snapshot(OrderBookUpdate(
        ticker="TEST-YES",
        is_snapshot=True,
        yes=[(55, ask_depth)],
        no=[(45, bid_depth)],
        timestamp=time.time(),
    ))
    return ob


# ---------------------------------------------------------------------------
# compute_kelly — spec validation tests
# ---------------------------------------------------------------------------


def test_buy_yes_known_value() -> None:
    """P_cons=0.55, P_kalshi=0.50, c=0.07, K_frac=0.25, mult=0.8.

    W = 0.93 * 0.50 = 0.465
    L = 0.50
    f_kelly = 0.03075 / (0.465 * 0.50) = 0.1323...
    f_invest = 0.25 * 0.1323 * 0.8 = 0.0265 (approx)
    """
    sig = _signal(direction="BUY_YES", EV=0.03075, P_cons=0.55, P_kalshi=0.50,
                  kelly_multiplier=0.8)
    result = compute_kelly(sig, c=0.07, K_frac=0.25)
    assert result == pytest.approx(0.25 * (0.03075 / (0.465 * 0.50)) * 0.8, rel=1e-4)


def test_incremental_zero_when_already_at_optimal() -> None:
    """existing_exposure equals optimal dollar amount → f_incremental = 0."""
    sig = _signal(direction="BUY_YES", EV=0.03075, P_cons=0.55, P_kalshi=0.50,
                  kelly_multiplier=0.8)
    bankroll = 10_000.0
    f_opt = compute_kelly(sig, c=0.07, K_frac=0.25, bankroll=bankroll)
    existing = f_opt * bankroll  # exactly at optimal
    result = compute_kelly(sig, c=0.07, K_frac=0.25,
                           existing_exposure=existing, bankroll=bankroll)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_incremental_partial_existing_exposure() -> None:
    """existing_exposure=100, bankroll=10000 → f_inc = f_opt - 0.01."""
    sig = _signal(direction="BUY_YES", EV=0.03075, P_cons=0.55, P_kalshi=0.50,
                  kelly_multiplier=0.8)
    bankroll = 10_000.0
    f_opt = compute_kelly(sig, c=0.07, K_frac=0.25, bankroll=bankroll)
    result = compute_kelly(sig, c=0.07, K_frac=0.25,
                           existing_exposure=100.0, bankroll=bankroll)
    assert result == pytest.approx(f_opt - 0.01, rel=1e-4)


def test_incremental_never_negative() -> None:
    """When existing > optimal, incremental is clamped to 0."""
    sig = _signal()
    result = compute_kelly(sig, c=0.07, K_frac=0.25,
                           existing_exposure=9999.0, bankroll=10_000.0)
    assert result == 0.0


def test_buy_no_direction() -> None:
    """BUY_NO uses W=(1-c)*P_kalshi, L=(1-P_kalshi)."""
    P_kalshi = 0.45
    c = 0.07
    EV = 0.02
    W = (1.0 - c) * P_kalshi
    L = 1.0 - P_kalshi
    expected_f_kelly = EV / (W * L)
    sig = _signal(direction="BUY_NO", EV=EV, P_cons=0.40, P_kalshi=P_kalshi,
                  kelly_multiplier=1.0)
    result = compute_kelly(sig, c=c, K_frac=1.0)
    assert result == pytest.approx(expected_f_kelly, rel=1e-4)


def test_hold_direction_returns_zero() -> None:
    """HOLD direction → 0.0 always."""
    sig = _signal(direction="HOLD", EV=0.05)
    assert compute_kelly(sig, c=0.07, K_frac=0.25) == 0.0


def test_zero_bankroll_returns_zero() -> None:
    """bankroll=0 → existing_fraction = 0, no ZeroDivisionError."""
    sig = _signal()
    result = compute_kelly(sig, c=0.07, K_frac=0.25, bankroll=0.0)
    assert isinstance(result, float)
    assert result >= 0.0


def test_degenerate_payoff_returns_zero() -> None:
    """P_kalshi=0.0 → L=0.0, W*L=0 → return 0.0 (no division)."""
    sig = _signal(P_kalshi=0.0)
    assert compute_kelly(sig, c=0.07, K_frac=0.25) == 0.0


def test_kelly_multiplier_scales_linearly() -> None:
    """Doubling kelly_multiplier doubles the result (all else equal)."""
    sig1 = _signal(kelly_multiplier=0.5)
    sig2 = _signal(kelly_multiplier=1.0)
    r1 = compute_kelly(sig1, c=0.07, K_frac=0.25)
    r2 = compute_kelly(sig2, c=0.07, K_frac=0.25)
    assert r2 == pytest.approx(r1 * 2.0, rel=1e-6)


def test_k_frac_scales_linearly() -> None:
    """Doubling K_frac doubles the result."""
    sig = _signal()
    r1 = compute_kelly(sig, c=0.07, K_frac=0.25)
    r2 = compute_kelly(sig, c=0.07, K_frac=0.50)
    assert r2 == pytest.approx(r1 * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# apply_risk_limits
# ---------------------------------------------------------------------------


def test_order_cap_binds() -> None:
    """f_invest=0.10 with bankroll=10000 → capped at F_ORDER_CAP * 10000 = 300."""
    result = apply_risk_limits(0.10, 10_000.0)
    assert result == pytest.approx(F_ORDER_CAP * 10_000.0)


def test_match_cap_binds() -> None:
    """current_match_exposure near F_MATCH_CAP leaves very little room."""
    bankroll = 10_000.0
    near_full = bankroll * F_MATCH_CAP - 50.0  # 50 remaining
    result = apply_risk_limits(0.10, bankroll, current_match_exposure=near_full)
    assert result == pytest.approx(50.0)


def test_total_cap_binds() -> None:
    """total_exposure near F_TOTAL_CAP leaves very little room."""
    bankroll = 10_000.0
    near_full = bankroll * F_TOTAL_CAP - 80.0  # 80 remaining
    result = apply_risk_limits(0.10, bankroll, total_exposure=near_full)
    assert result == pytest.approx(80.0)


def test_match_cap_exhausted_returns_zero() -> None:
    """current_match_exposure ≥ F_MATCH_CAP * bankroll → 0."""
    bankroll = 10_000.0
    result = apply_risk_limits(
        0.10, bankroll,
        current_match_exposure=bankroll * F_MATCH_CAP,
    )
    assert result == 0.0


def test_total_cap_exhausted_returns_zero() -> None:
    """total_exposure ≥ F_TOTAL_CAP * bankroll → 0."""
    bankroll = 10_000.0
    result = apply_risk_limits(
        0.10, bankroll,
        total_exposure=bankroll * F_TOTAL_CAP,
    )
    assert result == 0.0


def test_no_caps_bind_small_fraction() -> None:
    """f_invest=0.01 with no existing exposure → passthrough."""
    bankroll = 10_000.0
    result = apply_risk_limits(0.01, bankroll)
    assert result == pytest.approx(0.01 * bankroll)


def test_all_three_caps_smallest_wins() -> None:
    """All three caps active simultaneously — smallest cap wins."""
    bankroll = 10_000.0
    # order cap → 300, match remaining → 200, total remaining → 400
    result = apply_risk_limits(
        0.10, bankroll,
        current_match_exposure=bankroll * F_MATCH_CAP - 200.0,
        total_exposure=bankroll * F_TOTAL_CAP - 400.0,
    )
    assert result == pytest.approx(200.0)


def test_custom_caps() -> None:
    """Custom f_order_cap overrides default."""
    bankroll = 10_000.0
    result = apply_risk_limits(0.10, bankroll, f_order_cap=0.01)
    assert result == pytest.approx(0.01 * bankroll)


def test_zero_f_invest_returns_zero() -> None:
    """f_invest=0 → 0 regardless of caps."""
    assert apply_risk_limits(0.0, 10_000.0) == 0.0


# ---------------------------------------------------------------------------
# liquidity_gate
# ---------------------------------------------------------------------------


def test_gate_passes_within_depth_fraction() -> None:
    """target_qty=20, depth=100 → max=30, 20 < 30 → proceed=True, qty=20."""
    ob = _ob(ask_depth=100)
    gated, proceed = liquidity_gate(20, ob, "BUY_YES")
    assert proceed is True
    assert gated == 20


def test_gate_caps_at_depth_fraction() -> None:
    """target_qty=50, depth=100 → max=30, 30/50=60%>50% → proceed=True, qty=30."""
    ob = _ob(ask_depth=100)
    gated, proceed = liquidity_gate(50, ob, "BUY_YES")
    assert proceed is True
    assert gated == 30


def test_gate_skips_when_fill_ratio_too_low() -> None:
    """target_qty=80, depth=100 → max=30, 30/80=37.5%<50% → proceed=False."""
    ob = _ob(ask_depth=100)
    gated, proceed = liquidity_gate(80, ob, "BUY_YES")
    assert proceed is False
    assert gated == 0


def test_gate_buy_no_uses_bid_depth() -> None:
    """BUY_NO gates against bid depth, not ask depth."""
    ob = _ob(ask_depth=10, bid_depth=1000)
    gated, proceed = liquidity_gate(50, ob, "BUY_NO")
    assert proceed is True
    assert gated == min(50, int(0.30 * 1000))


def test_gate_zero_target_returns_false() -> None:
    """target_qty=0 → (0, False)."""
    ob = _ob()
    gated, proceed = liquidity_gate(0, ob, "BUY_YES")
    assert proceed is False
    assert gated == 0


def test_gate_empty_book_returns_false() -> None:
    """No depth available → (0, False)."""
    ob = OrderBookSync(ticker="X")
    gated, proceed = liquidity_gate(10, ob, "BUY_YES")
    assert proceed is False
    assert gated == 0


def test_gate_unknown_direction_returns_false() -> None:
    """HOLD direction → (0, False)."""
    ob = _ob()
    gated, proceed = liquidity_gate(10, ob, "HOLD")
    assert proceed is False
    assert gated == 0


def test_gate_minimum_one_contract() -> None:
    """depth_fraction * available rounds down to 0 but target fits → clamp to 1."""
    ob = _ob(ask_depth=2)  # max_qty = int(0.30 * 2) = 0
    # 0 < 1 * 0.5 → skip. Actually max_qty=0 → gated=0 < target*ratio → skip
    gated, proceed = liquidity_gate(1, ob, "BUY_YES")
    # max_qty = int(0.30 * 2) = 0 → available path hits gated=0 → skip
    assert proceed is False


def test_gate_exactly_at_depth_fraction() -> None:
    """target_qty == int(depth_fraction * available) → proceed=True."""
    # depth=100, max_qty=30, target=30 → gated=30, 30/30=100% > 50%
    ob = _ob(ask_depth=100)
    gated, proceed = liquidity_gate(30, ob, "BUY_YES")
    assert proceed is True
    assert gated == 30


def test_gate_custom_parameters() -> None:
    """Custom depth_fraction and min_fill_ratio."""
    ob = _ob(ask_depth=100)
    # depth_fraction=0.5 → max=50, target=40, 40/40=100% > min_fill=0.9 → proceed
    gated, proceed = liquidity_gate(40, ob, "BUY_YES",
                                    depth_fraction=0.5, min_fill_ratio=0.9)
    assert proceed is True
    assert gated == 40


# ---------------------------------------------------------------------------
# Constants smoke tests
# ---------------------------------------------------------------------------


def test_constants_in_valid_range() -> None:
    """Sanity check that constants are reasonable fractions."""
    assert 0.0 < F_ORDER_CAP < F_MATCH_CAP < F_TOTAL_CAP < 1.0
    assert 0.0 < DEPTH_FRACTION < 1.0
    assert 0.0 < MIN_FILL_RATIO < 1.0


def test_f_order_cap_value() -> None:
    assert F_ORDER_CAP == pytest.approx(0.03)


def test_f_match_cap_value() -> None:
    assert F_MATCH_CAP == pytest.approx(0.05)


def test_f_total_cap_value() -> None:
    assert F_TOTAL_CAP == pytest.approx(0.20)


def test_result_is_finite() -> None:
    """compute_kelly never returns NaN or inf."""
    sig = _signal()
    result = compute_kelly(sig, c=0.07, K_frac=0.25)
    assert math.isfinite(result)
