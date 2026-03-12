"""Gap-filling tests for kelly — BUY_NO W*L=0 edge cases, directional coverage.

Supplements test_kelly.py with degenerate payoff tests for BUY_NO direction.

Reference: docs/phase4.md Step 4.3
"""

from __future__ import annotations

import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import Signal
from src.execution.kelly import compute_kelly, liquidity_gate
from src.execution.order_book_sync import OrderBookSync


def _signal(
    *,
    direction: str = "BUY_NO",
    EV: float = 0.02,
    P_kalshi: float = 0.45,
    kelly_multiplier: float = 1.0,
) -> Signal:
    return Signal(
        direction=direction,
        EV=EV,
        P_cons=0.40,
        P_kalshi=P_kalshi,
        rough_qty=100,
        alignment_status="ALIGNED",
        kelly_multiplier=kelly_multiplier,
        market_ticker="T",
    )


def _ob(*, ask_depth: int = 100, bid_depth: int = 100) -> OrderBookSync:
    ob = OrderBookSync(ticker="T")
    ob._apply_snapshot(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, ask_depth)],
        no=[(45, bid_depth)],
        timestamp=time.time(),
    ))
    return ob


# ---------------------------------------------------------------------------
# BUY_NO degenerate payoffs (W*L=0)
# ---------------------------------------------------------------------------


def test_buy_no_p_kalshi_1_returns_zero() -> None:
    """BUY_NO with P_kalshi=1.0 → L = 1-1.0 = 0 → W*L=0 → 0.0."""
    sig = _signal(P_kalshi=1.0)
    assert compute_kelly(sig, c=0.07, K_frac=0.25) == 0.0


def test_buy_no_p_kalshi_0_returns_zero() -> None:
    """BUY_NO with P_kalshi=0.0 → W = (1-c)*0 = 0 → W*L=0 → 0.0."""
    sig = _signal(P_kalshi=0.0)
    assert compute_kelly(sig, c=0.07, K_frac=0.25) == 0.0


def test_buy_no_with_low_ev() -> None:
    """BUY_NO with small EV → small but positive fraction."""
    sig = _signal(EV=0.001, P_kalshi=0.45, kelly_multiplier=0.8)
    result = compute_kelly(sig, c=0.07, K_frac=0.25)
    assert result >= 0.0
    assert result < 0.01  # Very small


def test_buy_no_with_existing_exposure_near_optimal() -> None:
    """BUY_NO: existing_exposure near optimal → incremental ≈ 0."""
    sig = _signal(EV=0.02, P_kalshi=0.45, kelly_multiplier=0.8)
    bankroll = 10_000.0
    f_opt = compute_kelly(sig, c=0.07, K_frac=0.25, bankroll=bankroll)
    existing = f_opt * bankroll * 0.99  # 99% there
    result = compute_kelly(sig, c=0.07, K_frac=0.25,
                           existing_exposure=existing, bankroll=bankroll)
    assert result >= 0.0
    assert result < f_opt


# ---------------------------------------------------------------------------
# liquidity_gate — directional depth
# ---------------------------------------------------------------------------


def test_gate_buy_no_empty_bid_returns_false() -> None:
    """BUY_NO with zero bid depth → (0, False)."""
    ob = _ob(ask_depth=1000, bid_depth=0)
    gated, proceed = liquidity_gate(10, ob, "BUY_NO")
    assert proceed is False
    assert gated == 0


def test_gate_buy_yes_empty_ask_returns_false() -> None:
    """BUY_YES with zero ask depth → (0, False)."""
    ob = OrderBookSync(ticker="T")
    gated, proceed = liquidity_gate(10, ob, "BUY_YES")
    assert proceed is False
    assert gated == 0


def test_gate_asymmetric_depth() -> None:
    """Asymmetric: thin ask, deep bid — BUY_YES fails, BUY_NO succeeds."""
    ob = _ob(ask_depth=5, bid_depth=500)
    _, proceed_yes = liquidity_gate(10, ob, "BUY_YES")
    _, proceed_no = liquidity_gate(10, ob, "BUY_NO")
    # ask=5, max_qty=int(0.3*5)=1, 1/10=0.1 < 0.5 → skip
    assert proceed_yes is False
    # bid=500, max_qty=int(0.3*500)=150, 10<150 → proceed
    assert proceed_no is True
