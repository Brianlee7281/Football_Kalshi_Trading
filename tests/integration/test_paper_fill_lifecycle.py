"""Integration: Paper fill lifecycle — signal → Kelly → paper executor → position.

Verifies the end-to-end flow from a trading signal through Kelly sizing,
paper execution, and fill result validation.

Reference: docs/phase4.md Steps 4.2–4.5
"""

from __future__ import annotations

import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import PaperFill
from src.execution.edge_detection import generate_signal
from src.execution.kelly import compute_kelly, liquidity_gate
from src.execution.order_book_sync import OrderBookSync
from src.execution.paper_executor import PaperExecutionLayer

from .conftest import make_model, make_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_ob(ticker: str = "T") -> OrderBookSync:
    """Create an order book with deep liquidity at multiple levels."""
    ob = OrderBookSync(ticker=ticker)
    ob.update_from_kalshi(OrderBookUpdate(
        ticker=ticker,
        is_snapshot=True,
        yes=[(50, 500), (51, 500), (52, 500), (53, 500)],
        no=[(50, 500), (49, 500), (48, 500), (47, 500)],
        timestamp=time.time(),
    ))
    return ob


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_buy_yes_full_lifecycle() -> None:
    """BUY_YES: signal → Kelly > 0 → paper fill at ask + slippage."""
    # Low ask (35¢) with P_true=0.65 → strong BUY_YES edge
    # Deep book (100k contracts) so VWAP succeeds for Kelly quantity
    ob = OrderBookSync(ticker="T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(35, 100_000)],
        no=[(65, 100_000)],
        timestamp=time.time(),
    ))


    model = make_model()

    signal = generate_signal(
        P_true=0.65,
        sigma_MC=0.005,
        ob_sync=ob,
        P_bet365=0.60,
        c=0.07,
        z=1.645,
        K_frac=0.25,
        bankroll=10_000.0,
        market_ticker="T",
    )

    assert signal.direction == "BUY_YES", f"Expected BUY_YES, got {signal.direction}"

    f_kelly = compute_kelly(signal, c=0.07, K_frac=0.25, bankroll=10_000.0)
    assert f_kelly > 0.0

    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    amount = f_kelly * 10_000.0
    fill = await layer.execute_order(signal, amount, ob, model, urgent=True)

    assert fill is not None
    assert isinstance(fill, PaperFill)
    assert fill.quantity > 0
    assert fill.is_paper is True
    assert fill.price > 0.0


@pytest.mark.anyio
async def test_buy_no_full_lifecycle() -> None:
    """BUY_NO: signal → Kelly > 0 → paper fill at bid - slippage."""
    # High ask (80¢) with P_true=0.20 → strong BUY_NO edge
    # Deep book so VWAP succeeds for Kelly quantity
    ob = OrderBookSync(ticker="T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(80, 100_000)],
        no=[(22, 100_000)],
        timestamp=time.time(),
    ))


    model = make_model()

    signal = generate_signal(
        P_true=0.20,
        sigma_MC=0.005,
        ob_sync=ob,
        P_bet365=0.25,
        c=0.07,
        z=1.645,
        K_frac=0.25,
        bankroll=10_000.0,
        market_ticker="T",
    )

    assert signal.direction == "BUY_NO", f"Expected BUY_NO, got {signal.direction}"

    f_kelly = compute_kelly(signal, c=0.07, K_frac=0.25, bankroll=10_000.0)
    assert f_kelly > 0.0

    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    fill = await layer.execute_order(signal, f_kelly * 10_000.0, ob, model, urgent=True)

    assert fill is not None
    assert fill.quantity > 0


@pytest.mark.anyio
async def test_ob_freeze_cancels_paper_fill() -> None:
    """ob_freeze during fill delay → None."""
    ob = _deep_ob()
    model = make_model()
    model.ob_freeze = True

    signal = make_signal(direction="BUY_YES", P_kalshi=0.50)
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    fill = await layer.execute_order(signal, 100.0, ob, model, urgent=True)

    assert fill is None


@pytest.mark.anyio
async def test_event_state_cancels_paper_fill() -> None:
    """Non-IDLE event_state during fill → None."""
    ob = _deep_ob()
    model = make_model()
    model.event_state = "PRELIMINARY_DETECTED"

    signal = make_signal(direction="BUY_YES", P_kalshi=0.50)
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    fill = await layer.execute_order(signal, 100.0, ob, model, urgent=True)

    assert fill is None


@pytest.mark.anyio
async def test_hold_signal_skips_execution() -> None:
    """HOLD signal → Kelly returns 0, no execution."""
    signal = make_signal(direction="HOLD", EV=0.0)
    f = compute_kelly(signal, c=0.07, K_frac=0.25)
    assert f == 0.0


@pytest.mark.anyio
async def test_liquidity_gate_blocks_thin_market() -> None:
    """Thin ask depth → liquidity gate blocks the order."""
    ob = OrderBookSync(ticker="T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, 5)],
        no=[(45, 5)],
        timestamp=time.time(),
    ))


    gated_qty, proceed = liquidity_gate(100, ob, "BUY_YES")
    # max_qty = int(0.3 * 5) = 1, 1/100 = 0.01 < 0.5 → skip
    assert proceed is False
    assert gated_qty == 0


@pytest.mark.anyio
async def test_partial_fill_when_depth_limited() -> None:
    """When depth at fill price < target_qty → partial fill."""
    ob = OrderBookSync(ticker="T")
    # 20 at 55¢, 100 at 60¢ → VWAP for 30 contracts succeeds,
    # but with slippage of 2 ticks, fill_price=VWAP+2 ≈ 57¢,
    # only 20 contracts available at ≤ 57¢
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, 20), (60, 100)],
        no=[(45, 1000)],
        timestamp=time.time(),
    ))


    model = make_model()
    signal = make_signal(direction="BUY_YES", P_kalshi=0.55)

    layer = PaperExecutionLayer(slippage_ticks=2, fill_delay_range=(0.0, 0.0))
    # $16.50 / 0.55 = 30 contracts target
    fill = await layer.execute_order(signal, 16.50, ob, model, urgent=True)

    assert fill is not None
    assert fill.quantity <= 20  # Only 20 at ≤ fill price
    assert fill.partial is True


@pytest.mark.anyio
async def test_fill_price_buy_yes_slippage() -> None:
    """BUY_YES fill price = VWAP + slippage (adverse)."""
    ob = OrderBookSync(ticker="T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, 1000)],
        no=[(45, 1000)],
        timestamp=time.time(),
    ))


    model = make_model()
    signal = make_signal(direction="BUY_YES", P_kalshi=0.55)

    layer = PaperExecutionLayer(slippage_ticks=2, fill_delay_range=(0.0, 0.0))
    fill = await layer.execute_order(signal, 100.0, ob, model, urgent=True)

    assert fill is not None
    # VWAP = 55¢ = 0.55, slippage +2¢ → 0.57
    assert fill.price == pytest.approx(0.57, abs=0.01)


@pytest.mark.anyio
async def test_fill_price_buy_no_slippage() -> None:
    """BUY_NO fill price = VWAP - slippage (adverse)."""
    ob = OrderBookSync(ticker="T")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, 1000)],
        no=[(45, 1000)],
        timestamp=time.time(),
    ))


    model = make_model()
    signal = make_signal(direction="BUY_NO", P_kalshi=0.55)

    layer = PaperExecutionLayer(slippage_ticks=2, fill_delay_range=(0.0, 0.0))
    fill = await layer.execute_order(signal, 100.0, ob, model, urgent=True)

    assert fill is not None
    # VWAP sell at 55¢ = 0.55, slippage -2¢ → 0.53
    assert fill.price == pytest.approx(0.53, abs=0.01)
