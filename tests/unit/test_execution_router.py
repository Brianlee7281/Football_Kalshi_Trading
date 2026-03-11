"""Unit tests for ExecutionRouter, PaperExecutionLayer, and LiveExecutionLayer.

Reference: docs/phase4.md Step 4.5
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import FillResult, PaperFill, Signal
from src.execution.execution_router import ExecutionRouter
from src.execution.live_executor import LiveExecutionLayer
from src.execution.order_book_sync import OrderBookSync
from src.execution.paper_executor import PaperExecutionLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _signal(
    *,
    direction: str = "BUY_YES",
    EV: float = 0.03,
    P_cons: float = 0.55,
    P_kalshi: float = 0.55,
    kelly_multiplier: float = 0.8,
    alignment_status: str = "ALIGNED",
    rough_qty: int = 10,
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
    """Return a fresh (non-stale) OrderBookSync with uniform depth."""
    ob = OrderBookSync(ticker="TEST-YES")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="TEST-YES",
        is_snapshot=True,
        yes=[(55, ask_depth)],
        no=[(45, bid_depth)],
        timestamp=time.time(),
    ))
    return ob


def _ob_multilevel() -> OrderBookSync:
    """Ask levels: 55¢ × 50 and 70¢ × 1000.  Used for partial-fill tests."""
    ob = OrderBookSync(ticker="TEST-YES")
    ob.update_from_kalshi(OrderBookUpdate(
        ticker="TEST-YES",
        is_snapshot=True,
        yes=[(55, 50), (70, 1000)],
        no=[(45, 1000)],
        timestamp=time.time(),
    ))
    return ob


def _model(*, ob_freeze: bool = False, event_state: str = "IDLE") -> MagicMock:
    model = MagicMock()
    model.ob_freeze = ob_freeze
    model.event_state = event_state
    model.match_id = "test-match-001"
    return model


# ---------------------------------------------------------------------------
# PaperExecutionLayer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_buy_yes_slippage_positive() -> None:
    """BUY_YES fill price = VWAP + slip (adverse = higher price)."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    ob = _ob(ask_depth=1000)
    model = _model()

    with patch("asyncio.sleep"):
        fill = await layer.execute_order(_signal(direction="BUY_YES", P_kalshi=0.55),
                                         amount=55.0, ob_sync=ob, model=model)

    assert fill is not None
    assert isinstance(fill, PaperFill)
    # VWAP at 55¢ → P_effective = 0.55, slip = 0.01 → fill_price = 0.56
    assert fill.price == pytest.approx(0.56, abs=1e-4)
    assert fill.slippage == pytest.approx(0.01, abs=1e-4)
    assert fill.is_paper is True


@pytest.mark.asyncio
async def test_paper_buy_no_slippage_negative() -> None:
    """BUY_NO fill price = VWAP - slip (adverse = lower price for seller).

    Bid stored as 100 - no_price: no=45 → bid at 55¢.
    VWAP sell returns 55¢ → P_effective = 0.55.
    Adverse slip: fill_price = 0.55 - 0.01 = 0.54.
    """
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    ob = _ob(bid_depth=1000)
    model = _model()

    with patch("asyncio.sleep"):
        fill = await layer.execute_order(
            _signal(direction="BUY_NO", P_kalshi=0.55),
            amount=55.0, ob_sync=ob, model=model,
        )

    assert fill is not None
    # bid VWAP at 55¢ → P_effective=0.55, BUY_NO slip= -0.01 → fill=0.54
    assert fill.price == pytest.approx(0.54, abs=1e-4)
    assert fill.slippage == pytest.approx(0.01, abs=1e-4)


@pytest.mark.asyncio
async def test_paper_aborts_on_ob_freeze() -> None:
    """If ob_freeze is set during wait, order is cancelled → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.001, 0.001))
    ob = _ob()
    model = _model(ob_freeze=True)  # already frozen

    fill = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob, model=model)
    assert fill is None


@pytest.mark.asyncio
async def test_paper_aborts_on_event_state_not_idle() -> None:
    """If event_state != IDLE during wait, order is cancelled → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.001, 0.001))
    ob = _ob()
    model = _model(event_state="PRELIMINARY_DETECTED")

    fill = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob, model=model)
    assert fill is None


@pytest.mark.asyncio
async def test_paper_partial_fill_when_depth_insufficient() -> None:
    """Multi-level book: VWAP spans both levels, but fill_price clears only lower level.

    Setup: 55¢ × 50 contracts, 70¢ × 1000 contracts.
    target_qty = 60 → VWAP uses 50@55 + 10@70 = 3450/60 = 57.5¢.
    P_effective = 0.575, fill_price = 0.585, fill_price_cents = round(58.5) = 58.
    Depth at price ≤ 58: only 55¢ level → 50 contracts.
    filled_qty = min(60, 50) = 50 → partial=True.
    """
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    ob = _ob_multilevel()
    model = _model()

    # amount = 60 * 0.575 = 34.5 (target_qty = int(34.5/0.575) = 60)
    with patch("asyncio.sleep"):
        fill = await layer.execute_order(
            _signal(direction="BUY_YES", P_kalshi=0.575),
            amount=34.5, ob_sync=ob, model=model,
        )

    assert fill is not None
    assert fill.partial is True
    assert fill.quantity == 50


@pytest.mark.asyncio
async def test_paper_zero_target_qty_returns_none() -> None:
    """amount too small for even 1 contract → None."""
    layer = PaperExecutionLayer(fill_delay_range=(0.0, 0.0))
    ob = _ob()
    model = _model()

    fill = await layer.execute_order(
        _signal(P_kalshi=0.55), amount=0.10, ob_sync=ob, model=model
    )
    assert fill is None


@pytest.mark.asyncio
async def test_paper_fill_delay_in_range() -> None:
    """fill_delay is within the configured range."""
    layer = PaperExecutionLayer(fill_delay_range=(0.0, 0.0))
    ob = _ob()
    model = _model()

    with patch("asyncio.sleep"):
        fill = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob, model=model)

    assert fill is not None
    assert 0.0 <= fill.fill_delay <= 3.0


@pytest.mark.asyncio
async def test_paper_urgent_skips_delay() -> None:
    """urgent=True: asyncio.sleep is not called."""
    layer = PaperExecutionLayer(fill_delay_range=(1.0, 3.0))
    ob = _ob()
    model = _model()

    with patch("asyncio.sleep") as mock_sleep:
        fill = await layer.execute_order(
            _signal(), amount=55.0, ob_sync=ob, model=model, urgent=True
        )

    mock_sleep.assert_not_called()
    assert fill is not None
    assert fill.fill_delay == 0.0


@pytest.mark.asyncio
async def test_paper_returns_none_when_depth_empty() -> None:
    """Empty order book (no depth) → None."""
    layer = PaperExecutionLayer(fill_delay_range=(0.0, 0.0))
    ob = OrderBookSync(ticker="EMPTY")  # no depth
    model = _model()

    with patch("asyncio.sleep"):
        fill = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob, model=model)

    assert fill is None


# ---------------------------------------------------------------------------
# ExecutionRouter tests
# ---------------------------------------------------------------------------


def test_router_paper_mode_creates_paper_layer() -> None:
    """Paper mode initializes PaperExecutionLayer, not live."""
    model = _model()
    router = ExecutionRouter("paper", model)
    assert router.mode == "paper"
    assert router.paper is not None
    assert router.live is None


def test_router_live_mode_requires_kalshi_client() -> None:
    """Live mode without kalshi_client raises ValueError."""
    model = _model()
    with pytest.raises(ValueError, match="kalshi_client"):
        ExecutionRouter("live", model)


def test_router_unknown_mode_raises() -> None:
    """Unknown trading_mode raises ValueError."""
    model = _model()
    with pytest.raises(ValueError, match="Unknown trading_mode"):
        ExecutionRouter("sandbox", model)


def test_router_live_mode_with_client() -> None:
    """Live mode with kalshi_client initializes LiveExecutionLayer."""
    model = _model()
    kalshi = MagicMock()
    router = ExecutionRouter("live", model, kalshi_client=kalshi)
    assert router.live is not None
    assert router.paper is None


@pytest.mark.asyncio
async def test_router_hold_signal_returns_none() -> None:
    """HOLD direction skips execution and returns None."""
    model = _model()
    router = ExecutionRouter("paper", model)
    ob = _ob()

    result = await router.submit_order(
        _signal(direction="HOLD"), amount=100.0, ob_sync=ob
    )
    assert result is None


@pytest.mark.asyncio
async def test_router_paper_delegates_to_paper_layer() -> None:
    """Paper router calls PaperExecutionLayer.execute_order."""
    model = _model()
    router = ExecutionRouter("paper", model, fill_delay_range=(0.0, 0.0))
    ob = _ob()

    with patch("asyncio.sleep"):
        result = await router.submit_order(_signal(), amount=55.0, ob_sync=ob)

    assert result is not None
    assert isinstance(result, PaperFill)


@pytest.mark.asyncio
async def test_router_live_delegates_to_live_layer() -> None:
    """Live router calls LiveExecutionLayer.execute_order."""
    model = _model()
    kalshi = MagicMock()
    router = ExecutionRouter("live", model, kalshi_client=kalshi)

    mock_live = AsyncMock(return_value=FillResult(
        success=True, fill_price=0.55, fill_quantity=10, order_id="abc123"
    ))
    assert router.live is not None
    router.live.execute_order = mock_live  # type: ignore[method-assign]

    ob = _ob()
    result = await router.submit_order(_signal(), amount=55.0, ob_sync=ob)

    mock_live.assert_called_once()
    assert isinstance(result, FillResult)


# ---------------------------------------------------------------------------
# LiveExecutionLayer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_stale_book_returns_none() -> None:
    """Stale order book → skip order, return None."""
    kalshi = MagicMock()
    layer = LiveExecutionLayer(kalshi)
    ob = OrderBookSync(ticker="X")  # no data → stale

    result = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob)
    assert result is None


@pytest.mark.asyncio
async def test_live_no_ask_returns_none() -> None:
    """No best_ask available → None."""
    kalshi = MagicMock()
    layer = LiveExecutionLayer(kalshi)

    ob = OrderBookSync(ticker="X")
    # Force fresh timestamp but no ask (bid-only snapshot)
    ob.kalshi_last_update = time.monotonic()

    result = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob)
    assert result is None


@pytest.mark.asyncio
async def test_live_api_error_market_closed_returns_none() -> None:
    """market_closed KalshiApiError → None (no exception raised)."""
    from src.clients.kalshi import KalshiApiError

    kalshi = MagicMock()
    kalshi.submit_order = AsyncMock(
        side_effect=KalshiApiError("market_closed", "Market is closed")
    )
    layer = LiveExecutionLayer(kalshi)
    ob = _ob()

    result = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob)
    assert result is None


@pytest.mark.asyncio
async def test_live_full_fill_returns_fill_result() -> None:
    """Happy path: submitted order gets fully filled → FillResult(success=True)."""
    kalshi = MagicMock()
    kalshi.submit_order = AsyncMock(return_value={"order": {"id": "order-abc"}})
    kalshi.get_order = AsyncMock(return_value={
        "order": {
            "id": "order-abc",
            "status": "filled",
            "filled_count": 10,
            "yes_price": 55,
        }
    })
    layer = LiveExecutionLayer(kalshi)
    ob = _ob()

    with patch("asyncio.sleep"):
        result = await layer.execute_order(_signal(), amount=55.0, ob_sync=ob)

    assert result is not None
    assert isinstance(result, FillResult)
    assert result.success is True
    assert result.fill_quantity == 10
    assert result.order_id == "order-abc"
