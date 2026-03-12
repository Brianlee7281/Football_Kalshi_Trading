"""Unit tests for PaperExecutionLayer.

Reference: docs/phase4.md Step 4.5 (PaperExecutionLayer)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import Signal
from src.engine.model import EVENT_IDLE
from src.execution.order_book_sync import OrderBookSync
from src.execution.paper_executor import PaperExecutionLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    direction: str = "BUY_YES",
    P_kalshi: float = 0.50,
    EV: float = 0.03,
    market_ticker: str = "T",
) -> Signal:
    return Signal(
        direction=direction,
        EV=EV,
        P_cons=0.55,
        P_kalshi=P_kalshi,
        rough_qty=100,
        alignment_status="ALIGNED",
        kelly_multiplier=0.8,
        market_ticker=market_ticker,
    )


def _ob(*, ask_depth: int = 1000, bid_depth: int = 1000) -> OrderBookSync:
    ob = OrderBookSync(ticker="T")
    ob._apply_snapshot(OrderBookUpdate(
        ticker="T",
        is_snapshot=True,
        yes=[(55, ask_depth)],
        no=[(45, bid_depth)],
        timestamp=time.time(),
    ))
    return ob


def _model(*, ob_freeze: bool = False, event_state: str = EVENT_IDLE) -> MagicMock:
    m = MagicMock()
    m.ob_freeze = ob_freeze
    m.event_state = event_state
    m.match_id = "test_match"
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_p_kalshi_zero_returns_none() -> None:
    """P_kalshi <= 0 → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    sig = _signal(P_kalshi=0.0)
    result = await layer.execute_order(sig, 100.0, _ob(), _model())
    assert result is None


@pytest.mark.anyio
async def test_target_qty_zero_returns_none() -> None:
    """Amount too small to buy 1 contract → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    sig = _signal(P_kalshi=0.50)
    result = await layer.execute_order(sig, 0.10, _ob(), _model())  # 0.10/0.50 = 0
    assert result is None


@pytest.mark.anyio
async def test_buy_yes_slippage_increases_price() -> None:
    """BUY_YES: slippage makes fill price higher (adverse)."""
    layer = PaperExecutionLayer(slippage_ticks=2, fill_delay_range=(0.0, 0.0))
    sig = _signal(direction="BUY_YES", P_kalshi=0.55)
    result = await layer.execute_order(sig, 100.0, _ob(), _model(), urgent=True)
    assert result is not None
    # VWAP at 55¢ = 0.55, slippage +2¢ → fill_price ≈ 0.57
    assert result.price > 0.55


@pytest.mark.anyio
async def test_buy_no_slippage_decreases_price() -> None:
    """BUY_NO: slippage makes fill price lower (adverse for No buyer)."""
    layer = PaperExecutionLayer(slippage_ticks=2, fill_delay_range=(0.0, 0.0))
    sig = _signal(direction="BUY_NO", P_kalshi=0.45)
    ob = _ob(bid_depth=1000)
    result = await layer.execute_order(sig, 100.0, ob, _model(), urgent=True)
    assert result is not None
    # VWAP sell at 55¢ bid = 0.55, slippage -2¢ → fill_price ≈ 0.53
    assert result.price < 0.55 + 0.001  # less than unslipped


@pytest.mark.anyio
async def test_ob_freeze_aborts() -> None:
    """ob_freeze during delay → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    sig = _signal()
    model = _model(ob_freeze=True)
    result = await layer.execute_order(sig, 100.0, _ob(), model)
    assert result is None


@pytest.mark.anyio
async def test_event_state_non_idle_aborts() -> None:
    """event_state != IDLE during delay → None."""
    layer = PaperExecutionLayer(slippage_ticks=1, fill_delay_range=(0.0, 0.0))
    sig = _signal()
    model = _model(event_state="PRELIMINARY_DETECTED")
    result = await layer.execute_order(sig, 100.0, _ob(), model)
    assert result is None


@pytest.mark.anyio
async def test_is_paper_flag_set() -> None:
    """PaperFill always has is_paper=True."""
    layer = PaperExecutionLayer(slippage_ticks=0, fill_delay_range=(0.0, 0.0))
    sig = _signal(direction="BUY_YES", P_kalshi=0.55)
    result = await layer.execute_order(sig, 100.0, _ob(), _model(), urgent=True)
    assert result is not None
    assert result.is_paper is True


@pytest.mark.anyio
async def test_urgent_skips_delay() -> None:
    """urgent=True → fill_delay = 0.0."""
    layer = PaperExecutionLayer(slippage_ticks=0, fill_delay_range=(5.0, 10.0))
    sig = _signal(P_kalshi=0.55)
    result = await layer.execute_order(sig, 100.0, _ob(), _model(), urgent=True)
    assert result is not None
    assert result.fill_delay == 0.0
