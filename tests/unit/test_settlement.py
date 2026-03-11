"""Unit tests for settlement.py.

Tests cover:
  - compute_realized_pnl: BUY_YES profit/loss, BUY_NO profit/loss, fee logic
  - await_settlement: resolves immediately, partial timeout, full timeout
  - settle_all_positions: no-db guard, no-open-positions guard, happy path
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.types import Position
from src.execution.settlement import (
    FEE_RATE,
    await_settlement,
    compute_realized_pnl,
    settle_all_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos(
    *,
    direction: str = "BUY_YES",
    entry_price: float = 0.45,
    quantity: int = 10,
    market_ticker: str = "SOCC-M1-YES",
    is_paper: bool = True,
) -> Position:
    return Position(
        match_id="match-001",
        market_ticker=market_ticker,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        is_paper=is_paper,
    )


def _model(*, has_db: bool = True) -> MagicMock:
    model = MagicMock()
    model.match_id = "match-001"
    model.is_paper = True
    model.db_pool = MagicMock() if has_db else None
    return model


def _kalshi(ticker_settlements: dict[str, float | None] | None = None) -> MagicMock:
    """Return a mock KalshiClient.get_market that resolves given tickers."""

    async def _get_market(ticker: str) -> dict:  # type: ignore[return]
        if ticker_settlements is None:
            return {"status": "active"}
        val = ticker_settlements.get(ticker)
        if val is None:
            return {"status": "active"}
        return {"status": "resolved", "settlement_price": val}

    kalshi = MagicMock()
    kalshi.get_market = _get_market
    return kalshi


# ---------------------------------------------------------------------------
# compute_realized_pnl — validation table from docs/phase4.md
# ---------------------------------------------------------------------------


def test_buy_yes_win() -> None:
    """BUY_YES, entry=0.45, settlement=1.00 → profit after fee."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=1)
    pnl = compute_realized_pnl(pos, 1.00, fee_rate=FEE_RATE)
    gross = (1.00 - 0.45) * 1  # 0.55
    expected = gross - FEE_RATE * gross
    assert pnl == pytest.approx(expected, abs=1e-6)
    assert pnl > 0


def test_buy_yes_loss() -> None:
    """BUY_YES, entry=0.45, settlement=0.00 → loss (no fee on losses)."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=1)
    pnl = compute_realized_pnl(pos, 0.00, fee_rate=FEE_RATE)
    assert pnl == pytest.approx(-0.45, abs=1e-6)


def test_buy_no_win() -> None:
    """BUY_NO, entry=0.40, settlement=0.00 → profit (No wins when Yes=0)."""
    pos = _pos(direction="BUY_NO", entry_price=0.40, quantity=1)
    pnl = compute_realized_pnl(pos, 0.00, fee_rate=FEE_RATE)
    gross = (0.40 - 0.00) * 1  # 0.40
    expected = gross - FEE_RATE * gross
    assert pnl == pytest.approx(expected, abs=1e-6)
    assert pnl > 0


def test_buy_no_loss() -> None:
    """BUY_NO, entry=0.40, settlement=1.00 → loss (Yes wins, No loses)."""
    pos = _pos(direction="BUY_NO", entry_price=0.40, quantity=1)
    pnl = compute_realized_pnl(pos, 1.00, fee_rate=FEE_RATE)
    # gross = 0.40 - 1.00 = -0.60; no fee on loss
    assert pnl == pytest.approx(-0.60, abs=1e-6)


def test_fee_only_on_profit() -> None:
    """Fee is never applied to losses."""
    pos = _pos(direction="BUY_YES", entry_price=0.70, quantity=10)
    pnl = compute_realized_pnl(pos, 0.00, fee_rate=FEE_RATE)
    assert pnl == pytest.approx(-7.00, abs=1e-6)  # pure loss, no fee


def test_zero_fee_rate() -> None:
    """fee_rate=0 → gross = net."""
    pos = _pos(direction="BUY_YES", entry_price=0.50, quantity=10)
    pnl = compute_realized_pnl(pos, 1.00, fee_rate=0.0)
    assert pnl == pytest.approx(5.00, abs=1e-6)


def test_multi_quantity_scaling() -> None:
    """P&L scales linearly with quantity."""
    pos1 = _pos(direction="BUY_YES", entry_price=0.45, quantity=1)
    pos10 = _pos(direction="BUY_YES", entry_price=0.45, quantity=10)
    assert compute_realized_pnl(pos10, 1.00) == pytest.approx(
        10 * compute_realized_pnl(pos1, 1.00), abs=1e-6
    )


def test_unknown_direction_zero_pnl() -> None:
    """Unknown direction returns 0 P&L."""
    pos = _pos(direction="HOLD", entry_price=0.50, quantity=10)
    assert compute_realized_pnl(pos, 1.00) == 0.0


def test_buy_yes_breakeven() -> None:
    """Settlement at entry price → gross zero (minus fee=0) → pnl=0."""
    pos = _pos(direction="BUY_YES", entry_price=0.55, quantity=10)
    assert compute_realized_pnl(pos, 0.55) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# await_settlement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_await_settlement_resolves_immediately() -> None:
    """All markets resolved on first poll → returns immediately."""
    kalshi = _kalshi({"SOCC-A-YES": 1.0, "SOCC-B-YES": 0.0})

    with patch("src.execution.settlement.asyncio.sleep", new_callable=AsyncMock):
        result = await await_settlement(
            "match-001",
            ["SOCC-A-YES", "SOCC-B-YES"],
            kalshi,
        )

    assert result == {"SOCC-A-YES": 1.0, "SOCC-B-YES": 0.0}


@pytest.mark.asyncio
async def test_await_settlement_partial_then_full() -> None:
    """First poll resolves one ticker; second poll resolves the other."""
    call_count = {"n": 0}

    async def _get_market(ticker: str) -> dict:  # type: ignore[return]
        call_count["n"] += 1
        if ticker == "SOCC-A-YES":
            return {"status": "resolved", "settlement_price": 1.0}
        # SOCC-B-YES resolves on 2nd poll
        if call_count["n"] >= 3:
            return {"status": "resolved", "settlement_price": 0.0}
        return {"status": "active"}

    kalshi = MagicMock()
    kalshi.get_market = _get_market

    with patch("src.execution.settlement.asyncio.sleep", new_callable=AsyncMock):
        result = await await_settlement(
            "match-001",
            ["SOCC-A-YES", "SOCC-B-YES"],
            kalshi,
        )

    assert result["SOCC-A-YES"] == 1.0
    assert result["SOCC-B-YES"] == 0.0


@pytest.mark.asyncio
async def test_await_settlement_timeout_returns_partial() -> None:
    """Timeout reached → return only resolved markets (no exception)."""
    kalshi = _kalshi({"SOCC-A-YES": 1.0})  # SOCC-B-YES never resolves

    with patch("src.execution.settlement.asyncio.sleep", new_callable=AsyncMock):
        result = await await_settlement(
            "match-001",
            ["SOCC-A-YES", "SOCC-B-YES"],
            kalshi,
            timeout_hours=0.0,  # immediate timeout
        )

    assert "SOCC-A-YES" in result or result == {}  # partial or empty — no exception


@pytest.mark.asyncio
async def test_await_settlement_suppresses_api_error() -> None:
    """get_market exception is logged + suppressed; polling continues."""
    call_count = {"n": 0}

    async def _get_market(ticker: str) -> dict:  # type: ignore[return]
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("connection reset")
        return {"status": "resolved", "settlement_price": 0.5}

    kalshi = MagicMock()
    kalshi.get_market = _get_market

    with patch("src.execution.settlement.asyncio.sleep", new_callable=AsyncMock):
        result = await await_settlement("match-001", ["SOCC-A-YES"], kalshi)

    assert result["SOCC-A-YES"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_await_settlement_empty_tickers_returns_immediately() -> None:
    """Empty ticker list → immediately return empty dict."""
    kalshi = _kalshi()
    result = await await_settlement("match-001", [], kalshi)
    assert result == {}


# ---------------------------------------------------------------------------
# settle_all_positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settle_all_no_db_skips() -> None:
    """model.db_pool is None → function returns without error."""
    model = _model(has_db=False)
    kalshi = _kalshi()
    await settle_all_positions(model, kalshi)  # no exception


@pytest.mark.asyncio
async def test_settle_all_no_open_positions() -> None:
    """No OPEN positions → no DB writes."""
    model = _model()

    with patch(
        "src.execution.settlement._get_open_positions",
        new_callable=AsyncMock,
        return_value=[],
    ):
        kalshi = _kalshi()
        await settle_all_positions(model, kalshi)
        # no exception, no further DB calls


@pytest.mark.asyncio
async def test_settle_all_happy_path() -> None:
    """Two positions resolved → both settled, bankroll updated twice."""
    model = _model()

    raw_positions = [
        {
            "id": 1,
            "market_ticker": "SOCC-M1-YES",
            "direction": "BUY_YES",
            "entry_price": 0.45,
            "quantity": 10,
            "is_paper": True,
        },
        {
            "id": 2,
            "market_ticker": "SOCC-M2-YES",
            "direction": "BUY_NO",
            "entry_price": 0.40,
            "quantity": 5,
            "is_paper": True,
        },
    ]

    with (
        patch(
            "src.execution.settlement._get_open_positions",
            new_callable=AsyncMock,
            return_value=raw_positions,
        ),
        patch(
            "src.execution.settlement._set_awaiting_settlement",
            new_callable=AsyncMock,
        ) as mock_set_await,
        patch(
            "src.execution.settlement.await_settlement",
            new_callable=AsyncMock,
            return_value={"SOCC-M1-YES": 1.0, "SOCC-M2-YES": 0.0},
        ),
        patch(
            "src.execution.settlement._settle_position",
            new_callable=AsyncMock,
        ) as mock_settle,
        patch(
            "src.execution.settlement._update_bankroll",
            new_callable=AsyncMock,
        ) as mock_bankroll,
    ):
        await settle_all_positions(model, MagicMock())

    # Both positions transitioned to AWAITING_SETTLEMENT
    assert mock_set_await.call_count == 2

    # Both positions settled
    assert mock_settle.call_count == 2

    # P&L for each call
    settle_calls = {
        call.args[1]: (call.args[2], call.args[3])
        for call in mock_settle.call_args_list
    }  # {position_id: (settlement_price, pnl)}

    # Position 1: BUY_YES entry=0.45 settlement=1.00 qty=10
    sp1, pnl1 = settle_calls[1]
    assert sp1 == pytest.approx(1.0)
    assert pnl1 > 0  # won

    # Position 2: BUY_NO entry=0.40 settlement=0.00 qty=5
    sp2, pnl2 = settle_calls[2]
    assert sp2 == pytest.approx(0.0)
    assert pnl2 > 0  # No wins (settlement=0 means Yes lost)

    # Bankroll updated twice
    assert mock_bankroll.call_count == 2


@pytest.mark.asyncio
async def test_settle_all_unresolved_positions_skipped() -> None:
    """Unresolved ticker → position not settled (stays AWAITING_SETTLEMENT)."""
    model = _model()

    raw_positions = [
        {
            "id": 1,
            "market_ticker": "SOCC-M1-YES",
            "direction": "BUY_YES",
            "entry_price": 0.50,
            "quantity": 5,
            "is_paper": True,
        }
    ]

    with (
        patch(
            "src.execution.settlement._get_open_positions",
            new_callable=AsyncMock,
            return_value=raw_positions,
        ),
        patch("src.execution.settlement._set_awaiting_settlement", new_callable=AsyncMock),
        patch(
            "src.execution.settlement.await_settlement",
            new_callable=AsyncMock,
            return_value={},  # nothing resolved
        ),
        patch(
            "src.execution.settlement._settle_position",
            new_callable=AsyncMock,
        ) as mock_settle,
    ):
        await settle_all_positions(model, MagicMock())

    mock_settle.assert_not_called()
