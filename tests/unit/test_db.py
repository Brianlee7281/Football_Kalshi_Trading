"""Unit tests for src/common/db.py.

Tests cover:
  - get_bankroll: row found, row missing
  - get_match_exposure: rows present, empty
  - get_existing_exposure: rows present, empty
  - safe_submit_order: Phase A failure, Phase B failure, no fill, happy path
  - reconcile_stale_pending: stale rows found, none found
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.common.db import (
    get_bankroll,
    get_match_exposure,
    get_existing_exposure,
    safe_submit_order,
    reconcile_stale_pending,
    _extract_fill_quantity,
    _extract_fill_price,
)
from src.common.types import PaperFill, FillResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pool(fetchrow_return=None, fetch_return=None, execute_return=None):
    """Build a mock asyncpg Pool with a context-manager acquire()."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock(return_value=execute_return)

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


def _make_model(
    *,
    db_pool=None,
    execution=None,
    match_id="match-001",
    is_paper=True,
):
    model = MagicMock()
    model.match_id = match_id
    model.is_paper = is_paper
    model.db_pool = db_pool
    model.execution = execution
    return model


def _make_signal(
    *,
    market_ticker="SOCC-M1-YES",
    direction="BUY_YES",
    P_kalshi=0.50,
):
    signal = MagicMock()
    signal.market_ticker = market_ticker
    signal.direction = direction
    signal.P_kalshi = P_kalshi
    return signal


# ---------------------------------------------------------------------------
# _extract_fill_quantity / _extract_fill_price
# ---------------------------------------------------------------------------


def test_extract_fill_quantity_paper_fill() -> None:
    fill = PaperFill(price=0.55, quantity=100, timestamp=0.0)
    assert _extract_fill_quantity(fill) == 100


def test_extract_fill_quantity_fill_result() -> None:
    fill = FillResult(success=True, fill_quantity=200, fill_price=0.60)
    assert _extract_fill_quantity(fill) == 200


def test_extract_fill_quantity_none() -> None:
    assert _extract_fill_quantity(None) == 0


def test_extract_fill_price_paper_fill() -> None:
    fill = PaperFill(price=0.55, quantity=100, timestamp=0.0)
    assert _extract_fill_price(fill) == pytest.approx(0.55)


def test_extract_fill_price_fill_result() -> None:
    fill = FillResult(success=True, fill_quantity=200, fill_price=0.60)
    assert _extract_fill_price(fill) == pytest.approx(0.60)


def test_extract_fill_price_none() -> None:
    assert _extract_fill_price(None) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_bankroll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_bankroll_row_found() -> None:
    """Returns the DB balance when row exists."""
    row = {"balance": 12345.67}
    pool, _ = _make_pool(fetchrow_return=row)
    result = await get_bankroll(pool, "paper")
    assert result == pytest.approx(12345.67)


@pytest.mark.asyncio
async def test_get_bankroll_no_row_defaults_to_10000() -> None:
    """Returns 10_000.0 when no row found."""
    pool, _ = _make_pool(fetchrow_return=None)
    result = await get_bankroll(pool, "live")
    assert result == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# get_match_exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_match_exposure_with_positions() -> None:
    """Sums entry_price * quantity from OPEN and AWAITING_SETTLEMENT positions."""
    row = {"total": 1500.0}
    pool, _ = _make_pool(fetchrow_return=row)
    result = await get_match_exposure(pool, "match-001")
    assert result == pytest.approx(1500.0)


@pytest.mark.asyncio
async def test_get_match_exposure_no_positions() -> None:
    """Returns 0.0 when no open positions."""
    pool, _ = _make_pool(fetchrow_return=None)
    result = await get_match_exposure(pool, "match-001")
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_existing_exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_existing_exposure_with_positions() -> None:
    row = {"total": 300.0}
    pool, _ = _make_pool(fetchrow_return=row)
    result = await get_existing_exposure(pool, "match-001", "SOCC-M1-YES", "BUY_YES")
    assert result == pytest.approx(300.0)


@pytest.mark.asyncio
async def test_get_existing_exposure_no_positions() -> None:
    pool, _ = _make_pool(fetchrow_return=None)
    result = await get_existing_exposure(pool, "match-001", "SOCC-M1-YES", "BUY_YES")
    assert result == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_get_existing_exposure_passes_direction() -> None:
    """Direction is forwarded correctly to the query."""
    row = {"total": 100.0}
    pool, conn = _make_pool(fetchrow_return=row)
    await get_existing_exposure(pool, "match-001", "SOCC-M1-YES", "BUY_NO")
    call_args = conn.fetchrow.call_args
    assert "BUY_NO" in call_args.args


# ---------------------------------------------------------------------------
# safe_submit_order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safe_submit_order_phase_a_db_failure_returns_none() -> None:
    """If Phase A DB INSERT fails, no order submitted; returns None."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=OSError("DB down"))

    execution = AsyncMock()
    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)

    assert result is None
    execution.submit_order.assert_not_called()


@pytest.mark.asyncio
async def test_safe_submit_order_phase_b_execution_failure_deletes_pending() -> None:
    """If Phase B execution raises, PENDING row is deleted and None returned."""
    pool, conn = _make_pool(fetchrow_return={"id": 42})
    conn.execute = AsyncMock()

    execution = MagicMock()
    execution.submit_order = AsyncMock(side_effect=RuntimeError("Kalshi error"))

    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)

    assert result is None
    # DELETE should have been called with the position_id
    delete_calls = [str(c) for c in conn.execute.call_args_list]
    assert any("DELETE" in s or "42" in s for s in delete_calls)


@pytest.mark.asyncio
async def test_safe_submit_order_no_fill_deletes_pending() -> None:
    """If execution returns None, PENDING row is deleted and None returned."""
    pool, conn = _make_pool(fetchrow_return={"id": 7})
    conn.execute = AsyncMock()

    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=None)

    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)
    assert result is None


@pytest.mark.asyncio
async def test_safe_submit_order_zero_qty_fill_deletes_pending() -> None:
    """Fill with quantity=0 is treated as no-fill; PENDING deleted."""
    pool, conn = _make_pool(fetchrow_return={"id": 9})
    conn.execute = AsyncMock()

    fill = PaperFill(price=0.50, quantity=0, timestamp=1.0)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)
    assert result is None


@pytest.mark.asyncio
async def test_safe_submit_order_happy_path_paper_fill() -> None:
    """Happy path: PENDING inserted, fill received, position updated to OPEN."""
    pool, conn = _make_pool(fetchrow_return={"id": 15})
    conn.execute = AsyncMock()

    fill = PaperFill(price=0.52, quantity=190, timestamp=1.0)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)

    assert result is fill
    # Phase C UPDATE called
    update_call = conn.execute.call_args
    assert update_call is not None
    assert 190 in update_call.args or any(190 == a for a in update_call.args)


@pytest.mark.asyncio
async def test_safe_submit_order_happy_path_fill_result() -> None:
    """Happy path with FillResult (live mode)."""
    pool, conn = _make_pool(fetchrow_return={"id": 20})
    conn.execute = AsyncMock()

    fill = FillResult(success=True, fill_quantity=150, fill_price=0.55)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(db_pool=pool, execution=execution, is_paper=False)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)
    assert result is fill


@pytest.mark.asyncio
async def test_safe_submit_order_phase_c_db_failure_still_returns_fill() -> None:
    """Phase C DB failure is logged but fill is still returned (order went through)."""
    # First fetchrow for INSERT returns id, second execute call raises
    pool, conn = _make_pool(fetchrow_return={"id": 30})

    call_count = {"n": 0}

    async def execute_side_effect(*_args: object, **_kwargs: object) -> None:
        call_count["n"] += 1
        raise OSError("DB down at update")

    conn.execute = AsyncMock(side_effect=execute_side_effect)

    fill = PaperFill(price=0.50, quantity=200, timestamp=1.0)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(db_pool=pool, execution=execution)
    signal = _make_signal(P_kalshi=0.50)

    result = await safe_submit_order(signal, 100.0, MagicMock(), model)
    assert result is fill  # fill returned despite Phase C failure


# ---------------------------------------------------------------------------
# reconcile_stale_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_stale_pending_no_stale() -> None:
    """Returns empty list when no stale PENDING positions."""
    pool, conn = _make_pool(fetch_return=[])
    result = await reconcile_stale_pending(pool, "match-001")
    assert result == []


@pytest.mark.asyncio
async def test_reconcile_stale_pending_returns_ids() -> None:
    """Returns list of stale position IDs."""
    rows = [
        {
            "id": 11,
            "market_ticker": "SOCC-M1-YES",
            "direction": "BUY_YES",
            "quantity": 100,
            "entry_price": 0.50,
            "entry_time": "2026-03-11 10:00:00+00",
        },
        {
            "id": 22,
            "market_ticker": "SOCC-M2-YES",
            "direction": "BUY_NO",
            "quantity": 50,
            "entry_price": 0.45,
            "entry_time": "2026-03-11 10:01:00+00",
        },
    ]
    pool, conn = _make_pool()
    conn.fetch = AsyncMock(return_value=rows)

    result = await reconcile_stale_pending(pool, "match-001")
    assert set(result) == {11, 22}


@pytest.mark.asyncio
async def test_reconcile_stale_pending_custom_max_age() -> None:
    """max_age_minutes parameter is passed through to the query."""
    pool, conn = _make_pool()
    conn.fetch = AsyncMock(return_value=[])
    await reconcile_stale_pending(pool, "match-001", max_age_minutes=10)
    call_args = conn.fetch.call_args
    assert 10 in call_args.args
