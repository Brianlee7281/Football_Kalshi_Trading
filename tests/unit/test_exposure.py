"""Unit tests for src/common/exposure.py.

Tests cover:
  - reserve_exposure: all three cap layers, happy path, lock usage
  - confirm_reservation: DB UPDATE called with correct args
  - release_reservation: DB UPDATE called
  - execute_with_reservation: no-infra guard, cap exceeded, execution error,
    no-fill release, happy path confirm, no-execution release
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.common.exposure import (
    confirm_reservation,
    execute_with_reservation,
    release_reservation,
    reserve_exposure,
    _extract_fill_quantity,
    _extract_fill_price,
)
from src.common.types import FillResult, PaperFill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(fetchrow_return=None):
    """Mock asyncpg pool."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock()

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


def _make_redis():
    """Mock Redis client with a no-op exposure_lock."""
    redis = MagicMock()
    lock_cm = MagicMock()
    lock_cm.__aenter__ = AsyncMock(return_value=None)
    lock_cm.__aexit__ = AsyncMock(return_value=False)
    redis.lock = MagicMock(return_value=lock_cm)
    return redis


def _make_model(
    *,
    db_pool=None,
    redis=None,
    execution=None,
    match_id="match-001",
    bankroll=10_000.0,
    is_paper=True,
):
    model = MagicMock()
    model.match_id = match_id
    model.bankroll = bankroll
    model.is_paper = is_paper
    model.db_pool = db_pool
    model.redis = redis
    model.execution = execution
    return model


def _make_signal(*, market_ticker="SOCC-M1-YES", direction="BUY_YES", P_kalshi=0.50):
    signal = MagicMock()
    signal.market_ticker = market_ticker
    signal.direction = direction
    signal.P_kalshi = P_kalshi
    return signal


# ---------------------------------------------------------------------------
# _extract_fill helpers
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
# reserve_exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserve_exposure_happy_path() -> None:
    """All caps pass → inserts reservation row and returns id."""
    # get_match_exposure and get_total_exposure return 0 (plenty of room)
    # INSERT RESERVED returns id=42
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(
        side_effect=[
            # get_match_exposure query
            {"total": 0.0},
            # get_total_exposure query
            {"total": 0.0},
            # INSERT reservation
            {"id": 42},
        ]
    )
    redis = _make_redis()

    result = await reserve_exposure(
        pool, redis, "match-001", "SOCC-M1-YES",
        f_invest=0.02, bankroll=10_000.0, is_paper=True,
    )

    assert result == 42


@pytest.mark.asyncio
async def test_reserve_exposure_returns_none_when_all_caps_consumed() -> None:
    """Returns None when match cap is fully consumed."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(
        side_effect=[
            # get_match_exposure → match is at full cap (5% of 10k = 500)
            {"total": 500.0},
            # get_total_exposure → doesn't matter, match cap blocks first
            {"total": 0.0},
        ]
    )
    redis = _make_redis()

    result = await reserve_exposure(
        pool, redis, "match-001", "SOCC-M1-YES",
        f_invest=0.02, bankroll=10_000.0, is_paper=True,
    )

    assert result is None


@pytest.mark.asyncio
async def test_reserve_exposure_order_cap_limits_amount() -> None:
    """Amount is capped at F_ORDER_CAP (3%) even if f_invest is higher."""
    pool, conn = _make_pool()
    # Calls in order: get_match_exposure, get_total_exposure, INSERT reservation
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"total": 0.0},   # get_match_exposure
            {"total": 0.0},   # get_total_exposure
            {"id": 1},        # INSERT reservation
        ]
    )
    redis = _make_redis()

    bankroll = 10_000.0
    reservation_id = await reserve_exposure(
        pool, redis, "match-001", "SOCC-M1-YES",
        f_invest=0.10,  # requested 10%, but order cap is 3%
        bankroll=bankroll,
        is_paper=True,
    )

    # INSERT was called — capture the amount arg from the INSERT call
    assert reservation_id == 1
    # The third fetchrow call is the INSERT; args are (match_id, ticker, amount, is_paper)
    insert_call = conn.fetchrow.call_args_list[2]
    amount_arg = insert_call.args[3]  # sql, match_id, ticker, amount, ...
    assert float(amount_arg) <= bankroll * 0.03 + 1e-9


@pytest.mark.asyncio
async def test_reserve_exposure_uses_exposure_lock() -> None:
    """reserve_exposure acquires the exposure_lock on the redis client."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(
        side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 99}]
    )
    redis = _make_redis()

    await reserve_exposure(
        pool, redis, "match-001", "SOCC-M1-YES",
        f_invest=0.01, bankroll=10_000.0, is_paper=True,
    )

    # The exposure_lock context manager uses redis.lock("exposure_lock", ...)
    redis.lock.assert_called_once_with("exposure_lock", timeout=2.0)


# ---------------------------------------------------------------------------
# confirm_reservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_reservation_updates_db() -> None:
    """UPDATE is called with CONFIRMED status and actual amount."""
    pool, conn = _make_pool()

    await confirm_reservation(pool, reservation_id=7, actual_amount=123.45)

    conn.execute.assert_called_once()
    sql, *args = conn.execute.call_args.args
    assert "CONFIRMED" in sql
    assert 7 in args
    assert 123.45 in args


# ---------------------------------------------------------------------------
# release_reservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_reservation_updates_db() -> None:
    """UPDATE is called with RELEASED status."""
    pool, conn = _make_pool()

    await release_reservation(pool, reservation_id=5)

    conn.execute.assert_called_once()
    sql, *args = conn.execute.call_args.args
    assert "RELEASED" in sql
    assert 5 in args


# ---------------------------------------------------------------------------
# execute_with_reservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_reservation_no_db_returns_none() -> None:
    """Returns None immediately when db_pool is None."""
    model = _make_model(db_pool=None, redis=_make_redis())
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)
    assert result is None


@pytest.mark.asyncio
async def test_execute_with_reservation_no_redis_returns_none() -> None:
    """Returns None immediately when redis is None."""
    pool, _ = _make_pool()
    model = _make_model(db_pool=pool, redis=None)
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)
    assert result is None


@pytest.mark.asyncio
async def test_execute_with_reservation_cap_exceeded_returns_none() -> None:
    """Returns None when reserve_exposure returns None (cap exceeded)."""
    pool, conn = _make_pool()
    # match exposure at full cap
    conn.fetchrow = AsyncMock(side_effect=[{"total": 500.0}, {"total": 0.0}])
    redis = _make_redis()
    execution = MagicMock()
    model = _make_model(db_pool=pool, redis=redis, execution=execution, bankroll=10_000.0)
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)

    assert result is None
    execution.submit_order.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_reservation_execution_error_releases() -> None:
    """Execution exception → reservation released and exception re-raised."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 11}])
    conn.execute = AsyncMock()
    redis = _make_redis()

    execution = MagicMock()
    execution.submit_order = AsyncMock(side_effect=RuntimeError("Kalshi error"))

    model = _make_model(db_pool=pool, redis=redis, execution=execution, bankroll=10_000.0)
    signal = _make_signal()

    with pytest.raises(RuntimeError, match="Kalshi error"):
        await execute_with_reservation(signal, 100.0, MagicMock(), model)

    # RELEASED update must have been called
    release_calls = [
        str(c) for c in conn.execute.call_args_list
    ]
    assert any("RELEASED" in s for s in release_calls)


@pytest.mark.asyncio
async def test_execute_with_reservation_no_fill_releases() -> None:
    """No fill (None return) → reservation released, None returned."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 15}])
    conn.execute = AsyncMock()
    redis = _make_redis()

    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=None)

    model = _make_model(db_pool=pool, redis=redis, execution=execution, bankroll=10_000.0)
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)

    assert result is None
    release_calls = [str(c) for c in conn.execute.call_args_list]
    assert any("RELEASED" in s for s in release_calls)


@pytest.mark.asyncio
async def test_execute_with_reservation_happy_path_paper_fill() -> None:
    """Happy path: fill received → reservation CONFIRMED, fill returned."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 20}])
    conn.execute = AsyncMock()
    redis = _make_redis()

    fill = PaperFill(price=0.52, quantity=190, timestamp=1.0)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(db_pool=pool, redis=redis, execution=execution, bankroll=10_000.0)
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)

    assert result is fill
    confirm_calls = [str(c) for c in conn.execute.call_args_list]
    assert any("CONFIRMED" in s for s in confirm_calls)


@pytest.mark.asyncio
async def test_execute_with_reservation_happy_path_fill_result() -> None:
    """Happy path with FillResult (live mode)."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 25}])
    conn.execute = AsyncMock()
    redis = _make_redis()

    fill = FillResult(success=True, fill_quantity=150, fill_price=0.55)
    execution = MagicMock()
    execution.submit_order = AsyncMock(return_value=fill)

    model = _make_model(
        db_pool=pool, redis=redis, execution=execution,
        bankroll=10_000.0, is_paper=False,
    )
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)

    assert result is fill


@pytest.mark.asyncio
async def test_execute_with_reservation_no_execution_releases() -> None:
    """model.execution is None → reservation released, None returned."""
    pool, conn = _make_pool()
    conn.fetchrow = AsyncMock(side_effect=[{"total": 0.0}, {"total": 0.0}, {"id": 30}])
    conn.execute = AsyncMock()
    redis = _make_redis()

    model = _make_model(db_pool=pool, redis=redis, execution=None, bankroll=10_000.0)
    signal = _make_signal()

    result = await execute_with_reservation(signal, 100.0, MagicMock(), model)

    assert result is None
    release_calls = [str(c) for c in conn.execute.call_args_list]
    assert any("RELEASED" in s for s in release_calls)
