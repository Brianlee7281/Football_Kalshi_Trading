"""Integration: Reserve-Confirm-Release concurrency pattern.

Verifies the 3-phase exposure reservation flow:
  1. Reserve (short lock): check caps, INSERT RESERVED row
  2. Execute (no lock): submit order
  3. Confirm/Release: update reservation status

Also tests concurrent reservations and cap enforcement.

Reference: docs/orchestration.md Risk Limit Enforcement, docs/phase4.md Step 4.3
"""

from __future__ import annotations

import time

import pytest

from src.common.types import PaperFill
from src.execution.signal_generator import (
    _confirm_reservation,
    _fill_price,
    _fill_quantity,
    _release_reservation,
    _reserve_exposure,
    execute_with_reservation,
)

from .conftest import MockPool, MockRedis, make_model, make_ob, make_signal

# ---------------------------------------------------------------------------
# Basic reserve-confirm-release
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reserve_creates_reservation() -> None:
    """Reserve inserts a RESERVED row and returns an ID."""
    pool = MockPool()
    redis = MockRedis()
    rid = await _reserve_exposure(
        pool, redis, "m1", "T-HW", 0.02, 10_000.0, True,
    )
    assert rid is not None
    assert rid > 0
    assert pool._store["reservations"][rid]["status"] == "RESERVED"


@pytest.mark.anyio
async def test_confirm_marks_confirmed() -> None:
    """Confirm updates reservation to CONFIRMED with actual amount."""
    pool = MockPool()
    redis = MockRedis()
    rid = await _reserve_exposure(pool, redis, "m1", "T-HW", 0.02, 10_000.0, True)
    assert rid is not None

    await _confirm_reservation(pool, rid, 150.0)
    assert pool._store["reservations"][rid]["status"] == "CONFIRMED"


@pytest.mark.anyio
async def test_release_marks_released() -> None:
    """Release updates reservation to RELEASED."""
    pool = MockPool()
    redis = MockRedis()
    rid = await _reserve_exposure(pool, redis, "m1", "T-HW", 0.02, 10_000.0, True)
    assert rid is not None

    await _release_reservation(pool, rid)
    assert pool._store["reservations"][rid]["status"] == "RELEASED"


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_order_cap_limits_amount() -> None:
    """Reserve caps at F_ORDER_CAP (3%) of bankroll."""
    pool = MockPool()
    redis = MockRedis()
    bankroll = 10_000.0
    # Request 10% of bankroll — should be capped to 3%
    rid = await _reserve_exposure(pool, redis, "m1", "T-HW", 0.10, bankroll, True)
    assert rid is not None
    reserved_amount = pool._store["reservations"][rid]["amount"]
    assert reserved_amount <= bankroll * 0.03 + 0.01


@pytest.mark.anyio
async def test_zero_bankroll_returns_none() -> None:
    """Zero bankroll → amount = 0 → None."""
    pool = MockPool()
    redis = MockRedis()
    rid = await _reserve_exposure(pool, redis, "m1", "T-HW", 0.02, 0.0, True)
    assert rid is None


# ---------------------------------------------------------------------------
# execute_with_reservation integration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_with_reservation_full_flow() -> None:
    """Full flow: reserve → execute → confirm."""
    model = make_model(bankroll=10_000.0)
    ob = make_ob()

    # Set up a mock execution router that returns a fill
    from .conftest import MockExecutionRouter
    model.execution = MockExecutionRouter(fill_price=0.55, fill_qty=10)  # type: ignore[assignment]

    signal = make_signal(direction="BUY_YES", P_kalshi=0.55)
    fill = await execute_with_reservation(signal, 200.0, ob, model)

    assert fill is not None
    assert fill.quantity == 10  # type: ignore[union-attr]
    assert fill.price == 0.55  # type: ignore[union-attr]

    # Reservation should be confirmed
    reservations = model.db_pool._store.get("reservations", {})  # type: ignore[union-attr]
    confirmed = [r for r in reservations.values() if r["status"] == "CONFIRMED"]
    assert len(confirmed) == 1


@pytest.mark.anyio
async def test_execute_with_reservation_no_fill_releases() -> None:
    """When execution returns None → reservation is released."""
    model = make_model(bankroll=10_000.0)
    ob = make_ob()

    # Mock router that returns no fill
    class NoFillRouter:
        async def submit_order(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

    model.execution = NoFillRouter()  # type: ignore[assignment]

    signal = make_signal(direction="BUY_YES", P_kalshi=0.55)
    fill = await execute_with_reservation(signal, 200.0, ob, model)

    assert fill is None

    # Reservation should be released
    reservations = model.db_pool._store.get("reservations", {})  # type: ignore[union-attr]
    released = [r for r in reservations.values() if r["status"] == "RELEASED"]
    assert len(released) == 1


@pytest.mark.anyio
async def test_execute_with_reservation_exception_releases() -> None:
    """When execution raises → reservation is released and exception propagates."""
    model = make_model(bankroll=10_000.0)
    ob = make_ob()

    class FailRouter:
        async def submit_order(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("Kalshi API error")

    model.execution = FailRouter()  # type: ignore[assignment]

    signal = make_signal(direction="BUY_YES", P_kalshi=0.55)

    with pytest.raises(RuntimeError, match="Kalshi API error"):
        await execute_with_reservation(signal, 200.0, ob, model)

    reservations = model.db_pool._store.get("reservations", {})  # type: ignore[union-attr]
    released = [r for r in reservations.values() if r["status"] == "RELEASED"]
    assert len(released) == 1


@pytest.mark.anyio
async def test_no_db_pool_returns_none() -> None:
    """No db_pool → execute_with_reservation returns None."""
    model = make_model()
    model.db_pool = None
    ob = make_ob()

    signal = make_signal()
    fill = await execute_with_reservation(signal, 200.0, ob, model)
    assert fill is None


@pytest.mark.anyio
async def test_no_redis_returns_none() -> None:
    """No redis → execute_with_reservation returns None."""
    model = make_model()
    model.redis = None
    ob = make_ob()

    signal = make_signal()
    fill = await execute_with_reservation(signal, 200.0, ob, model)
    assert fill is None


# ---------------------------------------------------------------------------
# Concurrent reservations
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sequential_reservations_get_unique_ids() -> None:
    """Multiple sequential reservations get unique IDs."""
    pool = MockPool()
    redis = MockRedis()

    ids = []
    for i in range(5):
        rid = await _reserve_exposure(
            pool, redis, "m1", f"T-{i}", 0.005, 10_000.0, True,
        )
        if rid is not None:
            ids.append(rid)

    assert len(ids) == len(set(ids)), "Reservation IDs must be unique"
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------


def test_fill_quantity_paper() -> None:
    """_fill_quantity extracts quantity from PaperFill."""
    fill = PaperFill(price=0.55, quantity=42, timestamp=time.time())
    assert _fill_quantity(fill) == 42


def test_fill_price_paper() -> None:
    """_fill_price extracts price from PaperFill."""
    fill = PaperFill(price=0.55, quantity=42, timestamp=time.time())
    assert _fill_price(fill) == 0.55
