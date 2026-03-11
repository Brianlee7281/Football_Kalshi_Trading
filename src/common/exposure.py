"""Cross-container exposure reservation — Reserve → Execute → Confirm/Release.

Prevents over-allocation when multiple match containers run concurrently.
The Redis lock is held only while reading exposure totals and inserting the
reservation row (<10ms). Order execution (1-5s) happens *outside* the lock,
so other containers can reserve concurrently.

Pattern:
  1. reserve_exposure()      — acquire lock, check limits, INSERT RESERVED row
  2. execute (caller's code) — no lock held
  3. confirm_reservation()   — UPDATE to CONFIRMED with actual fill amount
     OR
     release_reservation()   — UPDATE to RELEASED on failure / no fill

Stale cleanup: a CRON job releases RESERVED rows older than 60s so leaked
reservations never permanently reduce available exposure.

Reference: docs/orchestration.md Risk Limit Enforcement (Cross-Container)
"""

from __future__ import annotations

from typing import Any

import asyncpg
import redis.asyncio as aioredis

from src.common.db import get_match_exposure
from src.common.logging import get_logger
from src.common.redis_client import exposure_lock
from src.execution.kelly import F_MATCH_CAP, F_ORDER_CAP, F_TOTAL_CAP

logger = get_logger("exposure")

# ---------------------------------------------------------------------------
# Reservation helpers
# ---------------------------------------------------------------------------


async def reserve_exposure(
    db_pool: asyncpg.Pool,
    redis: aioredis.Redis,
    match_id: str,
    market_ticker: str,
    f_invest: float,
    bankroll: float,
    is_paper: bool,
) -> int | None:
    """Reserve exposure under a short Redis lock (<10ms).

    Applies all three risk-limit layers before inserting the reservation:
      Layer 1 — single order cap    (F_ORDER_CAP  = 3% of bankroll)
      Layer 2 — per-match cap       (F_MATCH_CAP  = 5% of bankroll)
      Layer 3 — total portfolio cap (F_TOTAL_CAP  = 20% of bankroll,
                                     includes outstanding RESERVED rows)

    The lock is released immediately after the INSERT, so other containers
    can reserve concurrently while this container executes its order.

    Args:
        db_pool: asyncpg connection pool.
        redis: Connected Redis client (used for the exposure lock).
        match_id: Goalserve match identifier.
        market_ticker: Kalshi market ticker being reserved.
        f_invest: Requested Kelly fraction of bankroll to invest.
        bankroll: Current bankroll balance.
        is_paper: Whether this is a paper-trading reservation.

    Returns:
        Reservation ID (int) on success, or ``None`` if any cap is exceeded.
    """
    async with exposure_lock(redis):
        # ── Layer 1: single order cap ────────────────────────────────────────
        amount = min(f_invest * bankroll, bankroll * F_ORDER_CAP)

        # ── Layer 2: per-match cap ───────────────────────────────────────────
        match_exposure = await get_match_exposure(db_pool, match_id)
        remaining_match = bankroll * F_MATCH_CAP - match_exposure
        amount = min(amount, max(0.0, remaining_match))

        # ── Layer 3: total portfolio cap (includes RESERVED amounts) ─────────
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT get_total_exposure($1) AS total",
                is_paper,
            )
        total_exposure = float(row["total"]) if row and row["total"] is not None else 0.0
        remaining_total = bankroll * F_TOTAL_CAP - total_exposure
        amount = min(amount, max(0.0, remaining_total))

        if amount <= 0.0:
            logger.info(
                "exposure_limit_exceeded",
                match_id=match_id,
                ticker=market_ticker,
                f_invest=round(f_invest, 4),
                bankroll=round(bankroll, 2),
            )
            return None

        # ── INSERT RESERVED row ──────────────────────────────────────────────
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO exposure_reservation
                    (match_id, market_ticker, amount, is_paper, status)
                VALUES ($1, $2, $3, $4, 'RESERVED')
                RETURNING id
                """,
                match_id,
                market_ticker,
                amount,
                is_paper,
            )

        reservation_id: int | None = int(row["id"]) if row else None

        logger.info(
            "exposure_reserved",
            match_id=match_id,
            ticker=market_ticker,
            amount=round(amount, 2),
            reservation_id=reservation_id,
        )
        return reservation_id
    # Lock released here — other containers can now reserve concurrently


async def confirm_reservation(
    db_pool: asyncpg.Pool,
    reservation_id: int,
    actual_amount: float,
) -> None:
    """Mark a reservation CONFIRMED and update to the actual fill amount.

    Called after a successful fill.  Updates the reservation to reflect
    the true dollar amount committed (which may be less than reserved if
    the fill was partial).

    Args:
        db_pool: asyncpg connection pool.
        reservation_id: ID returned by ``reserve_exposure``.
        actual_amount: Actual fill amount (fill_price × fill_quantity).
    """
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE exposure_reservation
            SET status = 'CONFIRMED', amount = $2
            WHERE id = $1
            """,
            reservation_id,
            actual_amount,
        )
    logger.info(
        "exposure_confirmed",
        reservation_id=reservation_id,
        actual_amount=round(actual_amount, 2),
    )


async def release_reservation(
    db_pool: asyncpg.Pool,
    reservation_id: int,
) -> None:
    """Mark a reservation RELEASED (order did not fill or was cancelled).

    Called when an order fails, is rejected, or returns zero quantity.
    Releasing removes the amount from the pessimistic exposure total so
    subsequent orders can use that capacity.

    Args:
        db_pool: asyncpg connection pool.
        reservation_id: ID returned by ``reserve_exposure``.
    """
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE exposure_reservation SET status = 'RELEASED' WHERE id = $1",
            reservation_id,
        )
    logger.info("exposure_released", reservation_id=reservation_id)


# ---------------------------------------------------------------------------
# Full reserve → execute → confirm/release flow
# ---------------------------------------------------------------------------


async def execute_with_reservation(
    signal: Any,
    amount: float,
    ob_sync: Any,
    model: Any,
) -> Any:
    """Full reserve → execute → confirm/release cycle.

    Phase 1 (Reserve, <10ms lock):
        Call ``reserve_exposure``.  If any cap is exceeded, return None.

    Phase 2 (Execute, no lock):
        Submit order via ``model.execution.submit_order``.  If an exception
        is raised, release the reservation and re-raise.

    Phase 3 (Confirm or Release):
        If fill received with qty > 0 → confirm with actual fill amount.
        Otherwise → release the reservation.

    Args:
        signal: Trading signal (provides market_ticker, direction, P_kalshi).
        amount: Dollar amount to invest (post-Kelly, pre-cap).
        ob_sync: Order book snapshot for this market.
        model: Live match model (provides db_pool, redis, execution,
               match_id, bankroll, is_paper).

    Returns:
        PaperFill | FillResult on success, ``None`` if cap exceeded or no fill.
    """
    db_pool: Any = getattr(model, "db_pool", None)
    redis: Any = getattr(model, "redis", None)
    execution: Any = getattr(model, "execution", None)

    if db_pool is None or redis is None:
        logger.warning(
            "execute_with_reservation_skipped_no_infra",
            match_id=getattr(model, "match_id", "unknown"),
            ticker=getattr(signal, "market_ticker", "unknown"),
        )
        return None

    bankroll: float = float(getattr(model, "bankroll", 0.0))
    f_invest = amount / bankroll if bankroll > 0.0 else 0.0

    # ── Phase 1: Reserve ─────────────────────────────────────────────────────
    reservation_id = await reserve_exposure(
        db_pool,
        redis,
        getattr(model, "match_id", ""),
        getattr(signal, "market_ticker", ""),
        f_invest,
        bankroll,
        bool(getattr(model, "is_paper", True)),
    )
    if reservation_id is None:
        return None

    if execution is None:
        await release_reservation(db_pool, reservation_id)
        return None

    # ── Phase 2: Execute (no lock held) ──────────────────────────────────────
    fill: Any = None
    try:
        fill = await execution.submit_order(signal, amount, ob_sync)
    except Exception:
        await release_reservation(db_pool, reservation_id)
        raise

    # ── Phase 3: Confirm or Release ───────────────────────────────────────────
    fill_qty = _extract_fill_quantity(fill)
    if fill is not None and fill_qty > 0:
        fill_price = _extract_fill_price(fill)
        actual_amount = fill_price * fill_qty
        await confirm_reservation(db_pool, reservation_id, actual_amount)
    else:
        await release_reservation(db_pool, reservation_id)
        return None

    return fill


# ---------------------------------------------------------------------------
# Internal fill helpers
# ---------------------------------------------------------------------------


def _extract_fill_quantity(fill: Any) -> int:
    """Extract fill quantity from PaperFill or FillResult."""
    if fill is None:
        return 0
    qty = getattr(fill, "quantity", None)
    if qty is not None:
        return int(qty)
    return int(getattr(fill, "fill_quantity", None) or 0)


def _extract_fill_price(fill: Any) -> float:
    """Extract fill price from PaperFill or FillResult."""
    if fill is None:
        return 0.0
    price = getattr(fill, "price", None)
    if price is not None:
        return float(price)
    return float(getattr(fill, "fill_price", None) or 0.0)
