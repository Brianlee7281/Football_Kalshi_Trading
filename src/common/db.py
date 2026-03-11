"""asyncpg connection pool and DB helper functions.

All direct PostgreSQL access in the system flows through this module.
No raw SQL should appear outside of these helpers.

Reference: docs/orchestration.md Database Connection Resilience
"""

from __future__ import annotations

from typing import Any

import asyncpg

from src.common.logging import get_logger

logger = get_logger("db")

# ---------------------------------------------------------------------------
# Pool factory
# ---------------------------------------------------------------------------


async def create_pool(db_url: str) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Args:
        db_url: asyncpg-compatible PostgreSQL DSN.

    Returns:
        asyncpg.Pool with min=2, max=5 connections.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=2,
        max_size=5,
        command_timeout=10,
        max_inactive_connection_lifetime=300,
    )
    return pool


# ---------------------------------------------------------------------------
# Helper queries
# ---------------------------------------------------------------------------


async def get_bankroll(pool: asyncpg.Pool, mode: str) -> float:
    """Read current bankroll balance for paper or live mode.

    Args:
        pool: asyncpg connection pool.
        mode: ``"paper"`` or ``"live"``.

    Returns:
        Balance as a float, or 10_000.0 if no row exists.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT balance FROM bankroll WHERE mode = $1",
            mode,
        )
    return float(row["balance"]) if row else 10_000.0


async def get_match_exposure(pool: asyncpg.Pool, match_id: str) -> float:
    """Total open position cost for a single match (all tickers, all directions).

    Includes OPEN and AWAITING_SETTLEMENT positions; excludes PENDING
    (those are tracked separately via exposure_reservation).

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match identifier.

    Returns:
        Dollar amount currently at risk for this match.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(entry_price * quantity), 0.0) AS total
            FROM positions
            WHERE match_id = $1
              AND status IN ('OPEN', 'AWAITING_SETTLEMENT')
            """,
            match_id,
        )
    return float(row["total"]) if row else 0.0


async def get_existing_exposure(
    pool: asyncpg.Pool,
    match_id: str,
    ticker: str,
    direction: str,
) -> float:
    """Dollar exposure for a specific (match, ticker, direction) combination.

    Used by the incremental Kelly calculation to determine how much has
    already been allocated before reserving additional exposure.

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match identifier.
        ticker: Kalshi market ticker.
        direction: ``"BUY_YES"`` or ``"BUY_NO"``.

    Returns:
        Existing dollar exposure (entry_price × quantity summed).
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(entry_price * quantity), 0.0) AS total
            FROM positions
            WHERE match_id = $1
              AND market_ticker = $2
              AND direction = $3
              AND status IN ('OPEN', 'AWAITING_SETTLEMENT')
            """,
            match_id,
            ticker,
            direction,
        )
    return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# 2-phase write — safe_submit_order
# ---------------------------------------------------------------------------


async def safe_submit_order(
    signal: Any,
    amount: float,
    ob_sync: Any,
    model: Any,
) -> Any:
    """Submit an order using the 2-phase write pattern.

    Prevents ghost positions (order filled but not recorded) by writing
    PENDING intent to the DB *before* submitting to Kalshi.

    Phase A — record intent (PENDING):
        INSERT position with status='PENDING'.  If DB is down here,
        no order is submitted → safe.

    Phase B — execute:
        Submit order to Kalshi via model.execution.submit_order.
        DB not touched; no lock held.

    Phase C — confirm or cancel:
        If fill received → UPDATE position to OPEN with actual qty/price.
        If no fill or exception → DELETE the PENDING row.
        If DB is down here → PENDING row remains; reconciliation catches it.

    Args:
        signal: Trading signal (provides market_ticker, direction, P_kalshi).
        amount: Dollar amount to invest.
        ob_sync: Order book snapshot for this market.
        model: Live match model (provides match_id, is_paper, db_pool, execution).

    Returns:
        PaperFill | FillResult on success, None if Phase A fails or no fill.
    """
    db_pool: Any = getattr(model, "db_pool", None)
    execution: Any = getattr(model, "execution", None)
    match_id: str = getattr(model, "match_id", "")
    is_paper: bool = getattr(model, "is_paper", True)

    market_ticker: str = getattr(signal, "market_ticker", "")
    direction: str = getattr(signal, "direction", "")
    p_kalshi: float = getattr(signal, "P_kalshi", 0.0)
    quantity = int(amount / p_kalshi) if p_kalshi > 0 else 0

    # ── Phase A: write PENDING ───────────────────────────────────────────────
    position_id: int | None = None
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO positions
                    (match_id, market_ticker, direction,
                     entry_price, quantity, status, is_paper, entry_time)
                VALUES ($1, $2, $3, $4, $5, 'PENDING', $6, NOW())
                RETURNING id
                """,
                match_id,
                market_ticker,
                direction,
                p_kalshi,
                quantity,
                is_paper,
            )
        position_id = int(row["id"]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "db_write_failed_blocking_order",
            match_id=match_id,
            ticker=market_ticker,
            error=str(exc),
        )
        return None  # safe: no order submitted

    if position_id is None:
        logger.error(
            "db_insert_no_id",
            match_id=match_id,
            ticker=market_ticker,
        )
        return None

    # ── Phase B: execute (no lock, no DB touch) ──────────────────────────────
    fill: Any = None
    try:
        fill = await execution.submit_order(signal, amount, ob_sync)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "order_execution_failed",
            match_id=match_id,
            ticker=market_ticker,
            position_id=position_id,
            error=str(exc),
        )
        await _delete_pending(db_pool, position_id, match_id, market_ticker)
        return None

    # ── Phase C: confirm or cancel ───────────────────────────────────────────
    fill_qty = _extract_fill_quantity(fill)
    fill_price = _extract_fill_price(fill)

    if fill is not None and fill_qty > 0:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE positions
                    SET status = 'OPEN', quantity = $2, entry_price = $3
                    WHERE id = $1
                    """,
                    position_id,
                    fill_qty,
                    fill_price,
                )
            logger.info(
                "position_opened",
                match_id=match_id,
                ticker=market_ticker,
                position_id=position_id,
                quantity=fill_qty,
                price=round(fill_price, 4),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "db_update_failed_pending_remains",
                match_id=match_id,
                ticker=market_ticker,
                position_id=position_id,
                error=str(exc),
            )
        return fill
    else:
        await _delete_pending(db_pool, position_id, match_id, market_ticker)
        return None


async def _delete_pending(
    db_pool: Any,
    position_id: int,
    match_id: str,
    market_ticker: str,
) -> None:
    """Delete a PENDING position row after an execution failure or no-fill."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM positions WHERE id = $1",
                position_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "db_delete_pending_failed",
            match_id=match_id,
            ticker=market_ticker,
            position_id=position_id,
            error=str(exc),
        )


def _extract_fill_quantity(fill: Any) -> int:
    """Extract fill quantity from PaperFill or FillResult."""
    if fill is None:
        return 0
    # PaperFill has .quantity; FillResult has .fill_quantity
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


# ---------------------------------------------------------------------------
# Stale PENDING reconciliation
# ---------------------------------------------------------------------------


async def reconcile_stale_pending(
    pool: asyncpg.Pool,
    match_id: str,
    *,
    max_age_minutes: int = 5,
) -> list[int]:
    """Find and log PENDING positions older than ``max_age_minutes``.

    Called on container startup to detect positions that were left in PENDING
    state by a previous crash (Phase C DB failure).  Logs each one for manual
    review; does not automatically modify or delete them.

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match identifier.
        max_age_minutes: Age threshold in minutes (default 5).

    Returns:
        List of stale position IDs (empty if none found).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, market_ticker, direction, quantity, entry_price, entry_time
            FROM positions
            WHERE match_id = $1
              AND status = 'PENDING'
              AND entry_time < NOW() - ($2 * INTERVAL '1 minute')
            ORDER BY entry_time
            """,
            match_id,
            max_age_minutes,
        )

    stale_ids: list[int] = []
    for row in rows:
        position_id = int(row["id"])
        stale_ids.append(position_id)
        logger.warning(
            "stale_pending_position_detected",
            match_id=match_id,
            position_id=position_id,
            ticker=row["market_ticker"],
            direction=row["direction"],
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            entry_time=str(row["entry_time"]),
            action="manual_review_required",
        )

    if stale_ids:
        logger.error(
            "stale_pending_positions_found",
            match_id=match_id,
            count=len(stale_ids),
            position_ids=stale_ids,
        )

    return stale_ids
