"""Post-match settlement and realized P&L computation.

After match FINISHED, the engine waits for Kalshi to resolve each market,
computes directional P&L for every position, and persists the results.

Key design decisions:
  - Poll every 60 s, up to 6 hours (configurable via timeout_hours).
  - settlement_price is from the Yes perspective: 1.00 = Yes won, 0.00 = No won.
  - BUY_YES profit = (settlement - entry) × qty;  BUY_NO profit = (entry - settlement) × qty.
  - Fee (7%) applies only to gross profits, never to losses.
  - Positions remain in AWAITING_SETTLEMENT until resolution or timeout.
  - Timeout triggers an alert; already-resolved positions are still settled.

Reference: docs/phase4.md Step 4.6
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.clients.kalshi import KalshiClient
    from src.common.types import Position
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("settlement")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEE_RATE: float = 0.07          # Kalshi fee rate applied to gross profits
POLL_INTERVAL_S: float = 60.0   # seconds between settlement polls
DEFAULT_TIMEOUT_HOURS: float = 6.0


# ---------------------------------------------------------------------------
# compute_realized_pnl
# ---------------------------------------------------------------------------


def compute_realized_pnl(
    position: Position,
    settlement_price: float,
    fee_rate: float = FEE_RATE,
) -> float:
    """Compute direction-correct realized P&L for a settled position.

    Args:
        position: The closed position (entry_price, quantity, direction).
        settlement_price: Kalshi settlement from the Yes perspective.
                          1.00 = Yes outcome won, 0.00 = No outcome won.
        fee_rate: Fractional fee applied to gross profits (default 0.07).

    Returns:
        Net P&L in dollars (positive = profit, negative = loss).

    Examples (from docs/phase4.md validation table):
        BUY_YES  entry=0.45  settlement=1.00  qty=1 → gross=+0.55  fee=0.039  net=+0.511
        BUY_YES  entry=0.45  settlement=0.00  qty=1 → gross=-0.45  fee=0      net=-0.45
        BUY_NO   entry=0.40  settlement=0.00  qty=1 → gross=+0.40  fee=0.028  net=+0.372
        BUY_NO   entry=0.40  settlement=1.00  qty=1 → gross=-0.60  fee=0      net=-0.60
    """
    if position.direction == "BUY_YES":
        gross_pnl = (settlement_price - position.entry_price) * position.quantity
    elif position.direction == "BUY_NO":
        gross_pnl = (position.entry_price - settlement_price) * position.quantity
    else:
        gross_pnl = 0.0

    fee = fee_rate * max(0.0, gross_pnl)
    return gross_pnl - fee


# ---------------------------------------------------------------------------
# await_settlement
# ---------------------------------------------------------------------------


async def await_settlement(
    match_id: str,
    market_tickers: list[str],
    kalshi_client: KalshiClient,
    *,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
) -> dict[str, float]:
    """Poll Kalshi until all markets are resolved or timeout is reached.

    Args:
        match_id: Goalserve match ID (for logging).
        market_tickers: List of Kalshi tickers to poll.
        kalshi_client: Authenticated Kalshi API client.
        timeout_hours: Maximum time to wait before giving up (default 6h).

    Returns:
        Dict mapping ticker → settlement_price (Yes perspective: 1.00 or 0.00).
        Unresolved markets are absent from the returned dict.
    """
    deadline = time.time() + timeout_hours * 3600.0
    resolved: dict[str, float] = {}

    while time.time() < deadline:
        for ticker in market_tickers:
            if ticker in resolved:
                continue
            try:
                market: dict[str, Any] = await kalshi_client.get_market(ticker)
                if market.get("status") == "resolved":
                    settlement_price = float(market.get("settlement_price", 0.0))
                    resolved[ticker] = settlement_price
                    logger.info(
                        "market_resolved",
                        match_id=match_id,
                        ticker=ticker,
                        settlement_price=settlement_price,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "settlement_poll_failed",
                    match_id=match_id,
                    ticker=ticker,
                    error=str(exc),
                )

        if len(resolved) == len(market_tickers):
            logger.info(
                "all_markets_resolved",
                match_id=match_id,
                count=len(resolved),
            )
            return resolved

        await asyncio.sleep(POLL_INTERVAL_S)

    # Timeout — alert, return partial results
    unresolved = [t for t in market_tickers if t not in resolved]
    logger.error(
        "settlement_timeout",
        match_id=match_id,
        unresolved=unresolved,
        timeout_hours=timeout_hours,
    )
    logger.error(
        "manual_settlement_needed",
        match_id=match_id,
        unresolved=unresolved,
    )
    return resolved  # return what we have; unresolved positions stay AWAITING_SETTLEMENT


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _get_open_positions(db_pool: Any, match_id: str) -> list[dict[str, Any]]:
    """Return OPEN positions for the match as raw dicts."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, market_ticker, direction, entry_price, quantity,
                   status, is_paper, entry_time
            FROM positions
            WHERE match_id = $1
              AND status = 'OPEN'
            """,
            match_id,
        )
    return [dict(r) for r in rows]


async def _set_awaiting_settlement(db_pool: Any, position_id: int) -> None:
    """Transition a position to AWAITING_SETTLEMENT."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE positions SET status = 'AWAITING_SETTLEMENT' WHERE id = $1",
            position_id,
        )


async def _settle_position(
    db_pool: Any,
    position_id: int,
    settlement_price: float,
    realized_pnl: float,
) -> None:
    """Persist settlement price and P&L, transition to SETTLED."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE positions
            SET status            = 'SETTLED',
                settlement_price  = $1,
                realized_pnl      = $2,
                exit_time         = NOW()
            WHERE id = $3
            """,
            settlement_price,
            realized_pnl,
            position_id,
        )


async def _update_bankroll(db_pool: Any, is_paper: bool, pnl: float) -> None:
    """Increment the paper or live bankroll by realized P&L."""
    mode = "paper" if is_paper else "live"
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE bankroll
            SET balance = balance + $1
            WHERE mode = $2
            """,
            pnl,
            mode,
        )


# ---------------------------------------------------------------------------
# settle_all_positions
# ---------------------------------------------------------------------------


async def settle_all_positions(
    model: LiveFootballQuantModel,
    kalshi_client: KalshiClient,
) -> None:
    """Settle all open positions after match finishes.

    Flow:
      1. Fetch OPEN positions from DB.
      2. Transition each to AWAITING_SETTLEMENT.
      3. Poll Kalshi until all markets resolve (or timeout).
      4. Compute P&L per position and persist SETTLED status.
      5. Update bankroll.

    Args:
        model: Live match model (provides match_id, is_paper, db_pool).
        kalshi_client: Authenticated Kalshi API client.
    """
    if model.db_pool is None:
        logger.warning("settle_all_positions_skipped_no_db", match_id=model.match_id)
        return

    raw_positions = await _get_open_positions(model.db_pool, model.match_id)
    if not raw_positions:
        logger.info("settle_all_positions_no_open", match_id=model.match_id)
        return

    # Transition all to AWAITING_SETTLEMENT
    for row in raw_positions:
        await _set_awaiting_settlement(model.db_pool, row["id"])

    # Collect unique tickers to poll
    tickers = list({row["market_ticker"] for row in raw_positions})

    logger.info(
        "settlement_started",
        match_id=model.match_id,
        position_count=len(raw_positions),
        tickers=tickers,
    )

    settlements = await await_settlement(
        model.match_id,
        tickers,
        kalshi_client,
    )

    # Settle resolved positions
    for row in raw_positions:
        ticker = row["market_ticker"]
        if ticker not in settlements:
            continue  # unresolved — stays AWAITING_SETTLEMENT

        settlement_price = settlements[ticker]

        # Build a minimal Position-compatible object for P&L calc
        from src.common.types import Position

        pos = Position(
            match_id=model.match_id,
            market_ticker=ticker,
            direction=row["direction"],
            entry_price=float(row["entry_price"]),
            quantity=int(row["quantity"]),
            is_paper=bool(row["is_paper"]),
        )

        pnl = compute_realized_pnl(pos, settlement_price)

        await _settle_position(model.db_pool, row["id"], settlement_price, pnl)
        await _update_bankroll(model.db_pool, model.is_paper, pnl)

        logger.info(
            "position_settled",
            match_id=model.match_id,
            position_id=row["id"],
            ticker=ticker,
            direction=row["direction"],
            entry_price=float(row["entry_price"]),
            quantity=int(row["quantity"]),
            settlement_price=settlement_price,
            realized_pnl=round(pnl, 4),
        )

    logger.info(
        "settlement_complete",
        match_id=model.match_id,
        settled=len(settlements),
        total=len(raw_positions),
    )
