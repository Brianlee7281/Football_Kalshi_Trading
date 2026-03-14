"""Phase 4 signal generator — multi-market orchestration loop.

Consumes TickData from model.phase4_queue (emitted by Phase 3 each tick),
decomposes the P_true / σ_MC dicts into per-ticker floats, generates signals,
computes incremental Kelly sizing, and executes orders via the reservation
pattern:

    Reserve  (DB lock <10ms) → Execute (no lock, 1-5s) → Confirm/Release

The loop is sequential per tick: ticker B's Kelly reads correct exposure
only after ticker A's fill has been confirmed.  This prevents within-container
race conditions.

Reference: docs/phase4.md signal_generator section
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger
from src.execution.edge_detection import generate_signal
from src.execution.kelly import F_MATCH_CAP, F_ORDER_CAP, F_TOTAL_CAP, compute_kelly

if TYPE_CHECKING:
    from src.common.types import FillResult, PaperFill, Signal
    from src.engine.model import LiveFootballQuantModel
    from src.execution.order_book_sync import OrderBookSync

logger = get_logger("signal_generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINISHED = "FINISHED"
_DIAG_LOG_INTERVAL = 30  # Log diagnostics every N ticks (1 tick ≈ 1 sec)


# ---------------------------------------------------------------------------
# DB helpers (thin wrappers over model.db_pool)
# ---------------------------------------------------------------------------


async def _get_existing_exposure(
    db_pool: Any,
    match_id: str,
    ticker: str,
    direction: str,
) -> float:
    """Return total open + reserved dollar exposure for (match, ticker, direction).

    Queries only OPEN positions — exposure reservations are tracked separately
    via the exposure_reservation table and included in reserve_exposure checks.
    """
    async with db_pool.acquire() as conn:
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


async def _get_match_exposure(db_pool: Any, match_id: str) -> float:
    """Total open position cost for the match (all tickers, all directions)."""
    async with db_pool.acquire() as conn:
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


async def _get_total_exposure(db_pool: Any, is_paper: bool) -> float:
    """Total cross-match exposure including outstanding reservations."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT get_total_exposure($1) AS total",
            is_paper,
        )
    return float(row["total"]) if row else 0.0


async def _reserve_exposure(
    db_pool: Any,
    redis: Any,
    match_id: str,
    ticker: str,
    f_invest: float,
    bankroll: float,
    is_paper: bool,
) -> int | None:
    """Phase 1 of reserve-confirm-release. Returns reservation_id or None.

    Holds a short Redis lock while reading cross-container exposure totals
    and inserting the reservation row.  Lock is released before execution.
    """
    async with redis.lock("exposure_lock", timeout=2):
        amount = min(f_invest * bankroll, bankroll * F_ORDER_CAP)

        match_exposure = await _get_match_exposure(db_pool, match_id)
        remaining_match = bankroll * F_MATCH_CAP - match_exposure
        amount = min(amount, max(0.0, remaining_match))

        total_exposure = await _get_total_exposure(db_pool, is_paper)
        remaining_total = bankroll * F_TOTAL_CAP - total_exposure
        amount = min(amount, max(0.0, remaining_total))

        if amount <= 0.0:
            return None

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO exposure_reservation
                    (match_id, market_ticker, amount, is_paper, status)
                VALUES ($1, $2, $3, $4, 'RESERVED')
                RETURNING id
                """,
                match_id,
                ticker,
                amount,
                is_paper,
            )
        return int(row["id"]) if row else None


async def _confirm_reservation(db_pool: Any, reservation_id: int, actual_amount: float) -> None:
    """Mark reservation CONFIRMED and update to actual fill amount."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE exposure_reservation
            SET status = 'CONFIRMED', amount = $1
            WHERE id = $2
            """,
            actual_amount,
            reservation_id,
        )


async def _release_reservation(db_pool: Any, reservation_id: int) -> None:
    """Mark reservation RELEASED (order did not fill)."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE exposure_reservation SET status = 'RELEASED' WHERE id = $1",
            reservation_id,
        )


# ---------------------------------------------------------------------------
# execute_with_reservation
# ---------------------------------------------------------------------------


async def execute_with_reservation(
    signal: Signal,
    amount: float,
    ob_sync: OrderBookSync,
    model: LiveFootballQuantModel,
) -> PaperFill | FillResult | None:
    """Full reserve → execute → confirm/release cycle.

    Phase 1 (Reserve, <10ms lock):  write RESERVED row; lock released
    Phase 2 (Execute, no lock):     submit order to Kalshi (1–5s)
    Phase 3 (Confirm/Release):      update reservation to actual fill amount

    Args:
        signal: Trading signal for this market.
        amount: Dollar amount to invest (post-Kelly, pre-reservation).
        ob_sync: Order book for this market.
        model: Live match model (provides db_pool, redis, execution router).

    Returns:
        PaperFill | FillResult on success, None if reservation denied or
        order not filled.
    """
    if model.db_pool is None or model.redis is None:
        logger.warning(
            "execute_with_reservation_skipped_no_infra",
            match_id=model.match_id,
            ticker=signal.market_ticker,
        )
        return None

    f_invest = amount / model.bankroll if model.bankroll > 0.0 else 0.0

    reservation_id = await _reserve_exposure(
        model.db_pool,
        model.redis,
        model.match_id,
        signal.market_ticker,
        f_invest,
        model.bankroll,
        model.is_paper,
    )
    if reservation_id is None:
        logger.info(
            "exposure_limit_exceeded",
            match_id=model.match_id,
            ticker=signal.market_ticker,
            f_invest=round(f_invest, 4),
        )
        return None

    if model.execution is None:
        await _release_reservation(model.db_pool, reservation_id)
        return None

    fill: PaperFill | FillResult | None = None
    try:
        fill = await model.execution.submit_order(signal, amount, ob_sync)
    except Exception:
        await _release_reservation(model.db_pool, reservation_id)
        raise

    if fill is not None and _fill_quantity(fill) > 0:
        actual_amount = _fill_price(fill) * _fill_quantity(fill)
        await _confirm_reservation(model.db_pool, reservation_id, actual_amount)
    else:
        await _release_reservation(model.db_pool, reservation_id)

    return fill


def _fill_quantity(fill: PaperFill | FillResult) -> int:
    """Extract fill quantity from either fill type."""
    from src.common.types import PaperFill

    if isinstance(fill, PaperFill):
        return fill.quantity
    return fill.fill_quantity or 0


def _fill_price(fill: PaperFill | FillResult) -> float:
    """Extract fill price from either fill type."""
    from src.common.types import PaperFill

    if isinstance(fill, PaperFill):
        return fill.price
    return fill.fill_price or 0.0


# ---------------------------------------------------------------------------
# Redis publish (fire-and-forget)
# ---------------------------------------------------------------------------


async def _publish_signal_to_redis(
    model: LiveFootballQuantModel,
    ticker: str,
    signal: Signal,
    fill: PaperFill | FillResult,
) -> None:
    """Publish signal + fill to Redis for live dashboard.

    Publishes to two channels:
      signal:{match_id}  — signal log stream
      position_update    — position table refresh trigger
    """
    if model.redis is None:
        return
    try:
        await model.redis.publish(
            f"signal:{model.match_id}",
            json.dumps(
                {
                    "type": "signal",
                    "match_id": model.match_id,
                    "ticker": ticker,
                    "direction": signal.direction,
                    "EV": signal.EV,
                    "P_cons": signal.P_cons,
                    "P_kalshi": signal.P_kalshi,
                    "alignment": signal.alignment_status,
                    "kelly_fraction": signal.kelly_multiplier,
                    "fill_qty": _fill_quantity(fill),
                    "fill_price": _fill_price(fill),
                    "timestamp": time.time(),
                }
            ),
        )
        await model.redis.publish(
            "position_update",
            json.dumps(
                {
                    "type": "new_fill",
                    "match_id": model.match_id,
                    "ticker": ticker,
                    "direction": signal.direction,
                    "quantity": _fill_quantity(fill),
                    "price": _fill_price(fill),
                }
            ),
        )
    except Exception:  # noqa: BLE001
        pass  # fire-and-forget; Prometheus counter tracked in emit.py


# ---------------------------------------------------------------------------
# Signal generator — main loop
# ---------------------------------------------------------------------------


async def signal_generator(model: LiveFootballQuantModel) -> None:
    """Core Phase 4 loop.

    Runs continuously until match FINISHED.  Each iteration consumes one
    TickData from model.phase4_queue (pushed by Phase 3 each second), then
    processes ALL active Kalshi markets sequentially.

    Per-tick per-market flow:
      1. Decompose P_true dict → float for this market key
      2. Get order book snapshot (skip if stale or no liquidity)
      3. Generate trading signal (EV, direction, alignment)
      4. Compute incremental Kelly fraction
      5. execute_with_reservation (reserve → execute → confirm/release)
      6. Decrement model.bankroll by fill cost
      7. Publish fill to Redis (fire-and-forget)
    """
    logger.info("signal_generator_started", match_id=model.match_id)

    tick_count = 0

    while model.engine_phase != FINISHED:
        tick_data = await model.phase4_queue.get()
        tick_count += 1

        p_true_dict: dict[str, float] = tick_data.P_true
        sigma_mc_dict: dict[str, float] = tick_data.sigma_MC
        order_allowed: bool = tick_data.order_allowed
        should_log_diag = (tick_count % _DIAG_LOG_INTERVAL == 0)

        if not order_allowed:
            if should_log_diag:
                logger.info(
                    "signal_diag",
                    match_id=model.match_id,
                    tick=tick_count,
                    reason="ORDER_BLOCKED",
                    order_allowed=False,
                )
            continue

        for ticker in model.active_tickers:
            market_key = model.ticker_to_model_key.get(ticker)
            if market_key is None or market_key not in p_true_dict:
                continue

            # ── Dict → float decomposition ─────────────────────────────
            p_true_float = p_true_dict[market_key]
            sigma_mc_float = sigma_mc_dict.get(market_key, 0.0)

            # ── Order book gate ────────────────────────────────────────
            ob_sync = model.ob_syncs.get(ticker)
            if ob_sync is None or not ob_sync.liquidity_ok():
                if should_log_diag:
                    logger.info(
                        "signal_diag",
                        match_id=model.match_id,
                        tick=tick_count,
                        ticker=ticker,
                        P_true=round(p_true_float, 4),
                        reason="NO_OB" if ob_sync is None else "LOW_LIQUIDITY",
                    )
                continue

            p_bet365 = model.bet365_implied.get(market_key)

            # ── Step 4.2: Signal generation ────────────────────────────
            signal = generate_signal(
                p_true_float,
                sigma_mc_float,
                ob_sync,
                p_bet365,
                c=model.config.fee_rate,
                z=model.config.z,
                K_frac=model.config.K_frac,
                bankroll=model.bankroll,
                market_ticker=ticker,
            )

            if signal.direction == "HOLD":
                if should_log_diag:
                    # Extract best bid/ask for diagnostics
                    best_ask = ob_sync.kalshi_best_ask
                    best_bid = ob_sync.kalshi_best_bid
                    logger.info(
                        "signal_diag",
                        match_id=model.match_id,
                        tick=tick_count,
                        ticker=ticker,
                        P_true=round(p_true_float, 4),
                        sigma_MC=round(sigma_mc_float, 5),
                        P_kalshi_ask=best_ask,
                        P_kalshi_bid=best_bid,
                        direction="HOLD",
                        EV=0.0,
                        reason="NO_EDGE",
                    )
                continue

            # ── BUY signal: always log full details ────────────────────
            logger.info(
                "signal_buy",
                match_id=model.match_id,
                tick=tick_count,
                ticker=ticker,
                direction=signal.direction,
                P_true=round(p_true_float, 4),
                sigma_MC=round(sigma_mc_float, 5),
                P_cons=round(signal.P_cons, 4),
                P_kalshi=round(signal.P_kalshi, 4),
                EV=round(signal.EV, 5),
                rough_qty=signal.rough_qty,
                alignment=signal.alignment_status,
                kelly_multiplier=signal.kelly_multiplier,
            )

            # ── Step 4.3: Incremental Kelly ────────────────────────────
            existing = 0.0
            if model.db_pool is not None:
                with contextlib.suppress(Exception):
                    existing = await _get_existing_exposure(
                        model.db_pool,
                        model.match_id,
                        ticker,
                        signal.direction,
                    )

            f_incremental = compute_kelly(
                signal,
                model.config.fee_rate,
                model.config.K_frac,
                existing_exposure=existing,
                bankroll=model.bankroll,
            )

            if f_incremental <= 0.0:
                logger.info(
                    "signal_kelly_zero",
                    match_id=model.match_id,
                    ticker=ticker,
                    direction=signal.direction,
                    existing_exposure=round(existing, 2),
                    bankroll=round(model.bankroll, 2),
                )
                continue  # already at or above optimal allocation

            amount = f_incremental * model.bankroll

            # ── Reserve → Execute → Confirm/Release ───────────────────
            fill = await execute_with_reservation(signal, amount, ob_sync, model)

            # ── Bankroll refresh ───────────────────────────────────────
            if fill is not None and _fill_quantity(fill) > 0:
                fill_cost = _fill_price(fill) * _fill_quantity(fill)
                model.bankroll -= fill_cost
                logger.info(
                    "bankroll_updated",
                    match_id=model.match_id,
                    ticker=ticker,
                    fill_cost=round(fill_cost, 2),
                    remaining=round(model.bankroll, 2),
                )

                # ── Redis publish (fire-and-forget) ────────────────────
                asyncio.create_task(
                    _publish_signal_to_redis(model, ticker, signal, fill)
                )

    logger.info("signal_generator_finished", match_id=model.match_id)
