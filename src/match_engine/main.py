"""Match container entry point.

Runs Phase 3 (pricing) + Phase 4 (execution) concurrently for a single match.
Controlled by environment variables injected by the orchestrator.

Startup sequence:
  1. Load MatchEngineConfig from environment
  2. Connect to PostgreSQL (asyncpg pool) and Redis
  3. Build LiveFootballQuantModel from Phase 2 results in DB
  4. Inject ExecutionRouter (paper or live)
  5. asyncio.gather all Phase 3 + Phase 4 coroutines

If any coroutine raises an unhandled exception, all others are cancelled and
the process exits with code 1 so the orchestrator can detect the failure.

Reference: docs/orchestration.md Container Entry Point
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import asyncpg
import redis.asyncio as aioredis

from src.common.logging import get_logger
from src.engine.event_sources import (
    live_odds_listener,
    live_score_poller,
    order_book_sync_loop,
)
from src.engine.model import LiveFootballQuantModel, Phase4Config
from src.engine.tick_loop import tick_loop
from src.execution.execution_router import ExecutionRouter
from src.execution.exit_monitor import exit_monitor
from src.execution.order_book_sync import OrderBookSync
from src.execution.settlement import settle_all_positions
from src.execution.signal_generator import signal_generator
from src.match_engine.config import MatchEngineConfig
from src.match_engine.heartbeat import _emit_final_heartbeat, heartbeat_emitter

logger = get_logger("match_engine")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


async def _connect_db(db_url: str) -> asyncpg.Pool:
    """Create asyncpg connection pool (min=2, max=5)."""
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=db_url,
        min_size=2,
        max_size=5,
        command_timeout=10,
    )
    return pool


async def _connect_redis(redis_url: str) -> aioredis.Redis:
    """Create Redis client."""
    client: aioredis.Redis = aioredis.from_url(  # type: ignore[no-untyped-call]
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    return client


async def _load_model(
    config: MatchEngineConfig,
    db_pool: asyncpg.Pool,
) -> LiveFootballQuantModel:
    """Build LiveFootballQuantModel from Phase 2 DB results and pinned params.

    Loads production_params from the DB (pinned by param_version), then
    initializes the model with a_H, a_A, C_time from environment variables
    (injected by the orchestrator from Phase 2 results).
    """
    import json as _json

    import numpy as np

    from src.common.types import Phase2Result

    # Load production_params from DB (global — no league_id column)
    params: dict[str, Any] = {}
    try:
        async with db_pool.acquire() as conn:
            if config.param_version > 0:
                row = await conn.fetchrow(
                    "SELECT params FROM production_params "
                    "WHERE version = $1",
                    config.param_version,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT params FROM production_params "
                    "WHERE is_active = TRUE "
                    "ORDER BY version DESC LIMIT 1",
                )
            if row:
                raw = row["params"]
                params = _json.loads(raw) if isinstance(raw, str) else dict(raw)
                logger.info(
                    "production_params_loaded",
                    match_id=config.match_id,
                    param_version=config.param_version,
                    param_keys=list(params.keys())[:10],
                )
            else:
                logger.warning(
                    "production_params_not_found",
                    match_id=config.match_id,
                    param_version=config.param_version,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "production_params_load_failed",
            match_id=config.match_id,
            error=str(exc),
        )

    # Build Phase2Result from env vars (injected by orchestrator)
    phase2_result = Phase2Result(
        a_H=config.a_H,
        a_A=config.a_A,
        C_time=config.C_time,
        verdict="GO",
    )

    # If we have production_params, use from_phase2 for full initialization
    if params and "b" in params:
        model = LiveFootballQuantModel.from_phase2(
            phase2_result,
            params,
            match_id=config.match_id,
            league_id=int(config.league_id) if config.league_id else 0,
            trading_mode=config.trading_mode,
        )
    else:
        # Fallback: minimal model with Phase 2 intensities but no MMPP params
        logger.warning(
            "model_fallback_no_params",
            match_id=config.match_id,
            a_H=config.a_H,
            a_A=config.a_A,
        )
        model = LiveFootballQuantModel(
            match_id=config.match_id,
            league_id=int(config.league_id) if config.league_id else 0,
            trading_mode=config.trading_mode,
            a_H=config.a_H,
            a_A=config.a_A,
            C_time=config.C_time,
            b=np.ones(6, dtype=np.float64),
            gamma_H=np.ones(4, dtype=np.float64),
            gamma_A=np.ones(4, dtype=np.float64),
        )

    model.config = Phase4Config(
        fee_rate=config.fee_rate,
        z=config.z,
        K_frac=config.K_frac,
    )

    return model


async def _load_bankroll(
    db_pool: asyncpg.Pool,
    trading_mode: str,
) -> float:
    """Read current bankroll for paper or live mode from DB.

    Defaults to 10,000 if the bankroll table doesn't exist yet (Sprint 5).
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT balance FROM bankroll WHERE mode = $1",
                trading_mode,
            )
        return float(row["balance"]) if row else 10_000.0
    except Exception:  # noqa: BLE001
        return 10_000.0


async def _load_ticker_mapping(
    db_pool: asyncpg.Pool,
    match_id: str,
) -> dict[str, str]:
    """Load Kalshi ticker → model key mapping from DB.

    Returns empty dict if no mapping registered yet (Sprint 5 fallback).
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT kalshi_ticker, model_key FROM ticker_mapping WHERE match_id = $1",
                match_id,
            )
        return {row["kalshi_ticker"]: row["model_key"] for row in rows}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Async entry point for the match container."""
    # ── Top-level try/except: catch ANY startup crash ──────────────────────
    # Without this, import errors, missing env vars, or DB connection failures
    # produce an unlogged traceback to stderr and exit(1).
    try:
        config = MatchEngineConfig.from_env()
    except Exception:
        import traceback

        print(f"FATAL: MatchEngineConfig.from_env() failed:\n{traceback.format_exc()}", flush=True)
        logger.error("config_load_failed", error=traceback.format_exc())
        sys.exit(1)

    logger.info(
        "match_engine_starting",
        match_id=config.match_id,
        trading_mode=config.trading_mode,
        param_version=config.param_version,
        tickers=config.kalshi_tickers,
        a_H=config.a_H,
        a_A=config.a_A,
        C_time=config.C_time,
        db_url=config.db_url[:30] + "***" if config.db_url else "(empty)",
        redis_url=config.redis_url[:20] + "***" if config.redis_url else "(empty)",
        kalshi_key_path=config.kalshi_private_key_path,
    )

    # ── Infrastructure connections ──────────────────────────────────────────
    try:
        db_pool = await _connect_db(config.db_url)
    except Exception:
        import traceback

        logger.error(
            "db_connect_failed",
            match_id=config.match_id,
            db_url=config.db_url[:30] + "***",
            error=traceback.format_exc(),
        )
        sys.exit(1)

    try:
        redis = await _connect_redis(config.redis_url)
    except Exception:
        import traceback

        logger.error(
            "redis_connect_failed",
            match_id=config.match_id,
            redis_url=config.redis_url[:20] + "***",
            error=traceback.format_exc(),
        )
        await db_pool.close()
        sys.exit(1)

    # ── Model initialization ────────────────────────────────────────────────
    try:
        model = await _load_model(config, db_pool)
    except Exception:
        import traceback

        logger.error(
            "model_load_failed",
            match_id=config.match_id,
            error=traceback.format_exc(),
        )
        await db_pool.close()
        await redis.aclose()
        sys.exit(1)

    model.db_pool = db_pool
    model.redis = redis
    model.bankroll = await _load_bankroll(db_pool, config.trading_mode)

    # ── Parameter version pinning ───────────────────────────────────────────
    # Pin at startup — never reload mid-match (see patterns.md #5)
    # Sprint 6: load MMPP params from production_params table.
    logger.info(
        "param_version_pinned",
        match_id=config.match_id,
        param_version=config.param_version,
    )

    # ── Multi-market setup ──────────────────────────────────────────────────
    model.active_tickers = list(config.kalshi_tickers)
    model.ticker_to_model_key = await _load_ticker_mapping(db_pool, config.match_id)
    model.ob_syncs = {ticker: OrderBookSync(ticker=ticker) for ticker in model.active_tickers}

    # ── Kalshi client (needed for OB streaming in both paper and live) ───────
    kalshi_client = None
    if config.kalshi_api_key:
        import os
        from pathlib import Path as _Path

        from src.clients.kalshi import KalshiClient

        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        if private_key_path and _Path(private_key_path).is_file():
            kalshi_client = KalshiClient(
                api_key=config.kalshi_api_key,
                private_key_path=private_key_path,
            )
            logger.info(
                "kalshi_client_created",
                match_id=config.match_id,
                trading_mode=config.trading_mode,
                private_key_path=private_key_path,
            )
        else:
            logger.warning(
                "kalshi_private_key_not_found",
                match_id=config.match_id,
                private_key_path=private_key_path,
            )
    else:
        logger.warning(
            "kalshi_client_missing",
            match_id=config.match_id,
            reason="KALSHI_API_KEY not set — order book will not update",
        )

    model.kalshi_client = kalshi_client

    # ── Execution router (paper / live) ─────────────────────────────────────
    model.execution = ExecutionRouter(
        config.trading_mode,
        model,
        kalshi_client=kalshi_client,
    )

    logger.info(
        "match_engine_ready",
        match_id=config.match_id,
        bankroll=model.bankroll,
        trading_mode=config.trading_mode,
        market_count=len(model.active_tickers),
    )

    # ── Phase 3 + Phase 4: concurrent live engine ───────────────────────────
    # All coroutines run until model.engine_phase == FINISHED.
    # If any coroutine raises, cancel all others and exit with code 1.
    tasks: list[asyncio.Task[None]] = []
    try:
        tasks = [
            asyncio.create_task(tick_loop(model), name="tick_loop"),
            asyncio.create_task(live_odds_listener(model), name="live_odds"),
            asyncio.create_task(live_score_poller(model), name="live_score"),
            asyncio.create_task(order_book_sync_loop(model), name="ob_sync"),
            asyncio.create_task(signal_generator(model), name="signal_gen"),
            asyncio.create_task(exit_monitor(model), name="exit_monitor"),
            asyncio.create_task(heartbeat_emitter(model), name="heartbeat"),
        ]

        # Wait for all; the first exception propagates
        await asyncio.gather(*tasks)

    except Exception as exc:
        logger.error(
            "match_engine_crashed",
            match_id=config.match_id,
            error=str(exc),
            exc_info=True,
        )
        for t in tasks:
            if not t.done():
                t.cancel()
        # Wait for cancellations to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        await _emit_final_heartbeat(model)
        await db_pool.close()
        await redis.aclose()
        sys.exit(1)

    # ── Post-match settlement ───────────────────────────────────────────────
    logger.info(
        "settlement_phase_start",
        match_id=config.match_id,
        trading_mode=config.trading_mode,
        has_kalshi_client=kalshi_client is not None,
    )

    if kalshi_client is not None:
        try:
            await settle_all_positions(model, kalshi_client)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "settlement_failed",
                match_id=config.match_id,
                error=str(exc),
            )
    else:
        logger.info(
            "settlement_skipped_no_client",
            match_id=config.match_id,
            reason="No Kalshi client — paper positions stay OPEN until manual settlement",
        )

    await _emit_final_heartbeat(model)

    logger.info("match_engine_finished", match_id=config.match_id)

    await db_pool.close()
    await redis.aclose()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Synchronous wrapper called by ``python -m src.match_engine.main``."""
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception:
        # Last-resort safety net: if main() throws before any structured
        # logging is set up, print the traceback to stderr so it appears
        # in container logs (captured by orchestrator before removal).
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
