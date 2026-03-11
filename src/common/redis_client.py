"""Redis connection, pubsub helpers, and lock wrapper.

Centralises all Redis interactions except the heartbeat (match_engine/heartbeat.py)
and raw signal publishing already wired in signal_generator.py.

Channels used:
  tick:{match_id}     — Phase 3 tick data for the dashboard (every 1s)
  signal:{match_id}   — Phase 4 signal + fill events for the dashboard
  position_update     — Trigger for dashboard position-table refresh

Reference: docs/orchestration.md Container-Orchestrator Communication
           docs/dashboard.md Container → Redis Publishing
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis

from src.common.logging import get_logger

logger = get_logger("redis_client")

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


async def create_client(redis_url: str) -> aioredis.Redis:
    """Create and return an async Redis client.

    Args:
        redis_url: Redis URL (e.g. ``redis://redis:6379``).

    Returns:
        Connected ``redis.asyncio.Redis`` instance with UTF-8 decode.
    """
    client: aioredis.Redis = aioredis.from_url(  # type: ignore[no-untyped-call]
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    return client


# ---------------------------------------------------------------------------
# Lock wrapper — used by exposure reservation
# ---------------------------------------------------------------------------


@asynccontextmanager
async def exposure_lock(
    redis: aioredis.Redis,
    *,
    timeout: float = 2.0,
) -> AsyncGenerator[None, None]:
    """Async context manager that acquires the cross-container exposure lock.

    The lock is held only for the duration of the DB read + reservation INSERT
    (<10ms).  Order execution happens *outside* the lock.

    Args:
        redis: Connected Redis client.
        timeout: Lock timeout in seconds (default 2.0).  If the lock cannot
                 be acquired within this window, ``redis.lock`` raises
                 ``LockNotOwnedError``.

    Yields:
        Nothing — callers use it as ``async with exposure_lock(redis):``.

    Example::

        async with exposure_lock(redis):
            # read exposure, insert reservation
            ...
        # lock released — order execution starts here
    """
    lock = redis.lock("exposure_lock", timeout=timeout)
    async with lock:
        yield


# ---------------------------------------------------------------------------
# Dashboard publish helpers
# ---------------------------------------------------------------------------


async def publish_tick_to_dashboard(
    redis: aioredis.Redis,
    model: Any,
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
) -> None:
    """Publish a pricing tick to Redis for live dashboard consumption.

    Publishes to channel ``tick:{match_id}``.  Called every second from the
    Phase 3 tick loop (fire-and-forget; exceptions are logged and suppressed).

    JSON key is ``sigma_MC`` (not the Greek σ) to match TypeScript types.

    Args:
        redis: Connected Redis client.
        model: Live match model (provides match_id, t, engine_phase, etc.).
        P_true: Per-market true probabilities (e.g. ``{"home_win": 0.55}``).
        sigma_MC: Per-market MC uncertainty (e.g. ``{"home_win": 0.0022}``).
        order_allowed: Whether orders are allowed this tick.
    """
    try:
        payload = json.dumps(
            {
                "type": "tick",
                "match_id": model.match_id,
                "t": model.t,
                "engine_phase": model.engine_phase,
                "P_true": P_true,
                "sigma_MC": sigma_MC,
                "P_bet365": getattr(model, "bet365_implied", {}),
                "order_allowed": order_allowed,
                "cooldown": getattr(model, "cooldown", False),
                "ob_freeze": getattr(model, "ob_freeze", False),
                "event_state": getattr(model, "event_state", "IDLE"),
                "mu_H": getattr(model, "μ_H", 0.0),
                "mu_A": getattr(model, "μ_A", 0.0),
                "score": list(getattr(model, "S", [0, 0])),
            }
        )
        await redis.publish(f"tick:{model.match_id}", payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "redis_tick_publish_failed",
            match_id=getattr(model, "match_id", "unknown"),
            error=str(exc),
        )


async def publish_signal_to_dashboard(
    redis: aioredis.Redis,
    model: Any,
    ticker: str,
    signal: Any,
) -> None:
    """Publish a trading signal event to Redis for live dashboard consumption.

    Publishes to channel ``signal:{match_id}``.  Called from Phase 4
    signal_generator after a fill is confirmed (fire-and-forget).

    Args:
        redis: Connected Redis client.
        model: Live match model (provides match_id).
        ticker: Kalshi market ticker (e.g. ``"SOCC-EPL-ARS-CHE-WINNER"``).
        signal: Trading signal (provides direction, EV, P_cons, P_kalshi,
                alignment_status, kelly_multiplier).
    """
    try:
        payload = json.dumps(
            {
                "type": "signal",
                "match_id": model.match_id,
                "ticker": ticker,
                "direction": signal.direction,
                "EV": signal.EV,
                "P_cons": signal.P_cons,
                "P_kalshi": signal.P_kalshi,
                "alignment": signal.alignment_status,
                "kelly_multiplier": signal.kelly_multiplier,
                "timestamp": time.time(),
            }
        )
        await redis.publish(f"signal:{model.match_id}", payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "redis_signal_publish_failed",
            match_id=getattr(model, "match_id", "unknown"),
            ticker=ticker,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Subscribe helper (used by dashboard WebSocket route)
# ---------------------------------------------------------------------------


async def subscribe_to_channels(
    redis: aioredis.Redis,
    channels: list[str],
) -> aioredis.client.PubSub:
    """Create a PubSub object subscribed to the given channels.

    The caller is responsible for unsubscribing and closing the PubSub
    when done (e.g. on WebSocket disconnect).

    Args:
        redis: Connected Redis client.
        channels: List of channel names to subscribe to.

    Returns:
        An active ``PubSub`` instance.

    Example::

        pubsub = await subscribe_to_channels(redis, [f"tick:{match_id}"])
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                ...
        await pubsub.unsubscribe(*channels)
    """
    pubsub = redis.pubsub()
    await pubsub.subscribe(*channels)
    logger.info("redis_subscribed", channels=channels)
    return pubsub


async def unsubscribe_from_channels(
    pubsub: aioredis.client.PubSub,
    channels: list[str],
) -> None:
    """Unsubscribe from channels and close the PubSub connection.

    Args:
        pubsub: Active PubSub instance returned by ``subscribe_to_channels``.
        channels: Channel names to unsubscribe from.
    """
    try:
        await pubsub.unsubscribe(*channels)
        await pubsub.aclose()  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis_unsubscribe_failed", channels=channels, error=str(exc))
