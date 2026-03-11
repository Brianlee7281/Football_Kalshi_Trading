"""Phase 3 → Phase 4 emit layer.

Pushes each tick's pricing output to two sinks:
  1. asyncio.Queue (maxsize=1) — consumed by Phase 4 signal_generator
  2. Redis PUBLISH — consumed by dashboard + monitoring

Queue maxsize=1 semantics:
  If Phase 4 hasn't consumed the previous tick, the old tick is replaced
  (always-fresh guarantee). This ensures Phase 4 sees the latest state
  even if it's momentarily slow.

Redis channel: "p_true:{match_id}"
  JSON payload: {"P_true": {...}, "sigma_MC": {...}, "order_allowed": bool,
                 "t": float, "pricing_mode": str}

Both sinks are fire-and-forget: a failure in either does not raise;
errors are logged and counted in Prometheus.

Reference: docs/phase3.md §emit_to_phase4, patterns.md #1
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.metrics import emit_queue_full_total, redis_publish_error_total
from src.common.types import TickData
from src.engine.model import LiveFootballQuantModel

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = get_logger("emit")


def emit_to_phase4(
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
    model: LiveFootballQuantModel,
    redis_client: aioredis.Redis | None = None,
) -> None:
    """Push tick data to Phase 4 and optionally publish to Redis.

    Constructs a TickData and pushes it to model.phase4_queue with
    stale-replacement semantics (non-blocking). If a Redis client is
    provided, fires an asyncio.Task to publish asynchronously.

    This function is intentionally synchronous for the queue push so that
    it matches the call site in tick_loop.py (no await needed). The Redis
    publish is handled via asyncio.create_task() as a background coroutine.

    Args:
        P_true: Market → probability dict for this tick.
        sigma_MC: Market → σ_MC dict for this tick.
        order_allowed: Whether Phase 4 may place orders this tick.
        model: Live match model (provides queue, match_id, t, etc.).
        redis_client: Optional async Redis client for dashboard publish.
    """
    tick_data = TickData(
        P_true=P_true,
        sigma_MC=sigma_MC,
        order_allowed=order_allowed,
        event_state=model.event_state,
        pricing_mode=model.pricing_mode,
        engine_phase=model.engine_phase,
        mu_H=model.mu_H,
        mu_A=model.mu_A,
        P_bet365=model.bet365_odds_prev or {},
    )

    # ── asyncio.Queue push (stale-replacement) ────────────────────────────
    if model.phase4_queue.full():
        try:
            model.phase4_queue.get_nowait()
            emit_queue_full_total.labels(match_id=model.match_id).inc()
        except asyncio.QueueEmpty:
            pass

    model.phase4_queue.put_nowait(tick_data)

    # ── Redis publish (fire-and-forget) ───────────────────────────────────
    if redis_client is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop running — skip Redis publish

        loop.create_task(
            _redis_publish(
                redis_client,
                model.match_id,
                model.t,
                P_true,
                sigma_MC,
                order_allowed,
                model.pricing_mode,
            )
        )


async def _redis_publish(
    redis_client: aioredis.Redis,
    match_id: str,
    t: float,
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
    pricing_mode: str,
) -> None:
    """Publish tick data to Redis channel p_true:{match_id}.

    Silently swallows errors to avoid disrupting the tick loop.
    Errors are logged and counted in Prometheus.

    Args:
        redis_client: Async Redis client.
        match_id: Match identifier (used as channel suffix).
        t: Current effective play time in minutes.
        P_true: Market probabilities.
        sigma_MC: Per-market MC standard errors.
        order_allowed: Whether orders are allowed this tick.
        pricing_mode: "analytical" or "mc".
    """
    channel = f"p_true:{match_id}"
    payload = json.dumps(
        {
            "P_true": P_true,
            "sigma_MC": sigma_MC,
            "order_allowed": order_allowed,
            "t": round(t, 4),
            "pricing_mode": pricing_mode,
        }
    )
    try:
        await redis_client.publish(channel, payload)
    except Exception:
        redis_publish_error_total.labels(match_id=match_id).inc()
        logger.warning(
            "redis_publish_failed",
            match_id=match_id,
            channel=channel,
        )
