"""Phase 3 → Phase 4 emit layer + infrastructure writes.

Pushes each tick's pricing output to three sinks:
  1. asyncio.Queue (maxsize=1) — consumed by Phase 4 signal_generator
  2. tick_snapshots DB table  — sampled write for analytics / replay
  3. Redis PUBLISH            — consumed by dashboard + monitoring

Also provides record_event() for writing confirmed in-match events to
event_log (DB) and publishing to the Redis event:{match_id} channel.

Queue maxsize=1 semantics:
  If Phase 4 hasn't consumed the previous tick, the old tick is replaced
  (always-fresh guarantee). This ensures Phase 4 sees the latest state
  even if it's momentarily slow.

tick_snapshots sampling:
  - During events (event_state != IDLE) or cooldown: every tick (1s)
  - Normal play: every 10s (int(t_seconds) % 10 == 0)

Redis channels:
  tick:{match_id}   — full tick payload for live dashboard
  event:{match_id}  — confirmed match events (goals, cards, etc.)

All DB and Redis operations are fire-and-forget (asyncio.create_task).
Failures are logged and counted via Prometheus; they never raise.

Reference: docs/phase3.md §emit_to_phase4, §record_event, patterns.md #1
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from src.common.logging import get_logger
from src.common.metrics import emit_queue_full_total, redis_publish_error_total
from src.common.types import NormalizedEvent, TickData
from src.engine.model import LiveFootballQuantModel

logger = get_logger("emit")


def emit_to_phase4(
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
    model: LiveFootballQuantModel,
) -> None:
    """Push tick data to Phase 4, tick_snapshots DB, and Redis.

    Constructs a TickData and pushes it to model.phase4_queue with
    stale-replacement semantics (non-blocking).

    If model.db_pool is set, fires an asyncio.Task to write a sampled
    row to tick_snapshots (every 10s in normal play, every 1s during
    events / cooldown).

    If model.redis is set, fires an asyncio.Task to publish the tick
    payload to Redis channel tick:{match_id}.

    This function is intentionally synchronous for the queue push so
    it matches the call site in tick_loop.py (no await needed).

    Args:
        P_true: Market → probability dict for this tick.
        sigma_MC: Market → σ_MC dict for this tick.
        order_allowed: Whether Phase 4 may place orders this tick.
        model: Live match model (provides queue, db_pool, redis, etc.).
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

    # ── Sampling decision for tick_snapshot write ─────────────────────────
    # Always write during events/cooldown; every 10s in normal play.
    t_seconds = model.t * 60.0
    during_event = (model.event_state != "IDLE") or model.cooldown
    should_write = during_event or (int(t_seconds) % 10 == 0)

    # ── DB write: tick_snapshots (fire-and-forget) ────────────────────────
    if model.db_pool is not None and should_write:
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(
                _write_tick_snapshot(model, P_true, sigma_MC, order_allowed)
            )

    # ── Redis publish: tick:{match_id} (fire-and-forget) ──────────────────
    if model.redis is not None:
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(
                _publish_tick_to_redis(model, P_true, sigma_MC, order_allowed)
            )


async def record_event(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Write a confirmed match event to event_log and Redis.

    Called (via asyncio.create_task) after every confirmed event:
    goal_confirmed, score_rollback, red_card, period_change,
    match_finished, source_failure.

    Both operations are fire-and-forget within this coroutine —
    failures are logged but never re-raised so they don't disrupt
    the event pipeline.

    Args:
        model: Live match model with db_pool and redis attributes.
        event: The confirmed NormalizedEvent to record.
    """
    payload = {
        "score": [model.score_home, model.score_away],
        "team": event.team,
        "minute": event.minute if event.minute is not None else model.t,
        "var_cancelled": event.var_cancelled or False,
        "t": model.t,
    }

    # ── DB: INSERT INTO event_log ─────────────────────────────────────────
    if model.db_pool is not None:
        try:
            async with model.db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO event_log (match_id, event_type, source, payload)
                       VALUES ($1, $2, $3, $4)""",
                    model.match_id,
                    event.type,
                    event.source,
                    json.dumps(payload),
                )
        except Exception:
            logger.error(
                "event_log_write_failed",
                match_id=model.match_id,
                event_type=event.type,
            )

    # ── Redis: PUBLISH event:{match_id} ───────────────────────────────────
    if model.redis is not None:
        try:
            await model.redis.publish(
                f"event:{model.match_id}",
                json.dumps(
                    {
                        "type": "event",
                        "match_id": model.match_id,
                        "event_type": event.type,
                        "t": model.t,
                        "payload": payload,
                    }
                ),
            )
        except Exception:
            redis_publish_error_total.labels(match_id=model.match_id).inc()
            logger.warning(
                "redis_event_publish_failed",
                match_id=model.match_id,
                event_type=event.type,
            )


# ---------------------------------------------------------------------------
# Internal fire-and-forget coroutines
# ---------------------------------------------------------------------------


async def _write_tick_snapshot(
    model: LiveFootballQuantModel,
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
) -> None:
    """Insert a sampled tick row into tick_snapshots.

    Silently swallows errors so a DB hiccup never disrupts the tick loop.

    P_kalshi is omitted here (Phase 4 territory). P_bet365 is taken from
    model.bet365_odds_prev (the previous tick's Odds-API bet365 prices).
    """
    if model.db_pool is None:
        return
    try:
        async with model.db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO tick_snapshots
                   (match_id, t, mu_H, mu_A, P_true, P_kalshi, P_bet365,
                    sigma_MC, engine_phase, event_state, cooldown, ob_freeze,
                    order_allowed)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                model.match_id,
                float(model.t),
                model.mu_H,
                model.mu_A,
                json.dumps(P_true),
                None,   # P_kalshi tracked by Phase 4, not Phase 3
                json.dumps(model.bet365_odds_prev) if model.bet365_odds_prev else None,
                json.dumps(sigma_MC),
                model.engine_phase,
                model.event_state,
                model.cooldown,
                model.ob_freeze,
                order_allowed,
            )
    except Exception:
        logger.error(
            "tick_snapshot_write_failed",
            match_id=model.match_id,
            t=model.t,
        )


async def _publish_tick_to_redis(
    model: LiveFootballQuantModel,
    P_true: dict[str, float],
    sigma_MC: dict[str, float],
    order_allowed: bool,
) -> None:
    """Publish tick payload to Redis channel tick:{match_id}.

    Used by the live dashboard to stream real-time P_true updates.
    Silently swallows errors; errors counted via Prometheus.
    """
    if model.redis is None:
        return
    channel = f"tick:{model.match_id}"
    payload = json.dumps(
        {
            "type": "tick",
            "match_id": model.match_id,
            "t": round(model.t, 4),
            "engine_phase": model.engine_phase,
            "P_true": P_true,
            "sigma_MC": sigma_MC,
            "P_bet365": model.bet365_odds_prev or {},
            "order_allowed": order_allowed,
            "cooldown": model.cooldown,
            "ob_freeze": model.ob_freeze,
            "event_state": model.event_state,
            "mu_H": model.mu_H,
            "mu_A": model.mu_A,
            "score": [model.score_home, model.score_away],
        }
    )
    try:
        await model.redis.publish(channel, payload)
    except Exception:
        redis_publish_error_total.labels(match_id=model.match_id).inc()
        logger.warning(
            "redis_tick_publish_failed",
            match_id=model.match_id,
            channel=channel,
        )
