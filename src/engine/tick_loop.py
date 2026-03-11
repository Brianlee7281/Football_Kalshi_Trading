"""Phase 3 tick loop — wall-clock synchronized, 1-second cadence.

Coroutine 1 of run_engine(). Runs every second and:
  1. Updates model.t (effective play time, halftime excluded) from wall clock
  2. Computes remaining expected goals μ_H, μ_A (Step 3.2)
  3. Prices P_true + σ_MC via analytical or MC (Step 3.4)
  4. Emits TickData to Phase 4 via asyncio.Queue + Redis

Wall-clock / play-clock relationship:
  Wall: 0 ──── 47min ──── 62min ──── 110min
  Play: 0 ──── 47min ┃HT┃ 47min ──── 95min
                        (excluded via halftime_accumulated)

Key design invariants:
  - model.t = (wall_elapsed - halftime_accumulated) / 60.0   (never += 1/60)
  - Halftime is tracked once on entry; accumulated on second-half start
  - MC stale results (None) are silently skipped (tick still advances)
  - Backpressure: tick_latency Prometheus histogram, warnings at >1s / >3s
  - Phase 3 is completely mode-invariant (paper == live)

Reference: docs/phase3.md §run_engine, §tick_loop
"""

from __future__ import annotations

import asyncio
import time

import structlog

from src.common.logging import get_logger
from src.common.metrics import tick_latency, tick_overrun_total
from src.engine.compute_mu import compute_remaining_mu
from src.engine.emit import emit_to_phase4
from src.engine.mc_pricing import step_3_4_async
from src.engine.model import (
    FINISHED,
    FIRST_HALF,
    HALFTIME,
    SECOND_HALF,
    LiveFootballQuantModel,
)

logger = get_logger("tick_loop")


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------


async def run_engine(
    model: LiveFootballQuantModel,
) -> None:
    """Run Phase 3: tick loop + event source coroutines concurrently.

    Note: event source coroutines (live_odds_listener, live_score_poller)
    are defined in event_sources.py and plugged in by the match engine.
    Only the tick loop is started here to allow unit testing in isolation.
    Redis is injected via model.redis before calling run_engine.

    Args:
        model: Live match state container (model.redis set by caller).
    """
    await tick_loop(model)


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------


async def tick_loop(
    model: LiveFootballQuantModel,
) -> None:
    """Main 1-second pricing loop, wall-clock synchronized.

    Updates model.t from wall clock on every tick to prevent drift
    from slow MC computation or GC pauses. Halftime duration is
    excluded from model.t via halftime_accumulated.
    Redis publishing uses model.redis (injected by caller).

    Args:
        model: Mutable live match state.
    """
    model.kickoff_wall_clock = time.monotonic()
    model.halftime_accumulated = 0.0
    model.halftime_start = None
    tick_count = 0

    log: structlog.stdlib.BoundLogger = logger.bind(match_id=model.match_id)
    log.info("tick_loop_started", T_exp=model.T_exp, pricing_mode=model.pricing_mode)

    while model.engine_phase != FINISHED:
        tick_start = time.monotonic()

        if model.engine_phase in (FIRST_HALF, SECOND_HALF):
            # ── Step 3.1: update effective play time from wall clock ──────
            wall_elapsed = time.monotonic() - model.kickoff_wall_clock
            model.t = (wall_elapsed - model.halftime_accumulated) / 60.0

            # ── Step 3.2: remaining expected goals ────────────────────────
            mu_H, mu_A = compute_remaining_mu(model)
            model.mu_H = mu_H
            model.mu_A = mu_A

            # ── Step 3.4: pricing (analytical or MC) ─────────────────────
            P_true, sigma_MC = await step_3_4_async(model, mu_H, mu_A)

            # MC may return None if result is stale or PRELIMINARY event
            if P_true is None or sigma_MC is None:
                tick_count += 1
                await _sleep_until_next_tick(model, tick_count)
                continue

            # ── Emit to Phase 4 ───────────────────────────────────────────
            emit_to_phase4(
                P_true,
                sigma_MC,
                model.order_allowed,
                model,
            )

            # ── Backpressure monitoring ───────────────────────────────────
            tick_duration = time.monotonic() - tick_start
            tick_latency.observe(tick_duration)

            if tick_duration > 3.0:
                tick_overrun_total.labels(severity="critical").inc()
                log.warning(
                    "tick_critical_overrun",
                    tick=tick_count,
                    duration_s=round(tick_duration, 3),
                )
            elif tick_duration > 1.0:
                tick_overrun_total.labels(severity="warn").inc()
                log.info(
                    "tick_overrun",
                    tick=tick_count,
                    duration_s=round(tick_duration, 3),
                )

        elif model.engine_phase == HALFTIME:
            # Track halftime start on first tick in HALFTIME
            if model.halftime_start is None:
                model.halftime_start = time.monotonic()
                log.info("halftime_start_recorded", t=round(model.t, 2))

        tick_count += 1
        await _sleep_until_next_tick(model, tick_count)

    log.info(
        "tick_loop_finished",
        ticks_completed=tick_count,
        final_t=round(model.t, 2),
        score=model.score,
    )


# ---------------------------------------------------------------------------
# Halftime accumulation helper (called by period_handler on second-half start)
# ---------------------------------------------------------------------------


def record_halftime_end(model: LiveFootballQuantModel) -> None:
    """Accumulate halftime wall-clock duration into model.halftime_accumulated.

    Must be called exactly once when the second half starts (period_handler
    transitions engine_phase from HALFTIME → SECOND_HALF).

    After this call, model.t will resume from the correct play-minute
    (not including halftime in the elapsed count).
    """
    if model.halftime_start is not None:
        elapsed_halftime = time.monotonic() - model.halftime_start
        model.halftime_accumulated += elapsed_halftime
        logger.info(
            "halftime_accumulated",
            match_id=model.match_id,
            halftime_duration_s=round(elapsed_halftime, 1),
            total_accumulated_s=round(model.halftime_accumulated, 1),
        )
        model.halftime_start = None


# ---------------------------------------------------------------------------
# Absolute-time tick scheduler
# ---------------------------------------------------------------------------


async def _sleep_until_next_tick(
    model: LiveFootballQuantModel,
    tick_count: int,
) -> None:
    """Sleep until the next absolute 1-second tick boundary.

    Uses model.kickoff_wall_clock as the origin so ticks stay aligned
    to wall time regardless of per-tick computation cost. If we're
    already past the next tick boundary, return immediately (no catch-up).

    Args:
        model: Model with kickoff_wall_clock set.
        tick_count: The upcoming tick number (0-indexed from kickoff).
    """
    next_tick_time = model.kickoff_wall_clock + float(tick_count)
    sleep_duration = next_tick_time - time.monotonic()
    if sleep_duration > 0.0:
        await asyncio.sleep(sleep_duration)
    # else: tick was late — proceed immediately, no catch-up accumulation
