"""Phase 3 event handlers — two-stage preliminary → confirmed processing.

Handles all discrete events that occur during a live match:
  - Goal (preliminary ob_freeze → confirmed commit → cooldown)
  - Score rollback / VAR cancellation
  - Odds spike (early warning → ob_freeze)
  - Red card (Markov state transition)
  - Live score source failure

Key design patterns (docs/phase3.md §Rapid Sequential Events):
  Rule 1: State updates (S, ΔS, μ) are NEVER blocked by cooldown.
  Rule 2: Cooldown timer RESETS (not extends) on each confirmed event.
  Rule 3: Events during PRELIMINARY state are QUEUED via EventQueue,
          drained when the current event resolves.

ob_freeze release conditions (check_ob_freeze_release, called each tick):
  1. Event confirmed and cooldown active → ob_freeze = False
  2. 3 consecutive stable odds ticks (no movement above threshold)
  3. 10-second timeout (false-positive protection)

Reference: docs/phase3.md §Event Handlers, §ob_freeze Release Conditions
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import TYPE_CHECKING

import structlog

from src.common.logging import get_logger
from src.common.types import NormalizedEvent
from src.engine.compute_mu import compute_remaining_mu
from src.engine.event_queue import EventQueue as _EventQueue  # re-exported
from src.engine.model import EVENT_IDLE, EVENT_PRELIMINARY, FINISHED

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("event_handlers")

# Re-export EventQueue so callers can import from here
EventQueue = _EventQueue

# Cooldown duration in seconds after a confirmed event
_COOLDOWN_SECONDS = 15

# ob_freeze timeout in seconds (false-positive protection)
_OB_FREEZE_TIMEOUT = 10.0

# Number of stable ticks before ob_freeze is released
_OB_STABLE_TICKS_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Early warning (Odds-API WebSocket → ob_freeze)
# ---------------------------------------------------------------------------


def handle_odds_spike(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Detect abrupt odds move → set ob_freeze, await Goalserve confirmation.

    Odds-API WS fires <1s after a real event (goal/red card). We don't know
    what happened yet — just block new orders until Goalserve confirms.
    """
    if not model.ob_freeze:
        model._ob_freeze_start = time.monotonic()
        model._ob_stable_ticks = 0

    model.ob_freeze = True
    log: structlog.stdlib.BoundLogger = logger.bind(match_id=model.match_id)
    log.warning(
        "odds_spike_ob_freeze",
        delta=round(event.delta or 0.0, 4),
    )


# ---------------------------------------------------------------------------
# Preliminary goal (score change first seen — provisional)
# ---------------------------------------------------------------------------


def handle_preliminary_goal(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Score change detected (preliminary — VAR might still cancel).

    Sets PRELIMINARY_DETECTED and starts async μ precompute so confirmation
    latency is ~0ms (cached result ready when Goalserve next confirms).
    """
    if event.score is None:
        return

    # Queue if already in preliminary (rapid events)
    if model.event_state == EVENT_PRELIMINARY:
        model.event_queue.enqueue(event)
        return

    model.ob_freeze = True
    model._ob_freeze_start = time.monotonic()
    model.event_state = EVENT_PRELIMINARY

    preliminary_score = event.score
    if preliminary_score[0] > model.score_home:
        scoring_team = "home"
    elif preliminary_score[1] > model.score_away:
        scoring_team = "away"
    else:
        logger.warning(
            "score_changed_no_increase",
            match_id=model.match_id,
            score=preliminary_score,
        )
        model.event_state = EVENT_IDLE
        model.ob_freeze = False
        return

    preliminary_delta_S = preliminary_score[0] - preliminary_score[1]

    # Fire-and-forget μ precompute so confirmation is ~0ms
    asyncio.create_task(
        _precompute_preliminary_mu(model, preliminary_delta_S, scoring_team)
    )

    model.preliminary_cache = {
        "score": preliminary_score,
        "delta_S": preliminary_delta_S,
        "scoring_team": scoring_team,
        "timestamp": event.timestamp,
    }

    logger.info(
        "preliminary_goal",
        match_id=model.match_id,
        score_before=model.score,
        score_after=preliminary_score,
        team=scoring_team,
    )


# ---------------------------------------------------------------------------
# Score rollback / VAR cancellation
# ---------------------------------------------------------------------------


def handle_score_rollback(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Score decreased in Goalserve → likely VAR cancellation.

    Rolls back PRELIMINARY state immediately.
    """
    if model.event_state == EVENT_PRELIMINARY:
        logger.warning(
            "score_rollback_var_cancel",
            match_id=model.match_id,
            preliminary_score=model.preliminary_cache.get("score"),
            rollback_score=event.score,
        )
        model.event_state = EVENT_IDLE
        model.ob_freeze = False
        model.preliminary_cache = {}
    else:
        logger.warning(
            "score_rollback_unexpected_state",
            match_id=model.match_id,
            event_state=model.event_state,
            score=event.score,
        )


# ---------------------------------------------------------------------------
# Confirmed goal (authoritative — commits state, resets cooldown)
# ---------------------------------------------------------------------------


def handle_confirmed_goal(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Goal confirmed by Goalserve Live Score.

    Implements Rule 1 (state always updates) and Rule 2 (cooldown resets).
    Drains EventQueue after resolving (Rule 3).
    """
    # VAR cancellation: roll back and release
    if event.var_cancelled:
        model.event_state = EVENT_IDLE
        model.ob_freeze = False
        model.preliminary_cache = {}
        logger.info("goal_var_cancelled", match_id=model.match_id)
        return

    # ── Rule 1: always commit state, even during cooldown ────────────────
    if event.team == "localteam":
        model.update_score(model.score_home + 1, model.score_away)
    else:
        model.update_score(model.score_home, model.score_away + 1)

    # Reuse preliminary μ cache if ΔS matches (0ms), else recompute
    cached_delta_S = model.preliminary_cache.get("delta_S")
    if (
        model.preliminary_cache
        and cached_delta_S == model.delta_S
        and "mu_H" in model.preliminary_cache
    ):
        model.mu_H = float(model.preliminary_cache["mu_H"])
        model.mu_A = float(model.preliminary_cache["mu_A"])
        logger.debug("mu_from_cache", match_id=model.match_id)
    else:
        model.mu_H, model.mu_A = compute_remaining_mu(model)

    # ── Rule 2: reset cooldown timer (cancel previous, start new) ────────
    model.cooldown = True
    model.ob_freeze = False
    model.event_state = EVENT_IDLE
    model.preliminary_cache = {}

    if model._cooldown_task is not None:
        model._cooldown_task.cancel()
    model._cooldown_task = asyncio.create_task(
        _cooldown_timer(model, _COOLDOWN_SECONDS)
    )

    logger.info(
        "goal_confirmed",
        match_id=model.match_id,
        score=model.score,
        delta_S=model.delta_S,
        team=event.team,
        scorer=event.scorer_id,
    )

    # ── Rule 3: drain queued events ───────────────────────────────────────
    for queued_event in model.event_queue.drain():
        dispatch_event(model, queued_event)


# ---------------------------------------------------------------------------
# Confirmed red card
# ---------------------------------------------------------------------------


def handle_confirmed_red_card(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Red card confirmed — Markov state X transitions, cooldown starts.

    State transitions:
      0 → 1: home dismissal (10v11)
      0 → 2: away dismissal (11v10)
      1 → 3: additional away dismissal (10v10)
      2 → 3: additional home dismissal (10v10)
    """
    old_X = model.current_state_X

    if event.team == "localteam":
        if model.current_state_X == 0:
            model.transition_state(1)
        elif model.current_state_X == 2:
            model.transition_state(3)
    else:
        if model.current_state_X == 0:
            model.transition_state(2)
        elif model.current_state_X == 1:
            model.transition_state(3)

    # Recompute μ to reflect updated gamma^H / gamma^A via new X
    model.mu_H, model.mu_A = compute_remaining_mu(model)

    # Cooldown reset
    model.cooldown = True
    model.ob_freeze = False
    model.event_state = EVENT_IDLE

    if model._cooldown_task is not None:
        model._cooldown_task.cancel()
    model._cooldown_task = asyncio.create_task(
        _cooldown_timer(model, _COOLDOWN_SECONDS)
    )

    logger.info(
        "red_card_confirmed",
        match_id=model.match_id,
        team=event.team,
        X_before=old_X,
        X_after=model.current_state_X,
    )


# ---------------------------------------------------------------------------
# Source failure
# ---------------------------------------------------------------------------


def handle_live_score_failure(model: LiveFootballQuantModel) -> None:
    """5+ consecutive Goalserve poll failures → freeze all orders (safe mode)."""
    model.ob_freeze = True
    model._ob_freeze_start = time.monotonic()
    logger.error(
        "live_score_source_failure",
        match_id=model.match_id,
    )


# ---------------------------------------------------------------------------
# ob_freeze release conditions (called each tick from tick_loop)
# ---------------------------------------------------------------------------


def check_ob_freeze_release(model: LiveFootballQuantModel) -> None:
    """Check and release ob_freeze if release conditions are met.

    Called once per tick in tick_loop (before pricing). Three conditions:
      1. Cooldown active → event was explained + state committed → release
      2. 3 stable ticks (odds not moving) → likely false positive → release
      3. 10-second timeout → force release (false-positive protection)
    """
    if not model.ob_freeze:
        return

    # Condition 1: event was confirmed, cooldown started
    if model.cooldown:
        model.ob_freeze = False
        logger.debug("ob_freeze_released_cooldown", match_id=model.match_id)
        return

    # Condition 2: 3 consecutive stable ticks
    if model._ob_stable_ticks >= _OB_STABLE_TICKS_THRESHOLD:
        model.ob_freeze = False
        model._ob_stable_ticks = 0
        logger.info(
            "ob_freeze_released_stabilized",
            match_id=model.match_id,
            stable_ticks=_OB_STABLE_TICKS_THRESHOLD,
        )
        return

    # Condition 3: 10-second timeout
    elapsed = time.monotonic() - model._ob_freeze_start
    if elapsed >= _OB_FREEZE_TIMEOUT:
        model.ob_freeze = False
        logger.info(
            "ob_freeze_released_timeout",
            match_id=model.match_id,
            elapsed_s=round(elapsed, 1),
        )


def record_ob_stable_tick(model: LiveFootballQuantModel) -> None:
    """Increment ob_stable_ticks counter if odds are stable this tick.

    Called by tick_loop when the latest odds update shows no movement.
    Resets counter to 0 on any spike.
    """
    if model.ob_freeze:
        model._ob_stable_ticks += 1


def reset_ob_stable_ticks(model: LiveFootballQuantModel) -> None:
    """Reset stable ticks counter on odds movement."""
    model._ob_stable_ticks = 0


# ---------------------------------------------------------------------------
# Event dispatcher
# ---------------------------------------------------------------------------


def dispatch_event(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Route a NormalizedEvent to the appropriate handler.

    Used both by live_score_poller (direct dispatch) and EventQueue.drain()
    (deferred dispatch after PRELIMINARY resolves).
    """
    if model.engine_phase == FINISHED:
        return

    t = event.type
    if t == "odds_spike":
        handle_odds_spike(model, event)
    elif t == "goal_detected":
        handle_preliminary_goal(model, event)
    elif t == "goal_confirmed":
        handle_confirmed_goal(model, event)
    elif t == "score_rollback":
        handle_score_rollback(model, event)
    elif t == "red_card":
        handle_confirmed_red_card(model, event)
    elif t == "source_failure":
        handle_live_score_failure(model)
    else:
        logger.debug(
            "unknown_event_type",
            match_id=model.match_id,
            event_type=t,
        )


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def _cooldown_timer(
    model: LiveFootballQuantModel,
    duration: int,
) -> None:
    """Release cooldown after ``duration`` seconds."""
    await asyncio.sleep(duration)
    model.cooldown = False
    model._cooldown_task = None
    logger.info(
        "cooldown_expired",
        match_id=model.match_id,
        duration_s=duration,
    )


async def _precompute_preliminary_mu(
    model: LiveFootballQuantModel,
    preliminary_delta_S: int,
    scoring_team: str,
) -> None:
    """Asynchronously precompute μ_H / μ_A for the preliminary ΔS.

    Runs compute_remaining_mu in a thread executor so it doesn't block
    the event loop. Stores results in model.preliminary_cache for ~0ms
    access at confirmation time.
    """
    loop = asyncio.get_running_loop()
    try:
        mu_H, mu_A = await loop.run_in_executor(
            None,
            partial(compute_remaining_mu, model, preliminary_delta_S),
        )
        model.preliminary_cache["mu_H"] = mu_H
        model.preliminary_cache["mu_A"] = mu_A
        logger.info(
            "preliminary_mu_precomputed",
            match_id=model.match_id,
            mu_H=round(mu_H, 3),
            mu_A=round(mu_A, 3),
            scoring_team=scoring_team,
        )
    except Exception:
        logger.exception(
            "preliminary_mu_precompute_failed",
            match_id=model.match_id,
        )
