"""Phase 3 period handler — halftime tracking + second-half resume.

Handles period-change events from Goalserve Live Score and manages
the halftime_accumulated counter used by the tick loop to compute
effective play time (halftime excluded from model.t).

Wall-clock / play-clock relationship:
  Wall: 0 ──── 47min ──── 62min ──── 110min
  Play: 0 ──── 47min ┃HT┃ 47min ──── 95min
                        (halftime_accumulated excludes HT from model.t)

key rules:
  - HALFTIME entry: record model.halftime_start = time.monotonic()
  - SECOND HALF start: accumulate elapsed into halftime_accumulated,
    reset halftime_start, transition engine_phase → SECOND_HALF
  - model.t = (wall_elapsed - halftime_accumulated) / 60 — never drifts

Reference: docs/phase3.md §handle_period_change, patterns.md #4
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import NormalizedEvent
from src.engine.model import FINISHED, FIRST_HALF, HALFTIME, SECOND_HALF, WAITING_FOR_KICKOFF

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("period_handler")


def handle_period_change(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Handle a period-change event from Goalserve Live Score.

    This is the authoritative transition: Goalserve status field
    transitions from "1st Half" → "HT" → "2nd Half" → "Finished".

    On HALFTIME entry:
      - Set engine_phase = HALFTIME
      - Record halftime_start (wall clock) for accumulation later

    On SECOND HALF start:
      - Accumulate halftime_accumulated from halftime_start
      - Set engine_phase = SECOND_HALF
      - model.t will now resume from correct play-minute

    On FINISHED:
      - Set engine_phase = FINISHED (tick loop exits)

    Args:
        model: Mutable live match state.
        event: NormalizedEvent with type="period_change" and period field set.
    """
    period = event.period or ""

    if period in ("1st Half", "1H", "First Half"):
        _enter_first_half(model)

    elif period in ("Halftime", "HT", "Half Time", "Paused", "Half"):
        _enter_halftime(model)

    elif period in ("2nd Half", "2H", "Second Half"):
        _enter_second_half(model)

    elif period in ("Finished", "FT", "Full Time"):
        _finish_match(model)

    else:
        logger.warning(
            "unknown_period",
            match_id=model.match_id,
            period=period,
        )


def handle_match_finished(
    model: LiveFootballQuantModel,
    event: NormalizedEvent,
) -> None:
    """Handle match_finished event (Goalserve status = Finished).

    Args:
        model: Mutable live match state.
        event: NormalizedEvent with type="match_finished".
    """
    _finish_match(model)


# ---------------------------------------------------------------------------
# Internal transitions
# ---------------------------------------------------------------------------


def _enter_first_half(model: LiveFootballQuantModel) -> None:
    """Transition engine from WAITING_FOR_KICKOFF to FIRST_HALF."""
    if model.engine_phase == FIRST_HALF:
        return  # already in first half — idempotent

    model.engine_phase = FIRST_HALF

    logger.info(
        "first_half_started",
        match_id=model.match_id,
        previous_phase=WAITING_FOR_KICKOFF,
    )


def _enter_halftime(model: LiveFootballQuantModel) -> None:
    """Transition engine to HALFTIME and record halftime start wall clock."""
    if model.engine_phase == HALFTIME:
        # Already in halftime — idempotent
        return

    model.engine_phase = HALFTIME

    # Record halftime start for later accumulation
    # (tick_loop may also set this on the first tick in HALFTIME)
    if model.halftime_start is None:
        model.halftime_start = time.monotonic()

    logger.info(
        "halftime_entered",
        match_id=model.match_id,
        t_play_min=round(model.t, 2),
        score=model.score,
    )


def _enter_second_half(model: LiveFootballQuantModel) -> None:
    """Accumulate halftime duration and transition to SECOND_HALF."""
    if model.engine_phase == SECOND_HALF:
        # Already in second half — idempotent
        return

    # Accumulate halftime wall-clock duration
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

    model.engine_phase = SECOND_HALF

    logger.info(
        "second_half_started",
        match_id=model.match_id,
        t_play_min=round(model.t, 2),
        halftime_excluded_s=round(model.halftime_accumulated, 1),
        score=model.score,
    )


def _finish_match(model: LiveFootballQuantModel) -> None:
    """Transition engine to FINISHED."""
    if model.engine_phase == FINISHED:
        return

    model.finish()

    logger.info(
        "match_finished",
        match_id=model.match_id,
        final_score=model.score,
        t_play_min=round(model.t, 2),
    )
