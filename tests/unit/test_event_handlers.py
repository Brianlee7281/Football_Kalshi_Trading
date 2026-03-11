"""Unit tests for event handlers — goals, red cards, VAR, rapid sequential.

Tests the core event processing invariants:
  Rule 1: State (S, ΔS, μ) always updates, even during cooldown.
  Rule 2: Cooldown timer RESETS on each confirmed event.
  Rule 3: Events during PRELIMINARY are queued, drained on confirmation.

Reference: docs/phase3.md §Event Handlers, patterns.md #9
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.common.types import NormalizedEvent
from src.engine.event_handlers import (
    EventQueue,
    dispatch_event,
    handle_confirmed_goal,
    handle_confirmed_red_card,
    handle_score_rollback,
)
from src.engine.model import (
    EVENT_IDLE,
    EVENT_PRELIMINARY,
    FIRST_HALF,
    LiveFootballQuantModel,
)


@pytest.fixture(autouse=True)
def _mock_create_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch asyncio.create_task to a no-op so tests don't need an event loop."""
    monkeypatch.setattr(
        "asyncio.create_task",
        lambda coro, **_kw: coro.close() or None,
    )


def _make_model() -> LiveFootballQuantModel:
    model = LiveFootballQuantModel()
    model.engine_phase = FIRST_HALF
    model.kickoff_wall_clock = time.monotonic()
    model.halftime_accumulated = 0.0
    model.t = 30.0
    return model


def _goal_event(
    team: str,
    score: tuple[int, int],
    *,
    var_cancelled: bool = False,
) -> NormalizedEvent:
    return NormalizedEvent(
        type="goal_confirmed",
        source="live_score",
        confidence="confirmed",
        score=score,
        team=team,
        var_cancelled=var_cancelled,
        timestamp=time.time(),
    )


def _red_card_event(team: str) -> NormalizedEvent:
    return NormalizedEvent(
        type="red_card",
        source="live_score",
        confidence="confirmed",
        team=team,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Home goal: S=(0,0) → S=(1,0), delta_S=+1, cooldown=True
# ---------------------------------------------------------------------------


def test_home_goal_updates_score() -> None:
    """Home goal increments score_home and sets delta_S=+1."""
    model = _make_model()
    event = _goal_event("localteam", (1, 0))

    handle_confirmed_goal(model, event)

    assert model.score_home == 1
    assert model.score_away == 0
    assert model.delta_S == 1
    assert model.cooldown is True


def test_home_goal_releases_ob_freeze() -> None:
    """ob_freeze is released on goal confirmation."""
    model = _make_model()
    model.ob_freeze = True
    event = _goal_event("localteam", (1, 0))

    handle_confirmed_goal(model, event)

    assert model.ob_freeze is False


def test_home_goal_event_state_idle() -> None:
    """event_state returns to IDLE after confirmed goal."""
    model = _make_model()
    event = _goal_event("localteam", (1, 0))

    handle_confirmed_goal(model, event)

    assert model.event_state == EVENT_IDLE


# ---------------------------------------------------------------------------
# Away goal: S=(1,0) → S=(1,1)
# ---------------------------------------------------------------------------


def test_away_goal_updates_score() -> None:
    """Away goal increments score_away and sets delta_S=0."""
    model = _make_model()
    model.update_score(1, 0)
    event = _goal_event("visitorteam", (1, 1))

    handle_confirmed_goal(model, event)

    assert model.score_home == 1
    assert model.score_away == 1
    assert model.delta_S == 0


# ---------------------------------------------------------------------------
# Red card: X incremented via Markov transition
# ---------------------------------------------------------------------------


def test_home_red_card_transitions_state() -> None:
    """First home red card: X=0 → X=1 (10v11)."""
    model = _make_model()
    event = _red_card_event("localteam")

    handle_confirmed_red_card(model, event)

    assert model.current_state_X == 1


def test_away_red_card_transitions_state() -> None:
    """First away red card: X=0 → X=2 (11v10)."""
    model = _make_model()
    event = _red_card_event("visitorteam")

    handle_confirmed_red_card(model, event)

    assert model.current_state_X == 2


def test_second_home_red_after_away_transitions_to_3() -> None:
    """Home red after away red: X=2 → X=3 (10v10)."""
    model = _make_model()
    model.current_state_X = 2

    handle_confirmed_red_card(model, _red_card_event("localteam"))

    assert model.current_state_X == 3


def test_second_away_red_after_home_transitions_to_3() -> None:
    """Away red after home red: X=1 → X=3 (10v10)."""
    model = _make_model()
    model.current_state_X = 1

    handle_confirmed_red_card(model, _red_card_event("visitorteam"))

    assert model.current_state_X == 3


def test_red_card_starts_cooldown() -> None:
    """Red card confirmation starts cooldown."""
    model = _make_model()
    handle_confirmed_red_card(model, _red_card_event("localteam"))
    assert model.cooldown is True


# ---------------------------------------------------------------------------
# VAR cancellation: S unchanged, event_state=IDLE
# ---------------------------------------------------------------------------


def test_var_cancelled_score_unchanged() -> None:
    """VAR cancellation leaves score unchanged and resets state."""
    model = _make_model()
    model.update_score(1, 0)
    model.ob_freeze = True
    model.event_state = EVENT_PRELIMINARY
    model.preliminary_cache = {"score": (1, 0), "delta_S": 1}

    event = _goal_event("localteam", (1, 0), var_cancelled=True)
    handle_confirmed_goal(model, event)

    # Score should NOT have changed (goal was cancelled)
    assert model.score == (1, 0)  # unchanged from before
    assert model.event_state == EVENT_IDLE
    assert model.ob_freeze is False
    assert model.preliminary_cache == {}


# ---------------------------------------------------------------------------
# Rapid sequential (Rule 2: cooldown resets; Rule 3: queue drains)
# ---------------------------------------------------------------------------


def test_second_goal_during_cooldown_still_updates_score() -> None:
    """Rule 1: State updates even during cooldown. Score increments twice."""
    model = _make_model()

    # First goal
    handle_confirmed_goal(model, _goal_event("localteam", (1, 0)))
    assert model.score == (1, 0)
    assert model.cooldown is True

    # Second goal arrives while cooldown is active (state must still update)
    handle_confirmed_goal(model, _goal_event("localteam", (2, 0)))
    assert model.score == (2, 0)
    assert model.delta_S == 2


def test_event_queue_drained_after_confirmation() -> None:
    """Rule 3: Events queued during PRELIMINARY are dispatched after confirmation."""
    model = _make_model()

    # Manually queue a red card event (simulates arrival during PRELIMINARY)
    red_event = _red_card_event("localteam")
    model.event_queue.enqueue(red_event)

    assert len(model.event_queue) == 1

    # Confirm a goal — this triggers queue drain via dispatch_event
    goal_event = _goal_event("localteam", (1, 0))
    handle_confirmed_goal(model, goal_event)

    # Queue should be drained after confirmation
    assert len(model.event_queue) == 0


def test_score_rollback_resets_preliminary() -> None:
    """handle_score_rollback clears PRELIMINARY state."""
    model = _make_model()
    model.event_state = EVENT_PRELIMINARY
    model.preliminary_cache = {"score": (1, 0), "delta_S": 1}
    model.ob_freeze = True

    rollback_event = NormalizedEvent(
        type="score_rollback",
        source="live_score",
        confidence="confirmed",
        score=(0, 0),
        var_cancelled=True,
        timestamp=time.time(),
    )
    handle_score_rollback(model, rollback_event)

    assert model.event_state == EVENT_IDLE
    assert model.ob_freeze is False
    assert model.preliminary_cache == {}


# ---------------------------------------------------------------------------
# dispatch_event routing
# ---------------------------------------------------------------------------


def test_dispatch_goal_confirmed_routes_correctly() -> None:
    """dispatch_event routes goal_confirmed to handle_confirmed_goal."""
    model = _make_model()
    event = _goal_event("localteam", (1, 0))
    dispatch_event(model, event)
    assert model.score == (1, 0)


def test_dispatch_red_card_routes_correctly() -> None:
    """dispatch_event routes red_card to handle_confirmed_red_card."""
    model = _make_model()
    dispatch_event(model, _red_card_event("localteam"))
    assert model.current_state_X == 1
