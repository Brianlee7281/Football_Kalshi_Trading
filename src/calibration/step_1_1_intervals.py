"""Step 1.1 — Time-Series Event Segmentation and Intervalization.

Converts Goalserve match data into a list of IntervalRecord objects,
splitting the match timeline at every state-changing event (goals,
red cards, halftime boundary).

Reference: docs/phase1.md Step 1.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.clients.goalserve import (
    extract_goals,
    extract_red_cards,
    extract_stoppage_time,
)
from src.common.types import IntervalRecord, RedCardTransition

# ---------------------------------------------------------------------------
# Internal event representation
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    """Sortable match event used during interval construction."""

    kind: str  # "goal", "red_card"
    minute: float
    team: str  # "localteam" / "visitorteam"
    is_owngoal: bool = False


# ---------------------------------------------------------------------------
# Red-card Markov state transitions
# State 0: 11v11, State 1: 10v11 (home sent off),
# State 2: 11v10 (away sent off), State 3: 10v10
# ---------------------------------------------------------------------------

_RC_TRANSITIONS: dict[tuple[str, int], int] = {
    ("localteam", 0): 1,
    ("localteam", 2): 3,
    ("visitorteam", 0): 2,
    ("visitorteam", 1): 3,
}


def _advance_red_card_state(team: str, current: int) -> int:
    """Return the new Markov state after a red card for *team*."""
    return _RC_TRANSITIONS.get((team, current), current)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_intervals_from_goalserve(
    match_data: dict[str, Any],
) -> list[IntervalRecord]:
    """Convert a Goalserve match dict into interval records.

    Args:
        match_data: A single match dict from Goalserve (fixture or stats)
                    with keys ``id``, ``matchinfo``, ``summary``, and the
                    team blocks ``localteam`` / ``visitorteam``.

    Returns:
        Ordered list of :class:`IntervalRecord` covering the entire match.
        Halftime is not emitted as a separate interval; the first-half
        last interval ends at ``45 + alpha_1`` and the second-half first
        interval starts at the same value.
    """
    match_id = str(match_data.get("@id", match_data.get("id", "")))
    alpha_1, alpha_2 = extract_stoppage_time(match_data)
    T_m = 90.0 + alpha_1 + alpha_2

    events = _collect_events(match_data)
    ht_boundary = 45.0 + alpha_1
    reg_end = 90.0 + alpha_2

    # Determine match end time
    max_event = max((e.minute for e in events), default=0.0)
    match_end = max(120.0, max_event) if max_event > reg_end else reg_end

    # Build sorted list of boundary times
    boundaries: set[float] = {0.0, ht_boundary, match_end}
    for e in events:
        boundaries.add(e.minute)
    sorted_bounds = sorted(boundaries)

    # Index events by minute for quick lookup
    events_at: dict[float, list[_Event]] = {}
    for e in events:
        events_at.setdefault(e.minute, []).append(e)

    # Walk consecutive boundary pairs to create intervals
    intervals: list[IntervalRecord] = []
    delta_S = 0
    state_X = 0

    for i in range(len(sorted_bounds) - 1):
        t_start = sorted_bounds[i]
        t_end = sorted_bounds[i + 1]

        if t_start >= t_end:
            continue

        # Skip intervals that span the halftime boundary
        if t_start >= ht_boundary and t_end <= ht_boundary:
            continue

        # Process events at t_start to determine this interval's state
        home_goals: list[float] = []
        away_goals: list[float] = []
        goal_deltas: list[int] = []
        rc_transitions: list[RedCardTransition] = []

        for ev in events_at.get(t_start, []):
            if ev.kind == "goal":
                pre_delta = delta_S
                if ev.team == "localteam":
                    delta_S += 1
                    home_goals.append(ev.minute)
                else:
                    delta_S -= 1
                    away_goals.append(ev.minute)
                goal_deltas.append(pre_delta)

            elif ev.kind == "red_card":
                from_state = state_X
                state_X = _advance_red_card_state(ev.team, state_X)
                rc_transitions.append(
                    RedCardTransition(
                        minute=ev.minute,
                        team=ev.team,
                        from_state=from_state,
                        to_state=state_X,
                    )
                )

        intervals.append(
            IntervalRecord(
                match_id=match_id,
                t_start=t_start,
                t_end=t_end,
                state_X=state_X,
                delta_S=delta_S,
                home_goal_times=home_goals,
                away_goal_times=away_goals,
                goal_delta_before=goal_deltas,
                T_m=T_m,
                is_halftime=False,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                red_card_transitions=rc_transitions,
            )
        )

    return intervals


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------


def _collect_events(match_data: dict[str, Any]) -> list[_Event]:
    """Gather goals and red cards from match summary."""
    events: list[_Event] = []
    summary = match_data.get("summary", {})

    for team_key in ("localteam", "visitorteam"):
        for g in extract_goals(summary, team_key):
            if g["is_var_cancelled"]:
                continue
            events.append(
                _Event(
                    kind="goal",
                    minute=g["parsed_minute"],
                    team=g["scoring_team"],
                    is_owngoal=g["is_owngoal"],
                )
            )

        for c in extract_red_cards(summary, team_key):
            events.append(
                _Event(
                    kind="red_card",
                    minute=c["parsed_minute"],
                    team=c["team"],
                )
            )

    return events
