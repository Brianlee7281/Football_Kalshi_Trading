"""Tests for src/calibration/step_1_1_intervals.py — interval segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.calibration.step_1_1_intervals import build_intervals_from_goalserve

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_match() -> dict[str, Any]:
    """2022 World Cup Final synthetic fixture from docs/phase1.md."""
    with open(FIXTURES_DIR / "synthetic_match_phase1_example.json") as f:
        return json.load(f)


def _make_minimal_match(
    *,
    home_goals: list[dict[str, str]] | None = None,
    away_goals: list[dict[str, str]] | None = None,
    home_redcards: list[dict[str, str]] | None = None,
    away_redcards: list[dict[str, str]] | None = None,
    alpha_1: str = "3",
    alpha_2: str = "4",
    match_id: str = "test_001",
) -> dict[str, Any]:
    """Build a minimal Goalserve-shaped match dict for testing."""

    def _goals_block(goals: list[dict[str, str]] | None) -> Any:
        if not goals:
            return None
        if len(goals) == 1:
            return {"player": goals[0]}
        return {"player": goals}

    def _cards_block(cards: list[dict[str, str]] | None) -> Any:
        if not cards:
            return None
        if len(cards) == 1:
            return {"player": cards[0]}
        return {"player": cards}

    return {
        "id": match_id,
        "status": "Full-time",
        "localteam": {"name": "Home", "ht_score": "0", "ft_score": "0"},
        "visitorteam": {"name": "Away", "ht_score": "0", "ft_score": "0"},
        "matchinfo": {
            "time": {
                "addedTime_period1": alpha_1,
                "addedTime_period2": alpha_2,
            }
        },
        "summary": {
            "localteam": {
                "goals": _goals_block(home_goals),
                "redcards": _cards_block(home_redcards),
            },
            "visitorteam": {
                "goals": _goals_block(away_goals),
                "redcards": _cards_block(away_redcards),
            },
        },
    }


def _goal(
    minute: str,
    name: str = "Player",
    *,
    extra_min: str = "",
    penalty: str = "False",
    owngoal: str = "False",
    var_cancelled: str = "False",
) -> dict[str, str]:
    return {
        "id": "1",
        "minute": minute,
        "extra_min": extra_min,
        "name": name,
        "penalty": penalty,
        "owngoal": owngoal,
        "var_cancelled": var_cancelled,
    }


def _redcard(minute: str, name: str = "Player", extra_min: str = "") -> dict[str, str]:
    return {
        "id": "1",
        "minute": minute,
        "extra_min": extra_min,
        "name": name,
    }


# ---------------------------------------------------------------------------
# Synthetic fixture (WC Final) — from docs/phase1.md interval table
# ---------------------------------------------------------------------------


class TestSyntheticMatch:
    def test_interval_count(self, synthetic_match: dict[str, Any]) -> None:
        """Doc table has 6 regulation intervals + extra-time intervals."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert len(play) >= 6

    def test_interval_1(self, synthetic_match: dict[str, Any]) -> None:
        """Interval 1: [0, 23), ΔS=0, X=0, no goals."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        iv = intervals[0]
        assert iv.t_start == 0.0
        assert iv.t_end == 23.0
        assert iv.delta_S == 0
        assert iv.state_X == 0
        assert iv.home_goal_times == []
        assert iv.away_goal_times == []

    def test_interval_2(self, synthetic_match: dict[str, Any]) -> None:
        """Interval 2: [23, 36), ΔS=+1, home goal point event at 23."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        iv = intervals[1]
        assert iv.t_start == 23.0
        assert iv.t_end == 36.0
        assert iv.delta_S == 1
        assert iv.home_goal_times == [23.0]
        assert iv.goal_delta_before == [0]  # pre-goal ΔS

    def test_interval_3_end_includes_stoppage(
        self, synthetic_match: dict[str, Any]
    ) -> None:
        """Interval 3: t_end = 45 + alpha_1 = 52."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        iv = intervals[2]
        assert iv.t_start == 36.0
        assert iv.t_end == pytest.approx(45.0 + 7.0)

    def test_halftime_excluded(self, synthetic_match: dict[str, Any]) -> None:
        """No play interval should have t_start inside halftime."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        play = [iv for iv in intervals if not iv.is_halftime]
        ht_start = 45.0 + 7.0
        # No interval starts strictly inside halftime
        for iv in play:
            # All intervals either end at ht_start or start at ht_start
            # (no interval bridges halftime)
            assert not (iv.t_start > ht_start and iv.t_start < ht_start)

    def test_interval_after_halftime(self, synthetic_match: dict[str, Any]) -> None:
        """Interval 4: [52, 80), ΔS=+2 (continues from first half)."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        play = [iv for iv in intervals if not iv.is_halftime]
        iv4 = play[3]
        assert iv4.t_start == pytest.approx(52.0)
        assert iv4.t_end == 80.0
        assert iv4.delta_S == 2

    def test_interval_5_away_goal(self, synthetic_match: dict[str, Any]) -> None:
        """Interval 5: [80, 81), ΔS=+1 (away goal drops ΔS)."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        play = [iv for iv in intervals if not iv.is_halftime]
        iv5 = play[4]
        assert iv5.t_start == 80.0
        assert iv5.t_end == 81.0
        assert iv5.delta_S == 1
        assert iv5.away_goal_times == [80.0]
        assert iv5.goal_delta_before == [2]  # pre-goal ΔS was +2

    def test_T_m(self, synthetic_match: dict[str, Any]) -> None:
        """T_m = 90 + 7 + 8 = 105."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        for iv in intervals:
            assert iv.T_m == pytest.approx(105.0)

    def test_alpha_values(self, synthetic_match: dict[str, Any]) -> None:
        intervals = build_intervals_from_goalserve(synthetic_match)
        for iv in intervals:
            assert iv.alpha_1 == pytest.approx(7.0)
            assert iv.alpha_2 == pytest.approx(8.0)

    def test_match_id(self, synthetic_match: dict[str, Any]) -> None:
        intervals = build_intervals_from_goalserve(synthetic_match)
        for iv in intervals:
            assert iv.match_id == "227491"


# ---------------------------------------------------------------------------
# 0-0 match — exactly 2 intervals (one per half)
# ---------------------------------------------------------------------------


class TestZeroZeroMatch:
    def test_two_intervals(self) -> None:
        match = _make_minimal_match()
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert len(play) == 2

    def test_first_half_boundaries(self) -> None:
        match = _make_minimal_match(alpha_1="3", alpha_2="4")
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert play[0].t_start == 0.0
        assert play[0].t_end == pytest.approx(48.0)  # 45 + 3

    def test_second_half_boundaries(self) -> None:
        match = _make_minimal_match(alpha_1="3", alpha_2="4")
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert play[1].t_start == pytest.approx(48.0)
        assert play[1].t_end == pytest.approx(94.0)  # 90 + 4

    def test_no_goals_recorded(self) -> None:
        match = _make_minimal_match()
        intervals = build_intervals_from_goalserve(match)
        for iv in intervals:
            assert iv.home_goal_times == []
            assert iv.away_goal_times == []
            assert iv.goal_delta_before == []

    def test_delta_S_zero(self) -> None:
        match = _make_minimal_match()
        intervals = build_intervals_from_goalserve(match)
        for iv in intervals:
            assert iv.delta_S == 0


# ---------------------------------------------------------------------------
# Own goal handling
# ---------------------------------------------------------------------------


class TestOwnGoal:
    def test_owngoal_flips_scoring_team(self) -> None:
        """owngoal='True' on localteam → scoring_team=visitorteam → ΔS decreases."""
        match = _make_minimal_match(
            home_goals=[_goal("30", owngoal="True")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # After own goal by home team, visitorteam scores → ΔS = -1
        iv_after_goal = play[1]
        assert iv_after_goal.delta_S == -1
        assert iv_after_goal.away_goal_times == [30.0]


# ---------------------------------------------------------------------------
# VAR cancelled goal
# ---------------------------------------------------------------------------


class TestVARCancelled:
    def test_var_cancelled_no_split(self) -> None:
        """var_cancelled='True' → no interval split, no ΔS change."""
        match = _make_minimal_match(
            home_goals=[_goal("55", var_cancelled="True")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # Should still be exactly 2 intervals (one per half, no split at 55)
        assert len(play) == 2
        # ΔS should remain 0 throughout
        for iv in play:
            assert iv.delta_S == 0

    def test_var_cancelled_mixed_with_real(self) -> None:
        """Only non-cancelled goals create splits."""
        match = _make_minimal_match(
            home_goals=[
                _goal("20"),
                _goal("55", var_cancelled="True"),
                _goal("70"),
            ],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # Splits at: 0, 20, 48(ht), 70, 94 → 4 play intervals
        assert len(play) == 4
        assert play[0].delta_S == 0  # [0, 20)
        assert play[1].delta_S == 1  # [20, 48)
        assert play[2].delta_S == 1  # [48, 70) — VAR goal at 55 ignored
        assert play[3].delta_S == 2  # [70, 94)


# ---------------------------------------------------------------------------
# Red card handling
# ---------------------------------------------------------------------------


class TestRedCard:
    def test_red_card_creates_split(self) -> None:
        match = _make_minimal_match(
            home_redcards=[_redcard("35")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # Splits: 0, 35, 48(ht), 94 → 3 play intervals
        assert len(play) == 3

    def test_red_card_state_transition_home(self) -> None:
        """Home red card: state 0 → 1."""
        match = _make_minimal_match(
            home_redcards=[_redcard("35")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert play[0].state_X == 0  # before red card
        assert play[1].state_X == 1  # after home red card
        assert play[2].state_X == 1  # continues

    def test_red_card_state_transition_away(self) -> None:
        """Away red card: state 0 → 2."""
        match = _make_minimal_match(
            away_redcards=[_redcard("60")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert play[1].state_X == 0  # second half before red card
        assert play[2].state_X == 2  # after away red card

    def test_both_red_cards_state_3(self) -> None:
        """Home + away red cards → state 3."""
        match = _make_minimal_match(
            home_redcards=[_redcard("30")],
            away_redcards=[_redcard("60")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # After home red at 30: state 1
        # After away red at 60: state 1 → 3
        states = [iv.state_X for iv in play]
        assert 3 in states

    def test_red_card_transition_recorded(self) -> None:
        match = _make_minimal_match(
            home_redcards=[_redcard("35")],
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # The interval starting at 35 (after red card) should have the transition
        rc_interval = play[1]
        assert len(rc_interval.red_card_transitions) == 1
        rc = rc_interval.red_card_transitions[0]
        assert rc.minute == 35.0
        assert rc.team == "localteam"
        assert rc.from_state == 0
        assert rc.to_state == 1


# ---------------------------------------------------------------------------
# Real fixture (goalserve_match_stats.json — same WC Final data)
# ---------------------------------------------------------------------------


class TestRealFixture:
    def test_intervals_from_saved_fixture(self) -> None:
        """Build intervals from the saved goalserve_match_stats.json fixture."""
        with open(FIXTURES_DIR / "goalserve_match_stats.json") as f:
            raw = json.load(f)
        match = raw["match"]
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert len(play) >= 6
        # Same match as synthetic — verify consistency
        assert intervals[0].match_id == "227491"
        assert intervals[0].T_m == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_stoppage_time_goal(self) -> None:
        """Goal in first-half stoppage time (45+2')."""
        match = _make_minimal_match(
            home_goals=[_goal("45", extra_min="2")],
            alpha_1="3",
        )
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        # Goal at minute 47, ht at 48 → intervals: [0,47), [47,48), [48, 94)
        assert len(play) == 3
        assert play[1].t_start == pytest.approx(47.0)
        assert play[1].t_end == pytest.approx(48.0)

    def test_missing_summary_returns_two_intervals(self) -> None:
        """Match with empty summary → 2 intervals (one per half)."""
        match = {
            "id": "empty",
            "matchinfo": {"time": {"addedTime_period1": "2", "addedTime_period2": "3"}},
            "summary": {},
        }
        intervals = build_intervals_from_goalserve(match)
        play = [iv for iv in intervals if not iv.is_halftime]
        assert len(play) == 2

    def test_intervals_contiguous(self, synthetic_match: dict[str, Any]) -> None:
        """All intervals should be contiguous (no gaps, no overlaps)."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        play = [iv for iv in intervals if not iv.is_halftime]
        for i in range(len(play) - 1):
            assert play[i].t_end == pytest.approx(play[i + 1].t_start)

    def test_no_zero_width_intervals(self, synthetic_match: dict[str, Any]) -> None:
        """No interval should have t_start == t_end."""
        intervals = build_intervals_from_goalserve(synthetic_match)
        for iv in intervals:
            assert iv.t_end > iv.t_start
