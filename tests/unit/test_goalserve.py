"""Tests for src/clients/goalserve.py — parsing, extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.clients.goalserve import (
    GoalserveClient,
    ensure_list,
    extract_goals,
    extract_red_cards,
    extract_stoppage_time,
    parse_minute,
    resolve_scoring_team,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestParseMinute:
    def test_regular_minute(self) -> None:
        assert parse_minute("23", "") == 23.0

    def test_stoppage_time(self) -> None:
        assert parse_minute("90", "3") == 93.0

    def test_first_half_stoppage(self) -> None:
        assert parse_minute("45", "2") == 47.0

    def test_empty_strings(self) -> None:
        assert parse_minute("", "") == 0.0

    def test_extra_min_none_equivalent(self) -> None:
        assert parse_minute("80", "") == 80.0


class TestResolveScoringTeam:
    def test_regular_goal_local(self) -> None:
        goal = {"owngoal": "False"}
        assert resolve_scoring_team(goal, "localteam") == "localteam"

    def test_regular_goal_visitor(self) -> None:
        goal = {"owngoal": "False"}
        assert resolve_scoring_team(goal, "visitorteam") == "visitorteam"

    def test_own_goal_flips_local_to_visitor(self) -> None:
        goal = {"owngoal": "True"}
        assert resolve_scoring_team(goal, "localteam") == "visitorteam"

    def test_own_goal_flips_visitor_to_local(self) -> None:
        goal = {"owngoal": "True"}
        assert resolve_scoring_team(goal, "visitorteam") == "localteam"

    def test_missing_owngoal_field(self) -> None:
        goal: dict[str, str] = {}
        assert resolve_scoring_team(goal, "localteam") == "localteam"


class TestEnsureList:
    def test_none_returns_empty(self) -> None:
        assert ensure_list(None) == []

    def test_dict_returns_single_item_list(self) -> None:
        result = ensure_list({"id": "1"})
        assert result == [{"id": "1"}]

    def test_list_returned_as_is(self) -> None:
        result = ensure_list([{"id": "1"}, {"id": "2"}])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert ensure_list([]) == []

    def test_scalar_wrapped(self) -> None:
        assert ensure_list("hello") == ["hello"]


# ---------------------------------------------------------------------------
# Extract functions with real fixture data
# ---------------------------------------------------------------------------


@pytest.fixture
def match_stats_data() -> dict[str, Any]:
    with open(FIXTURES_DIR / "goalserve_match_stats.json") as f:
        raw = json.load(f)
    return raw["match"]


@pytest.fixture
def fixtures_data() -> dict[str, Any]:
    with open(FIXTURES_DIR / "goalserve_fixtures.json") as f:
        return json.load(f)


class TestExtractGoals:
    def test_home_goals_count(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "localteam")
        assert len(goals) == 3  # Messi 23', Di Maria 36', Messi 108'

    def test_away_goals_count(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "visitorteam")
        assert len(goals) == 3  # Mbappe 80', 81', 118'

    def test_goal_minutes_parsed(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "localteam")
        minutes = [g["parsed_minute"] for g in goals]
        assert minutes == [23.0, 36.0, 108.0]

    def test_scoring_team_set(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "localteam")
        assert all(g["scoring_team"] == "localteam" for g in goals)

    def test_penalty_flag(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "localteam")
        assert goals[0]["is_penalty"] is True  # Messi 23' penalty
        assert goals[1]["is_penalty"] is False  # Di Maria 36'

    def test_var_cancelled_flag(self, match_stats_data: dict[str, Any]) -> None:
        goals = extract_goals(match_stats_data["summary"], "localteam")
        assert all(g["is_var_cancelled"] is False for g in goals)

    def test_no_goals_returns_empty(self) -> None:
        summary: dict[str, Any] = {"localteam": {"goals": None}}
        assert extract_goals(summary, "localteam") == []

    def test_missing_team_returns_empty(self) -> None:
        summary: dict[str, Any] = {}
        assert extract_goals(summary, "localteam") == []


class TestExtractRedCards:
    def test_no_red_cards(self, match_stats_data: dict[str, Any]) -> None:
        cards = extract_red_cards(match_stats_data["summary"], "localteam")
        assert cards == []

    def test_null_redcards(self, match_stats_data: dict[str, Any]) -> None:
        cards = extract_red_cards(match_stats_data["summary"], "visitorteam")
        assert cards == []

    def test_red_card_from_fixtures(self, fixtures_data: dict[str, Any]) -> None:
        # Liverpool vs Man City — van Dijk red card at 35'
        matches = fixtures_data["scores"]["category"]["tournament"]["match"]
        liverpool_match = matches[1]
        cards = extract_red_cards(liverpool_match["summary"], "localteam")
        assert len(cards) == 1
        assert cards[0]["parsed_minute"] == 35.0
        assert cards[0]["team"] == "localteam"
        assert cards[0]["name"] == "Virgil van Dijk"


class TestExtractStoppageTime:
    def test_wc_final_stoppage(self, match_stats_data: dict[str, Any]) -> None:
        alpha_1, alpha_2 = extract_stoppage_time(match_stats_data)
        assert alpha_1 == 7.0
        assert alpha_2 == 8.0

    def test_missing_stoppage(self) -> None:
        data: dict[str, Any] = {"matchinfo": {"time": {}}}
        alpha_1, alpha_2 = extract_stoppage_time(data)
        assert alpha_1 == 0.0
        assert alpha_2 == 0.0

    def test_no_matchinfo(self) -> None:
        data: dict[str, Any] = {}
        alpha_1, alpha_2 = extract_stoppage_time(data)
        assert alpha_1 == 0.0
        assert alpha_2 == 0.0


# ---------------------------------------------------------------------------
# Client response parsing (mock HTTP)
# ---------------------------------------------------------------------------


def _make_goalserve_client(handler: Any) -> GoalserveClient:
    """Create a GoalserveClient with a mock HTTP transport."""
    client = GoalserveClient(api_key="test_key")
    client._http._client = httpx.AsyncClient(
        base_url="https://www.goalserve.com/getfeed",
        transport=httpx.MockTransport(handler),
    )
    return client


class TestGoalserveClientGetFixtures:
    async def test_parses_fixture_response(self, fixtures_data: dict[str, Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=fixtures_data)

        client = _make_goalserve_client(handler)
        matches = await client.get_fixtures(league_id=1204)
        assert len(matches) == 2
        assert matches[0]["id"] == "300001"
        assert matches[1]["id"] == "300002"
        await client.close()

    async def test_season_param_sent(self, fixtures_data: dict[str, Any]) -> None:
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.update(dict(request.url.params))
            return httpx.Response(200, json=fixtures_data)

        client = _make_goalserve_client(handler)
        await client.get_fixtures(league_id=1204, season="2024-2025")
        assert captured_params.get("season") == "2024-2025"
        assert captured_params.get("json") == "1"
        await client.close()


class TestGoalserveClientGetMatchStats:
    async def test_parses_match_stats(self) -> None:
        with open(FIXTURES_DIR / "goalserve_match_stats.json") as f:
            raw = json.load(f)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        client = _make_goalserve_client(handler)
        stats = await client.get_match_stats("227491", league_id=1204)
        assert stats["id"] == "227491"
        assert stats["localteam"]["name"] == "Argentina"
        assert stats["visitorteam"]["name"] == "France"
        assert stats["status"] == "Full-time"
        # Verify nested data accessible
        assert "summary" in stats
        assert "stats" in stats
        assert "player_stats" in stats
        assert "substitutions" in stats
        await client.close()


class TestGoalserveClientGetLiveScore:
    async def test_returns_matching_match(self) -> None:
        live_data = {
            "scores": {
                "category": {
                    "match": [
                        {"id": "999", "status": "1H", "localteam": {"score": "1"}, "visitorteam": {"score": "0"}},
                        {"id": "888", "status": "2H", "localteam": {"score": "0"}, "visitorteam": {"score": "2"}},
                    ]
                }
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=live_data)

        client = _make_goalserve_client(handler)
        result = await client.get_live_score("888")
        assert result is not None
        assert result["id"] == "888"
        assert result["status"] == "2H"
        await client.close()

    async def test_returns_none_for_missing_match(self) -> None:
        live_data: dict[str, Any] = {"scores": {"category": {"match": []}}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=live_data)

        client = _make_goalserve_client(handler)
        result = await client.get_live_score("nonexistent")
        assert result is None
        await client.close()


# ---------------------------------------------------------------------------
# Edge cases: Goalserve inconsistent response shapes
# ---------------------------------------------------------------------------


class TestGoalserveResponseShapes:
    async def test_single_match_as_dict(self) -> None:
        """Goalserve sometimes returns a single match as a dict, not a list."""
        data: dict[str, Any] = {
            "scores": {
                "category": {
                    "match": {
                        "id": "500001",
                        "status": "Full-time",
                    }
                }
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=data)

        client = _make_goalserve_client(handler)
        matches = await client.get_fixtures(league_id=1204)
        assert len(matches) == 1
        assert matches[0]["id"] == "500001"
        await client.close()

    async def test_single_goal_as_dict(self) -> None:
        """Goalserve returns a single goal as a dict instead of a list."""
        # The fixtures_data has this for Chelsea's goal (visitorteam)
        with open(FIXTURES_DIR / "goalserve_fixtures.json") as f:
            fixtures_data = json.load(f)

        matches = fixtures_data["scores"]["category"]["tournament"]["match"]
        arsenal_match = matches[0]
        away_goals = extract_goals(arsenal_match["summary"], "visitorteam")
        assert len(away_goals) == 1
        assert away_goals[0]["name"] == "Nicolas Jackson"

    async def test_empty_category(self) -> None:
        data: dict[str, Any] = {"scores": {"category": []}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=data)

        client = _make_goalserve_client(handler)
        matches = await client.get_fixtures(league_id=9999)
        assert matches == []
        await client.close()


# ---------------------------------------------------------------------------
# Own goal + VAR cancelled integration
# ---------------------------------------------------------------------------


class TestOwnGoalAndVAR:
    def test_own_goal_scoring_team_flipped(self) -> None:
        summary: dict[str, Any] = {
            "localteam": {
                "goals": {
                    "player": [
                        {
                            "id": "1",
                            "minute": "45",
                            "extra_min": "",
                            "name": "Player X",
                            "penalty": "False",
                            "owngoal": "True",
                            "var_cancelled": "False",
                        }
                    ]
                }
            }
        }
        goals = extract_goals(summary, "localteam")
        assert len(goals) == 1
        assert goals[0]["scoring_team"] == "visitorteam"
        assert goals[0]["is_owngoal"] is True

    def test_var_cancelled_goal_flagged(self) -> None:
        summary: dict[str, Any] = {
            "localteam": {
                "goals": {
                    "player": [
                        {
                            "id": "2",
                            "minute": "55",
                            "extra_min": "",
                            "name": "Player Y",
                            "penalty": "False",
                            "owngoal": "False",
                            "var_cancelled": "True",
                        }
                    ]
                }
            }
        }
        goals = extract_goals(summary, "localteam")
        assert len(goals) == 1
        assert goals[0]["is_var_cancelled"] is True
