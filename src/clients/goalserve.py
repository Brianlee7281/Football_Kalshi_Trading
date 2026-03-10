"""Goalserve REST client: fixtures, match stats, live scores."""

from __future__ import annotations

from typing import Any

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

# Goalserve returns XML by default; append ?json=1 for JSON.
_BASE_URL = "https://www.goalserve.com/getfeed"

logger = get_logger("goalserve")


class GoalserveClient:
    """Async client for Goalserve Soccer API.

    Endpoints used (per API reference):
        - soccerfixtures/leagueid/{league_id}       — current season fixtures & results
        - soccerhistory/leagueid/{league_id}-{season} — historical season data
        - commentaries/match?id={match_id}&league={league_id} — match stats
        - soccernew/home                            — live scores (Phase 3 polling)
        - soccernew/d-{n}                           — past day scores

    Args:
        api_key: Goalserve API key.
        timeout: Request timeout in seconds.
    """

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._http = BaseClient(
            _BASE_URL,
            timeout=timeout,
            max_retries=3,
            backoff_base=1.0,
        )
        self._logger = get_logger("goalserve")

    async def close(self) -> None:
        await self._http.close()

    async def __aenter__(self) -> GoalserveClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Fixtures / Results (Phase 1 — current season)
    # ------------------------------------------------------------------

    async def get_fixtures(
        self,
        league_id: int,
        *,
        season: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch fixtures for a league (current season).

        API ref: /getfeed/{key}/soccerfixtures/leagueid/{league_id}

        Args:
            league_id: Goalserve league ID (e.g. 1204 for EPL).
            season: Optional season filter (e.g. "2024-2025").

        Returns:
            List of match dicts from the Goalserve response.
        """
        path = f"/{self._api_key}/soccerfixtures/leagueid/{league_id}"
        params: dict[str, Any] = {"json": "1"}
        if season:
            params["season"] = season

        response = await self._http.get(path, params=params)
        data = response.json()

        return _extract_matches(data)

    # ------------------------------------------------------------------
    # Historical Fixtures (Phase 1 — past seasons)
    # ------------------------------------------------------------------

    async def get_historical_fixtures(
        self,
        league_id: int,
        season: str,
    ) -> list[dict[str, Any]]:
        """Fetch historical fixtures for a league and season.

        API ref: /getfeed/{key}/soccerhistory/leagueid/{league_id}-{season}

        Args:
            league_id: Goalserve league ID (e.g. 1204 for EPL).
            season: Season string (e.g. "2022-2023").

        Returns:
            List of historical match dicts.
        """
        path = f"/{self._api_key}/soccerhistory/leagueid/{league_id}-{season}"
        params: dict[str, Any] = {"json": "1"}

        response = await self._http.get(path, params=params)
        data = response.json()

        return _extract_matches(data)

    # ------------------------------------------------------------------
    # Match Stats (Phase 1 — detailed stats + player stats)
    # ------------------------------------------------------------------

    async def get_match_stats(
        self,
        match_id: str,
        league_id: int,
    ) -> dict[str, Any]:
        """Fetch detailed match statistics including player stats and xG.

        API ref: /getfeed/{key}/commentaries/match?id={match_id}&league={league_id}

        Args:
            match_id: Goalserve match ID.
            league_id: Goalserve league ID (required by API).

        Returns:
            Raw match stats dict with keys like 'stats', 'player_stats',
            'summary', 'matchinfo', 'teams', 'substitutions', etc.
        """
        path = f"/{self._api_key}/commentaries/match"
        params: dict[str, Any] = {"id": match_id, "league": str(league_id), "json": "1"}

        response = await self._http.get(path, params=params)
        data = response.json()

        return _extract_match_stats(data)

    # ------------------------------------------------------------------
    # Live Score (Phase 3 — real-time polling)
    # ------------------------------------------------------------------

    async def get_live_scores(self) -> list[dict[str, Any]]:
        """Fetch all current live match scores.

        API ref: /getfeed/{key}/soccernew/home

        Returns:
            List of live match dicts with score, minute, status, events.
        """
        path = f"/{self._api_key}/soccernew/home"
        params: dict[str, Any] = {"json": "1"}

        response = await self._http.get(path, params=params)
        data = response.json()

        return _extract_live_matches(data)

    async def get_live_score(self, match_id: str) -> dict[str, Any] | None:
        """Fetch live score for a specific match.

        Args:
            match_id: Goalserve match ID.

        Returns:
            Live match dict or None if match not found in live feed.
        """
        all_live = await self.get_live_scores()
        for match in all_live:
            if str(match.get("id")) == str(match_id):
                return match
        return None

    # ------------------------------------------------------------------
    # Past Scores (Phase 1 — recent results)
    # ------------------------------------------------------------------

    async def get_past_scores(self, days_ago: int = 1) -> list[dict[str, Any]]:
        """Fetch match scores from a past day.

        API ref: /getfeed/{key}/soccernew/d-{n}

        Args:
            days_ago: Number of days in the past (1-7).

        Returns:
            List of match dicts from that day.
        """
        path = f"/{self._api_key}/soccernew/d-{days_ago}"
        params: dict[str, Any] = {"json": "1"}

        response = await self._http.get(path, params=params)
        data = response.json()

        return _extract_live_matches(data)


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _extract_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract match list from Goalserve fixtures response.

    Goalserve nests matches under various structures depending on
    whether a single matchday or multiple are returned.
    """
    matches: list[dict[str, Any]] = []

    # Navigate into the response — structure varies by endpoint version
    tournaments = data.get("scores", data)
    if isinstance(tournaments, dict):
        # Could be nested under 'category' -> 'match' or 'tournament' -> 'match'
        categories = tournaments.get("category", tournaments)
        if isinstance(categories, dict):
            categories = [categories]
        if isinstance(categories, list):
            for cat in categories:
                _collect_matches_from_category(cat, matches)

    return matches


def _collect_matches_from_category(
    category: dict[str, Any],
    out: list[dict[str, Any]],
) -> None:
    """Recursively collect match dicts from a Goalserve category/tournament."""
    raw_matches = category.get("match", category.get("matches", []))
    if isinstance(raw_matches, dict):
        raw_matches = [raw_matches]
    if isinstance(raw_matches, list):
        out.extend(raw_matches)

    # Some responses nest under 'tournament'
    tournaments = category.get("tournament", [])
    if isinstance(tournaments, dict):
        tournaments = [tournaments]
    if isinstance(tournaments, list):
        for t in tournaments:
            raw = t.get("match", [])
            if isinstance(raw, dict):
                raw = [raw]
            if isinstance(raw, list):
                out.extend(raw)


def _extract_match_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the match stats payload from the Goalserve response wrapper."""
    # The response may wrap under 'scores' or 'match' or return directly
    if "match" in data:
        match = data["match"]
        if isinstance(match, list):
            return match[0] if match else {}
        if isinstance(match, dict):
            return match
        return {}
    if "scores" in data:
        return _extract_match_stats(data["scores"])
    return data


def _extract_live_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract live match list from Goalserve soccernew response."""
    matches: list[dict[str, Any]] = []

    scores = data.get("scores", data)
    if isinstance(scores, dict):
        categories = scores.get("category", [])
        if isinstance(categories, dict):
            categories = [categories]
        if isinstance(categories, list):
            for cat in categories:
                raw = cat.get("match", [])
                if isinstance(raw, dict):
                    raw = [raw]
                if isinstance(raw, list):
                    matches.extend(raw)

    return matches


# ---------------------------------------------------------------------------
# Typed extraction helpers (used by Step 1.1 and Phase 3)
# ---------------------------------------------------------------------------


def ensure_list(value: Any) -> list[Any]:
    """Normalize Goalserve's inconsistent list/dict returns to a list."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def parse_minute(minute: str, extra_min: str = "") -> float:
    """Parse Goalserve minute + extra_min into a float timestamp.

    Examples:
        parse_minute("23", "")   -> 23.0
        parse_minute("90", "3")  -> 93.0
        parse_minute("45", "2")  -> 47.0
    """
    base = float(minute) if minute else 0.0
    extra = float(extra_min) if extra_min else 0.0
    return base + extra


def resolve_scoring_team(goal_event: dict[str, Any], recorded_team: str) -> str:
    """Flip scoring team for own goals.

    Args:
        goal_event: Goalserve goal dict with 'owngoal' field.
        recorded_team: The team key under which the goal was recorded.

    Returns:
        Actual scoring team: "localteam" or "visitorteam".
    """
    if str(goal_event.get("owngoal", "")).lower() == "true":
        return "visitorteam" if recorded_team == "localteam" else "localteam"
    return recorded_team


def extract_goals(
    summary: dict[str, Any],
    team_key: str,
) -> list[dict[str, Any]]:
    """Extract goal events from a team's summary, filtering VAR-cancelled.

    Args:
        summary: The 'summary' dict from Goalserve match data.
        team_key: "localteam" or "visitorteam".

    Returns:
        List of valid (non-VAR-cancelled) goal dicts with 'minute',
        'scoring_team', 'recorded_team', and original fields.
    """
    goals: list[dict[str, Any]] = []
    team_data = summary.get(team_key, {})
    if not team_data:
        return goals

    raw_goals = team_data.get("goals", {})
    if not raw_goals:
        return goals

    for g in ensure_list(raw_goals.get("player", [])):
        goal: dict[str, Any] = dict(g)
        goal["recorded_team"] = team_key
        goal["scoring_team"] = resolve_scoring_team(g, team_key)
        goal["parsed_minute"] = parse_minute(
            g.get("minute", "0"), g.get("extra_min", "")
        )
        goal["is_var_cancelled"] = str(g.get("var_cancelled", "")).lower() == "true"
        goal["is_owngoal"] = str(g.get("owngoal", "")).lower() == "true"
        goal["is_penalty"] = str(g.get("penalty", "")).lower() == "true"
        goals.append(goal)

    return goals


def extract_red_cards(
    summary: dict[str, Any],
    team_key: str,
) -> list[dict[str, Any]]:
    """Extract red card events from a team's summary.

    Args:
        summary: The 'summary' dict from Goalserve match data.
        team_key: "localteam" or "visitorteam".

    Returns:
        List of red card dicts with 'minute', 'team', and original fields.
    """
    cards: list[dict[str, Any]] = []
    team_data = summary.get(team_key, {})
    if not team_data:
        return cards

    raw = team_data.get("redcards", {})
    if not raw:
        return cards

    for r in ensure_list(raw.get("player", [])):
        card: dict[str, Any] = dict(r)
        card["team"] = team_key
        card["parsed_minute"] = parse_minute(
            r.get("minute", "0"), r.get("extra_min", "")
        )
        cards.append(card)

    return cards


def extract_stoppage_time(match_data: dict[str, Any]) -> tuple[float, float]:
    """Extract first/second half stoppage times from matchinfo.

    Returns:
        (alpha_1, alpha_2) — stoppage time in minutes for each half.
    """
    time_info = match_data.get("matchinfo", {}).get("time", {})
    alpha_1 = float(time_info.get("addedTime_period1") or 0)
    alpha_2 = float(time_info.get("addedTime_period2") or 0)
    return alpha_1, alpha_2
