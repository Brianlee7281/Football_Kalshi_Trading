"""Goalserve REST client: fixtures, match stats, live scores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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

    def _safe_json(
        self,
        response: Any,
        caller: str,
        **ctx: Any,
    ) -> dict[str, Any]:
        """Decode JSON from an httpx response, returning {} on empty/invalid body."""
        body = response.text.strip()
        if not body:
            self._logger.warning(
                "empty_response_body",
                caller=caller,
                status_code=response.status_code,
                **ctx,
            )
            return {}
        try:
            return response.json()  # type: ignore[no-any-return]
        except Exception as exc:
            self._logger.error(
                "json_decode_failed",
                caller=caller,
                status_code=response.status_code,
                body_preview=body[:200],
                error=str(exc),
                **ctx,
            )
            return {}

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
        data = self._safe_json(response, "get_fixtures", league_id=league_id)

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
        data = self._safe_json(
            response, "get_historical_fixtures",
            league_id=league_id, season=season,
        )

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
            Returns empty dict if the endpoint returns no data (e.g. match
            hasn't started yet).
        """
        path = f"/{self._api_key}/commentaries/match"
        params: dict[str, Any] = {"id": match_id, "league": str(league_id), "json": "1"}

        response = await self._http.get(path, params=params)

        # Goalserve returns empty body for matches with no commentaries yet
        body = response.text.strip()
        if not body:
            self._logger.warning(
                "empty_commentaries_response",
                match_id=match_id,
                league_id=league_id,
                status_code=response.status_code,
            )
            return {}

        try:
            data = response.json()
        except Exception as exc:
            self._logger.error(
                "commentaries_json_decode_failed",
                match_id=match_id,
                league_id=league_id,
                status_code=response.status_code,
                body_preview=body[:200],
                error=str(exc),
            )
            return {}

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
        data = self._safe_json(response, "get_live_scores")

        return _extract_live_matches(data)

    async def get_live_score(self, match_id: str) -> dict[str, Any] | None:
        """Fetch live score for a specific match.

        Searches by @id, @fix_id, and @static_id since the match_schedule
        may store any of these Goalserve identifier fields.

        Args:
            match_id: Goalserve match ID (may be @id, @fix_id, or @static_id).

        Returns:
            Live match dict or None if match not found in live feed.
        """
        all_live = await self.get_live_scores()
        mid = str(match_id)
        for match in all_live:
            # Check all known ID fields
            if (
                str(match.get("@id", "")) == mid
                or str(match.get("id", "")) == mid
                or str(match.get("@fix_id", "")) == mid
                or str(match.get("@static_id", "")) == mid
            ):
                return match
        return None

    # ------------------------------------------------------------------
    # Commentaries by league (Phase 1 — historical with red cards)
    # ------------------------------------------------------------------

    async def get_commentaries_by_league(
        self,
        league_id: int,
        date: str,
    ) -> list[dict[str, Any]]:
        """Fetch commentaries for all matches in a league on a given date.

        Returns full match dicts with ``summary`` (goals, red cards),
        ``matchinfo`` (stoppage time), ``stats``, ``player_stats``, etc.

        API ref: /getfeed/{key}/commentaries/{league_id}?date={date}

        Args:
            league_id: Goalserve league ID (e.g. 1204 for EPL).
            date: Date string in DD.MM.YYYY format (e.g. "15.01.2024").

        Returns:
            List of match dicts with full commentary data.
        """
        path = f"/{self._api_key}/commentaries/{league_id}"
        params: dict[str, Any] = {"date": date, "json": "1"}

        response = await self._http.get(path, params=params)
        data = self._safe_json(
            response, "get_commentaries_by_league",
            league_id=league_id, date=date,
        )

        return _extract_commentaries_matches(data)

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
        data = self._safe_json(response, "get_past_scores", days_ago=days_ago)

        return _extract_live_matches(data)

    # ------------------------------------------------------------------
    # Upcoming Fixtures (Scheduler — Phase 2/3 trigger computation)
    # ------------------------------------------------------------------

    async def get_upcoming_fixtures(
        self,
        league_ids: list[int],
        *,
        hours_ahead: float = 48.0,
    ) -> list[dict[str, Any]]:
        """Fetch upcoming fixtures across multiple leagues within a time window.

        Calls ``get_fixtures`` for each league, parses kickoff timestamps,
        and returns only matches starting within ``hours_ahead`` hours from now.

        Each returned dict is the raw Goalserve match dict enriched with:
          ``_league_id``  — the integer league ID
          ``_kickoff_utc`` — a timezone-aware ``datetime`` in UTC

        Errors fetching a single league are logged and skipped so that a
        partial failure does not block discovery for other leagues.

        Args:
            league_ids: List of Goalserve league IDs to scan.
            hours_ahead: Forward window in hours (default 48).

        Returns:
            List of upcoming fixture dicts, sorted by ``_kickoff_utc``.
        """
        now = datetime.now(UTC)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming: list[dict[str, Any]] = []
        for league_id in league_ids:
            try:
                fixtures = await self.get_fixtures(league_id)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "get_fixtures_failed",
                    league_id=league_id,
                    error=str(exc),
                )
                continue

            for fix in fixtures:
                kickoff = _parse_fixture_kickoff(fix)
                if kickoff and now < kickoff <= cutoff:
                    enriched = dict(fix)
                    enriched["_league_id"] = league_id
                    enriched["_kickoff_utc"] = kickoff
                    upcoming.append(enriched)

        upcoming.sort(key=lambda f: f["_kickoff_utc"])
        return upcoming


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _extract_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract match list from Goalserve fixtures/history response.

    Real API structure: ``results.tournament.week[].match[]``
    Each match has @-prefixed fields: @id, @status, @date, etc.
    """
    matches: list[dict[str, Any]] = []

    # Primary path: results.tournament.week[].match[]
    results = data.get("results", {})
    if isinstance(results, dict):
        tournament = results.get("tournament", {})
        if isinstance(tournament, dict):
            weeks = tournament.get("week", [])
            if isinstance(weeks, dict):
                weeks = [weeks]
            if isinstance(weeks, list):
                for week in weeks:
                    raw = week.get("match", [])
                    if isinstance(raw, dict):
                        raw = [raw]
                    if isinstance(raw, list):
                        matches.extend(raw)
            # Also check for matches directly under tournament (history endpoint)
            if not matches:
                raw = tournament.get("match", [])
                if isinstance(raw, dict):
                    raw = [raw]
                if isinstance(raw, list):
                    matches.extend(raw)

    # Fallback: scores.category structure (some endpoints)
    if not matches:
        scores = data.get("scores", {})
        if isinstance(scores, dict):
            categories = scores.get("category", [])
            if isinstance(categories, dict):
                categories = [categories]
            if isinstance(categories, list):
                for cat in categories:
                    _collect_matches_from_node(cat, matches)

    return matches


def _collect_matches_from_node(
    node: dict[str, Any],
    out: list[dict[str, Any]],
) -> None:
    """Collect match dicts from a Goalserve category/tournament node."""
    # Direct match/matches keys
    for key in ("match", "matches"):
        raw = node.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict) and "match" in raw:
            # Nested: matches.match[]
            inner = raw["match"]
            if isinstance(inner, dict):
                inner = [inner]
            if isinstance(inner, list):
                out.extend(inner)
        elif isinstance(raw, dict):
            out.append(raw)
        elif isinstance(raw, list):
            out.extend(raw)

    # Recurse into tournament nodes
    tournaments = node.get("tournament", [])
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
    """Extract the match stats payload from the Goalserve commentaries response.

    Real API structure: ``commentaries.tournament.match``
    """
    # Primary path: commentaries.tournament.match
    if "commentaries" in data:
        commentaries = data["commentaries"]
        if isinstance(commentaries, dict):
            tournament = commentaries.get("tournament", {})
            if isinstance(tournament, dict) and "match" in tournament:
                match = tournament["match"]
                if isinstance(match, dict):
                    return match
                if isinstance(match, list):
                    return match[0] if match else {}

    # Fallback: direct 'match' key (legacy / test fixtures)
    if "match" in data:
        match = data["match"]
        if isinstance(match, list):
            return match[0] if match else {}
        if isinstance(match, dict):
            return match

    return data


def _extract_commentaries_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract match list from Goalserve commentaries-by-league response.

    Structure: ``commentaries.tournament.match`` (single dict or list).
    Falls back to ``_extract_matches`` if structure doesn't match.
    """
    matches: list[dict[str, Any]] = []

    commentaries = data.get("commentaries", {})
    if isinstance(commentaries, dict):
        tournament = commentaries.get("tournament", {})
        if isinstance(tournament, dict):
            raw = tournament.get("match", [])
            if isinstance(raw, dict):
                matches.append(raw)
            elif isinstance(raw, list):
                matches.extend(raw)
        elif isinstance(tournament, list):
            for t in tournament:
                raw = t.get("match", []) if isinstance(t, dict) else []
                if isinstance(raw, dict):
                    matches.append(raw)
                elif isinstance(raw, list):
                    matches.extend(raw)

    # Fallback: try generic extraction
    if not matches:
        matches = _extract_matches(data)

    return matches


def _extract_live_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract live match list from Goalserve soccernew response.

    Real API structure: ``scores.category[].matches.match[]``
    The ``matches`` node contains a ``@date`` and nested ``match`` list.
    """
    matches: list[dict[str, Any]] = []

    scores = data.get("scores", data)
    if isinstance(scores, dict):
        categories = scores.get("category", [])
        if isinstance(categories, dict):
            categories = [categories]
        if isinstance(categories, list):
            for cat in categories:
                # Primary path: category.matches.match[]
                matches_node = cat.get("matches", {})
                if isinstance(matches_node, dict):
                    raw = matches_node.get("match", [])
                    if isinstance(raw, dict):
                        raw = [raw]
                    if isinstance(raw, list):
                        matches.extend(raw)
                elif isinstance(matches_node, list):
                    for mn in matches_node:
                        raw = mn.get("match", []) if isinstance(mn, dict) else []
                        if isinstance(raw, dict):
                            raw = [raw]
                        if isinstance(raw, list):
                            matches.extend(raw)

                # Fallback: category.match[] (some endpoints)
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
        parse_minute("23", "")          -> 23.0
        parse_minute("90", "3")         -> 93.0
        parse_minute("45", "2")         -> 47.0
        parse_minute("90+5", "")        -> 95.0
        parse_minute("pen miss 22", "") -> 22.0
    """
    # Handle "90+5" format (common in historical fixtures)
    if minute and "+" in minute:
        parts = minute.split("+", 1)
        try:
            return float(parts[0]) + float(parts[1])
        except (ValueError, TypeError):
            pass
    try:
        base = float(minute) if minute else 0.0
    except (ValueError, TypeError):
        # Extract trailing number from strings like "pen miss 22"
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*$", minute)
        base = float(match.group(1)) if match else 0.0
    try:
        extra = float(extra_min) if extra_min else 0.0
    except (ValueError, TypeError):
        extra = 0.0
    return base + extra


def _get_field(d: dict[str, Any], name: str, default: Any = "") -> Any:
    """Get a field from a Goalserve dict, trying @-prefixed key first."""
    return d.get(f"@{name}", d.get(name, default))


def resolve_scoring_team(goal_event: dict[str, Any], recorded_team: str) -> str:
    """Flip scoring team for own goals.

    Args:
        goal_event: Goalserve goal dict with 'owngoal' or '@owngoal' field.
        recorded_team: The team key under which the goal was recorded.

    Returns:
        Actual scoring team: "localteam" or "visitorteam".
    """
    if str(_get_field(goal_event, "owngoal")).lower() == "true":
        return "visitorteam" if recorded_team == "localteam" else "localteam"
    return recorded_team


def extract_goals(
    summary: dict[str, Any],
    team_key: str,
) -> list[dict[str, Any]]:
    """Extract goal events from a team's summary.

    Handles both @-prefixed (real API) and plain (legacy) field names.

    Args:
        summary: The 'summary' dict from Goalserve match data.
        team_key: "localteam" or "visitorteam".

    Returns:
        List of goal dicts with 'parsed_minute', 'scoring_team',
        'recorded_team', and boolean flags.
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
            str(_get_field(g, "minute", "0")),
            str(_get_field(g, "extra_min", "")),
        )
        goal["is_var_cancelled"] = str(_get_field(g, "var_cancelled")).lower() == "true"
        goal["is_owngoal"] = str(_get_field(g, "owngoal")).lower() == "true"
        goal["is_penalty"] = str(_get_field(g, "penalty")).lower() == "true"
        goal["name"] = _get_field(g, "name", "")
        goals.append(goal)

    return goals


def extract_red_cards(
    summary: dict[str, Any],
    team_key: str,
) -> list[dict[str, Any]]:
    """Extract red card events from a team's summary.

    Handles both @-prefixed (real API) and plain (legacy) field names.

    Args:
        summary: The 'summary' dict from Goalserve match data.
        team_key: "localteam" or "visitorteam".

    Returns:
        List of red card dicts with 'parsed_minute', 'team', and original fields.
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
            str(_get_field(r, "minute", "0")),
            str(_get_field(r, "extra_min", "")),
        )
        card["name"] = _get_field(r, "name", "")
        cards.append(card)

    return cards


def extract_stoppage_time(match_data: dict[str, Any]) -> tuple[float, float]:
    """Extract first/second half stoppage times from matchinfo.

    Handles both @-prefixed (real API) and plain (legacy) field names.

    Returns:
        (alpha_1, alpha_2) — stoppage time in minutes for each half.
    """
    time_info = match_data.get("matchinfo", {}).get("time", {})
    alpha_1 = float(
        _get_field(time_info, "addedTime_period1", 0) or 0
    )
    alpha_2 = float(
        _get_field(time_info, "addedTime_period2", 0) or 0
    )
    return alpha_1, alpha_2


def _parse_fixture_kickoff(fix: dict[str, Any]) -> datetime | None:
    """Parse kickoff datetime from a Goalserve fixture dict.

    Goalserve encodes dates in ``@date`` (DD.MM.YYYY or MM/DD/YYYY) and
    ``@time`` (HH:MM) fields, both in UTC.  Returns ``None`` if the date
    field is absent or unparseable.

    Args:
        fix: Raw Goalserve fixture dict (``@``-prefixed keys).

    Returns:
        Timezone-aware UTC datetime, or ``None`` on parse failure.
    """
    date_str: str = fix.get("@date", "") or ""
    time_str: str = fix.get("@time", "00:00") or "00:00"

    if not date_str:
        return None

    for fmt in ("%d.%m.%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            date_part = datetime.strptime(date_str.strip(), fmt)
            parts = time_str.strip().split(":")
            hour = int(parts[0]) if parts else 0
            minute = int(parts[1]) if len(parts) > 1 else 0
            return datetime(
                date_part.year, date_part.month, date_part.day,
                hour, minute, tzinfo=UTC,
            )
        except (ValueError, AttributeError):
            continue

    return None
