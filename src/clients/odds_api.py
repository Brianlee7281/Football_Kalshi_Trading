"""Odds-API.io client: REST odds + live odds WebSocket.

API reference: docs/api_reference_odds_api.md
Base URL: https://api.odds-api.io/v3
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets
import websockets.exceptions

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

_REST_BASE_URL = "https://api.odds-api.io/v3"
_WS_BASE_URL = "wss://api.odds-api.io/v3/ws"

# Bookmaker display names as used by the odds-api.io API.
# These are the 5 bookmakers our system tracks for odds features.
SELECTED_BOOKMAKERS = frozenset({
    "Bet365",
    "Betfair Exchange",
    "Sbobet",
    "1xbet",
    "DraftKings",
})

logger = get_logger("odds_api")


class OddsApiClient:
    """Async client for Odds-API.io (REST + WebSocket).

    REST endpoints:
        - /events                — list events (filter by sport/league/status)
        - /odds                  — odds for a single event
        - /odds/multi            — odds for multiple events (1 API call)
        - /odds/movements        — historical odds movements
        - /value-bets            — pre-computed value bets

    WebSocket:
        - wss://api.odds-api.io/v3/ws — live in-play odds push

    Args:
        api_key: Odds-API.io API key.
        timeout: REST request timeout in seconds.
    """

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._http = BaseClient(
            _REST_BASE_URL,
            timeout=timeout,
            max_retries=3,
            backoff_base=1.0,
        )
        self._logger = get_logger("odds_api")

    async def close(self) -> None:
        await self._http.close()

    async def __aenter__(self) -> OddsApiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Events (discover matches)
    # ------------------------------------------------------------------

    async def get_events(
        self,
        sport: str,
        *,
        league: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch events for a sport.

        API ref: GET /events?apiKey=...&sport=...

        Args:
            sport: Sport slug (e.g. "football").
            league: Optional league slug filter (e.g. "england-premier-league").
            status: Optional status filter (e.g. "pending", "live", "settled").

        Returns:
            List of event dicts with id, home, away, date, status, etc.
        """
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "sport": sport,
        }
        if league:
            params["league"] = league
        if status:
            params["status"] = status

        response = await self._http.get("/events", params=params)
        data = response.json()
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Odds (single event)
    # ------------------------------------------------------------------

    async def get_odds(
        self,
        event_id: int | str,
        bookmakers: str,
    ) -> dict[str, Any]:
        """Fetch odds for a single event from specified bookmakers.

        API ref: GET /odds?apiKey=...&eventId=...&bookmakers=...

        Args:
            event_id: Event ID.
            bookmakers: Comma-separated bookmaker names (max 30).

        Returns:
            Event odds dict with ``bookmakers`` object keyed by name.
        """
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "eventId": str(event_id),
            "bookmakers": bookmakers,
        }

        response = await self._http.get("/odds", params=params)
        data = response.json()
        return _filter_bookmakers(data) if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Odds Multi (batch, counts as 1 API call)
    # ------------------------------------------------------------------

    async def get_odds_multi(
        self,
        event_ids: list[int | str],
        bookmakers: str,
    ) -> list[dict[str, Any]]:
        """Fetch odds for multiple events in a single API call (max 10).

        API ref: GET /odds/multi?apiKey=...&eventIds=...&bookmakers=...

        Args:
            event_ids: List of event IDs (max 10).
            bookmakers: Comma-separated bookmaker names (max 30).

        Returns:
            List of event odds dicts.
        """
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "eventIds": ",".join(str(eid) for eid in event_ids),
            "bookmakers": bookmakers,
        }

        response = await self._http.get("/odds/multi", params=params)
        data = response.json()
        if not isinstance(data, list):
            return []
        return [_filter_bookmakers(ev) for ev in data if isinstance(ev, dict)]

    # ------------------------------------------------------------------
    # Odds Movements (historical — Phase 1)
    # ------------------------------------------------------------------

    async def get_odds_movements(
        self,
        event_id: int | str,
        bookmaker: str,
        market: str = "ML",
    ) -> dict[str, Any]:
        """Fetch historical odds movements for an event/bookmaker/market.

        API ref: GET /odds/movements?apiKey=...&eventId=...&bookmaker=...&market=...

        Args:
            event_id: Event ID.
            bookmaker: Bookmaker name (e.g. "Bet365").
            market: Market name (e.g. "ML", "Totals").

        Returns:
            Dict with ``opening``, ``latest``, and ``movements`` fields.
        """
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "eventId": str(event_id),
            "bookmaker": bookmaker,
            "market": market,
        }

        response = await self._http.get("/odds/movements", params=params)
        data = response.json()
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Live Odds WebSocket (Phase 3)
    # ------------------------------------------------------------------

    async def connect_live_ws(
        self,
        *,
        markets: str = "ML,Totals",
        event_ids: set[str] | None = None,
        odds_threshold_pct: float = 0.10,
    ) -> AsyncIterator[dict[str, Any]]:
        """Connect to Odds-API.io live odds WebSocket.

        Yields parsed messages filtered to tracked event IDs.

        Args:
            markets: Comma-separated market list for subscription.
            event_ids: Optional set of event IDs to track. None = all.
            odds_threshold_pct: Threshold for odds spike detection.

        Yields:
            Parsed WebSocket message dicts with added 'odds_delta' field
            when an abrupt odds movement is detected.
        """
        url = f"{_WS_BASE_URL}?apiKey={self._api_key}&markets={markets}&status=live"
        last_home_odds: float | None = None

        try:
            async for ws in websockets.connect(url):
                try:
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type", "")

                        if msg_type == "welcome":
                            self._logger.info(
                                "ws_connected",
                                bookmakers=msg.get("bookmakers"),
                            )
                            yield msg
                            continue

                        if msg_type == "deleted":
                            yield msg
                            continue

                        if msg_type not in ("updated", "created"):
                            continue

                        # Filter to tracked events
                        event_id = str(msg.get("id", ""))
                        if event_ids and event_id not in event_ids:
                            continue

                        # Compute odds delta for spike detection
                        odds_delta, last_home_odds = _compute_odds_delta(
                            msg.get("markets", []),
                            last_home_odds,
                        )
                        msg["odds_delta"] = odds_delta
                        msg["is_spike"] = odds_delta >= odds_threshold_pct

                        yield msg

                except websockets.exceptions.ConnectionClosed:
                    self._logger.warning("ws_connection_closed", url=url)
                    await asyncio.sleep(1.0)
                    continue

        except Exception:
            self._logger.exception("ws_fatal_error")
            raise


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _filter_bookmakers(event: dict[str, Any]) -> dict[str, Any]:
    """Filter an event's bookmakers object to the selected set.

    Odds-API.io returns bookmakers as an object keyed by name:
    ``{"Bet365": [...markets...], "Pinnacle": [...markets...]}``

    We keep only bookmakers in SELECTED_BOOKMAKERS.
    """
    filtered_event = dict(event)
    bookmakers = event.get("bookmakers", {})
    if not isinstance(bookmakers, dict):
        filtered_event["bookmakers"] = {}
        return filtered_event

    filtered_event["bookmakers"] = {
        name: markets
        for name, markets in bookmakers.items()
        if name in SELECTED_BOOKMAKERS
    }
    return filtered_event


def _compute_odds_delta(
    markets: list[dict[str, Any]],
    last_home_odds: float | None,
) -> tuple[float, float | None]:
    """Compute home-odds change rate from ML market.

    WebSocket market format: ``{"name": "ML", "odds": [{"home": "1.85", ...}]}``

    Returns:
        (delta, updated_last_home_odds)
    """
    try:
        for market in markets:
            if market.get("name") == "ML":
                odds = market["odds"][0]
                current = float(odds["home"])
                if last_home_odds is not None and last_home_odds > 0:
                    delta = abs(current - last_home_odds) / last_home_odds
                    return delta, current
                return 0.0, current
    except (KeyError, ValueError, IndexError):
        pass
    return 0.0, last_home_odds


# ---------------------------------------------------------------------------
# Odds feature extraction (used by Step 1.3 and Phase 3/4)
# ---------------------------------------------------------------------------


def remove_overround(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Remove bookmaker overround to get true implied probabilities.

    Args:
        h: Home decimal odds.
        d: Draw decimal odds.
        a: Away decimal odds.

    Returns:
        (home_prob, draw_prob, away_prob) summing to 1.0.
    """
    total = 1.0 / h + 1.0 / d + 1.0 / a
    return (1.0 / h) / total, (1.0 / d) / total, (1.0 / a) / total


def build_odds_features(
    bookmakers: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Extract odds features from Odds-API.io bookmakers object.

    Betfair Exchange is the primary baseline. If unavailable, falls back to
    market average of the remaining bookmakers.

    Args:
        bookmakers: Dict mapping bookmaker name to list of market dicts.
                    Format: ``{"Bet365": [{"name": "ML", "odds": [...]}], ...}``

    Returns:
        Feature dict with exchange_home_prob, market_avg_home_prob, etc.
    """
    all_probs: list[tuple[float, float, float]] = []
    exchange_prob: tuple[float, float, float] | None = None

    for bm_name, markets in bookmakers.items():
        h, d, a = _extract_ml_odds(markets)
        if not (h > 0 and d > 0 and a > 0):
            continue

        prob = remove_overround(h, d, a)
        all_probs.append(prob)

        if bm_name == "Betfair Exchange":
            exchange_prob = prob

    if not all_probs:
        return {
            "exchange_home_prob": 0.0,
            "exchange_draw_prob": 0.0,
            "exchange_away_prob": 0.0,
            "market_avg_home_prob": 0.0,
            "market_avg_draw_prob": 0.0,
            "bookmaker_odds_std": 0.0,
        }

    # Fallback: market average if Betfair Exchange unavailable
    if exchange_prob is None:
        avg_h = sum(p[0] for p in all_probs) / len(all_probs)
        avg_d = sum(p[1] for p in all_probs) / len(all_probs)
        avg_a = sum(p[2] for p in all_probs) / len(all_probs)
        exchange_prob = (avg_h, avg_d, avg_a)

    home_probs = [p[0] for p in all_probs]
    draw_probs = [p[1] for p in all_probs]
    n = len(home_probs)
    mean_home = sum(home_probs) / n
    mean_draw = sum(draw_probs) / n
    variance = sum((x - mean_home) ** 2 for x in home_probs) / n
    std_home = variance ** 0.5

    return {
        "exchange_home_prob": exchange_prob[0],
        "exchange_draw_prob": exchange_prob[1],
        "exchange_away_prob": exchange_prob[2],
        "market_avg_home_prob": mean_home,
        "market_avg_draw_prob": mean_draw,
        "bookmaker_odds_std": std_home,
    }


def extract_bet365_implied_probs(
    bookmakers: dict[str, list[dict[str, Any]]],
) -> dict[str, float] | None:
    """Extract Bet365 implied probabilities for Phase 3/4 alignment checks.

    Args:
        bookmakers: Dict mapping bookmaker name to list of market dicts.

    Returns:
        Dict with home_win, draw, away_win probabilities, or None if Bet365 absent.
    """
    markets = bookmakers.get("Bet365")
    if not markets:
        return None
    h, d, a = _extract_ml_odds(markets)
    if not (h > 0 and d > 0 and a > 0):
        return None
    prob = remove_overround(h, d, a)
    return {
        "home_win": prob[0],
        "draw": prob[1],
        "away_win": prob[2],
    }


def _extract_ml_odds(markets: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Extract home/draw/away odds from an ML market.

    Odds-API.io format: ``[{"name": "ML", "odds": [{"home": "2.10", "draw": "3.40", "away": "3.20"}]}]``

    Returns:
        (home, draw, away) as floats, or (0, 0, 0) if ML not found.
    """
    for market in markets:
        if market.get("name") != "ML":
            continue
        odds_list = market.get("odds", [])
        if not odds_list:
            return 0.0, 0.0, 0.0
        odds = odds_list[0]
        try:
            return (
                float(odds.get("home", 0)),
                float(odds.get("draw", 0)),
                float(odds.get("away", 0)),
            )
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0
    return 0.0, 0.0, 0.0
