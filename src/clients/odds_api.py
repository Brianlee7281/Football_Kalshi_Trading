"""Odds-API client: historical odds REST + live odds WebSocket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets
import websockets.exceptions

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

_REST_BASE_URL = "https://api.the-odds-api.com"
_WS_BASE_URL = "wss://api.odds-api.io/v3/ws"

SELECTED_BOOKMAKERS = frozenset({
    "bet365",
    "betfair_exchange",
    "sbobet",
    "1xbet",
    "draftkings",
})

logger = get_logger("odds_api")


class OddsApiClient:
    """Async client for The Odds API (REST + WebSocket).

    REST endpoints:
        - /v4/historical/sports/{sport}/odds/ — historical odds snapshots
        - /v4/sports/{sport}/odds/             — current odds

    WebSocket:
        - wss://api.odds-api.io/v3/ws — live in-play odds push

    Args:
        api_key: The Odds API key.
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
    # Historical Odds (Phase 1)
    # ------------------------------------------------------------------

    async def get_historical_odds(
        self,
        sport: str,
        date: str,
        *,
        event_id: str | None = None,
        markets: str = "h2h,totals,spreads",
        regions: str = "eu,us",
    ) -> list[dict[str, Any]]:
        """Fetch historical odds snapshot for a given date.

        Args:
            sport: Sport key (e.g. "soccer_epl").
            date: ISO date string (e.g. "2024-03-15T12:00:00Z").
            event_id: Optional specific event ID to filter.
            markets: Comma-separated market types.
            regions: Comma-separated region codes.

        Returns:
            List of event dicts with bookmaker odds, filtered to selected bookmakers.
        """
        path = f"/v4/historical/sports/{sport}/odds/"
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": markets,
            "date": date,
        }
        if event_id:
            params["eventIds"] = event_id

        response = await self._http.get(path, params=params)
        data = response.json()

        return _filter_and_parse_events(data)

    # ------------------------------------------------------------------
    # Current Odds
    # ------------------------------------------------------------------

    async def get_odds(
        self,
        sport: str,
        *,
        markets: str = "h2h,totals,spreads",
        regions: str = "eu,us",
        event_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch current odds for a sport.

        Args:
            sport: Sport key (e.g. "soccer_epl").
            markets: Comma-separated market types.
            regions: Comma-separated region codes.
            event_id: Optional specific event ID.

        Returns:
            List of event dicts with bookmaker odds, filtered to selected bookmakers.
        """
        path = f"/v4/sports/{sport}/odds/"
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": markets,
        }
        if event_id:
            params["eventIds"] = event_id

        response = await self._http.get(path, params=params)
        data = response.json()

        return _filter_and_parse_events(data)

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
        """Connect to Odds-API live odds WebSocket.

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


def _filter_and_parse_events(
    data: Any,
) -> list[dict[str, Any]]:
    """Parse Odds-API response and filter bookmakers to selected set.

    Handles both historical (wrapped in 'data' key) and current (direct list) formats.
    """
    events: list[dict[str, Any]]

    if isinstance(data, dict):
        # Historical endpoint wraps in {"data": [...], "timestamp": ...}
        events = data.get("data", [])
        if not isinstance(events, list):
            events = []
    elif isinstance(data, list):
        events = data
    else:
        return []

    result: list[dict[str, Any]] = []
    for event in events:
        filtered = _filter_bookmakers(event)
        if filtered.get("bookmakers"):
            result.append(filtered)

    return result


def _filter_bookmakers(event: dict[str, Any]) -> dict[str, Any]:
    """Filter an event's bookmakers to the 5 selected ones."""
    filtered_event = dict(event)
    bookmakers = event.get("bookmakers", [])
    if not isinstance(bookmakers, list):
        bookmakers = []

    filtered_event["bookmakers"] = [
        bm for bm in bookmakers
        if bm.get("key") in SELECTED_BOOKMAKERS
    ]
    return filtered_event


def _compute_odds_delta(
    markets: list[dict[str, Any]],
    last_home_odds: float | None,
) -> tuple[float, float | None]:
    """Compute home-odds change rate from ML market.

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
# Odds feature extraction (used by Step 1.3)
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


def build_odds_features(bookmakers: list[dict[str, Any]]) -> dict[str, float]:
    """Extract odds features from Odds-API bookmaker list.

    Betfair Exchange is the primary baseline. If unavailable, falls back to
    market average of the remaining bookmakers.

    Args:
        bookmakers: List of bookmaker dicts from Odds-API response.

    Returns:
        Feature dict with exchange_home_prob, market_avg_home_prob, etc.
    """
    all_probs: list[tuple[float, float, float]] = []
    exchange_prob: tuple[float, float, float] | None = None

    for bm in bookmakers:
        h2h = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
        if not h2h:
            continue

        outcomes = {o["name"]: float(o["price"]) for o in h2h.get("outcomes", [])}

        # Odds-API uses team names as keys; handle both formats
        keys = list(outcomes.keys())
        h = outcomes.get("Home Team", outcomes.get(keys[0], 0.0) if keys else 0.0)
        d = outcomes.get("Draw", 0.0)
        a = outcomes.get("Away Team", outcomes.get(keys[-1], 0.0) if keys else 0.0)

        if not (h > 0 and d > 0 and a > 0):
            continue

        prob = remove_overround(h, d, a)
        all_probs.append(prob)

        if bm.get("key") == "betfair_exchange":
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
    bookmakers: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Extract bet365 implied probabilities for Phase 3/4 alignment checks.

    Args:
        bookmakers: List of bookmaker dicts from Odds-API response.

    Returns:
        Dict with home_win, draw, away_win probabilities, or None if bet365 absent.
    """
    for bm in bookmakers:
        if bm.get("key") != "bet365":
            continue
        h2h = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
        if not h2h:
            return None
        outcomes = {o["name"]: float(o["price"]) for o in h2h.get("outcomes", [])}
        keys = list(outcomes.keys())
        h = outcomes.get("Home Team", outcomes.get(keys[0], 0.0) if keys else 0.0)
        d = outcomes.get("Draw", 0.0)
        a = outcomes.get("Away Team", outcomes.get(keys[-1], 0.0) if keys else 0.0)
        if not (h > 0 and d > 0 and a > 0):
            return None
        prob = remove_overround(h, d, a)
        return {
            "home_win": prob[0],
            "draw": prob[1],
            "away_win": prob[2],
        }
    return None
