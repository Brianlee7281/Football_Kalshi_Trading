"""Phase 3 event sources — abstraction layer over Goalserve REST + Odds-API WS.

Two concrete EventSource implementations:

1. OddsApiLiveOddsSource — WebSocket PUSH (<1s):
   - Detects abrupt odds movements → odds_spike → ob_freeze (early warning)
   - Does NOT provide score, period, or event details

2. GoalserveLiveScoreSource — REST polling (3s cadence):
   - Authoritative score, red cards, period, VAR cancellation
   - Multi-goal same-poll fix: yields one NormalizedEvent per goal using
     running_home / running_away intermediate score tracking

Multi-goal fix (docs/phase3.md):
  If 2 goals are scored in the same 3s polling interval, we yield them
  one at a time with INTERMEDIATE scores so each goal's ΔS transition
  is committed step-by-step. Without this, handle_confirmed_goal would
  receive the final score and miss the intermediate ΔS for μ precompute.

Reference: docs/phase3.md §Source 1, §Source 2
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.clients.goalserve import GoalserveClient
from src.clients.odds_api import OddsApiClient
from src.common.logging import get_logger
from src.common.types import NormalizedEvent

logger = get_logger("event_sources")

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EventSource(ABC):
    """Abstract event source — decouples the engine from data providers."""

    @abstractmethod
    async def connect(self, match_id: str) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    def listen(self) -> AsyncIterator[NormalizedEvent]:
        """Yield NormalizedEvents as they arrive."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection gracefully."""
        ...


# ---------------------------------------------------------------------------
# Source 1: Odds-API Live Odds WebSocket (early warning, <1s)
# ---------------------------------------------------------------------------


class OddsApiLiveOddsSource(EventSource):
    """Odds-API WebSocket — early warning via abrupt odds movement detection.

    Connects to the Odds-API.io live odds WebSocket and yields odds_spike
    events when the home ML odds change by more than ``odds_threshold_pct``.

    Does NOT provide score details — only signals that something happened.
    The engine sets ob_freeze on spike and waits for Goalserve to confirm.

    Reference: docs/phase3.md §Source 1
    """

    def __init__(
        self,
        client: OddsApiClient,
        odds_event_id: str,
        *,
        odds_threshold_pct: float = 0.10,
    ) -> None:
        self._client = client
        self._event_id = odds_event_id
        self._odds_threshold = odds_threshold_pct

    async def connect(self, match_id: str) -> None:
        # Connection is handled inside listen() via the client's context manager
        logger.info("odds_api_source_ready", match_id=match_id, event_id=self._event_id)

    async def disconnect(self) -> None:
        pass  # client manages its own connection lifecycle

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        """Yield odds_spike and match_removed events from the WebSocket."""
        event_ids: set[str] = {self._event_id}

        async for msg in self._client.connect_live_ws(
            markets="ML,Totals",
            event_ids=event_ids,
            odds_threshold_pct=self._odds_threshold,
        ):
            msg_type = msg.get("type", "")

            if msg_type == "deleted":
                yield NormalizedEvent(
                    type="match_removed",
                    source="live_odds",
                    confidence="preliminary",
                    timestamp=time.time(),
                )
                continue

            if msg_type not in ("updated", "created"):
                continue

            if msg.get("is_spike"):
                yield NormalizedEvent(
                    type="odds_spike",
                    source="live_odds",
                    confidence="preliminary",
                    delta=float(msg.get("odds_delta", 0.0)),
                    timestamp=time.time(),
                )


# ---------------------------------------------------------------------------
# Source 2: Goalserve Live Score REST (authoritative, 3s poll)
# ---------------------------------------------------------------------------

_MAX_CONSECUTIVE_FAILURES = 5


class GoalserveLiveScoreSource(EventSource):
    """Goalserve REST poller — authoritative score + event confirmation.

    Polls every ``poll_interval`` seconds and computes a diff against the
    previous poll to yield individual NormalizedEvents.

    Multi-goal same-poll fix:
        If N > 1 goals are detected in a single 3s interval, yields N
        separate goal_confirmed events with INTERMEDIATE scores using
        running_home/running_away tracking so each ΔS step is committed
        individually by the event handler.

    Reference: docs/phase3.md §Source 2
    """

    def __init__(
        self,
        client: GoalserveClient,
        match_id: str,
        *,
        poll_interval: float = 3.0,
    ) -> None:
        self._client = client
        self._match_id = match_id
        self._poll_interval = poll_interval
        self._running = True

        # Diff tracking
        self._last_score: dict[str, int] = {"home": 0, "away": 0}
        self._last_red_cards: dict[str, int] = {"home": 0, "away": 0}
        self._last_period: str | None = None
        self._consecutive_failures = 0

    async def connect(self, match_id: str) -> None:
        self._running = True
        logger.info("goalserve_source_ready", match_id=match_id)

    async def disconnect(self) -> None:
        self._running = False

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        """Poll Goalserve every poll_interval seconds, yield diff events."""
        async with httpx.AsyncClient(timeout=10.0):
            while self._running:
                try:
                    data = await self._client.get_live_score(self._match_id)
                    if data:
                        async for event in self._diff(data):
                            yield event
                        self._consecutive_failures = 0
                    else:
                        logger.debug(
                            "live_score_no_data",
                            match_id=self._match_id,
                        )

                except (httpx.HTTPError, httpx.TimeoutException) as exc:
                    self._consecutive_failures += 1
                    logger.error(
                        "live_score_poll_failed",
                        match_id=self._match_id,
                        error=str(exc),
                        consecutive=self._consecutive_failures,
                    )
                    if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        yield NormalizedEvent(
                            type="source_failure",
                            source="live_score",
                            confidence="confirmed",
                            timestamp=time.time(),
                        )

                await _sleep_poll(self._poll_interval)

    async def _diff(self, match: dict[str, Any]) -> AsyncIterator[NormalizedEvent]:
        """Compute diff between current and last poll, yield changed events."""
        # ── Score change (multi-goal same-poll fix) ───────────────────────
        home_goals = _safe_int(match.get("localteam", {}).get("goals", 0))
        away_goals = _safe_int(match.get("visitorteam", {}).get("goals", 0))

        running_home = self._last_score["home"]
        running_away = self._last_score["away"]

        # Home goals scored this interval — yield one event per goal
        if home_goals > running_home:
            for _ in range(home_goals - running_home):
                running_home += 1
                yield NormalizedEvent(
                    type="goal_confirmed",
                    source="live_score",
                    confidence="confirmed",
                    score=(running_home, running_away),  # intermediate, not final
                    team="localteam",
                    var_cancelled=False,
                    timestamp=time.time(),
                )

        # Away goals scored this interval — yield one event per goal
        if away_goals > running_away:
            for _ in range(away_goals - running_away):
                running_away += 1
                yield NormalizedEvent(
                    type="goal_confirmed",
                    source="live_score",
                    confidence="confirmed",
                    score=(running_home, running_away),  # intermediate, not final
                    team="visitorteam",
                    var_cancelled=False,
                    timestamp=time.time(),
                )

        # Score rollback (VAR cancellation): score decreased
        if home_goals < self._last_score["home"] or away_goals < self._last_score["away"]:
            yield NormalizedEvent(
                type="score_rollback",
                source="live_score",
                confidence="confirmed",
                score=(home_goals, away_goals),
                var_cancelled=True,
                timestamp=time.time(),
            )

        self._last_score = {"home": home_goals, "away": away_goals}

        # ── Red card detection ────────────────────────────────────────────
        live_stats = match.get("live_stats", {})
        home_reds = _extract_red_cards(live_stats, "home")
        away_reds = _extract_red_cards(live_stats, "away")

        if home_reds > self._last_red_cards["home"]:
            for _ in range(home_reds - self._last_red_cards["home"]):
                yield NormalizedEvent(
                    type="red_card",
                    source="live_score",
                    confidence="confirmed",
                    team="localteam",
                    timestamp=time.time(),
                )

        if away_reds > self._last_red_cards["away"]:
            for _ in range(away_reds - self._last_red_cards["away"]):
                yield NormalizedEvent(
                    type="red_card",
                    source="live_score",
                    confidence="confirmed",
                    team="visitorteam",
                    timestamp=time.time(),
                )

        self._last_red_cards = {"home": home_reds, "away": away_reds}

        # ── Period change ─────────────────────────────────────────────────
        status = str(match.get("status", "") or "")
        if status and status != self._last_period:
            if status in ("HT", "Half Time", "Paused"):
                yield NormalizedEvent(
                    type="period_change",
                    source="live_score",
                    confidence="confirmed",
                    period="Halftime",
                    timestamp=time.time(),
                )
            elif status in ("2nd Half", "2H", "Second Half"):
                yield NormalizedEvent(
                    type="period_change",
                    source="live_score",
                    confidence="confirmed",
                    period="2nd Half",
                    timestamp=time.time(),
                )
            elif status in ("Finished", "FT", "Full Time"):
                yield NormalizedEvent(
                    type="match_finished",
                    source="live_score",
                    confidence="confirmed",
                    timestamp=time.time(),
                )
            self._last_period = status

        # ── Stoppage time entry (minute > 45 in first half, etc.) ─────────
        timer = _safe_float(match.get("timer", ""))
        if timer is not None:
            period = str(match.get("period", "") or "")
            if (period in ("1st Half", "1H") and timer > 45.0) or (
                period in ("2nd Half", "2H") and timer > 90.0
            ):
                yield NormalizedEvent(
                    type="stoppage_entered",
                    source="live_score",
                    confidence="confirmed",
                    minute=timer,
                    period=period,
                    timestamp=time.time(),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(val: object) -> int:
    """Safely convert a value to int, defaulting to 0."""
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val: object) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return None


def _extract_red_cards(live_stats: dict[str, Any], side: str) -> int:
    """Extract red card count from Goalserve live_stats dict."""
    try:
        val = live_stats.get("value", "")
        if isinstance(val, list):
            for stat in val:
                if isinstance(stat, dict) and stat.get("@type") == "IRedCard":
                    return _safe_int(stat.get(f"@{side}", 0))
    except (TypeError, AttributeError):
        pass
    return 0


async def _sleep_poll(interval: float) -> None:
    """Sleep for poll interval — extracted for testability."""
    import asyncio

    await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Top-level coroutines — wired into asyncio.gather by match_engine/main.py
# ---------------------------------------------------------------------------


async def live_odds_listener(model: Any) -> None:
    """Phase 3 Odds-API WebSocket listener coroutine.

    Connects to the Odds-API WebSocket stream, normalises incoming odds
    updates into NormalizedEvent objects, and dispatches them via
    dispatch_event(model, event).

    Stub for Sprint 5 — full implementation in Sprint 6.
    """
    import asyncio

    from src.engine.model import FINISHED

    while getattr(model, "engine_phase", None) != FINISHED:
        await asyncio.sleep(1.0)


async def live_score_poller(model: Any) -> None:
    """Phase 3 Goalserve REST polling coroutine.

    Polls Goalserve every 3 seconds for score updates, red cards, period
    changes, and VAR cancellations.  Dispatches confirmed events via
    dispatch_event(model, event).

    Stub for Sprint 5 — full implementation in Sprint 6.
    """
    import asyncio

    from src.engine.model import FINISHED

    while getattr(model, "engine_phase", None) != FINISHED:
        await asyncio.sleep(3.0)


async def order_book_sync_loop(model: Any) -> None:
    """Phase 4 Kalshi WebSocket order-book sync coroutine.

    Subscribes to Kalshi order-book WebSocket streams for all active tickers
    and feeds updates into the per-ticker OrderBookSync instances stored in
    model.ob_syncs.

    Stub for Sprint 5 — full implementation in Sprint 6.
    """
    import asyncio

    from src.engine.model import FINISHED

    while getattr(model, "engine_phase", None) != FINISHED:
        await asyncio.sleep(1.0)
