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


def _gs(d: dict[str, Any], key: str, default: Any = "") -> Any:
    """Get a value from a Goalserve dict, trying both ``@key`` and ``key``.

    Goalserve XML-to-JSON conversion prefixes attribute keys with ``@``.
    Some endpoints use ``@goals``, others use ``goals``.  This helper
    checks both variants so the code works regardless of format.
    """
    return d.get(f"@{key}", d.get(key, default))

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
        self._last_period: str | None = None  # classified period (e.g. "1st Half")
        self._last_raw_status: str | None = None  # raw status for logging
        self._consecutive_failures = 0

    async def connect(self, match_id: str) -> None:
        self._running = True
        logger.info("goalserve_source_ready", match_id=match_id)

    async def disconnect(self) -> None:
        self._running = False

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        """Poll Goalserve every poll_interval seconds, yield diff events."""
        _poll_count = 0
        async with httpx.AsyncClient(timeout=10.0):
            while self._running:
                _poll_count += 1
                try:
                    data = await self._client.get_live_score(self._match_id)
                    if data:
                        status = str(_gs(data, "status", "") or "")
                        localteam = data.get("localteam", {})
                        visitorteam = data.get("visitorteam", {})
                        if isinstance(localteam, str):
                            localteam = {}
                        if isinstance(visitorteam, str):
                            visitorteam = {}
                        # Log every 10th poll or on raw status change
                        if _poll_count % 10 == 1 or status != self._last_raw_status:
                            logger.info(
                                "live_score_poll",
                                match_id=self._match_id,
                                poll_count=_poll_count,
                                status=status,
                                score_home=_gs(localteam, "goals", "?"),
                                score_away=_gs(visitorteam, "goals", "?"),
                                raw_keys=list(data.keys())[:15] if _poll_count <= 3 else None,
                            )
                        async for event in self._diff(data):
                            yield event
                        self._last_raw_status = status
                        self._consecutive_failures = 0
                    else:
                        if _poll_count <= 3 or _poll_count % 20 == 0:
                            # Log all available IDs on first few polls to
                            # help diagnose match_id mapping issues.
                            try:
                                all_live = await self._client.get_live_scores()
                                avail_ids = [
                                    f"{m.get('@id','?')}/{m.get('@fix_id','?')}/{m.get('@static_id','?')}"
                                    for m in all_live[:15]
                                ]
                            except Exception:  # noqa: BLE001
                                avail_ids = ["<fetch_failed>"]
                            logger.warning(
                                "live_score_no_data",
                                match_id=self._match_id,
                                poll_count=_poll_count,
                                available_ids=avail_ids,
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
        localteam = match.get("localteam", {})
        visitorteam = match.get("visitorteam", {})
        if isinstance(localteam, str):
            localteam = {}
        if isinstance(visitorteam, str):
            visitorteam = {}
        home_goals = _safe_int(_gs(localteam, "goals", 0))
        away_goals = _safe_int(_gs(visitorteam, "goals", 0))

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
        live_stats = _gs(match, "live_stats", {})
        if not isinstance(live_stats, dict):
            live_stats = {}
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
        # Only emit when the *classified* period changes, not on every
        # minute tick (e.g. "33" → "34" are both "1st Half" — no event).
        status = str(_gs(match, "status", "") or "")
        if status:
            period_event = _classify_status(status)
            current_minute = _parse_minute(status)

            if period_event and period_event != self._last_period:
                if period_event == "Finished":
                    yield NormalizedEvent(
                        type="match_finished",
                        source="live_score",
                        confidence="confirmed",
                        minute=current_minute,
                        timestamp=time.time(),
                    )
                else:
                    yield NormalizedEvent(
                        type="period_change",
                        source="live_score",
                        confidence="confirmed",
                        period=period_event,
                        minute=current_minute,
                        timestamp=time.time(),
                    )
                self._last_period = period_event

        # ── Stoppage time entry (minute > 45 in first half, etc.) ─────────
        timer = _safe_float(_gs(match, "timer", ""))
        if timer is not None:
            period = str(_gs(match, "period", "") or "")
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


def _classify_status(status: str) -> str | None:
    """Map a Goalserve status string to a canonical period name.

    Goalserve uses several formats:
      - Text: ``"1st Half"``, ``"HT"``, ``"2nd Half"``, ``"Finished"``
      - Numeric minute: ``"33"`` (minute 33 of 1st half), ``"67"`` (2nd half)
      - ``"45+2"`` (stoppage time format)

    Returns:
        One of ``"1st Half"``, ``"Halftime"``, ``"2nd Half"``, ``"Finished"``
        or None if the status is unrecognised.
    """
    if status in ("1st Half", "1H", "First Half"):
        return "1st Half"
    if status in ("HT", "Half Time", "Paused"):
        return "Halftime"
    if status in ("2nd Half", "2H", "Second Half"):
        return "2nd Half"
    if status in ("Finished", "FT", "Full Time", "AET", "Pen."):
        return "Finished"

    # Numeric status = match minute (e.g. "33", "67", "45+2")
    # Strip stoppage suffix for parsing
    numeric_part = status.split("+")[0].strip()
    try:
        minute = int(numeric_part)
        if 0 < minute <= 45:
            return "1st Half"
        if minute > 45:
            return "2nd Half"
    except (ValueError, TypeError):
        pass

    return None


def _parse_minute(status: str) -> float | None:
    """Extract the match minute from a Goalserve status string.

    Examples: ``"33"`` → 33.0, ``"45+2"`` → 47.0, ``"HT"`` → None.
    """
    parts = status.split("+")
    try:
        base = int(parts[0].strip())
        added = int(parts[1].strip()) if len(parts) > 1 else 0
        return float(base + added)
    except (ValueError, TypeError):
        return None


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
    """
    import asyncio
    import os

    from src.clients.goalserve import GoalserveClient
    from src.engine.event_handlers import dispatch_event
    from src.engine.model import FINISHED

    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        logger.error("live_score_poller_no_api_key", match_id=getattr(model, "match_id", ""))
        return

    client = GoalserveClient(api_key=api_key)
    match_id = str(getattr(model, "match_id", ""))
    source = GoalserveLiveScoreSource(client, match_id, poll_interval=3.0)
    await source.connect(match_id)

    logger.info("live_score_poller_started", match_id=match_id)

    # One-time diagnostic: log all live match IDs so we can debug
    # match_id field mapping issues.
    try:
        all_live = await client.get_live_scores()
        sample_ids = [
            {
                "@id": m.get("@id", ""),
                "@fix_id": m.get("@fix_id", ""),
                "@static_id": m.get("@static_id", ""),
                "status": m.get("status", ""),
            }
            for m in all_live[:20]
        ]
        logger.info(
            "live_score_diag",
            searching_for=match_id,
            total_live=len(all_live),
            sample_ids=sample_ids,
        )
    except Exception:  # noqa: BLE001
        pass

    async for event in source.listen():
        if getattr(model, "engine_phase", None) == FINISHED:
            break
        logger.info(
            "live_score_event",
            match_id=match_id,
            event_type=event.type,
            period=getattr(event, "period", None),
            score=getattr(event, "score", None),
        )
        dispatch_event(model, event)

    await source.disconnect()
    logger.info("live_score_poller_stopped", match_id=match_id)


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
