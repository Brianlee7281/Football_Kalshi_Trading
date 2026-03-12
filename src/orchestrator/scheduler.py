"""Match discovery scheduler and trigger executor.

Component 1 of the orchestration layer.

MatchDiscovery:
  Runs every 6 hours.  Scans Goalserve fixtures for all 8 tradable leagues,
  cross-checks Kalshi for active soccer events, computes Phase 2/3 trigger
  times, and writes new MatchSchedule rows to the DB.

TriggerExecutor:
  Runs every 30 seconds.  Checks match_schedule for rows whose trigger time
  has arrived and fires the appropriate orchestrator callback.

Reference: docs/orchestration.md Component 1: Scheduler
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from src.clients.goalserve import GoalserveClient
from src.clients.kalshi import KalshiClient
from src.common.logging import get_logger
from src.common.types import MatchSchedule

logger = get_logger("scheduler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Forward window for fixture scanning.
DISCOVERY_HOURS_AHEAD: float = 48.0

#: phase2_trigger = kickoff − 65 min (lineup check + buffer)
PHASE2_OFFSET_MINUTES: int = 65

#: phase3_trigger = kickoff − 2 min (final sanity check)
PHASE3_OFFSET_MINUTES: int = 2

#: TriggerExecutor polling interval
TRIGGER_INTERVAL_S: float = 30.0

#: 8 tradable leagues (see docs/orchestration.md TRADABLE_LEAGUES)
TRADABLE_LEAGUES: dict[int, str] = {
    # Europe (Tier 1)
    1204: "EPL",
    1399: "La Liga",
    1229: "Bundesliga",
    1269: "Serie A",
    1221: "Ligue 1",
    # Americas (Tier 2) — IDs verified via /soccerfixtures/data/mapping
    1440: "MLS",
    1141: "Brasileirão",
    1081: "Liga Argentina",
}


# ---------------------------------------------------------------------------
# MatchDiscovery
# ---------------------------------------------------------------------------


class MatchDiscovery:
    """Discover tradable matches and persist them to the match_schedule table.

    Scans Goalserve fixtures for all 8 tradable leagues, cross-checks Kalshi
    for active soccer events (by team name + date), computes trigger times,
    and upserts new ``SCHEDULED`` rows.  Already-known matches are skipped.

    Args:
        goalserve: Initialized GoalserveClient.
        kalshi: Initialized KalshiClient.
        db_pool: asyncpg connection pool.
        trading_mode: ``"paper"`` or ``"live"`` (injected into new rows).
        param_version: Phase 1 parameter version pinned for new matches.
    """

    def __init__(
        self,
        goalserve: GoalserveClient,
        kalshi: KalshiClient,
        db_pool: asyncpg.Pool,
        *,
        trading_mode: str = "paper",
        param_version: int = 1,
    ) -> None:
        self.goalserve = goalserve
        self.kalshi = kalshi
        self.db_pool = db_pool
        self.trading_mode = trading_mode
        self.param_version = param_version

    async def discover(self) -> list[MatchSchedule]:
        """Run one discovery cycle.

        Steps:
          1. Fetch Goalserve fixtures for all TRADABLE_LEAGUES (48h window).
          2. Fetch active Kalshi soccer events.
          3. Match fixtures to Kalshi events by date + team name.
          4. Compute phase2_trigger / phase3_trigger.
          5. Upsert new SCHEDULED rows to match_schedule (ON CONFLICT skip).

        Returns:
            All MatchSchedule objects produced this cycle (including skipped
            duplicates that were already in the DB).
        """
        # ── Step 1: Goalserve fixtures ────────────────────────────────────────
        fixtures = await self.goalserve.get_upcoming_fixtures(
            list(TRADABLE_LEAGUES.keys()),
            hours_ahead=DISCOVERY_HOURS_AHEAD,
        )
        logger.info("fixtures_fetched", count=len(fixtures))

        # ── Step 2: Kalshi active soccer events ───────────────────────────────
        kalshi_events = await self.kalshi.get_active_soccer_events()
        logger.info("kalshi_events_fetched", count=len(kalshi_events))

        # ── Step 3: Match fixtures → Kalshi events ────────────────────────────
        matched = _match_fixtures_to_markets(fixtures, kalshi_events)
        logger.info("fixtures_matched", count=len(matched))

        # ── Step 4 + 5: Build MatchSchedule and upsert ───────────────────────
        schedules: list[MatchSchedule] = []
        for m in matched:
            kickoff: datetime = m["kickoff_utc"]
            schedule = MatchSchedule(
                match_id=m["match_id"],
                league_id=m["league_id"],
                kickoff_utc=kickoff,
                phase2_trigger=kickoff - timedelta(minutes=PHASE2_OFFSET_MINUTES),
                phase3_trigger=kickoff - timedelta(minutes=PHASE3_OFFSET_MINUTES),
                kalshi_tickers=m["kalshi_tickers"],
                odds_api_event_id=m.get("odds_api_event_id"),
                trading_mode=self.trading_mode,
                param_version=self.param_version,
            )
            schedules.append(schedule)

        new_count = await _upsert_match_schedules(self.db_pool, schedules)
        logger.info(
            "discovery_complete",
            new_schedules=new_count,
            total_matched=len(schedules),
        )
        return schedules


# ---------------------------------------------------------------------------
# TriggerExecutor
# ---------------------------------------------------------------------------


class TriggerExecutor:
    """Fires Phase 2 and Phase 3 triggers as their times arrive.

    Polls ``match_schedule`` every ``TRIGGER_INTERVAL_S`` (30 seconds).
    Delegates to the orchestrator via ``start_match_lifecycle`` (Phase 2)
    and ``start_live_engine`` (Phase 3).

    The orchestrator argument is typed as ``Any`` to avoid a circular
    import; it is expected to implement both async methods.

    Args:
        db_pool: asyncpg connection pool.
        orchestrator: Object with ``start_match_lifecycle`` and
                      ``start_live_engine`` async methods.
    """

    def __init__(self, db_pool: asyncpg.Pool, orchestrator: Any) -> None:
        self.db_pool = db_pool
        self.orchestrator = orchestrator

    async def tick(self) -> None:
        """One trigger check: fire all due Phase 2 and Phase 3 triggers."""
        now = datetime.now(UTC)

        # ── Phase 2 triggers ─────────────────────────────────────────────────
        phase2_rows = await _fetch_ready_for_phase2(self.db_pool, now)
        for row in phase2_rows:
            logger.info("phase2_trigger_fired", match_id=row["match_id"])
            try:
                await self.orchestrator.start_match_lifecycle(row)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "phase2_trigger_error",
                    match_id=row["match_id"],
                    error=str(exc),
                )

        # ── Phase 3 triggers ─────────────────────────────────────────────────
        phase3_rows = await _fetch_ready_for_phase3(self.db_pool, now)
        for row in phase3_rows:
            logger.info("phase3_trigger_fired", match_id=row["match_id"])
            try:
                await self.orchestrator.start_live_engine(row)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "phase3_trigger_error",
                    match_id=row["match_id"],
                    error=str(exc),
                )

    async def run(self, *, interval_seconds: float = TRIGGER_INTERVAL_S) -> None:
        """Run the trigger loop continuously until cancelled.

        Errors in ``tick`` are caught and logged so the loop never exits
        silently on a transient DB failure.

        Args:
            interval_seconds: Poll interval (default ``TRIGGER_INTERVAL_S``).
        """
        logger.info("trigger_executor_started", interval_s=interval_seconds)
        while True:
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001
                logger.error("trigger_executor_error", error=str(exc))
            await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Fixture ↔ Kalshi event matching
# ---------------------------------------------------------------------------


def _match_fixtures_to_markets(
    fixtures: list[dict[str, Any]],
    kalshi_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match Goalserve fixtures to Kalshi events by date + team name.

    For each fixture, scans all Kalshi events that close within 24 hours
    of the kickoff and checks whether the home and away team names appear
    in the event title (first-word heuristic for compound names).

    Returns:
        List of dicts with keys:
          ``match_id``, ``league_id``, ``kickoff_utc``, ``kalshi_tickers``,
          ``odds_api_event_id`` (= Kalshi event_ticker).
    """
    matched: list[dict[str, Any]] = []

    for fix in fixtures:
        kickoff: datetime = fix["_kickoff_utc"]
        home: str = str(fix.get("@localteam_name", "")).lower().strip()
        away: str = str(fix.get("@visitorteam_name", "")).lower().strip()
        match_id: str = str(fix.get("@id", ""))

        if not match_id or not home or not away:
            continue

        for event in kalshi_events:
            if not _event_within_window(kickoff, event):
                continue
            title: str = str(event.get("title", "")).lower()
            if _name_in_title(home, title) and _name_in_title(away, title):
                tickers: list[str] = _extract_tickers(event)
                matched.append(
                    {
                        "match_id": match_id,
                        "league_id": int(fix.get("_league_id", 0)),
                        "kickoff_utc": kickoff,
                        "kalshi_tickers": tickers,
                        "odds_api_event_id": event.get("event_ticker"),
                    }
                )
                break  # one Kalshi event per fixture

    return matched


def _event_within_window(kickoff: datetime, event: dict[str, Any]) -> bool:
    """Return True if the Kalshi event closes within 24h of the kickoff."""
    close_str: str = (
        event.get("close_time", "")
        or event.get("end_date", "")
        or ""
    )
    if not close_str:
        return False
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        return abs((close_dt - kickoff).total_seconds()) < 86_400
    except (ValueError, AttributeError):
        return False


def _name_in_title(team_name: str, title: str) -> bool:
    """Return True if the team name (or its first word) appears in title."""
    if not team_name:
        return False
    if team_name in title:
        return True
    first_word = team_name.split()[0] if team_name.split() else ""
    return bool(first_word) and len(first_word) >= 4 and first_word in title


def _extract_tickers(event: dict[str, Any]) -> list[str]:
    """Extract market tickers from a Kalshi event dict."""
    # Kalshi API may use "markets" (list of tickers) or "market_tickers"
    tickers: list[str] = event.get("markets", []) or event.get("market_tickers", [])
    if isinstance(tickers, list):
        return [str(t) for t in tickers]
    return []


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _upsert_match_schedules(
    pool: asyncpg.Pool,
    schedules: list[MatchSchedule],
) -> int:
    """Insert new SCHEDULED rows; skip matches already in the table.

    Uses ``ON CONFLICT (match_id) DO NOTHING`` so re-running discovery is
    idempotent — existing rows (in any status) are never overwritten.

    Returns:
        Number of newly inserted rows.
    """
    if not schedules:
        return 0

    inserted = 0
    async with pool.acquire() as conn:
        for s in schedules:
            result = await conn.execute(
                """
                INSERT INTO match_schedule
                    (match_id, league_id, kickoff_utc,
                     phase2_trigger, phase3_trigger,
                     kalshi_tickers, odds_api_event_id,
                     trading_mode, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'SCHEDULED')
                ON CONFLICT (match_id) DO NOTHING
                """,
                s.match_id,
                s.league_id,
                s.kickoff_utc,
                s.phase2_trigger,
                s.phase3_trigger,
                json.dumps(s.kalshi_tickers),
                s.odds_api_event_id,
                s.trading_mode,
            )
            # asyncpg execute() returns "INSERT 0 N" or "INSERT 0 0"
            if result.endswith(" 1"):
                inserted += 1

    return inserted


async def _fetch_ready_for_phase2(
    pool: asyncpg.Pool,
    now: datetime,
) -> list[dict[str, Any]]:
    """Return SCHEDULED matches whose phase2_trigger has arrived."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT match_id, league_id, kickoff_utc,
                   phase2_trigger, phase3_trigger,
                   kalshi_tickers, odds_api_event_id,
                   trading_mode, status
            FROM match_schedule
            WHERE status = 'SCHEDULED'
              AND phase2_trigger <= $1
            ORDER BY kickoff_utc ASC
            """,
            now,
        )
    return [dict(r) for r in rows]


async def _fetch_ready_for_phase3(
    pool: asyncpg.Pool,
    now: datetime,
) -> list[dict[str, Any]]:
    """Return PHASE2_DONE matches whose phase3_trigger has arrived."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT match_id, league_id, kickoff_utc,
                   phase2_trigger, phase3_trigger,
                   kalshi_tickers, odds_api_event_id,
                   trading_mode, status
            FROM match_schedule
            WHERE status = 'PHASE2_DONE'
              AND phase3_trigger <= $1
            ORDER BY kickoff_utc ASC
            """,
            now,
        )
    return [dict(r) for r in rows]
