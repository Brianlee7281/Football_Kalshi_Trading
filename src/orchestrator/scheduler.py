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
import unicodedata
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
                home_team=m.get("home_team"),
                away_team=m.get("away_team"),
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


def _extract_team_name(fix: dict[str, Any], side: str) -> str:
    """Extract team name from a Goalserve fixture dict.

    Goalserve nests names under ``localteam.@name`` / ``visitorteam.@name``.
    Falls back to the flat ``@localteam_name`` key for older API versions.
    """
    team_obj = fix.get(side)
    if isinstance(team_obj, dict):
        name = team_obj.get("@name", "")
        if name:
            return str(name).lower().strip()
    # Fallback to flat key (legacy format)
    return str(fix.get(f"@{side}_name", "")).lower().strip()


def _match_fixtures_to_markets(
    fixtures: list[dict[str, Any]],
    kalshi_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match Goalserve fixtures to Kalshi events by team name.

    For each fixture, scans all Kalshi GAME events and checks whether
    the home and away team names appear in the event title.  Kalshi
    close_time is often weeks after the actual match (market expiry !=
    kickoff), so time filtering is relaxed: close_time must be AFTER
    the kickoff (market hasn't expired yet).

    Returns:
        List of dicts with keys:
          ``match_id``, ``league_id``, ``kickoff_utc``, ``kalshi_tickers``,
          ``odds_api_event_id`` (= Kalshi event_ticker).
    """
    matched: list[dict[str, Any]] = []

    for fix in fixtures:
        kickoff: datetime = fix["_kickoff_utc"]
        home = _extract_team_name(fix, "localteam")
        away = _extract_team_name(fix, "visitorteam")
        match_id: str = str(fix.get("@id", ""))

        if not match_id or not home or not away:
            continue

        for event in kalshi_events:
            if not _event_still_open(kickoff, event):
                continue
            title: str = str(event.get("title", "")).lower()
            if _name_in_title(home, title) and _name_in_title(away, title):
                tickers: list[str] = _extract_tickers(event)
                # Capitalise team names for display (stored lowercase internally)
                home_display = fix.get("localteam", {}).get("@name", "") if isinstance(fix.get("localteam"), dict) else ""
                away_display = fix.get("visitorteam", {}).get("@name", "") if isinstance(fix.get("visitorteam"), dict) else ""
                matched.append(
                    {
                        "match_id": match_id,
                        "league_id": int(fix.get("_league_id", 0)),
                        "kickoff_utc": kickoff,
                        "kalshi_tickers": tickers,
                        "odds_api_event_id": event.get("event_ticker"),
                        "home_team": home_display,
                        "away_team": away_display,
                    }
                )
                break  # one Kalshi event per fixture

    return matched


def _event_still_open(kickoff: datetime, event: dict[str, Any]) -> bool:
    """Return True if the Kalshi event hasn't expired yet.

    Kalshi sets close_time days or weeks after the actual match (market
    expiry window), so we only check that close_time is AFTER kickoff.
    """
    close_str: str = (
        event.get("close_time", "")
        or event.get("end_date", "")
        or ""
    )
    if not close_str:
        return True  # no close_time → assume still open
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        return close_dt >= kickoff
    except (ValueError, AttributeError):
        return True


#: Common Goalserve → Kalshi name aliases.
#: Maps Goalserve name fragments to the form Kalshi uses in event titles.
_NAME_ALIASES: dict[str, list[str]] = {
    "atl. madrid": ["atletico"],
    "atl madrid": ["atletico"],
    "atletico madrid": ["atletico"],
    "ath bilbao": ["athletic"],
    "b. monchengladbach": ["monchengladbach", "gladbach"],
    "hamburger sv": ["hamburg"],
    "fc koln": ["koln", "köln", "cologne"],
    "inter": ["inter milan"],
    "botafogo rj": ["botafogo"],
    "flamengo rj": ["flamengo"],
    "st. pauli": ["st pauli", "st. pauli", "pauli"],
}


def _strip_accents(text: str) -> str:
    """Remove unicode accents: ö → o, é → e, etc."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _name_in_title(team_name: str, title: str) -> bool:
    """Return True if the team name (or any word >=4 chars) appears in title.

    Handles format mismatches like:
      Goalserve: "Bayer Leverkusen"  →  Kalshi: "Leverkusen vs Bayern Munich"
      Goalserve: "Eintracht Frankfurt"  →  Kalshi: "Frankfurt vs ..."
      Goalserve: "Atl. Madrid"  →  Kalshi: "Atletico vs ..."
      Goalserve: "FC Koln"  →  Kalshi: "FC Köln"

    Checks: alias table → full name → each word (>=4 chars).
    Also strips accents for comparison (Köln ↔ Koln).
    """
    if not team_name:
        return False

    title_ascii = _strip_accents(title)

    # 1. Check alias table
    if team_name in _NAME_ALIASES:
        for alias in _NAME_ALIASES[team_name]:
            if alias in title or alias in title_ascii:
                return True

    # 2. Full name match (with and without accents)
    if team_name in title or team_name in title_ascii:
        return True

    name_ascii = _strip_accents(team_name)
    if name_ascii in title or name_ascii in title_ascii:
        return True

    # 3. Each word (>=4 chars) — handles compound names
    for word in team_name.split():
        clean = word.strip(".,;:()")
        if len(clean) >= 4:
            clean_ascii = _strip_accents(clean)
            if clean in title or clean_ascii in title_ascii:
                return True
    return False


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
    """Insert new SCHEDULED rows or update team names for existing rows.

    Uses ``ON CONFLICT (match_id) DO UPDATE`` to backfill home_team/away_team
    on re-discovery while preserving status and other fields.

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
                     home_team, away_team,
                     trading_mode, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'SCHEDULED')
                ON CONFLICT (match_id) DO UPDATE
                    SET home_team = COALESCE(EXCLUDED.home_team, match_schedule.home_team),
                        away_team = COALESCE(EXCLUDED.away_team, match_schedule.away_team)
                """,
                s.match_id,
                s.league_id,
                s.kickoff_utc,
                s.phase2_trigger,
                s.phase3_trigger,
                json.dumps(s.kalshi_tickers),
                s.odds_api_event_id,
                s.home_team,
                s.away_team,
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
