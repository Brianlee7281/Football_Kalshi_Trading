"""Unit tests for src/orchestrator/scheduler.py.

Tests cover:
  - _parse_fixture_kickoff: valid dates, invalid, missing
  - _name_in_title: exact match, first-word match, no match
  - _event_within_window: within/outside 24h, invalid date
  - _match_fixtures_to_markets: happy path, no Kalshi match, missing fields
  - MatchDiscovery.discover: happy path, zero fixtures, zero Kalshi events
  - TriggerExecutor.tick: phase2 triggered, phase3 triggered, none ready
  - _upsert_match_schedules: new row inserted, conflict skipped
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.scheduler import (
    PHASE2_OFFSET_MINUTES,
    PHASE3_OFFSET_MINUTES,
    MatchDiscovery,
    TriggerExecutor,
    _event_within_window,
    _match_fixtures_to_markets,
    _name_in_title,
    _upsert_match_schedules,
)
from src.clients.goalserve import _parse_fixture_kickoff
from src.common.types import MatchSchedule


# ---------------------------------------------------------------------------
# _parse_fixture_kickoff
# ---------------------------------------------------------------------------


def test_parse_fixture_kickoff_dd_mm_yyyy() -> None:
    fix = {"@date": "26.03.2026", "@time": "15:00"}
    result = _parse_fixture_kickoff(fix)
    assert result is not None
    assert result.year == 2026
    assert result.month == 3
    assert result.day == 26
    assert result.hour == 15
    assert result.minute == 0


def test_parse_fixture_kickoff_mm_slash_dd_slash_yyyy() -> None:
    fix = {"@date": "03/26/2026", "@time": "20:45"}
    result = _parse_fixture_kickoff(fix)
    assert result is not None
    assert result.day == 26
    assert result.hour == 20
    assert result.minute == 45


def test_parse_fixture_kickoff_iso_format() -> None:
    fix = {"@date": "2026-03-26", "@time": "18:00"}
    result = _parse_fixture_kickoff(fix)
    assert result is not None
    assert result.day == 26


def test_parse_fixture_kickoff_missing_date() -> None:
    assert _parse_fixture_kickoff({}) is None
    assert _parse_fixture_kickoff({"@date": ""}) is None


def test_parse_fixture_kickoff_invalid_date() -> None:
    assert _parse_fixture_kickoff({"@date": "not-a-date"}) is None


def test_parse_fixture_kickoff_default_time() -> None:
    """Missing @time defaults to 00:00."""
    fix = {"@date": "26.03.2026"}
    result = _parse_fixture_kickoff(fix)
    assert result is not None
    assert result.hour == 0
    assert result.minute == 0


def test_parse_fixture_kickoff_is_utc() -> None:
    fix = {"@date": "26.03.2026", "@time": "15:00"}
    result = _parse_fixture_kickoff(fix)
    assert result is not None
    assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# _name_in_title
# ---------------------------------------------------------------------------


def test_name_in_title_exact_match() -> None:
    assert _name_in_title("arsenal", "arsenal vs chelsea") is True


def test_name_in_title_first_word_match() -> None:
    assert _name_in_title("manchester city", "manchester vs liverpool") is True


def test_name_in_title_no_match() -> None:
    assert _name_in_title("arsenal", "manchester vs chelsea") is False


def test_name_in_title_empty_team() -> None:
    assert _name_in_title("", "arsenal vs chelsea") is False


def test_name_in_title_short_first_word_not_matched() -> None:
    """First word shorter than 4 chars shouldn't trigger first-word match."""
    assert _name_in_title("ac milan", "chelsea vs ac milan") is True  # full name matches


def test_name_in_title_short_team_not_in_title() -> None:
    """Short team name that doesn't appear in the title returns False."""
    assert _name_in_title("psg", "arsenal vs chelsea") is False


# ---------------------------------------------------------------------------
# _event_within_window
# ---------------------------------------------------------------------------

_BASE_KO = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)


def test_event_within_window_same_day() -> None:
    event = {"close_time": "2026-03-26T17:00:00Z"}
    assert _event_within_window(_BASE_KO, event) is True


def test_event_within_window_outside_24h() -> None:
    event = {"close_time": "2026-03-28T15:00:00Z"}  # 48h ahead
    assert _event_within_window(_BASE_KO, event) is False


def test_event_within_window_missing_close_time() -> None:
    assert _event_within_window(_BASE_KO, {}) is False


def test_event_within_window_invalid_date() -> None:
    assert _event_within_window(_BASE_KO, {"close_time": "not-a-date"}) is False


def test_event_within_window_uses_end_date_fallback() -> None:
    event = {"end_date": "2026-03-26T17:00:00Z"}
    assert _event_within_window(_BASE_KO, event) is True


# ---------------------------------------------------------------------------
# _match_fixtures_to_markets
# ---------------------------------------------------------------------------


def _make_fixture(
    *,
    match_id: str = "gs-001",
    home: str = "Arsenal",
    away: str = "Chelsea",
    league_id: int = 1204,
    kickoff: datetime | None = None,
) -> dict:
    if kickoff is None:
        kickoff = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    return {
        "@id": match_id,
        "@localteam_name": home,
        "@visitorteam_name": away,
        "_league_id": league_id,
        "_kickoff_utc": kickoff,
    }


def _make_kalshi_event(
    *,
    title: str = "Arsenal vs Chelsea",
    event_ticker: str = "SOCCER-EPL-ARS-CHE-20260326",
    close_time: str = "2026-03-26T17:00:00Z",
    tickers: list[str] | None = None,
) -> dict:
    return {
        "event_ticker": event_ticker,
        "title": title,
        "close_time": close_time,
        "markets": tickers or ["SOCCER-EPL-ARS-CHE-20260326-YES"],
    }


def test_match_fixtures_happy_path() -> None:
    fixtures = [_make_fixture()]
    events = [_make_kalshi_event()]
    matched = _match_fixtures_to_markets(fixtures, events)
    assert len(matched) == 1
    assert matched[0]["match_id"] == "gs-001"
    assert matched[0]["league_id"] == 1204
    assert matched[0]["kalshi_tickers"] == ["SOCCER-EPL-ARS-CHE-20260326-YES"]
    assert matched[0]["odds_api_event_id"] == "SOCCER-EPL-ARS-CHE-20260326"


def test_match_fixtures_no_kalshi_match() -> None:
    fixtures = [_make_fixture(home="Arsenal", away="Chelsea")]
    events = [_make_kalshi_event(title="Manchester City vs Liverpool")]
    matched = _match_fixtures_to_markets(fixtures, events)
    assert matched == []


def test_match_fixtures_date_outside_window() -> None:
    fixtures = [_make_fixture()]
    events = [_make_kalshi_event(close_time="2026-03-29T15:00:00Z")]
    matched = _match_fixtures_to_markets(fixtures, events)
    assert matched == []


def test_match_fixtures_missing_match_id_skipped() -> None:
    fix = _make_fixture()
    del fix["@id"]
    matched = _match_fixtures_to_markets([fix], [_make_kalshi_event()])
    assert matched == []


def test_match_fixtures_multiple_matches() -> None:
    kickoff2 = datetime(2026, 3, 27, 20, 0, tzinfo=UTC)
    fixtures = [
        _make_fixture(match_id="gs-001", home="Arsenal", away="Chelsea"),
        _make_fixture(
            match_id="gs-002",
            home="Liverpool",
            away="Manchester City",
            kickoff=kickoff2,
        ),
    ]
    events = [
        _make_kalshi_event(
            title="Arsenal vs Chelsea",
            close_time="2026-03-26T17:00:00Z",
        ),
        _make_kalshi_event(
            title="Liverpool vs Manchester City",
            event_ticker="SOCCER-EPL-LIV-MCI-20260327",
            close_time="2026-03-27T22:00:00Z",
            tickers=["SOCCER-EPL-LIV-MCI-20260327-YES"],
        ),
    ]
    matched = _match_fixtures_to_markets(fixtures, events)
    assert len(matched) == 2
    ids = {m["match_id"] for m in matched}
    assert ids == {"gs-001", "gs-002"}


# ---------------------------------------------------------------------------
# MatchDiscovery.discover
# ---------------------------------------------------------------------------


def _make_pool(fetch_return=None, execute_return="INSERT 0 1"):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock(return_value=execute_return)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


@pytest.mark.asyncio
async def test_discover_happy_path() -> None:
    """Discover returns MatchSchedule list and upserts to DB."""
    ko = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    fixture = _make_fixture(kickoff=ko)
    event = _make_kalshi_event()

    goalserve = MagicMock()
    goalserve.get_upcoming_fixtures = AsyncMock(return_value=[fixture])

    kalshi = MagicMock()
    kalshi.get_active_soccer_events = AsyncMock(return_value=[event])

    pool, _ = _make_pool()
    disc = MatchDiscovery(goalserve, kalshi, pool)
    schedules = await disc.discover()

    assert len(schedules) == 1
    s = schedules[0]
    assert s.match_id == "gs-001"
    assert s.phase2_trigger == ko - timedelta(minutes=PHASE2_OFFSET_MINUTES)
    assert s.phase3_trigger == ko - timedelta(minutes=PHASE3_OFFSET_MINUTES)
    assert s.kalshi_tickers == ["SOCCER-EPL-ARS-CHE-20260326-YES"]


@pytest.mark.asyncio
async def test_discover_no_fixtures() -> None:
    goalserve = MagicMock()
    goalserve.get_upcoming_fixtures = AsyncMock(return_value=[])
    kalshi = MagicMock()
    kalshi.get_active_soccer_events = AsyncMock(return_value=[])
    pool, _ = _make_pool()
    disc = MatchDiscovery(goalserve, kalshi, pool)
    schedules = await disc.discover()
    assert schedules == []


@pytest.mark.asyncio
async def test_discover_trading_mode_param_version() -> None:
    """trading_mode and param_version are forwarded to MatchSchedule."""
    ko = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    goalserve = MagicMock()
    goalserve.get_upcoming_fixtures = AsyncMock(return_value=[_make_fixture(kickoff=ko)])
    kalshi = MagicMock()
    kalshi.get_active_soccer_events = AsyncMock(return_value=[_make_kalshi_event()])
    pool, _ = _make_pool()
    disc = MatchDiscovery(goalserve, kalshi, pool, trading_mode="live", param_version=7)
    schedules = await disc.discover()
    assert schedules[0].trading_mode == "live"
    assert schedules[0].param_version == 7


# ---------------------------------------------------------------------------
# TriggerExecutor.tick
# ---------------------------------------------------------------------------


def _make_pool_with_rows(phase2_rows=None, phase3_rows=None):
    conn = MagicMock()
    call_n = {"n": 0}

    async def fetch_side(sql, *args):
        n = call_n["n"]
        call_n["n"] += 1
        if n == 0:
            return phase2_rows or []
        return phase3_rows or []

    conn.fetch = AsyncMock(side_effect=fetch_side)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


@pytest.mark.asyncio
async def test_trigger_executor_fires_phase2() -> None:
    row = {"match_id": "gs-001", "status": "SCHEDULED"}
    pool = _make_pool_with_rows(phase2_rows=[row], phase3_rows=[])

    orchestrator = MagicMock()
    orchestrator.start_match_lifecycle = AsyncMock()
    orchestrator.start_live_engine = AsyncMock()

    ex = TriggerExecutor(pool, orchestrator)
    await ex.tick()

    orchestrator.start_match_lifecycle.assert_called_once_with(row)
    orchestrator.start_live_engine.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_executor_fires_phase3() -> None:
    row = {"match_id": "gs-002", "status": "PHASE2_DONE"}
    pool = _make_pool_with_rows(phase2_rows=[], phase3_rows=[row])

    orchestrator = MagicMock()
    orchestrator.start_match_lifecycle = AsyncMock()
    orchestrator.start_live_engine = AsyncMock()

    ex = TriggerExecutor(pool, orchestrator)
    await ex.tick()

    orchestrator.start_live_engine.assert_called_once_with(row)
    orchestrator.start_match_lifecycle.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_executor_none_ready() -> None:
    pool = _make_pool_with_rows(phase2_rows=[], phase3_rows=[])

    orchestrator = MagicMock()
    orchestrator.start_match_lifecycle = AsyncMock()
    orchestrator.start_live_engine = AsyncMock()

    ex = TriggerExecutor(pool, orchestrator)
    await ex.tick()

    orchestrator.start_match_lifecycle.assert_not_called()
    orchestrator.start_live_engine.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_executor_orchestrator_error_does_not_propagate() -> None:
    """An orchestrator error is caught per-match; tick continues."""
    row1 = {"match_id": "gs-001"}
    row2 = {"match_id": "gs-002"}
    pool = _make_pool_with_rows(phase2_rows=[row1, row2], phase3_rows=[])

    orchestrator = MagicMock()
    orchestrator.start_match_lifecycle = AsyncMock(
        side_effect=[RuntimeError("boom"), None]
    )
    orchestrator.start_live_engine = AsyncMock()

    ex = TriggerExecutor(pool, orchestrator)
    await ex.tick()  # should not raise

    assert orchestrator.start_match_lifecycle.call_count == 2


# ---------------------------------------------------------------------------
# _upsert_match_schedules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_match_schedules_new_row_counted() -> None:
    pool, conn = _make_pool(execute_return="INSERT 0 1")
    ko = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    s = MatchSchedule(
        match_id="gs-001",
        league_id=1204,
        kickoff_utc=ko,
        phase2_trigger=ko - timedelta(minutes=65),
        phase3_trigger=ko - timedelta(minutes=2),
        kalshi_tickers=["TICKER-YES"],
    )
    count = await _upsert_match_schedules(pool, [s])
    assert count == 1


@pytest.mark.asyncio
async def test_upsert_match_schedules_conflict_not_counted() -> None:
    pool, conn = _make_pool(execute_return="INSERT 0 0")
    ko = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    s = MatchSchedule(
        match_id="gs-001",
        league_id=1204,
        kickoff_utc=ko,
        phase2_trigger=ko - timedelta(minutes=65),
        phase3_trigger=ko - timedelta(minutes=2),
        kalshi_tickers=[],
    )
    count = await _upsert_match_schedules(pool, [s])
    assert count == 0


@pytest.mark.asyncio
async def test_upsert_match_schedules_empty_list() -> None:
    pool, _ = _make_pool()
    count = await _upsert_match_schedules(pool, [])
    assert count == 0
