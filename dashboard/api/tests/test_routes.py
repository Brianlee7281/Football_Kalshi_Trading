"""Dashboard API route tests.

Tests cover:
  - Field presence on match list / detail / ticks / events / positions
  - Sort order (matches: kickoff DESC, ticks: t ASC, events: created_at ASC)
  - 0 trades edge case (PnL report with zero trades)
  - 0 ticks edge case (empty tick list for a match)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dashboard.api.routes.matches import router as matches_router
from dashboard.api.routes.positions import router as positions_router
from dashboard.api.routes.analytics import router as analytics_router
from dashboard.api.routes.system import router as system_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(pool_mock: Any) -> FastAPI:
    """Create a FastAPI app with mocked pool on app.state."""
    app = FastAPI()
    app.include_router(matches_router)
    app.include_router(positions_router)
    app.include_router(analytics_router)
    app.include_router(system_router)
    app.state.pool = pool_mock
    return app


def _make_match_row(
    match_id: str = "m1",
    status: str = "PHASE3_RUNNING",
    kickoff_utc: datetime | None = None,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "league_id": 39,
        "kickoff_utc": kickoff_utc or datetime(2026, 3, 10, 20, 0, tzinfo=UTC),
        "status": status,
        "trading_mode": "paper",
        "param_version": 1,
        "kalshi_tickers": json.dumps(["SOCCER-EPL-ARS-v-CHE-10MAR26-WINNER"]),
    }


def _make_tick_row(
    match_id: str = "m1",
    t: float = 45.0,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "t": t,
        "engine_phase": "RUNNING",
        "P_true": json.dumps({"home_win": 0.55}),
        "P_kalshi": json.dumps({"home_win": 0.52}),
        "P_bet365": None,
        "sigma_MC": json.dumps({"home_win": 0.0022}),
        "order_allowed": True,
        "cooldown": False,
        "ob_freeze": False,
        "event_state": "NORMAL",
        "mu_H": 1.2,
        "mu_A": 0.8,
        "rn": 1,
    }


def _make_position_row(
    pos_id: int = 1,
    match_id: str = "m1",
    direction: str = "BUY_YES",
    status: str = "OPEN",
) -> dict[str, Any]:
    return {
        "id": pos_id,
        "match_id": match_id,
        "market_ticker": "SOCCER-EPL-ARS-v-CHE-WINNER",
        "direction": direction,
        "entry_price": 0.55,
        "quantity": 10,
        "status": status,
        "is_paper": True,
        "entry_time": datetime(2026, 3, 10, 20, 5, tzinfo=UTC),
        "exit_time": None,
        "exit_price": None,
        "settlement_price": None,
        "realized_pnl": None,
    }


def _make_event_row(
    event_id: int = 1,
    match_id: str = "m1",
    event_type: str = "goal_confirmed",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "match_id": match_id,
        "event_type": event_type,
        "source": "goalserve",
        "payload": json.dumps({"team": "home", "minute": 23}),
        "created_at": datetime(2026, 3, 10, 20, 23, tzinfo=UTC),
    }


def _mock_pool(
    fetch_returns: list[list[dict[str, Any]]] | None = None,
    fetchrow_returns: list[dict[str, Any] | None] | None = None,
    fetchval_returns: list[Any] | None = None,
) -> MagicMock:
    """Create a mock asyncpg pool with configurable return values."""
    pool = MagicMock()
    conn = AsyncMock()

    fetch_returns = fetch_returns or [[]]
    fetchrow_returns = fetchrow_returns or [None]
    fetchval_returns = fetchval_returns or [None]

    conn.fetch = AsyncMock(side_effect=fetch_returns)
    conn.fetchrow = AsyncMock(side_effect=fetchrow_returns)
    conn.fetchval = AsyncMock(side_effect=fetchval_returns)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx

    return pool


# ---------------------------------------------------------------------------
# GET /api/matches — field presence & sort order
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_matches_field_presence() -> None:
    """Response items contain all MatchSummary fields."""
    rows = [
        _make_match_row("m1", kickoff_utc=datetime(2026, 3, 10, 20, 0, tzinfo=UTC)),
        _make_match_row("m2", kickoff_utc=datetime(2026, 3, 10, 18, 0, tzinfo=UTC)),
    ]
    pool = _mock_pool(fetch_returns=[rows])
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/matches")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    required_fields = {
        "match_id", "league_id", "kickoff_utc", "status",
        "trading_mode", "home_team", "away_team", "score", "param_version",
    }
    for item in data:
        assert required_fields.issubset(set(item.keys()))


@pytest.mark.anyio
async def test_list_matches_sort_order() -> None:
    """Matches returned in kickoff_utc DESC order (DB-side)."""
    rows = [
        _make_match_row("m_later", kickoff_utc=datetime(2026, 3, 10, 22, 0, tzinfo=UTC)),
        _make_match_row("m_earlier", kickoff_utc=datetime(2026, 3, 10, 18, 0, tzinfo=UTC)),
    ]
    pool = _mock_pool(fetch_returns=[rows])
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/matches")

    data = resp.json()
    assert data[0]["match_id"] == "m_later"
    assert data[1]["match_id"] == "m_earlier"


# ---------------------------------------------------------------------------
# GET /api/match/{id}/ticks — field presence, sort, 0-ticks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_match_ticks_fields() -> None:
    """Tick snapshots include all expected fields."""
    tick_rows = [_make_tick_row(t=10.0), _make_tick_row(t=20.0)]
    pool = _mock_pool(
        fetchval_returns=[1],  # match exists
        fetch_returns=[tick_rows],
    )
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/match/m1/ticks")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    expected_fields = {
        "match_id", "t", "engine_phase", "P_true", "P_kalshi",
        "sigma_MC", "order_allowed",
    }
    for tick in data:
        assert expected_fields.issubset(set(tick.keys()))


@pytest.mark.anyio
async def test_match_ticks_sort_ascending() -> None:
    """Ticks sorted by t ascending."""
    tick_rows = [_make_tick_row(t=5.0), _make_tick_row(t=15.0)]
    pool = _mock_pool(
        fetchval_returns=[1],
        fetch_returns=[tick_rows],
    )
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/match/m1/ticks")

    data = resp.json()
    assert data[0]["t"] == 5.0
    assert data[1]["t"] == 15.0


@pytest.mark.anyio
async def test_match_ticks_zero_ticks() -> None:
    """0 ticks returns empty list, not error."""
    pool = _mock_pool(
        fetchval_returns=[1],  # match exists
        fetch_returns=[[]],    # no ticks
    )
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/match/m1/ticks")

    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/match/{id}/events — sort order
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_match_events_sort_chronological() -> None:
    """Events sorted by created_at ascending."""
    event1 = _make_event_row(1, event_type="kickoff")
    event1["created_at"] = datetime(2026, 3, 10, 20, 0, tzinfo=UTC)
    event2 = _make_event_row(2, event_type="goal_confirmed")
    event2["created_at"] = datetime(2026, 3, 10, 20, 23, tzinfo=UTC)

    pool = _mock_pool(
        fetchval_returns=[1],  # match exists
        fetch_returns=[[event1, event2]],
    )
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/match/m1/events")

    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["event_type"] == "kickoff"
    assert data[1]["event_type"] == "goal_confirmed"


# ---------------------------------------------------------------------------
# GET /api/positions — field presence
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_positions_field_presence() -> None:
    """Position items contain all expected fields."""
    rows = [_make_position_row(1), _make_position_row(2, direction="BUY_NO")]
    pool = _mock_pool(fetch_returns=[rows])
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/positions")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    required = {
        "id", "match_id", "market_ticker", "direction",
        "entry_price", "quantity", "status", "is_paper",
    }
    for pos in data:
        assert required.issubset(set(pos.keys()))


# ---------------------------------------------------------------------------
# GET /api/analytics/pnl — 0 trades edge case
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pnl_zero_trades() -> None:
    """PnL report with 0 settled trades returns safe defaults."""
    summary = {"total_trades": 0, "wins": 0, "total_pnl": 0.0}
    dd_row = {"max_dd": 0.0}
    pool = _mock_pool(
        fetchrow_returns=[summary, dd_row],
        fetch_returns=[[], [], []],  # league, market, direction breakdowns
    )
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/analytics/pnl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trades"] == 0
    assert data["win_rate"] == 0.0
    assert data["total_pnl"] == 0.0
    assert data["max_drawdown_pct"] == 0.0
    assert "breakdown" in data


# ---------------------------------------------------------------------------
# GET /api/match/{id} — 404 for missing match
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_match_detail_404() -> None:
    """Unknown match_id returns 404."""
    pool = _mock_pool(fetchrow_returns=[None])
    app = _make_app(pool)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/match/unknown_id")

    assert resp.status_code == 404
