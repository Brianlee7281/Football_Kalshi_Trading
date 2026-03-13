# dashboard/api/routes/matches.py
#
# REST endpoints:
#   GET /api/matches                      → List[MatchSummary]
#   GET /api/match/{match_id}             → MatchDetail
#   GET /api/match/{match_id}/ticks       → List[TickSnapshot]
#   GET /api/match/{match_id}/events      → List[EventItem]

from __future__ import annotations

import json as _json
from typing import Any

from fastapi import APIRouter, HTTPException

from dashboard.api.deps import Pool
from dashboard.api.models import (
    EventItem,
    MarketProbs,
    MatchDetail,
    MatchSummary,
    PositionItem,
    Score,
    TickSnapshot,
)

router = APIRouter(prefix="/api", tags=["matches"])


# ── helpers ───────────────────────────────────────────────────────────────────


def _market_probs(v: Any) -> MarketProbs | None:
    """Decode a JSONB value from asyncpg (str or dict) → MarketProbs."""
    if v is None:
        return None
    d: dict[str, Any] = _json.loads(v) if isinstance(v, str) else dict(v)
    return {k: float(val) for k, val in d.items()}


def _parse_teams(kalshi_tickers: Any) -> tuple[str | None, str | None]:
    """Extract home/away team codes from the first Kalshi ticker.

    Format: SOCCER-EPL-ARS-v-CHE-25MAR15-WINNER → ('ARS', 'CHE')
    """
    try:
        tickers: list[str] = (
            _json.loads(kalshi_tickers)
            if isinstance(kalshi_tickers, str)
            else list(kalshi_tickers)
        )
        if not tickers:
            return None, None
        parts = tickers[0].split("-")
        v_idx = parts.index("v")
        return parts[v_idx - 1], parts[v_idx + 1]
    except (ValueError, IndexError, TypeError):
        return None, None


def _row_to_match_summary(row: Any) -> MatchSummary:
    # Prefer stored team names; fall back to parsing Kalshi tickers
    home = row.get("home_team") or None
    away = row.get("away_team") or None
    if not home or not away:
        home, away = _parse_teams(row["kalshi_tickers"])
    return MatchSummary(
        match_id=row["match_id"],
        league_id=int(row["league_id"]),
        kickoff_utc=row["kickoff_utc"],
        status=row["status"],
        trading_mode=row["trading_mode"],
        home_team=home,
        away_team=away,
        score=None,  # not stored in match_schedule
        param_version=(
            int(row["param_version"]) if row["param_version"] is not None else None
        ),
    )


def _row_to_tick(row: Any) -> TickSnapshot:
    return TickSnapshot(
        match_id=row["match_id"],
        t=float(row["t"]),
        engine_phase=row["engine_phase"],
        P_true=_market_probs(row["P_true"]),
        P_kalshi=_market_probs(row["P_kalshi"]),
        P_bet365=_market_probs(row["P_bet365"]),
        sigma_MC=_market_probs(row["sigma_MC"]),
        order_allowed=row["order_allowed"],
        cooldown=row["cooldown"],
        ob_freeze=row["ob_freeze"],
        event_state=row["event_state"],
        mu_H=float(row["mu_H"]) if row["mu_H"] is not None else None,
        mu_A=float(row["mu_A"]) if row["mu_A"] is not None else None,
        score=None,  # not stored per-tick; front-end derives from events
    )


def _row_to_position(row: Any) -> PositionItem:
    return PositionItem(
        id=int(row["id"]),
        match_id=row["match_id"],
        market_ticker=row["market_ticker"],
        direction=row["direction"],
        entry_price=float(row["entry_price"]),
        quantity=int(row["quantity"]),
        status=row["status"],
        is_paper=bool(row["is_paper"]),
        entry_time=row["entry_time"],
        exit_time=row["exit_time"],
        exit_price=(
            float(row["exit_price"]) if row["exit_price"] is not None else None
        ),
        settlement_price=(
            float(row["settlement_price"])
            if row["settlement_price"] is not None
            else None
        ),
        realized_pnl=(
            float(row["realized_pnl"]) if row["realized_pnl"] is not None else None
        ),
    )


def _row_to_event(row: Any) -> EventItem:
    payload_raw = row["payload"]
    payload: dict[str, Any] | None = None
    if payload_raw is not None:
        payload = (
            _json.loads(payload_raw)
            if isinstance(payload_raw, str)
            else dict(payload_raw)
        )
    return EventItem(
        id=int(row["id"]),
        match_id=row["match_id"],
        event_type=row["event_type"],
        source=row["source"],
        payload=payload,
        created_at=row["created_at"],
    )


def _compute_score(goal_rows: list[Any]) -> Score:
    """Derive current score from goal_confirmed event payloads."""
    home = 0
    away = 0
    for row in goal_rows:
        payload_raw = row["payload"]
        if payload_raw is None:
            continue
        payload: dict[str, Any] = (
            _json.loads(payload_raw)
            if isinstance(payload_raw, str)
            else dict(payload_raw)
        )
        team = payload.get("team", "")
        if team == "home":
            home += 1
        elif team == "away":
            away += 1
    return Score(home=home, away=away)


# ── endpoints ─────────────────────────────────────────────────────────────────


@router.get("/matches", response_model=list[MatchSummary])
async def list_matches(
    pool: Pool,
    status: str | None = None,
    date: str | None = None,
) -> list[MatchSummary]:
    """List matches from the last 24 hours, optionally filtered by status/date."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT match_id, league_id, kickoff_utc, status, trading_mode,
                   param_version, kalshi_tickers, home_team, away_team
            FROM match_schedule
            WHERE kickoff_utc > NOW() - INTERVAL '24 hours'
              AND ($1::text IS NULL OR status = $1)
              AND ($2::text IS NULL OR kickoff_utc::date = $2::date)
            ORDER BY kickoff_utc DESC
            """,
            status,
            date,
        )
    return [_row_to_match_summary(r) for r in rows]


@router.get("/match/{match_id}", response_model=MatchDetail)
async def match_detail(match_id: str, pool: Pool) -> MatchDetail:
    """Full match detail including latest tick, open positions, and recent events."""
    async with pool.acquire() as conn:
        # Match row
        match_row = await conn.fetchrow(
            """
            SELECT match_id, league_id, kickoff_utc, status, trading_mode,
                   param_version, kalshi_tickers, home_team, away_team
            FROM match_schedule
            WHERE match_id = $1
            """,
            match_id,
        )
        if match_row is None:
            raise HTTPException(status_code=404, detail=f"Match {match_id!r} not found")

        # Latest tick
        tick_row = await conn.fetchrow(
            """
            SELECT match_id, t, engine_phase, P_true, P_kalshi, P_bet365, sigma_MC,
                   order_allowed, cooldown, ob_freeze, event_state, mu_H, mu_A
            FROM tick_snapshots
            WHERE match_id = $1
            ORDER BY t DESC
            LIMIT 1
            """,
            match_id,
        )

        # Open positions
        pos_rows = await conn.fetch(
            """
            SELECT id, match_id, market_ticker, direction, entry_price, quantity,
                   status, is_paper, entry_time, exit_time, exit_price,
                   settlement_price, realized_pnl
            FROM positions
            WHERE match_id = $1
              AND status IN ('OPEN', 'AWAITING_SETTLEMENT')
            ORDER BY entry_time DESC
            """,
            match_id,
        )

        # Recent events (last 50)
        event_rows = await conn.fetch(
            """
            SELECT id, match_id, event_type, source, payload, created_at
            FROM event_log
            WHERE match_id = $1
            ORDER BY created_at DESC
            LIMIT 50
            """,
            match_id,
        )

        # Score from goal_confirmed events
        goal_rows = await conn.fetch(
            """
            SELECT payload FROM event_log
            WHERE match_id = $1 AND event_type = 'goal_confirmed'
            ORDER BY created_at
            """,
            match_id,
        )

    home = match_row.get("home_team") or None
    away = match_row.get("away_team") or None
    if not home or not away:
        home, away = _parse_teams(match_row["kalshi_tickers"])
    score = _compute_score(list(goal_rows))

    return MatchDetail(
        match_id=match_row["match_id"],
        league_id=int(match_row["league_id"]),
        kickoff_utc=match_row["kickoff_utc"],
        status=match_row["status"],
        trading_mode=match_row["trading_mode"],
        home_team=home,
        away_team=away,
        score=score,
        param_version=(
            int(match_row["param_version"])
            if match_row["param_version"] is not None
            else None
        ),
        latest_tick=_row_to_tick(tick_row) if tick_row is not None else None,
        positions=[_row_to_position(r) for r in pos_rows],
        recent_events=[_row_to_event(r) for r in event_rows],
    )


@router.get("/match/{match_id}/ticks", response_model=list[TickSnapshot])
async def match_ticks(
    match_id: str,
    pool: Pool,
    market: str | None = None,
    downsample: int = 1,
) -> list[TickSnapshot]:
    """Time series of tick snapshots.  downsample=10 returns every 10th tick."""
    # Validate match exists
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM match_schedule WHERE match_id = $1",
            match_id,
        )
        if exists is None:
            raise HTTPException(status_code=404, detail=f"Match {match_id!r} not found")

        # Clamp downsample to at least 1
        step = max(1, downsample)
        rows = await conn.fetch(
            """
            WITH ranked AS (
                SELECT match_id, t, engine_phase,
                       P_true, P_kalshi, P_bet365, sigma_MC,
                       order_allowed, cooldown, ob_freeze, event_state,
                       mu_H, mu_A,
                       ROW_NUMBER() OVER (ORDER BY t) AS rn
                FROM tick_snapshots
                WHERE match_id = $1
            )
            SELECT * FROM ranked
            WHERE (rn - 1) % $2 = 0
            ORDER BY t
            """,
            match_id,
            step,
        )

    return [_row_to_tick(r) for r in rows]


@router.get("/match/{match_id}/events", response_model=list[EventItem])
async def match_events(match_id: str, pool: Pool) -> list[EventItem]:
    """All events for a match, chronological order."""
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM match_schedule WHERE match_id = $1",
            match_id,
        )
        if exists is None:
            raise HTTPException(status_code=404, detail=f"Match {match_id!r} not found")

        rows = await conn.fetch(
            """
            SELECT id, match_id, event_type, source, payload, created_at
            FROM event_log
            WHERE match_id = $1
            ORDER BY created_at
            """,
            match_id,
        )

    return [_row_to_event(r) for r in rows]
