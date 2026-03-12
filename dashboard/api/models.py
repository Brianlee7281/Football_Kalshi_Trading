# dashboard/api/models.py
#
# Pydantic v2 response models for the MMPP Trading Dashboard API.
#
# Field names use snake_case to match FastAPI JSON serialisation defaults.
# Use sigma_MC (not σ_MC) for JSON keys — matches TypeScript / Redis format.
#
# Model list (15):
#   MarketProbs (TypeAlias), Score, MatchSummary, MatchDetail,
#   TickSnapshot, PositionItem, SignalItem, EventItem,
#   PnLBreakdown, PnLReport, ModelHealthReport, GraduationChecklist,
#   ContainerStatus, ConnectionHealth, SystemStatus

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

# ── Primitive helpers ─────────────────────────────────────────────────────────

# Market probability dict keyed by market name, e.g. {"home_win": 0.55}
MarketProbs = dict[str, float]


class Score(BaseModel):
    home: int
    away: int


# ── Match models ──────────────────────────────────────────────────────────────


class MatchSummary(BaseModel):
    match_id: str
    league_id: int
    kickoff_utc: datetime
    status: str
    trading_mode: str
    home_team: str | None = None
    away_team: str | None = None
    score: Score | None = None
    param_version: int | None = None


class TickSnapshot(BaseModel):
    match_id: str
    t: float
    engine_phase: str | None = None
    P_true: MarketProbs | None = None
    P_kalshi: MarketProbs | None = None
    P_bet365: MarketProbs | None = None
    sigma_MC: MarketProbs | None = None
    order_allowed: bool | None = None
    cooldown: bool | None = None
    ob_freeze: bool | None = None
    event_state: str | None = None
    mu_H: float | None = None
    mu_A: float | None = None
    score: Score | None = None


class PositionItem(BaseModel):
    id: int
    match_id: str
    market_ticker: str
    direction: str  # BUY_YES | BUY_NO
    entry_price: float
    quantity: int
    status: str  # PENDING | OPEN | CLOSED | AWAITING_SETTLEMENT | SETTLED
    is_paper: bool
    entry_time: datetime
    exit_time: datetime | None = None
    exit_price: float | None = None
    settlement_price: float | None = None
    realized_pnl: float | None = None


class SignalItem(BaseModel):
    match_id: str
    ticker: str
    direction: str  # BUY_YES | BUY_NO | HOLD
    EV: float
    P_cons: float
    P_kalshi: float
    alignment: str  # ALIGNED | DIVERGENT | UNAVAILABLE
    kelly_multiplier: float
    timestamp: float  # Unix epoch seconds


class EventItem(BaseModel):
    id: int
    match_id: str
    event_type: str
    source: str
    payload: dict[str, Any] | None = None
    created_at: datetime


class MatchDetail(BaseModel):
    match_id: str
    league_id: int
    kickoff_utc: datetime
    status: str
    trading_mode: str
    home_team: str | None = None
    away_team: str | None = None
    score: Score | None = None
    param_version: int | None = None
    latest_tick: TickSnapshot | None = None
    positions: list[PositionItem] = []
    recent_events: list[EventItem] = []


# ── Analytics models ──────────────────────────────────────────────────────────


class PnLBreakdown(BaseModel):
    by_league: dict[str, float]
    by_market: dict[str, float]
    by_direction: dict[str, float]
    by_alignment: dict[str, float]


class PnLReport(BaseModel):
    total_trades: int
    win_rate: float
    total_pnl: float
    edge_realization: float
    max_drawdown_pct: float
    sharpe: float | None = None
    avg_ev: float | None = None
    avg_slippage_cents: float | None = None
    brier_vs_exchange: float | None = None
    breakdown: PnLBreakdown


class ModelHealthReport(BaseModel):
    param_version: int
    param_trained_at: datetime
    brier_score: float
    brier_vs_exchange: float | None = None
    edge_realization: float
    matches_since_retrain: int
    brier_by_league: dict[str, float]
    edge_realization_rolling: list[float]  # last 20 matches


class GraduationChecklist(BaseModel):
    trade_count: int
    trades_ok: bool           # >= 50
    edge_realization_ok: bool  # in [0.6, 1.5]
    brier_ok: bool             # within Phase 1.5 baseline ± 0.03
    max_drawdown_ok: bool      # < 15%
    realism_score_ok: bool     # > 0.85
    directional_ok: bool       # 100% directional correctness
    no_crashes_ok: bool        # no system crashes in paper period
    theta_calibrated: bool     # THETA_ENTRY calibrated
    all_pass: bool


# ── System models ─────────────────────────────────────────────────────────────


class ContainerStatus(BaseModel):
    match_id: str
    status: str
    uptime_min: float | None = None       # minutes since container launch
    heartbeat_age_s: float | None = None  # seconds since last heartbeat
    container_id: str | None = None


class ConnectionHealth(BaseModel):
    service: str
    status: str  # connected | disconnected | polling
    last_message_age_s: float | None = None
    detail: str | None = None


class SystemStatus(BaseModel):
    containers: list[ContainerStatus]
    connections: list[ConnectionHealth]
    param_version: int | None = None
    param_trained_at: datetime | None = None
    matches_since_retrain: int | None = None
