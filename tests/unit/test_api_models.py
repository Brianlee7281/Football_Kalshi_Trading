# tests/unit/test_api_models.py
#
# Unit tests for dashboard/api/models.py
# Validates field presence, types, and serialisation for all 15 models.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dashboard.api.models import (
    ConnectionHealth,
    ContainerStatus,
    EventItem,
    GraduationChecklist,
    MatchDetail,
    MatchSummary,
    ModelHealthReport,
    PnLBreakdown,
    PnLReport,
    PositionItem,
    Score,
    SignalItem,
    SystemStatus,
    TickSnapshot,
)

_NOW = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)


# ── Score ─────────────────────────────────────────────────────────────────────


def test_score_fields() -> None:
    s = Score(home=1, away=0)
    assert s.home == 1
    assert s.away == 0


def test_score_serialise() -> None:
    s = Score(home=2, away=1)
    d = s.model_dump()
    assert d == {"home": 2, "away": 1}


# ── MatchSummary ──────────────────────────────────────────────────────────────


def test_match_summary_required_fields() -> None:
    ms = MatchSummary(
        match_id="m1",
        league_id=12,
        kickoff_utc=_NOW,
        status="PHASE3_RUNNING",
        trading_mode="paper",
    )
    assert ms.match_id == "m1"
    assert ms.home_team is None
    assert ms.score is None


def test_match_summary_with_score() -> None:
    ms = MatchSummary(
        match_id="m2",
        league_id=12,
        kickoff_utc=_NOW,
        status="PHASE3_RUNNING",
        trading_mode="paper",
        home_team="Arsenal",
        away_team="Chelsea",
        score=Score(home=1, away=0),
    )
    assert ms.score is not None
    assert ms.score.home == 1


# ── TickSnapshot ──────────────────────────────────────────────────────────────


def test_tick_snapshot_minimal() -> None:
    ts = TickSnapshot(match_id="m1", t=45.5)
    assert ts.t == 45.5
    assert ts.P_true is None
    assert ts.sigma_MC is None


def test_tick_snapshot_full() -> None:
    ts = TickSnapshot(
        match_id="m1",
        t=67.0,
        engine_phase="SECOND_HALF",
        P_true={"home_win": 0.72, "over_25": 0.58},
        P_kalshi={"home_win_ask": 0.70, "home_win_bid": 0.68},
        P_bet365={"home_win": 0.71},
        sigma_MC={"home_win": 0.0022, "over_25": 0.0021},
        order_allowed=True,
        cooldown=False,
        ob_freeze=False,
        event_state="IDLE",
        mu_H=0.45,
        mu_A=0.20,
        score=Score(home=1, away=0),
    )
    assert ts.sigma_MC is not None
    assert ts.sigma_MC["home_win"] == pytest.approx(0.0022)
    assert ts.score is not None
    assert ts.score.away == 0


def test_tick_snapshot_uses_sigma_mc_not_greek() -> None:
    """JSON key must be sigma_MC (ASCII), not σ_MC."""
    ts = TickSnapshot(match_id="m1", t=1.0, sigma_MC={"home_win": 0.002})
    d = ts.model_dump()
    assert "sigma_MC" in d
    assert "σ_MC" not in d


# ── PositionItem ──────────────────────────────────────────────────────────────


def test_position_item_required() -> None:
    p = PositionItem(
        id=1,
        match_id="m1",
        market_ticker="SOCCER-EPL-ARS-CHE-WINNER",
        direction="BUY_YES",
        entry_price=0.65,
        quantity=25,
        status="OPEN",
        is_paper=True,
        entry_time=_NOW,
    )
    assert p.direction == "BUY_YES"
    assert p.exit_time is None
    assert p.realized_pnl is None


def test_position_item_settled() -> None:
    p = PositionItem(
        id=2,
        match_id="m1",
        market_ticker="SOCCER-EPL-ARS-CHE-WINNER",
        direction="BUY_YES",
        entry_price=0.65,
        quantity=25,
        status="SETTLED",
        is_paper=True,
        entry_time=_NOW,
        settlement_price=1.0,
        realized_pnl=8.75,
    )
    assert p.settlement_price == pytest.approx(1.0)
    assert p.realized_pnl == pytest.approx(8.75)


# ── SignalItem ────────────────────────────────────────────────────────────────


def test_signal_item_fields() -> None:
    s = SignalItem(
        match_id="m1",
        ticker="SOCCER-EPL-ARS-CHE-WINNER",
        direction="BUY_YES",
        EV=0.032,
        P_cons=0.5336,
        P_kalshi=0.68,
        alignment="ALIGNED",
        kelly_multiplier=0.018,
        timestamp=1741694400.0,
    )
    assert pytest.approx(0.032) == s.EV
    assert s.alignment == "ALIGNED"


# ── EventItem ─────────────────────────────────────────────────────────────────


def test_event_item_minimal() -> None:
    e = EventItem(
        id=1,
        match_id="m1",
        event_type="goal_confirmed",
        source="odds_api",
        created_at=_NOW,
    )
    assert e.payload is None


def test_event_item_with_payload() -> None:
    e = EventItem(
        id=2,
        match_id="m1",
        event_type="goal_confirmed",
        source="live_score",
        payload={"minute": 34, "team": "home", "scorer": "Saka"},
        created_at=_NOW,
    )
    assert e.payload is not None
    assert e.payload["minute"] == 34


# ── MatchDetail ───────────────────────────────────────────────────────────────


def test_match_detail_defaults() -> None:
    md = MatchDetail(
        match_id="m1",
        league_id=12,
        kickoff_utc=_NOW,
        status="PHASE3_RUNNING",
        trading_mode="paper",
    )
    assert md.latest_tick is None
    assert md.positions == []
    assert md.recent_events == []


def test_match_detail_with_tick() -> None:
    tick = TickSnapshot(match_id="m1", t=67.0, engine_phase="SECOND_HALF")
    md = MatchDetail(
        match_id="m1",
        league_id=12,
        kickoff_utc=_NOW,
        status="PHASE3_RUNNING",
        trading_mode="paper",
        latest_tick=tick,
        positions=[],
        recent_events=[],
    )
    assert md.latest_tick is not None
    assert md.latest_tick.engine_phase == "SECOND_HALF"


# ── PnLBreakdown + PnLReport ─────────────────────────────────────────────────


def test_pnl_breakdown_fields() -> None:
    b = PnLBreakdown(
        by_league={"EPL": 145.0, "La Liga": 62.0},
        by_market={"home_win": 180.0, "over_25": 40.0},
        by_direction={"BUY_YES": 198.0, "BUY_NO": 36.0},
        by_alignment={"ALIGNED": 210.0, "DIVERGENT": 24.0},
    )
    assert b.by_league["EPL"] == pytest.approx(145.0)


def test_pnl_report_fields() -> None:
    r = PnLReport(
        total_trades=47,
        win_rate=0.596,
        total_pnl=234.0,
        edge_realization=0.87,
        max_drawdown_pct=8.2,
        sharpe=1.4,
        breakdown=PnLBreakdown(
            by_league={},
            by_market={},
            by_direction={},
            by_alignment={},
        ),
    )
    assert r.total_trades == 47
    assert r.win_rate == pytest.approx(0.596)


# ── ModelHealthReport ─────────────────────────────────────────────────────────


def test_model_health_report() -> None:
    r = ModelHealthReport(
        param_version=12,
        param_trained_at=_NOW,
        brier_score=0.198,
        edge_realization=0.87,
        matches_since_retrain=23,
        brier_by_league={"EPL": 0.192, "La Liga": 0.201},
        edge_realization_rolling=[0.85, 0.88, 0.90],
    )
    assert r.param_version == 12
    assert r.brier_vs_exchange is None
    assert len(r.edge_realization_rolling) == 3


# ── GraduationChecklist ───────────────────────────────────────────────────────


def test_graduation_checklist_all_pass() -> None:
    g = GraduationChecklist(
        trade_count=52,
        trades_ok=True,
        edge_realization_ok=True,
        brier_ok=True,
        max_drawdown_ok=True,
        realism_score_ok=True,
        directional_ok=True,
        no_crashes_ok=True,
        theta_calibrated=True,
        all_pass=True,
    )
    assert g.all_pass is True
    assert g.trade_count == 52


def test_graduation_checklist_partial_fail() -> None:
    g = GraduationChecklist(
        trade_count=30,
        trades_ok=False,
        edge_realization_ok=True,
        brier_ok=True,
        max_drawdown_ok=True,
        realism_score_ok=True,
        directional_ok=True,
        no_crashes_ok=True,
        theta_calibrated=False,
        all_pass=False,
    )
    assert g.all_pass is False
    assert g.trades_ok is False


# ── ContainerStatus ───────────────────────────────────────────────────────────


def test_container_status() -> None:
    c = ContainerStatus(
        match_id="m1",
        status="PHASE3_RUNNING",
        uptime_min=67.0,
        heartbeat_age_s=3.2,
        container_id="abc123def456",
    )
    assert c.heartbeat_age_s == pytest.approx(3.2)


def test_container_status_scheduled() -> None:
    c = ContainerStatus(match_id="m2", status="SCHEDULED")
    assert c.uptime_min is None
    assert c.heartbeat_age_s is None


# ── ConnectionHealth ──────────────────────────────────────────────────────────


def test_connection_health_connected() -> None:
    ch = ConnectionHealth(
        service="Odds-API WS",
        status="connected",
        last_message_age_s=0.3,
    )
    assert ch.status == "connected"
    assert ch.detail is None


def test_connection_health_disconnected() -> None:
    ch = ConnectionHealth(
        service="Kalshi WS",
        status="disconnected",
        last_message_age_s=45.0,
        detail="reconnect attempt 2/10",
    )
    assert ch.detail is not None


# ── SystemStatus ──────────────────────────────────────────────────────────────


def test_system_status() -> None:
    ss = SystemStatus(
        containers=[
            ContainerStatus(
                match_id="m1",
                status="PHASE3_RUNNING",
                heartbeat_age_s=1.0,
            )
        ],
        connections=[
            ConnectionHealth(service="PostgreSQL", status="connected"),
            ConnectionHealth(service="Redis", status="connected"),
        ],
        param_version=12,
        param_trained_at=_NOW,
        matches_since_retrain=23,
    )
    assert len(ss.containers) == 1
    assert len(ss.connections) == 2
    assert ss.param_version == 12


def test_system_status_empty() -> None:
    ss = SystemStatus(containers=[], connections=[])
    assert ss.param_version is None
    assert ss.param_trained_at is None
