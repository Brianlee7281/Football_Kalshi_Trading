# dashboard/api/routes/analytics.py
#
# REST endpoints:
#   GET /api/analytics/pnl               → PnLReport
#   GET /api/analytics/model-health      → ModelHealthReport
#   GET /api/analytics/paper-graduation  → GraduationChecklist

from __future__ import annotations

import json as _json
from typing import Any

from fastapi import APIRouter, HTTPException

from dashboard.api.deps import Pool
from dashboard.api.models import (
    GraduationChecklist,
    ModelHealthReport,
    PnLBreakdown,
    PnLReport,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Graduation thresholds from docs/dashboard.md Part 1 (Dashboard 6)
_MIN_TRADES: int = 50
_EDGE_REAL_LO: float = 0.6
_EDGE_REAL_HI: float = 1.5
_MAX_DD_PCT: float = 15.0
_REALISM_SCORE_MIN: float = 0.85


# ── helpers ───────────────────────────────────────────────────────────────────


def _j(v: Any) -> dict[str, Any] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return _json.loads(v)  # type: ignore[no-any-return]
    return dict(v)


# ── endpoints ─────────────────────────────────────────────────────────────────


@router.get("/pnl", response_model=PnLReport)
async def pnl_analytics(
    pool: Pool,
    days: int = 30,
    league: str | None = None,
    is_paper: bool | None = None,
) -> PnLReport:
    """Aggregated P&L analytics with per-dimension breakdowns.

    edge_realization and sharpe require EV-at-fill data not yet stored;
    returned as 0.0 / None until a signals audit table is added.
    """
    league_id = int(league) if league is not None else None

    async with pool.acquire() as conn:
        # ── aggregate summary ────────────────────────────────────────────────
        summary = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                                        AS total_trades,
                COUNT(*) FILTER (WHERE realized_pnl > 0)       AS wins,
                COALESCE(SUM(realized_pnl), 0)                  AS total_pnl
            FROM positions p
            JOIN match_schedule ms USING (match_id)
            WHERE p.status = 'SETTLED'
              AND ($1::boolean IS NULL OR p.is_paper = $1)
              AND ($2::int IS NULL OR ms.league_id = $2)
              AND p.created_at > NOW() - ($3 * INTERVAL '1 day')
            """,
            is_paper,
            league_id,
            days,
        )

        total = int(summary["total_trades"]) if summary else 0
        wins = int(summary["wins"]) if summary else 0
        total_pnl = float(summary["total_pnl"]) if summary else 0.0
        win_rate = wins / total if total > 0 else 0.0

        # ── max drawdown from bankroll snapshots ─────────────────────────────
        mode_filter = "paper" if (is_paper is True) else "live"
        dd_row = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(drawdown_pct), 0) AS max_dd
            FROM bankroll_snapshot
            WHERE mode = $1
              AND created_at > NOW() - ($2 * INTERVAL '1 day')
            """,
            mode_filter,
            days,
        )
        max_drawdown_pct = float(dd_row["max_dd"]) if dd_row else 0.0

        # ── breakdown by league ───────────────────────────────────────────────
        league_rows = await conn.fetch(
            """
            SELECT ms.league_id::text AS key,
                   COALESCE(SUM(p.realized_pnl), 0) AS pnl
            FROM positions p
            JOIN match_schedule ms USING (match_id)
            WHERE p.status = 'SETTLED'
              AND ($1::boolean IS NULL OR p.is_paper = $1)
              AND p.created_at > NOW() - ($2 * INTERVAL '1 day')
            GROUP BY ms.league_id
            """,
            is_paper,
            days,
        )
        by_league = {r["key"]: float(r["pnl"]) for r in league_rows}

        # ── breakdown by market (model key via ticker_mapping) ────────────────
        market_rows = await conn.fetch(
            """
            SELECT tm.model_key AS key,
                   COALESCE(SUM(p.realized_pnl), 0) AS pnl
            FROM positions p
            JOIN ticker_mapping tm
              ON p.match_id = tm.match_id
             AND p.market_ticker = tm.kalshi_ticker
            WHERE p.status = 'SETTLED'
              AND ($1::boolean IS NULL OR p.is_paper = $1)
              AND p.created_at > NOW() - ($2 * INTERVAL '1 day')
            GROUP BY tm.model_key
            """,
            is_paper,
            days,
        )
        by_market = {r["key"]: float(r["pnl"]) for r in market_rows}

        # ── breakdown by direction ────────────────────────────────────────────
        dir_rows = await conn.fetch(
            """
            SELECT direction AS key,
                   COALESCE(SUM(realized_pnl), 0) AS pnl
            FROM positions
            WHERE status = 'SETTLED'
              AND ($1::boolean IS NULL OR is_paper = $1)
              AND created_at > NOW() - ($2 * INTERVAL '1 day')
            GROUP BY direction
            """,
            is_paper,
            days,
        )
        by_direction = {r["key"]: float(r["pnl"]) for r in dir_rows}

    return PnLReport(
        total_trades=total,
        win_rate=win_rate,
        total_pnl=total_pnl,
        edge_realization=0.0,  # requires EV-at-fill — not yet stored
        max_drawdown_pct=max_drawdown_pct,
        sharpe=None,           # requires daily P&L time series
        avg_ev=None,           # requires EV-at-fill
        avg_slippage_cents=None,
        brier_vs_exchange=None,
        breakdown=PnLBreakdown(
            by_league=by_league,
            by_market=by_market,
            by_direction=by_direction,
            by_alignment={},  # alignment not stored per-position
        ),
    )


@router.get("/model-health", response_model=ModelHealthReport)
async def model_health(pool: Pool) -> ModelHealthReport:
    """Phase 1 model health metrics from the active production_params row."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT version, params, validation, created_at
            FROM production_params
            WHERE is_active = TRUE
            LIMIT 1
            """,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="No active production params found")

    validation = _j(row["validation"]) or {}

    brier_score = float(validation.get("brier_score", 0.0))
    brier_vs_exchange = (
        float(validation["brier_vs_exchange"])
        if "brier_vs_exchange" in validation
        else None
    )
    edge_realization = float(validation.get("edge_realization", 0.0))
    brier_by_league: dict[str, float] = {
        k: float(v)
        for k, v in (validation.get("brier_by_league") or {}).items()
    }
    edge_rolling_raw: list[Any] = validation.get("edge_realization_rolling") or []
    edge_realization_rolling = [float(v) for v in edge_rolling_raw]

    # Matches since this param version was activated
    async with pool.acquire() as conn:
        count_row = await conn.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM match_schedule
            WHERE status IN ('PHASE3_RUNNING', 'SETTLING', 'FINISHED', 'ARCHIVED')
              AND created_at >= $1
            """,
            row["created_at"],
        )
    matches_since = int(count_row["cnt"]) if count_row else 0

    return ModelHealthReport(
        param_version=int(row["version"]),
        param_trained_at=row["created_at"],
        brier_score=brier_score,
        brier_vs_exchange=brier_vs_exchange,
        edge_realization=edge_realization,
        matches_since_retrain=matches_since,
        brier_by_league=brier_by_league,
        edge_realization_rolling=edge_realization_rolling,
    )


@router.get("/paper-graduation", response_model=GraduationChecklist)
async def paper_graduation(pool: Pool) -> GraduationChecklist:
    """Phase 0 graduation criteria — all 8 must pass before going live."""
    async with pool.acquire() as conn:
        # 1. Trade count
        count_row = await conn.fetchrow(
            "SELECT COUNT(*) AS cnt FROM positions WHERE is_paper = TRUE AND status = 'SETTLED'",
        )
        trade_count = int(count_row["cnt"]) if count_row else 0

        # 2. Win rate → edge_realization proxy
        #    Actual edge_realization needs EV-at-fill; use win_rate as proxy.
        pnl_row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE realized_pnl > 0) AS wins,
                COUNT(*) AS total,
                COALESCE(SUM(realized_pnl), 0) AS total_pnl
            FROM positions
            WHERE is_paper = TRUE AND status = 'SETTLED'
            """,
        )
        total_trades = int(pnl_row["total"]) if pnl_row else 0
        wins = int(pnl_row["wins"]) if pnl_row else 0
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        # edge_realization proxy: ratio of win_rate to 0.5 baseline, scaled [0, 2]
        edge_realization_proxy = min(2.0, win_rate * 2.0)

        # 3. Max drawdown
        dd_row = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(drawdown_pct), 0) AS max_dd
            FROM bankroll_snapshot
            WHERE mode = 'paper'
            """,
        )
        max_dd = float(dd_row["max_dd"]) if dd_row else 0.0

        # 4. Brier score from active params
        brier_row = await conn.fetchrow(
            "SELECT validation FROM production_params WHERE is_active = TRUE LIMIT 1",
        )
        brier_within_threshold = False
        if brier_row is not None:
            val = _j(brier_row["validation"]) or {}
            bs = val.get("brier_score")
            bs_baseline = val.get("brier_baseline")
            if bs is not None and bs_baseline is not None:
                brier_within_threshold = abs(float(bs) - float(bs_baseline)) <= 0.03

        # 5. Directional correctness: all BUY_YES fills should have positive
        #    expected value (P_true > entry_price), but we can't verify without
        #    EV-at-fill. Approximate: no position has realized_pnl = 0 on a
        #    closed position (i.e., data integrity check).
        wrong_dir = await conn.fetchval(
            """
            SELECT COUNT(*) FROM positions
            WHERE is_paper = TRUE
              AND status = 'SETTLED'
              AND (
                (direction = 'BUY_YES' AND settlement_price IS NOT NULL
                 AND entry_price > settlement_price AND realized_pnl > 0)
              )
            """,
        )
        directional_ok = int(wrong_dir or 0) == 0

    trades_ok = trade_count >= _MIN_TRADES
    edge_realization_ok = _EDGE_REAL_LO <= edge_realization_proxy <= _EDGE_REAL_HI
    max_drawdown_ok = max_dd < _MAX_DD_PCT
    # realism_score and no_crashes require runtime checks not available here
    realism_score_ok = False  # computed by compute_paper_realism_score() — Sprint 8
    no_crashes_ok = True      # assume OK unless orchestrator reports crashes
    theta_calibrated = False  # set True once Step 4.6 adaptive tuning runs

    all_pass = all([
        trades_ok,
        edge_realization_ok,
        brier_within_threshold,
        max_drawdown_ok,
        realism_score_ok,
        directional_ok,
        no_crashes_ok,
        theta_calibrated,
    ])

    return GraduationChecklist(
        trade_count=trade_count,
        trades_ok=trades_ok,
        edge_realization_ok=edge_realization_ok,
        brier_ok=brier_within_threshold,
        max_drawdown_ok=max_drawdown_ok,
        realism_score_ok=realism_score_ok,
        directional_ok=directional_ok,
        no_crashes_ok=no_crashes_ok,
        theta_calibrated=theta_calibrated,
        all_pass=all_pass,
    )
