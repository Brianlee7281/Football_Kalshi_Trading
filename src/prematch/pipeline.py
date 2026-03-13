"""Phase 2 Pipeline — Pre-Match Initialization.

Orchestrates Steps 2.1 → 2.5 to produce an EngineState ready for
Phase 3 live trading. Called ~60 minutes before kickoff.

Pipeline:
    2.1  Data collection (lineups, stats, odds, context)
    2.2  Feature selection (apply Phase 1 mask)
    2.3  Back-solve intensity parameter a
    2.4  Sanity check vs market
    2.5  Engine initialization (load params, precompute grids)

Reference: docs/phase2.md, docs/implementation_roadmap.md Task 4.1
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

from src.calibration.step_1_5_validation import SanityThresholds
from src.clients.goalserve import GoalserveClient
from src.clients.odds_api import OddsApiClient
from src.common.logging import get_logger
from src.common.types import Phase2Result
from src.prematch.step_2_1_data_collection import (
    collect_pre_match_data,
)
from src.prematch.step_2_2_feature_selection import (
    apply_feature_mask,
    build_away_feature_vector,
)
from src.prematch.step_2_3_backsolve import (
    BacksolveResult,
    backsolve_from_mle,
    backsolve_intensity,
    predict_match_goals,
)
from src.prematch.step_2_4_sanity_check import (
    SanityResult,
    combined_sanity_check,
)
from src.prematch.step_2_5_initialization import (
    EngineState,
    initialize_engine,
)

logger = get_logger("phase2_pipeline")


async def run_phase2(
    gs_client: GoalserveClient,
    match_id: str,
    league_id: int,
    params: dict[str, Any],
    *,
    odds_client: OddsApiClient | None = None,
    odds_event_id: str | None = None,
    xgb_model_path: str | None = None,
    feature_mask: list[str] | None = None,
    median_values: dict[str, float] | None = None,
    thresholds: SanityThresholds | None = None,
    kickoff_date: str = "",
    recent_home_stats: list[dict[str, Any]] | None = None,
    recent_away_stats: list[dict[str, Any]] | None = None,
    home_fixtures: list[dict[str, Any]] | None = None,
    away_fixtures: list[dict[str, Any]] | None = None,
    h2h_matches: list[dict[str, Any]] | None = None,
    alpha_1_mean: float = 2.0,
    alpha_2_mean: float = 4.0,
    bankroll: float = 0.0,
    delta_significant: bool = False,
) -> tuple[Phase2Result, EngineState]:
    """Run the complete Phase 2 pipeline for a single match.

    Args:
        gs_client: Active Goalserve client.
        match_id: Goalserve match ID.
        league_id: Goalserve league ID.
        params: Phase 1 production_params JSON (b, gamma, delta, Q, etc.).
        odds_client: Optional Odds-API client.
        odds_event_id: Odds-API event ID for this match.
        xgb_model_path: Path to trained XGBoost model file.
        feature_mask: Phase 1 feature mask (ordered feature names).
        median_values: Phase 1 training medians for imputation.
        thresholds: Phase 1 calibrated sanity thresholds.
        kickoff_date: Match date string.
        recent_home_stats: Pre-fetched recent home match stats.
        recent_away_stats: Pre-fetched recent away match stats.
        home_fixtures: Pre-fetched home team fixtures.
        away_fixtures: Pre-fetched away team fixtures.
        h2h_matches: Pre-fetched H2H fixtures.
        alpha_1_mean: League mean 1st-half stoppage (from Phase 1).
        alpha_2_mean: League mean 2nd-half stoppage (from Phase 1).
        bankroll: Current account balance.
        delta_significant: Phase 1 LRT result.

    Returns:
        (Phase2Result, EngineState) — the Phase2Result for logging/DB,
        and the EngineState ready for Phase 3.
    """
    logger.info("phase2_start", match_id=match_id, league_id=league_id)

    # ---------------------------------------------------------------
    # Step 2.1: Data collection
    # ---------------------------------------------------------------
    logger.info("step_2_1_start", match_id=match_id)
    pre_match = await collect_pre_match_data(
        gs_client,
        odds_client,
        match_id,
        league_id,
        odds_event_id=odds_event_id,
        kickoff_date=kickoff_date,
        recent_home_stats=recent_home_stats,
        recent_away_stats=recent_away_stats,
        home_fixtures=home_fixtures,
        away_fixtures=away_fixtures,
        h2h_matches=h2h_matches,
    )
    logger.info("step_2_1_complete", match_id=match_id)

    # ---------------------------------------------------------------
    # Step 2.2: Feature selection
    # ---------------------------------------------------------------
    logger.info("step_2_2_start", match_id=match_id)
    b = np.array(params["b"], dtype=np.float64)

    # ---------------------------------------------------------------
    # Step 2.3: Back-solve a
    # ---------------------------------------------------------------
    backsolve: BacksolveResult

    has_xgb_model = (
        xgb_model_path
        and feature_mask
        and xgb_model_path != "mle_fallback"
    )
    if has_xgb_model:
        # XGBoost path
        X_home = apply_feature_mask(pre_match, feature_mask, median_values)
        X_away = build_away_feature_vector(
            pre_match, feature_mask, median_values,
        )

        model = xgb.Booster()
        model.load_model(xgb_model_path)

        mu_H, mu_A = predict_match_goals(
            model, X_home, X_away, feature_names=feature_mask,
        )

        backsolve = backsolve_intensity(
            mu_H, mu_A, b,
            alpha_1_mean=alpha_1_mean,
            alpha_2_mean=alpha_2_mean,
        )

        logger.info(
            "step_2_3_xgboost",
            mu_H=round(mu_H, 4),
            mu_A=round(mu_A, 4),
        )
    else:
        # MLE fallback (no XGBoost model)
        from src.calibration.step_1_3_ml_prior import _LEAGUE_AVG_GOALS

        backsolve = backsolve_from_mle(
            _LEAGUE_AVG_GOALS, _LEAGUE_AVG_GOALS, b,
            alpha_1_mean=alpha_1_mean,
            alpha_2_mean=alpha_2_mean,
        )

        logger.info(
            "step_2_3_mle_fallback",
            mu_H=round(backsolve.mu_H, 4),
            mu_A=round(backsolve.mu_A, 4),
        )

    # ---------------------------------------------------------------
    # Step 2.4: Sanity check
    # ---------------------------------------------------------------
    sanity: SanityResult

    if thresholds:
        exchange_prob = _extract_exchange_prob(pre_match.odds_features)
        market_avg = _extract_market_avg(pre_match.odds_features)

        sanity = combined_sanity_check(
            backsolve.mu_H,
            backsolve.mu_A,
            thresholds,
            exchange_prob=exchange_prob,
            market_avg=market_avg,
        )
    else:
        # No thresholds → skip sanity check
        sanity = SanityResult(
            verdict="GO",
            mu_H=backsolve.mu_H,
            mu_A=backsolve.mu_A,
            warning="No sanity thresholds available — skipping check",
        )

    # ---------------------------------------------------------------
    # Step 2.5: Engine initialization
    # ---------------------------------------------------------------
    engine_state = initialize_engine(
        match_id,
        params,
        backsolve,
        sanity,
        bankroll=bankroll,
        delta_significant=delta_significant,
    )

    # Build Phase2Result for logging / DB storage
    phase2_result = Phase2Result(
        a_H=backsolve.a_H,
        a_A=backsolve.a_A,
        C_time=backsolve.C_time,
        verdict=sanity.verdict,
        pre_match_data={
            "mu_H": backsolve.mu_H,
            "mu_A": backsolve.mu_A,
            "T_exp": backsolve.T_exp,
            "home_formation": pre_match.home_formation,
            "away_formation": pre_match.away_formation,
            "home_starting_11_count": len(pre_match.home_starting_11),
            "away_starting_11_count": len(pre_match.away_starting_11),
            "delta_match_winner": sanity.delta_match_winner,
            "delta_over_under": sanity.delta_over_under,
        },
        warning=sanity.warning,
    )

    logger.info(
        "phase2_complete",
        match_id=match_id,
        verdict=sanity.verdict,
        a_H=round(backsolve.a_H, 4),
        a_A=round(backsolve.a_A, 4),
    )

    return phase2_result, engine_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_exchange_prob(
    odds_features: dict[str, float],
) -> tuple[float, float, float] | None:
    """Extract Betfair Exchange implied probs from odds features."""
    h = odds_features.get("exchange_home_prob", 0.0)
    d = odds_features.get("exchange_draw_prob", 0.0)
    a = odds_features.get("exchange_away_prob", 0.0)
    if h > 0 and d > 0 and a > 0:
        return (h, d, a)
    return None


def _extract_market_avg(
    odds_features: dict[str, float],
) -> tuple[float, float, float] | None:
    """Extract market average implied probs from odds features."""
    h = odds_features.get("market_avg_home_prob", 0.0)
    d = odds_features.get("market_avg_draw_prob", 0.0)
    # market_avg_draw_prob + home + away should sum to ~1
    a = 1.0 - h - d if (h > 0 and d > 0) else 0.0
    if h > 0 and d > 0 and a > 0:
        return (h, d, a)
    return None
