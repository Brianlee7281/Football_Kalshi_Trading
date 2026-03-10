"""Phase 1 Worker — Orchestrates Steps 1.1 → 1.5.

Runs the full offline calibration pipeline:
    1.1  Intervalize match data
    1.2  Estimate Q matrix (red-card transitions)
    1.3  XGBoost Poisson prior → initial a_H, a_A
    1.4  Joint NLL optimization → MMPP parameters
    1.5  Validation → Go/No-Go

Results are returned as a Phase1Result dataclass. In production,
this would write to the ``production_params`` DB table.

Reference: docs/implementation_roadmap.md Sprint 3 Task 3.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.calibration.step_1_1_intervals import build_intervals_from_goalserve
from src.calibration.step_1_2_Q_estimation import (
    apply_state3_additivity,
    estimate_Q_by_delta_S,
    estimate_Q_global,
)
from src.calibration.step_1_3_ml_prior import (
    _LEAGUE_AVG_GOALS,
    FEATURE_COLUMNS,
    build_match_features,
    features_to_array,
    goals_from_intervals,
    mu_to_log_intensity,
    predict_expected_goals,
    select_features_by_importance,
    train_poisson_xgb,
)
from src.calibration.step_1_4_nll_optimize import (
    OptimizationResult,
    optimize_nll,
    prepare_match_data,
)
from src.calibration.step_1_5_validation import (
    ValidationResult,
    encode_outcome_1x2,
    poisson_1x2,
    run_validation,
)
from src.common.logging import get_logger
from src.common.types import IntervalRecord

logger = get_logger("phase1_worker")

# ---------------------------------------------------------------------------
# Phase 1 result
# ---------------------------------------------------------------------------

_DEFAULT_SIGMA_A = 0.3
_DEFAULT_NUM_EPOCHS = 1000


@dataclass
class Phase1Result:
    """Complete output of Phase 1 calibration pipeline."""

    # Step 1.2: Q matrix
    Q_global: np.ndarray  # (4, 4)
    Q_by_delta_S: dict[int, np.ndarray]  # bin → (4, 4)

    # Step 1.3: XGBoost model info
    feature_mask: list[str]
    n_train_matches: int

    # Step 1.4: MMPP parameters
    optimization: OptimizationResult

    # Step 1.5: Validation
    validation: ValidationResult | None

    # Metadata
    league_id: int
    n_matches: int
    n_goals: int


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_1_1_intervalize(
    matches: list[dict[str, Any]],
) -> tuple[dict[str, list[IntervalRecord]], list[IntervalRecord]]:
    """Step 1.1: Convert match dicts to interval records.

    Returns:
        (intervals_by_match, all_intervals)
    """
    intervals_by_match: dict[str, list[IntervalRecord]] = {}
    all_intervals: list[IntervalRecord] = []

    for match in matches:
        try:
            ivs = build_intervals_from_goalserve(match)
        except Exception:
            continue
        if not ivs:
            continue
        mid = ivs[0].match_id
        intervals_by_match[mid] = ivs
        all_intervals.extend(ivs)

    logger.info(
        "step_1_1_complete",
        n_matches=len(intervals_by_match),
        n_intervals=len(all_intervals),
    )
    return intervals_by_match, all_intervals


def step_1_2_estimate_Q(
    all_intervals: list[IntervalRecord],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Step 1.2: Estimate Q matrix and per-ΔS-bin Q matrices."""
    Q_global = estimate_Q_global(all_intervals)
    Q_global = apply_state3_additivity(Q_global)
    Q_by_ds = estimate_Q_by_delta_S(all_intervals)

    n_red = sum(len(iv.red_card_transitions) for iv in all_intervals)
    logger.info("step_1_2_complete", n_red_cards=n_red)
    return Q_global, Q_by_ds


def step_1_3_ml_prior(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Step 1.3: Train XGBoost prior and compute initial a_H, a_A.

    When match_stats are available, uses the full 4-tier feature pipeline.
    Otherwise, falls back to MLE from observed goals.

    Args:
        intervals_by_match: Match intervals from Step 1.1.
        match_stats: Optional per-match Goalserve stats dicts.

    Returns:
        (match_ids, a_H_init, a_A_init, feature_mask)
    """
    match_ids = list(intervals_by_match.keys())

    if match_stats and _has_enough_features(match_stats):
        return _step_1_3_xgboost(intervals_by_match, match_stats, match_ids)

    return _step_1_3_mle_fallback(intervals_by_match, match_ids)


def _has_enough_features(match_stats: dict[str, dict[str, Any]]) -> bool:
    """Check if match stats have enough data for XGBoost training."""
    if len(match_stats) < 20:
        return False
    sample = next(iter(match_stats.values()))
    return bool(sample.get("stats"))


def _step_1_3_xgboost(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_stats: dict[str, dict[str, Any]],
    match_ids: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Full XGBoost pipeline when detailed stats are available."""
    feature_dicts_H: list[dict[str, float]] = []
    feature_dicts_A: list[dict[str, float]] = []
    y_H: list[float] = []
    y_A: list[float] = []
    valid_ids: list[str] = []

    for mid in match_ids:
        stats = match_stats.get(mid)
        if not stats:
            continue
        ivs = intervals_by_match[mid]
        h_goals, a_goals = goals_from_intervals(ivs, mid)

        feature_dicts_H.append(build_match_features(stats, None, is_home=True))
        feature_dicts_A.append(build_match_features(stats, None, is_home=False))
        y_H.append(float(h_goals))
        y_A.append(float(a_goals))
        valid_ids.append(mid)

    if len(valid_ids) < 20:
        return _step_1_3_mle_fallback(intervals_by_match, match_ids)

    X_H = features_to_array(feature_dicts_H)
    X_A = features_to_array(feature_dicts_A)
    X = np.vstack([X_H, X_A])
    y = np.concatenate([np.array(y_H), np.array(y_A)])

    model = train_poisson_xgb(X, y, feature_names=list(FEATURE_COLUMNS), num_rounds=200)
    feature_mask = select_features_by_importance(model, list(FEATURE_COLUMNS))

    mu_H = predict_expected_goals(model, X_H, feature_names=list(FEATURE_COLUMNS))
    mu_A = predict_expected_goals(model, X_A, feature_names=list(FEATURE_COLUMNS))

    T_m_arr = np.array([
        intervals_by_match[mid][0].T_m if intervals_by_match[mid][0].T_m > 0 else 90.0
        for mid in valid_ids
    ])
    a_H = np.asarray(mu_to_log_intensity(mu_H, T_m_arr))  # type: ignore[arg-type]
    a_A = np.asarray(mu_to_log_intensity(mu_A, T_m_arr))  # type: ignore[arg-type]

    logger.info("step_1_3_xgboost_complete", n_matches=len(valid_ids), n_features=len(feature_mask))
    return valid_ids, a_H, a_A, feature_mask


def _step_1_3_mle_fallback(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Fallback: estimate initial a from observed goals via MLE."""
    a_H_list: list[float] = []
    a_A_list: list[float] = []
    valid_ids: list[str] = []

    for mid in match_ids:
        ivs = intervals_by_match[mid]
        if not ivs:
            continue
        T_m = ivs[0].T_m if ivs[0].T_m > 0 else 90.0
        h_goals = sum(len(iv.home_goal_times) for iv in ivs)
        a_goals = sum(len(iv.away_goal_times) for iv in ivs)

        mu_H = max(h_goals, 0.1) if (h_goals + a_goals) > 0 else _LEAGUE_AVG_GOALS
        mu_A = max(a_goals, 0.1) if (h_goals + a_goals) > 0 else _LEAGUE_AVG_GOALS

        valid_ids.append(mid)
        a_H_list.append(math.log(mu_H / T_m))
        a_A_list.append(math.log(mu_A / T_m))

    logger.info("step_1_3_mle_fallback", n_matches=len(valid_ids))
    return valid_ids, np.array(a_H_list), np.array(a_A_list), list(FEATURE_COLUMNS)


def step_1_4_optimize(
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
    a_H_init: np.ndarray,
    a_A_init: np.ndarray,
    *,
    sigma_a: float = _DEFAULT_SIGMA_A,
    num_epochs: int = _DEFAULT_NUM_EPOCHS,
) -> OptimizationResult:
    """Step 1.4: Joint NLL optimization."""
    match_data = prepare_match_data(intervals_by_match, match_ids)
    result = optimize_nll(
        match_data, a_H_init, a_A_init,
        sigma_a=sigma_a, num_epochs=num_epochs,
    )

    logger.info(
        "step_1_4_complete",
        initial_nll=result.loss_history[0] if result.loss_history else 0,
        final_nll=result.loss_history[-1] if result.loss_history else 0,
        n_epochs=len(result.loss_history),
    )
    return result


def step_1_5_validate(
    optimization: OptimizationResult,
    intervals_by_match: dict[str, list[IntervalRecord]],
    match_ids: list[str],
    exchange_preds: np.ndarray | None = None,
) -> ValidationResult:
    """Step 1.5: Validate model and compute Go/No-Go.

    Args:
        optimization: Step 1.4 result.
        intervals_by_match: Match intervals.
        match_ids: Ordered match IDs matching optimization.a_H / a_A.
        exchange_preds: (N, 3) Betfair Exchange close [H, D, A]. If None,
            uses uniform baseline (allows pipeline to run without odds data).

    Returns:
        ValidationResult with Go/No-Go decision.
    """
    n = len(match_ids)

    # Derive model predictions from fitted μ_H, μ_A
    model_preds = np.zeros((n, 3))
    outcomes = np.zeros((n, 3))

    for i, mid in enumerate(match_ids):
        ivs = intervals_by_match[mid]
        T_m = ivs[0].T_m if ivs[0].T_m > 0 else 90.0

        mu_H = float(np.exp(optimization.a_H[i]) * T_m)
        mu_A = float(np.exp(optimization.a_A[i]) * T_m)
        model_preds[i] = poisson_1x2(mu_H, mu_A)

        h_goals = sum(len(iv.home_goal_times) for iv in ivs)
        a_goals = sum(len(iv.away_goal_times) for iv in ivs)
        outcomes[i] = encode_outcome_1x2(h_goals, a_goals)

    # Exchange predictions: use provided or uniform baseline
    if exchange_preds is None:
        exchange_preds = np.full((n, 3), 1.0 / 3.0)

    result = run_validation(
        model_preds, exchange_preds, outcomes,
        gamma_H=optimization.gamma_H,
        gamma_A=optimization.gamma_A,
        delta_H=optimization.delta_H,
        delta_A=optimization.delta_A,
    )

    logger.info(
        "step_1_5_complete",
        bs_model=result.bs_model,
        bs_exchange=result.bs_exchange,
        delta_bs=result.delta_bs,
        go_decision=result.go_decision,
    )
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_phase1(
    matches: list[dict[str, Any]],
    *,
    league_id: int = 0,
    match_stats: dict[str, dict[str, Any]] | None = None,
    exchange_preds: np.ndarray | None = None,
    sigma_a: float = _DEFAULT_SIGMA_A,
    num_epochs: int = _DEFAULT_NUM_EPOCHS,
    skip_validation: bool = False,
) -> Phase1Result:
    """Run the complete Phase 1 calibration pipeline.

    Args:
        matches: List of Goalserve match dicts.
        league_id: League identifier.
        match_stats: Optional per-match detailed stats (for XGBoost).
        exchange_preds: Optional (N, 3) Betfair Exchange close probs.
        sigma_a: ML prior regularization strength.
        num_epochs: NLL optimization epochs.
        skip_validation: Skip Step 1.5 (for testing).

    Returns:
        Phase1Result with all calibrated parameters.
    """
    logger.info("phase1_start", n_matches=len(matches), league_id=league_id)

    # Step 1.1: Intervalize
    intervals_by_match, all_intervals = step_1_1_intervalize(matches)

    n_goals = sum(
        len(iv.home_goal_times) + len(iv.away_goal_times)
        for iv in all_intervals
    )

    # Step 1.2: Q matrix
    Q_global, Q_by_ds = step_1_2_estimate_Q(all_intervals)

    # Step 1.3: ML prior
    match_ids, a_H_init, a_A_init, feature_mask = step_1_3_ml_prior(
        intervals_by_match, match_stats,
    )

    # Step 1.4: NLL optimization
    optimization = step_1_4_optimize(
        intervals_by_match, match_ids, a_H_init, a_A_init,
        sigma_a=sigma_a, num_epochs=num_epochs,
    )

    # Step 1.5: Validation
    validation = None
    if not skip_validation:
        validation = step_1_5_validate(
            optimization, intervals_by_match, match_ids, exchange_preds,
        )

    result = Phase1Result(
        Q_global=Q_global,
        Q_by_delta_S=Q_by_ds,
        feature_mask=feature_mask,
        n_train_matches=len(match_ids),
        optimization=optimization,
        validation=validation,
        league_id=league_id,
        n_matches=len(intervals_by_match),
        n_goals=n_goals,
    )

    logger.info(
        "phase1_complete",
        n_matches=result.n_matches,
        n_goals=result.n_goals,
        go_decision=validation.go_decision if validation else "skipped",
    )
    return result


# ---------------------------------------------------------------------------
# Serialization (for DB storage)
# ---------------------------------------------------------------------------


def params_to_json(result: Phase1Result) -> dict[str, Any]:
    """Serialize Phase1Result parameters to JSON-compatible dict.

    Suitable for the ``production_params.params`` JSONB column.
    """
    opt = result.optimization
    return {
        "b": opt.b.tolist(),
        "gamma_H": opt.gamma_H.tolist(),
        "gamma_A": opt.gamma_A.tolist(),
        "delta_H": opt.delta_H.tolist(),
        "delta_A": opt.delta_A.tolist(),
        "beta_H": opt.beta_H,
        "kappa_H": opt.kappa_H,
        "tau_H": opt.tau_H,
        "beta_A": opt.beta_A,
        "kappa_A": opt.kappa_A,
        "tau_A": opt.tau_A,
        "Q_global": result.Q_global.tolist(),
        "Q_by_delta_S": {
            str(k): v.tolist() for k, v in result.Q_by_delta_S.items()
        },
        "n_matches": result.n_matches,
        "n_goals": result.n_goals,
        "league_id": result.league_id,
    }


def validation_to_json(result: Phase1Result) -> dict[str, Any]:
    """Serialize validation results to JSON-compatible dict.

    Suitable for the ``production_params.validation`` JSONB column.
    """
    if result.validation is None:
        return {"status": "skipped"}

    v = result.validation
    return {
        "bs_model": v.bs_model,
        "bs_exchange": v.bs_exchange,
        "delta_bs": v.delta_bs,
        "log_loss_model": v.log_loss_model,
        "go_decision": v.go_decision,
        "reasons": v.reasons,
    }


def thresholds_to_json(result: Phase1Result) -> dict[str, Any]:
    """Serialize sanity thresholds to JSON-compatible dict.

    Suitable for the ``production_params.sanity_thresholds`` JSONB column.
    """
    if result.validation is None:
        return {}

    t = result.validation.thresholds
    return {
        "go_threshold": t.go_threshold,
        "hold_threshold": t.hold_threshold,
        "median_delta": t.median_delta,
        "n_matches": t.n_matches,
    }
