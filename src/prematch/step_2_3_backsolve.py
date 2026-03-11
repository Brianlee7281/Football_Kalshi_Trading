"""Step 2.3 — Back-Solving Baseline Intensity Parameter *a*.

Converts XGBoost-predicted expected goals (μ_H, μ_A) into the
log-intensity parameter *a* used by the live engine:

    a_H = ln(μ̂_H) − ln(C_time)
    a_A = ln(μ̂_A) − ln(C_time)

where C_time = Σ exp(b_i) · Δt_i accounts for the piecewise time
profile learned in Phase 1.

Reference: docs/phase2.md Step 2.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from src.common.logging import get_logger

logger = get_logger("step_2_3")

# Number of piecewise basis intervals (6 × 15-min)
_NUM_BASIS = 6


@dataclass
class BacksolveResult:
    """Output of Step 2.3 back-solving."""

    a_H: float
    a_A: float
    mu_H: float  # XGBoost predicted home expected goals
    mu_A: float  # XGBoost predicted away expected goals
    C_time: float  # Time profile constant
    T_exp: float  # Expected match duration


def compute_C_time(
    b: np.ndarray,
    alpha_1_mean: float = 2.0,
    alpha_2_mean: float = 4.0,
) -> float:
    """Compute the time profile constant C_time.

    C_time = Σ_{i=1}^{6} exp(b_i) · Δt_i

    where Δt_i are the interval widths in minutes, incorporating
    expected stoppage times for the 3rd and 6th intervals.

    Args:
        b: (6,) piecewise time profile from Phase 1 Step 1.4.
        alpha_1_mean: League mean 1st-half stoppage time (minutes).
        alpha_2_mean: League mean 2nd-half stoppage time (minutes).

    Returns:
        C_time (sum of weighted basis contributions).
    """
    # Interval widths: 15, 15, 15+α₁, 15, 15, 15+α₂
    dt = np.array([
        15.0,
        15.0,
        15.0 + alpha_1_mean,
        15.0,
        15.0,
        15.0 + alpha_2_mean,
    ])
    return float(np.sum(np.exp(b) * dt))


def compute_T_exp(
    alpha_1_mean: float = 2.0,
    alpha_2_mean: float = 4.0,
) -> float:
    """Compute expected match duration.

    T_exp = 90 + E[α₁] + E[α₂]
    """
    return 90.0 + alpha_1_mean + alpha_2_mean


def predict_match_goals(
    model: xgb.Booster,
    X_home: np.ndarray,
    X_away: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> tuple[float, float]:
    """Predict full-match expected goals using XGBoost Poisson model.

    Args:
        model: Trained XGBoost Poisson model from Phase 1.
        X_home: Home feature vector (1-D).
        X_away: Away feature vector (1-D).
        feature_names: Feature names matching training.

    Returns:
        (mu_H, mu_A) — predicted expected goals.
    """
    dmat_H = xgb.DMatrix(
        X_home.reshape(1, -1), feature_names=feature_names,
    )
    dmat_A = xgb.DMatrix(
        X_away.reshape(1, -1), feature_names=feature_names,
    )
    mu_H = float(model.predict(dmat_H)[0])
    mu_A = float(model.predict(dmat_A)[0])
    return mu_H, mu_A


def backsolve_intensity(
    mu_H: float,
    mu_A: float,
    b: np.ndarray,
    *,
    alpha_1_mean: float = 2.0,
    alpha_2_mean: float = 4.0,
) -> BacksolveResult:
    """Back-solve baseline intensity parameters a_H, a_A from predicted μ.

    At kickoff (X=0, ΔS=0), the intensity function simplifies to:
        λ_H(t) = exp(a_H + b_{i(t)})

    So full-match expected goals:
        μ̂_H = exp(a_H) · C_time
        → a_H = ln(μ̂_H) − ln(C_time)

    Args:
        mu_H: Predicted home expected goals.
        mu_A: Predicted away expected goals.
        b: (6,) piecewise time profile from Phase 1.
        alpha_1_mean: League mean 1st-half stoppage.
        alpha_2_mean: League mean 2nd-half stoppage.

    Returns:
        BacksolveResult with a_H, a_A, and supporting values.
    """
    C = compute_C_time(b, alpha_1_mean, alpha_2_mean)
    T_exp = compute_T_exp(alpha_1_mean, alpha_2_mean)

    # Clamp μ to avoid log(0)
    mu_H_safe = max(mu_H, 1e-6)
    mu_A_safe = max(mu_A, 1e-6)

    a_H = math.log(mu_H_safe) - math.log(C)
    a_A = math.log(mu_A_safe) - math.log(C)

    logger.info(
        "backsolve_complete",
        mu_H=round(mu_H, 4),
        mu_A=round(mu_A, 4),
        a_H=round(a_H, 4),
        a_A=round(a_A, 4),
        C_time=round(C, 4),
        T_exp=round(T_exp, 1),
    )

    return BacksolveResult(
        a_H=a_H,
        a_A=a_A,
        mu_H=mu_H,
        mu_A=mu_A,
        C_time=C,
        T_exp=T_exp,
    )


def backsolve_from_mle(
    home_goals: float,
    away_goals: float,
    b: np.ndarray,
    *,
    alpha_1_mean: float = 2.0,
    alpha_2_mean: float = 4.0,
) -> BacksolveResult:
    """Fallback back-solving from observed/expected goals (no XGBoost).

    Used when no XGBoost model is available (e.g. MLE fallback path).

    Args:
        home_goals: Expected home goals (from league average or prior).
        away_goals: Expected away goals.
        b: (6,) time profile.
        alpha_1_mean: League mean 1st-half stoppage.
        alpha_2_mean: League mean 2nd-half stoppage.

    Returns:
        BacksolveResult.
    """
    from src.calibration.step_1_3_ml_prior import _LEAGUE_AVG_GOALS

    mu_H = max(home_goals, 0.1) if home_goals > 0 else _LEAGUE_AVG_GOALS
    mu_A = max(away_goals, 0.1) if away_goals > 0 else _LEAGUE_AVG_GOALS

    return backsolve_intensity(
        mu_H, mu_A, b,
        alpha_1_mean=alpha_1_mean,
        alpha_2_mean=alpha_2_mean,
    )
