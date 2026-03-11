"""Step 1.3 — XGBoost Poisson Prior for Base Intensities.

Trains an XGBoost model with ``count:poisson`` objective to predict
full-match expected goals (μ_H, μ_A) from a 4-tier feature vector.
The predicted μ values are converted to initial log-intensities:

    a_H = ln(μ_H / T_m),   a_A = ln(μ_A / T_m)

These serve as the prior for the joint NLL optimization in Step 1.4.

Reference: docs/phase1.md Step 1.3
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from src.clients.odds_api import build_odds_features
from src.common.types import IntervalRecord

# ---------------------------------------------------------------------------
# Default XGBoost hyperparameters for Poisson regression
# ---------------------------------------------------------------------------

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "count:poisson",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "poisson-nloglik",
    "seed": 42,
}
_DEFAULT_NUM_ROUNDS = 200

# Typical full-match duration (minutes) used when T_m is unknown.
_DEFAULT_T_M = 90.0

# League-average goals per team per match (fallback baseline).
_LEAGUE_AVG_GOALS = 1.3


# ---------------------------------------------------------------------------
# 4-Tier Feature Building
# ---------------------------------------------------------------------------


def build_match_features(
    match_stats: dict[str, Any],
    odds_bookmakers: dict[str, list[dict[str, Any]]] | None = None,
    *,
    is_home: bool = True,
    rest_days: float | None = None,
) -> dict[str, float]:
    """Build the full feature vector for one team in one match.

    Combines 4 feature tiers:
        Tier 1 — Team-level rolling stats (from Goalserve match stats)
        Tier 2 — Player-level aggregated features (from player_stats)
        Tier 3 — Odds features (from Odds-API, via build_odds_features)
        Tier 4 — Context features (home/away, rest days)

    Args:
        match_stats: Goalserve match dict with ``stats``, ``player_stats``, etc.
        odds_bookmakers: Odds-API bookmakers dict (or None if unavailable).
        is_home: Whether this is the home team's feature row.
        rest_days: Days since previous match (or None if unknown).

    Returns:
        Flat dict of feature_name → float value.
    """
    features: dict[str, float] = {}

    # Tier 1: Team-level stats
    team_key = "localteam" if is_home else "visitorteam"
    features.update(_build_tier1_features(match_stats, team_key))

    # Tier 2: Player-level aggregated stats
    features.update(_build_tier2_features(match_stats, team_key))

    # Tier 3: Odds features
    if odds_bookmakers:
        features.update(build_odds_features(odds_bookmakers))
    else:
        features.update(_zero_odds_features())

    # Tier 4: Context
    features["home_away_flag"] = 1.0 if is_home else 0.0
    features["rest_days"] = rest_days if rest_days is not None else 3.0

    return features


# ---------------------------------------------------------------------------
# Tier 1: Team-level stats
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _build_tier1_features(
    match_data: dict[str, Any],
    team_key: str,
) -> dict[str, float]:
    """Extract Tier 1 features from Goalserve team-level stats."""
    stats = match_data.get("stats", {}).get(team_key, {})
    if not stats:
        return _zero_tier1_features()

    shots_total = _safe_float(stats.get("shots", stats.get("shots_total", 0)))
    shots_on = _safe_float(stats.get("shots_ongoal", stats.get("ongoal", 0)))
    possession = _safe_float(stats.get("possestiontime", stats.get("possession", 0)))
    corners = _safe_float(stats.get("corners", 0))
    fouls = _safe_float(stats.get("fouls", 0))
    saves = _safe_float(stats.get("saves", 0))
    xg = _safe_float(stats.get("expected_goals", 0))

    passes_total = _safe_float(stats.get("passes", 0))
    # Some formats nest passes as dict
    if isinstance(stats.get("passes"), dict):
        passes_total = _safe_float(stats["passes"].get("total", 0))

    return {
        "xg": xg,
        "shots_total": shots_total,
        "shots_on_target": shots_on,
        "shots_insidebox_ratio": 0.0,  # requires insidebox data
        "possession": possession,
        "pass_accuracy": 0.0,  # requires accurate/total breakdown
        "corners": corners,
        "fouls": fouls,
        "saves": saves,
        "passes_total": passes_total,
    }


def _zero_tier1_features() -> dict[str, float]:
    return {
        "xg": 0.0,
        "shots_total": 0.0,
        "shots_on_target": 0.0,
        "shots_insidebox_ratio": 0.0,
        "possession": 0.0,
        "pass_accuracy": 0.0,
        "corners": 0.0,
        "fouls": 0.0,
        "saves": 0.0,
        "passes_total": 0.0,
    }


# ---------------------------------------------------------------------------
# Tier 2: Player-level aggregated stats
# ---------------------------------------------------------------------------


def _build_tier2_features(
    match_data: dict[str, Any],
    team_key: str,
) -> dict[str, float]:
    """Extract Tier 2 features from Goalserve player-level stats."""
    player_stats = match_data.get("player_stats", {}).get(team_key, {})
    if not player_stats:
        return _zero_tier2_features()

    players = player_stats.get("player", [])
    if isinstance(players, dict):
        players = [players]
    if not players:
        return _zero_tier2_features()

    ratings: list[float] = []
    for p in players:
        r = _safe_float(p.get("rating", p.get("@rating", 0)))
        if r > 0:
            ratings.append(r)

    return {
        "team_avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        "num_players_rated": float(len(ratings)),
    }


def _zero_tier2_features() -> dict[str, float]:
    return {
        "team_avg_rating": 0.0,
        "num_players_rated": 0.0,
    }


# ---------------------------------------------------------------------------
# Tier 3: Odds features (delegates to odds_api.build_odds_features)
# ---------------------------------------------------------------------------


def _zero_odds_features() -> dict[str, float]:
    """Return NaN for all odds features when odds are unavailable.

    XGBoost handles NaN natively as missing values, so partial odds
    coverage (e.g. only Dec 2025+ matches have odds) works correctly.
    """
    nan = float("nan")
    return {
        "exchange_home_prob": nan,
        "exchange_draw_prob": nan,
        "exchange_away_prob": nan,
        "market_avg_home_prob": nan,
        "market_avg_draw_prob": nan,
        "bookmaker_odds_std": nan,
    }


# ---------------------------------------------------------------------------
# Target extraction from intervals
# ---------------------------------------------------------------------------


def goals_from_intervals(
    intervals: list[IntervalRecord],
    match_id: str,
) -> tuple[int, int]:
    """Count home and away goals for a match from its intervals.

    Args:
        intervals: All intervals (may span multiple matches).
        match_id: The match ID to filter on.

    Returns:
        (home_goals, away_goals) tuple.
    """
    home = 0
    away = 0
    for iv in intervals:
        if iv.match_id != match_id:
            continue
        home += len(iv.home_goal_times)
        away += len(iv.away_goal_times)
    return home, away


# ---------------------------------------------------------------------------
# XGBoost Training
# ---------------------------------------------------------------------------


def train_poisson_xgb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    num_rounds: int = _DEFAULT_NUM_ROUNDS,
    feature_names: list[str] | None = None,
) -> xgb.Booster:
    """Train an XGBoost model with Poisson regression objective.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (goals scored, non-negative integers).
        params: XGBoost parameters (defaults to _DEFAULT_XGB_PARAMS).
        num_rounds: Number of boosting rounds.
        feature_names: Optional feature names for the DMatrix.

    Returns:
        Trained xgb.Booster model.
    """
    xgb_params = dict(_DEFAULT_XGB_PARAMS)
    if params:
        xgb_params.update(params)

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_rounds)
    return model


def predict_expected_goals(
    model: xgb.Booster,
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Predict full-match expected goals (μ) from features.

    Args:
        model: Trained XGBoost Poisson model.
        X: Feature matrix (n_samples, n_features).
        feature_names: Feature names matching training.

    Returns:
        Array of predicted μ values (shape: n_samples).
    """
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    return model.predict(dmat)


def mu_to_log_intensity(
    mu: float | np.ndarray,
    T_m: float = _DEFAULT_T_M,
) -> float | np.ndarray:
    """Convert full-match expected goals μ to log-intensity a.

    a = ln(μ / T_m)

    Args:
        mu: Predicted expected goals (scalar or array).
        T_m: Match duration in minutes.

    Returns:
        Log-intensity a (same shape as mu).
    """
    if isinstance(mu, np.ndarray):
        return np.log(np.clip(mu, 1e-6, None) / T_m)
    return math.log(max(mu, 1e-6) / T_m)


# ---------------------------------------------------------------------------
# Feature importance / mask
# ---------------------------------------------------------------------------


def select_features_by_importance(
    model: xgb.Booster,
    feature_names: list[str],
    *,
    cumulative_threshold: float = 0.95,
) -> list[str]:
    """Select top features reaching cumulative importance threshold.

    Uses Poisson deviance gain importance from the trained model.

    Args:
        model: Trained XGBoost model.
        feature_names: Full list of feature names.
        cumulative_threshold: Fraction of total importance to retain.

    Returns:
        List of selected feature names.
    """
    raw_importance = model.get_score(importance_type="gain")
    if not raw_importance:
        return list(feature_names)

    # Coerce values to float (get_score can return float | list[float])
    importance: dict[str, float] = {
        k: float(v) if isinstance(v, int | float) else float(v[0])
        for k, v in raw_importance.items()
    }

    # Sort by importance descending
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in sorted_feats)
    if total <= 0:
        return list(feature_names)

    selected: list[str] = []
    cumsum = 0.0
    for name, score in sorted_feats:
        selected.append(name)
        cumsum += score / total
        if cumsum >= cumulative_threshold:
            break

    return selected


def save_feature_mask(
    selected_features: list[str],
    path: str | Path,
) -> None:
    """Save selected feature names to a JSON file.

    Args:
        selected_features: List of feature names to keep.
        path: Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(selected_features, f, indent=2)


def load_feature_mask(path: str | Path) -> list[str]:
    """Load feature mask from a JSON file."""
    with open(path) as f:
        result: list[str] = json.load(f)
        return result


# ---------------------------------------------------------------------------
# Convenience: get feature column order
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # Tier 1
    "xg",
    "shots_total",
    "shots_on_target",
    "shots_insidebox_ratio",
    "possession",
    "pass_accuracy",
    "corners",
    "fouls",
    "saves",
    "passes_total",
    # Tier 2
    "team_avg_rating",
    "num_players_rated",
    # Tier 3
    "exchange_home_prob",
    "exchange_draw_prob",
    "exchange_away_prob",
    "market_avg_home_prob",
    "market_avg_draw_prob",
    "bookmaker_odds_std",
    # Tier 4
    "home_away_flag",
    "rest_days",
]


def features_to_array(
    feature_dicts: list[dict[str, float]],
    columns: list[str] | None = None,
) -> np.ndarray:
    """Convert a list of feature dicts to a numpy array.

    Args:
        feature_dicts: List of {feature_name: value} dicts.
        columns: Column order (defaults to FEATURE_COLUMNS).

    Returns:
        2D numpy array (n_samples, n_features).
    """
    cols = columns or FEATURE_COLUMNS
    rows = []
    for fd in feature_dicts:
        rows.append([fd.get(c, 0.0) for c in cols])
    return np.array(rows, dtype=np.float64)
