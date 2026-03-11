"""Step 2.2 — Feature Selection (Apply Phase 1 Feature Mask).

Extracts only the features selected by Phase 1's XGBoost importance
analysis, producing a feature vector with identical dimensionality
and ordering as Phase 1 training data.

Missing values are replaced with Phase 1 training medians.

Reference: docs/phase2.md Step 2.2
"""

from __future__ import annotations

import math

import numpy as np

from src.prematch.step_2_1_data_collection import PreMatchData


def apply_feature_mask(
    pre_match: PreMatchData,
    feature_mask: list[str],
    median_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Extract features matching Phase 1 feature_mask from pre-match data.

    Because Phase 1 and Phase 2 both use the same Goalserve schema,
    no manual feature-name mapping layer is required.

    Args:
        pre_match: Collected pre-match data from Step 2.1.
        feature_mask: Ordered list of feature names from Phase 1 Step 1.3.
        median_values: Phase 1 training median per feature (for imputation).
            If None, uses 0.0 for missing values.

    Returns:
        1-D numpy array of shape ``(len(feature_mask),)`` — the feature
        vector X_match ready for XGBoost inference.
    """
    medians = median_values or {}

    # Build the full (unmasked) feature vector from all tiers
    full_vec = _build_full_feature_vector(pre_match)

    # Apply mask: select only features in feature_mask, in order
    selected: list[float] = []
    for feat_name in feature_mask:
        val = full_vec.get(feat_name)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            selected.append(val)
        else:
            # Replace missing with median from Phase 1 training data
            selected.append(medians.get(feat_name, 0.0))

    return np.array(selected, dtype=np.float64)


def _build_full_feature_vector(pre_match: PreMatchData) -> dict[str, float]:
    """Build a complete feature dict from all 4 tiers of pre-match data.

    Feature names follow the same conventions as Phase 1 Step 1.3
    ``FEATURE_COLUMNS`` to ensure compatibility.
    """
    full_vec: dict[str, float] = {}

    # Tier 1: team rolling stats (direct — no home/away prefix needed,
    # because each team's stats are already from their perspective)
    for k, v in pre_match.home_team_rolling.items():
        full_vec[k] = v
    # Override with away if building away perspective (handled by caller)

    # Tier 1 also includes the basic stats features expected by Phase 1
    _merge_tier1_from_rolling(full_vec, pre_match.home_team_rolling)

    # Tier 2: player aggregates
    for k, v in pre_match.home_player_agg.items():
        full_vec[f"home_{k}"] = v
    for k, v in pre_match.away_player_agg.items():
        full_vec[f"away_{k}"] = v

    # Also map team_avg_rating to Phase 1's feature name
    full_vec["team_avg_rating"] = pre_match.home_player_agg.get(
        "team_avg_rating", 0.0,
    )
    full_vec["num_players_rated"] = 11.0  # always 11 starters

    # Tier 3: odds (not team-specific)
    for k, v in pre_match.odds_features.items():
        if not k.startswith("_"):  # exclude internal fields
            full_vec[k] = v

    # Tier 4: context
    full_vec["home_away_flag"] = 1.0  # always home perspective
    full_vec["rest_days"] = float(pre_match.home_rest_days)
    full_vec["home_rest_days"] = float(pre_match.home_rest_days)
    full_vec["away_rest_days"] = float(pre_match.away_rest_days)
    full_vec["h2h_goal_diff"] = pre_match.h2h_goal_diff

    return full_vec


def _merge_tier1_from_rolling(
    full_vec: dict[str, float],
    rolling: dict[str, float],
) -> None:
    """Map team rolling stats to Phase 1 Tier 1 feature names.

    Phase 1 uses feature names like ``xg``, ``shots_total``, etc.
    Team rolling stats from Step 2.1.4 use names like ``xg_per_90``.
    """
    mapping = {
        "xg": "xg_per_90",
        "shots_total": "shots_per_90",
        "possession": "possession_avg",
        "pass_accuracy": "pass_accuracy",
        "corners": "corners_per_90",
        "fouls": "fouls_per_90",
        "shots_insidebox_ratio": "shots_insidebox_ratio",
    }
    for phase1_name, rolling_name in mapping.items():
        if rolling_name in rolling:
            full_vec[phase1_name] = rolling[rolling_name]


def build_away_feature_vector(
    pre_match: PreMatchData,
    feature_mask: list[str],
    median_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Build feature vector for the away team's perspective.

    Mirrors ``apply_feature_mask`` but swaps home/away context.

    Args:
        pre_match: Collected pre-match data.
        feature_mask: Phase 1 feature mask.
        median_values: Phase 1 training medians.

    Returns:
        Away-perspective feature vector.
    """
    medians = median_values or {}

    full_vec: dict[str, float] = {}

    # Tier 1: use away team's rolling stats
    _merge_tier1_from_rolling(full_vec, pre_match.away_team_rolling)

    # Tier 2: swap home/away aggregates
    full_vec["team_avg_rating"] = pre_match.away_player_agg.get(
        "team_avg_rating", 0.0,
    )
    full_vec["num_players_rated"] = 11.0

    # Tier 3: same odds
    for k, v in pre_match.odds_features.items():
        if not k.startswith("_"):
            full_vec[k] = v

    # Tier 4: away perspective
    full_vec["home_away_flag"] = 0.0
    full_vec["rest_days"] = float(pre_match.away_rest_days)
    full_vec["home_rest_days"] = float(pre_match.home_rest_days)
    full_vec["away_rest_days"] = float(pre_match.away_rest_days)
    full_vec["h2h_goal_diff"] = -pre_match.h2h_goal_diff

    selected: list[float] = []
    for feat_name in feature_mask:
        val = full_vec.get(feat_name)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            selected.append(val)
        else:
            selected.append(medians.get(feat_name, 0.0))

    return np.array(selected, dtype=np.float64)
