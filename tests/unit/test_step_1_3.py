"""Tests for src/calibration/step_1_3_ml_prior.py."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.calibration.step_1_3_ml_prior import (
    FEATURE_COLUMNS,
    _LEAGUE_AVG_GOALS,
    build_match_features,
    features_to_array,
    goals_from_intervals,
    load_feature_mask,
    mu_to_log_intensity,
    predict_expected_goals,
    save_feature_mask,
    select_features_by_importance,
    train_poisson_xgb,
)
from src.clients.odds_api import build_odds_features
from src.common.types import IntervalRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval(
    *,
    match_id: str = "m1",
    t_start: float = 0.0,
    t_end: float = 45.0,
    state_X: int = 0,
    delta_S: int = 0,
    home_goal_times: list[float] | None = None,
    away_goal_times: list[float] | None = None,
) -> IntervalRecord:
    return IntervalRecord(
        match_id=match_id,
        t_start=t_start,
        t_end=t_end,
        state_X=state_X,
        delta_S=delta_S,
        home_goal_times=home_goal_times or [],
        away_goal_times=away_goal_times or [],
    )


def _synthetic_training_data(n: int = 50) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic features and Poisson-distributed goal counts."""
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_COLUMNS)
    X = rng.uniform(0, 1, size=(n, n_features))
    # Poisson targets with mean ~1.3
    y = rng.poisson(lam=_LEAGUE_AVG_GOALS, size=n).astype(np.float64)
    return X, y, list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# build_odds_features (Tier 3 — from odds_api.py)
# ---------------------------------------------------------------------------


class TestBuildOddsFeaturesIntegration:
    """Test build_odds_features with Betfair Exchange odds as used by Step 1.3."""

    def test_betfair_exchange_home_prob(self) -> None:
        """Betfair Exchange home=1.44, draw=3.50, away=12.00 → exchange_home_prob ≈ 0.653."""
        bookmakers = {
            "Betfair Exchange": [
                {
                    "name": "ML",
                    "odds": [{"home": "1.44", "draw": "3.50", "away": "12.00"}],
                }
            ],
        }
        features = build_odds_features(bookmakers)
        assert features["exchange_home_prob"] == pytest.approx(0.653, abs=0.01)
        assert features["exchange_draw_prob"] > 0.0
        assert features["exchange_away_prob"] > 0.0

    def test_all_probs_sum_to_one(self) -> None:
        bookmakers = {
            "Betfair Exchange": [
                {
                    "name": "ML",
                    "odds": [{"home": "2.00", "draw": "3.50", "away": "4.00"}],
                }
            ],
        }
        features = build_odds_features(bookmakers)
        total = (
            features["exchange_home_prob"]
            + features["exchange_draw_prob"]
            + features["exchange_away_prob"]
        )
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_match_features
# ---------------------------------------------------------------------------


class TestBuildMatchFeatures:
    def test_returns_all_feature_columns(self) -> None:
        features = build_match_features({}, None, is_home=True)
        for col in FEATURE_COLUMNS:
            assert col in features

    def test_home_away_flag(self) -> None:
        home_feats = build_match_features({}, None, is_home=True)
        away_feats = build_match_features({}, None, is_home=False)
        assert home_feats["home_away_flag"] == 1.0
        assert away_feats["home_away_flag"] == 0.0

    def test_rest_days_default(self) -> None:
        features = build_match_features({}, None)
        assert features["rest_days"] == 3.0

    def test_rest_days_explicit(self) -> None:
        features = build_match_features({}, None, rest_days=5.0)
        assert features["rest_days"] == 5.0

    def test_tier1_stats_extracted(self) -> None:
        match_stats = {
            "stats": {
                "localteam": {
                    "shots": "12",
                    "shots_ongoal": "5",
                    "possestiontime": "55",
                    "corners": "7",
                    "fouls": "10",
                    "saves": "3",
                    "expected_goals": "1.8",
                }
            }
        }
        features = build_match_features(match_stats, None, is_home=True)
        assert features["shots_total"] == 12.0
        assert features["shots_on_target"] == 5.0
        assert features["possession"] == 55.0
        assert features["xg"] == 1.8

    def test_tier2_player_ratings(self) -> None:
        match_stats = {
            "player_stats": {
                "localteam": {
                    "player": [
                        {"rating": "7.5"},
                        {"rating": "6.8"},
                        {"rating": "8.0"},
                    ]
                }
            }
        }
        features = build_match_features(match_stats, None, is_home=True)
        expected_avg = (7.5 + 6.8 + 8.0) / 3
        assert features["team_avg_rating"] == pytest.approx(expected_avg, abs=1e-6)
        assert features["num_players_rated"] == 3.0

    def test_empty_stats_returns_zeros(self) -> None:
        features = build_match_features({}, None)
        assert features["xg"] == 0.0
        assert features["shots_total"] == 0.0
        assert features["team_avg_rating"] == 0.0


# ---------------------------------------------------------------------------
# goals_from_intervals
# ---------------------------------------------------------------------------


class TestGoalsFromIntervals:
    def test_counts_goals(self) -> None:
        intervals = [
            _make_interval(match_id="m1", home_goal_times=[23.0], away_goal_times=[]),
            _make_interval(match_id="m1", home_goal_times=[67.0], away_goal_times=[55.0, 80.0]),
        ]
        h, a = goals_from_intervals(intervals, "m1")
        assert h == 2
        assert a == 2

    def test_filters_by_match_id(self) -> None:
        intervals = [
            _make_interval(match_id="m1", home_goal_times=[10.0]),
            _make_interval(match_id="m2", home_goal_times=[20.0]),
        ]
        h, a = goals_from_intervals(intervals, "m1")
        assert h == 1
        assert a == 0

    def test_no_goals(self) -> None:
        intervals = [_make_interval(match_id="m1")]
        h, a = goals_from_intervals(intervals, "m1")
        assert h == 0
        assert a == 0


# ---------------------------------------------------------------------------
# features_to_array
# ---------------------------------------------------------------------------


class TestFeaturesToArray:
    def test_shape(self) -> None:
        dicts = [
            {c: float(i) for i, c in enumerate(FEATURE_COLUMNS)},
            {c: float(i + 1) for i, c in enumerate(FEATURE_COLUMNS)},
        ]
        arr = features_to_array(dicts)
        assert arr.shape == (2, len(FEATURE_COLUMNS))

    def test_missing_keys_default_to_zero(self) -> None:
        dicts = [{"xg": 1.5}]
        arr = features_to_array(dicts)
        assert arr[0, 0] == 1.5  # xg is first column
        assert arr[0, 1] == 0.0  # shots_total missing → 0

    def test_custom_columns(self) -> None:
        dicts = [{"a": 1.0, "b": 2.0}]
        arr = features_to_array(dicts, columns=["a", "b"])
        assert arr.shape == (1, 2)
        assert arr[0, 0] == 1.0
        assert arr[0, 1] == 2.0


# ---------------------------------------------------------------------------
# mu_to_log_intensity
# ---------------------------------------------------------------------------


class TestMuToLogIntensity:
    def test_scalar(self) -> None:
        mu = 1.3
        a = mu_to_log_intensity(mu, T_m=90.0)
        assert a == pytest.approx(math.log(1.3 / 90.0), abs=1e-10)

    def test_array(self) -> None:
        mu = np.array([1.0, 2.0])
        a = mu_to_log_intensity(mu, T_m=90.0)
        expected = np.log(mu / 90.0)
        np.testing.assert_allclose(a, expected, atol=1e-10)

    def test_near_zero_clipped(self) -> None:
        """Very small mu should be clipped, not produce -inf."""
        a = mu_to_log_intensity(0.0, T_m=90.0)
        assert np.isfinite(a)

    def test_array_near_zero_clipped(self) -> None:
        a = mu_to_log_intensity(np.array([0.0, 1e-10]), T_m=90.0)
        assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# XGBoost train + predict
# ---------------------------------------------------------------------------


class TestXGBoostTrainPredict:
    def test_predictions_positive(self) -> None:
        """XGBoost Poisson predictions should be > 0 for all samples."""
        X, y, feat_names = _synthetic_training_data(50)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=20)
        preds = predict_expected_goals(model, X, feature_names=feat_names)
        assert np.all(preds > 0)

    def test_predictions_in_range(self) -> None:
        """Predictions should be in reasonable range [0.0, 5.0]."""
        X, y, feat_names = _synthetic_training_data(100)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=50)
        preds = predict_expected_goals(model, X, feature_names=feat_names)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 5.0)

    def test_zero_features_near_league_average(self) -> None:
        """With all-zero features → predictions near league average (~1.3)."""
        X, y, feat_names = _synthetic_training_data(100)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=50)
        X_zero = np.zeros((1, len(FEATURE_COLUMNS)))
        preds = predict_expected_goals(model, X_zero, feature_names=feat_names)
        # Should be within reasonable range of league average
        assert preds[0] > 0.3
        assert preds[0] < 3.0

    def test_log_intensity_from_predictions(self) -> None:
        """Log-intensity a should be finite for all predictions."""
        X, y, feat_names = _synthetic_training_data(50)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=20)
        preds = predict_expected_goals(model, X, feature_names=feat_names)
        a = mu_to_log_intensity(preds, T_m=90.0)
        assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# Feature importance / mask
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_select_features_returns_subset(self) -> None:
        X, y, feat_names = _synthetic_training_data(100)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=50)
        selected = select_features_by_importance(model, feat_names, cumulative_threshold=0.95)
        assert len(selected) > 0
        assert len(selected) <= len(feat_names)
        for s in selected:
            assert s in feat_names

    def test_threshold_1_returns_all(self) -> None:
        X, y, feat_names = _synthetic_training_data(100)
        model = train_poisson_xgb(X, y, feature_names=feat_names, num_rounds=50)
        selected = select_features_by_importance(model, feat_names, cumulative_threshold=1.0)
        # At threshold=1.0, should include all features that have any importance
        assert len(selected) >= 1

    def test_save_and_load_mask(self) -> None:
        features = ["xg", "shots_total", "possession"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.json"
            save_feature_mask(features, path)
            loaded = load_feature_mask(path)
            assert loaded == features
