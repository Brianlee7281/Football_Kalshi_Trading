"""Tests for src/calibration/step_1_5_validation.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.step_1_5_validation import (
    ValidationResult,
    brier_score,
    brier_score_binary,
    calibrate_sanity_thresholds,
    compute_league_stratified_bs,
    encode_outcome_1x2,
    generate_walk_forward_folds,
    log_loss,
    poisson_1x2,
    poisson_over_under,
    run_validation,
    validate_delta_signs,
    validate_gamma_signs,
)


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_known_case(self) -> None:
        """P_model = [0.7, 0.2, 0.1], outcome = home_win → BS = 0.14."""
        preds = np.array([[0.7, 0.2, 0.1]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        bs = brier_score(preds, outcomes)
        expected = (0.7 - 1) ** 2 + (0.2 - 0) ** 2 + (0.1 - 0) ** 2
        assert bs == pytest.approx(expected, abs=1e-10)
        assert bs == pytest.approx(0.14, abs=1e-10)

    def test_perfect_prediction(self) -> None:
        """P = [1, 0, 0], outcome = home_win → BS = 0.0."""
        preds = np.array([[1.0, 0.0, 0.0]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        bs = brier_score(preds, outcomes)
        assert bs == pytest.approx(0.0, abs=1e-10)

    def test_uniform_predictions(self) -> None:
        """P = [0.33, 0.33, 0.33] → BS ≈ 0.67 (baseline)."""
        p = 1.0 / 3.0
        preds = np.array([[p, p, p]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        bs = brier_score(preds, outcomes)
        expected = (p - 1) ** 2 + p ** 2 + p ** 2
        assert bs == pytest.approx(expected, abs=1e-6)
        assert bs == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_worst_prediction(self) -> None:
        """P = [0, 0, 1], outcome = home_win → BS = 2.0."""
        preds = np.array([[0.0, 0.0, 1.0]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        bs = brier_score(preds, outcomes)
        assert bs == pytest.approx(2.0, abs=1e-10)

    def test_multiple_matches(self) -> None:
        """Average BS across multiple matches."""
        preds = np.array([
            [0.7, 0.2, 0.1],
            [1.0, 0.0, 0.0],
        ])
        outcomes = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        bs = brier_score(preds, outcomes)
        expected = (0.14 + 0.0) / 2
        assert bs == pytest.approx(expected, abs=1e-10)

    def test_empty_returns_zero(self) -> None:
        preds = np.zeros((0, 3))
        outcomes = np.zeros((0, 3))
        assert brier_score(preds, outcomes) == 0.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            brier_score(np.zeros((2, 3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Binary Brier Score
# ---------------------------------------------------------------------------


class TestBrierScoreBinary:
    def test_perfect(self) -> None:
        assert brier_score_binary(np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)

    def test_known(self) -> None:
        bs = brier_score_binary(np.array([0.7]), np.array([1.0]))
        assert bs == pytest.approx(0.09, abs=1e-10)


# ---------------------------------------------------------------------------
# Log Loss
# ---------------------------------------------------------------------------


class TestLogLoss:
    def test_perfect_prediction(self) -> None:
        preds = np.array([[1.0, 0.0, 0.0]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        ll = log_loss(preds, outcomes)
        assert ll == pytest.approx(0.0, abs=1e-6)

    def test_uniform(self) -> None:
        p = 1.0 / 3.0
        preds = np.array([[p, p, p]])
        outcomes = np.array([[1.0, 0.0, 0.0]])
        ll = log_loss(preds, outcomes)
        assert ll == pytest.approx(-np.log(p), abs=1e-6)


# ---------------------------------------------------------------------------
# Sanity thresholds
# ---------------------------------------------------------------------------


class TestSanityThresholds:
    def test_go_less_than_hold(self) -> None:
        """go_threshold < hold_threshold always (90th < 99th percentile)."""
        rng = np.random.default_rng(42)
        n = 200
        model = rng.dirichlet([2, 1, 1], size=n)
        exchange = rng.dirichlet([2, 1, 1], size=n)
        thresholds = calibrate_sanity_thresholds(model, exchange)
        assert thresholds.go_threshold < thresholds.hold_threshold

    def test_identical_preds_zero_thresholds(self) -> None:
        """If model == exchange, all deltas are 0."""
        preds = np.array([[0.5, 0.3, 0.2]] * 100)
        thresholds = calibrate_sanity_thresholds(preds, preds)
        assert thresholds.go_threshold == pytest.approx(0.0, abs=1e-10)
        assert thresholds.hold_threshold == pytest.approx(0.0, abs=1e-10)

    def test_empty_returns_zero(self) -> None:
        thresholds = calibrate_sanity_thresholds(
            np.zeros((0, 3)), np.zeros((0, 3)),
        )
        assert thresholds.n_matches == 0

    def test_with_varied_deltas(self) -> None:
        """Thresholds should reflect the distribution of discrepancies."""
        n = 1000
        rng = np.random.default_rng(123)
        model = rng.dirichlet([3, 2, 1], size=n)
        exchange = rng.dirichlet([3, 2, 1], size=n)
        thresholds = calibrate_sanity_thresholds(model, exchange)
        assert thresholds.go_threshold > 0
        assert thresholds.hold_threshold > thresholds.go_threshold
        assert thresholds.median_delta > 0
        assert thresholds.n_matches == n


# ---------------------------------------------------------------------------
# Outcome encoding
# ---------------------------------------------------------------------------


class TestEncodeOutcome:
    def test_home_win(self) -> None:
        out = encode_outcome_1x2(2, 1)
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0])

    def test_draw(self) -> None:
        out = encode_outcome_1x2(1, 1)
        np.testing.assert_array_equal(out, [0.0, 1.0, 0.0])

    def test_away_win(self) -> None:
        out = encode_outcome_1x2(0, 3)
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Poisson 1X2
# ---------------------------------------------------------------------------


class TestPoisson1X2:
    def test_sums_to_one(self) -> None:
        probs = poisson_1x2(1.5, 1.2)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_strong_home_favorite(self) -> None:
        probs = poisson_1x2(3.0, 0.5)
        assert probs[0] > probs[1] > probs[2]

    def test_equal_teams(self) -> None:
        probs = poisson_1x2(1.3, 1.3)
        assert probs[0] == pytest.approx(probs[2], abs=1e-6)
        assert probs[1] > 0


class TestPoissonOverUnder:
    def test_high_scoring(self) -> None:
        p_over = poisson_over_under(2.0, 2.0, 2.5)
        assert p_over > 0.5  # high-scoring match → likely over 2.5

    def test_low_scoring(self) -> None:
        p_over = poisson_over_under(0.5, 0.5, 2.5)
        assert p_over < 0.5  # low-scoring → likely under 2.5

    def test_range(self) -> None:
        p = poisson_over_under(1.5, 1.2, 2.5)
        assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


class TestWalkForwardFolds:
    def test_basic_folds(self) -> None:
        folds = generate_walk_forward_folds(6)
        in_season = [f for f in folds if f.fold_type == "in_season"]
        assert len(in_season) >= 1

    def test_cross_season_folds_with_5_plus(self) -> None:
        folds = generate_walk_forward_folds(5)
        cross = [f for f in folds if f.fold_type == "cross_season"]
        assert len(cross) == 5

    def test_no_cross_season_under_5(self) -> None:
        folds = generate_walk_forward_folds(4)
        cross = [f for f in folds if f.fold_type == "cross_season"]
        assert len(cross) == 0

    def test_no_leakage(self) -> None:
        """Validation season should never appear in training set."""
        folds = generate_walk_forward_folds(6)
        for f in folds:
            assert f.val_season not in f.train_seasons


# ---------------------------------------------------------------------------
# Gamma / delta sign validation
# ---------------------------------------------------------------------------


class TestGammaSignValidation:
    def test_correct_signs_pass(self) -> None:
        gamma_H = np.array([0.0, -0.3, 0.2, -0.1])
        gamma_A = np.array([0.0, 0.2, -0.3, -0.1])
        results = validate_gamma_signs(gamma_H, gamma_A)
        assert all(r.passes for r in results)

    def test_wrong_signs_fail(self) -> None:
        gamma_H = np.array([0.0, 0.3, -0.2, 0.1])  # wrong signs
        gamma_A = np.array([0.0, -0.2, 0.3, 0.1])
        results = validate_gamma_signs(gamma_H, gamma_A)
        assert not any(r.passes for r in results)


class TestDeltaSignValidation:
    def test_correct_signs_pass(self) -> None:
        delta_H = np.array([0.5, 0.2, 0.0, -0.2, -0.5])
        delta_A = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
        results = validate_delta_signs(delta_H, delta_A)
        assert all(r.passes for r in results)


# ---------------------------------------------------------------------------
# League-stratified BS
# ---------------------------------------------------------------------------


class TestLeagueStratifiedBS:
    def test_basic(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        league_ids = np.full(n, 1204, dtype=np.int64)  # all EPL (tier_1)
        model_preds = rng.dirichlet([3, 1, 1], size=n)
        baseline_preds = rng.dirichlet([2, 1, 1], size=n)
        outcomes = np.zeros((n, 3))
        for i in range(n):
            outcomes[i, rng.integers(0, 3)] = 1.0

        results = compute_league_stratified_bs(
            league_ids, model_preds, baseline_preds, outcomes,
        )
        assert len(results) >= 1
        assert results[0].tier == "tier_1"
        assert results[0].n_matches == n

    def test_too_few_matches_skipped(self) -> None:
        results = compute_league_stratified_bs(
            np.array([1204], dtype=np.int64),
            np.array([[0.5, 0.3, 0.2]]),
            np.array([[0.4, 0.3, 0.3]]),
            np.array([[1.0, 0.0, 0.0]]),
            min_matches=10,
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_model_beats_exchange(self) -> None:
        """When model is strictly better, go_decision should be True."""
        n = 100
        rng = np.random.default_rng(42)
        outcomes = np.zeros((n, 3))
        for i in range(n):
            outcomes[i, rng.integers(0, 3)] = 1.0

        # Model: closer to outcomes
        model_preds = 0.7 * outcomes + 0.1 * rng.dirichlet([1, 1, 1], size=n)
        model_preds /= model_preds.sum(axis=1, keepdims=True)

        # Exchange: noisier
        exchange_preds = 0.4 * outcomes + 0.2 * rng.dirichlet([1, 1, 1], size=n)
        exchange_preds /= exchange_preds.sum(axis=1, keepdims=True)

        result = run_validation(model_preds, exchange_preds, outcomes)
        assert result.delta_bs < 0
        assert result.go_decision is True

    def test_model_worse_than_exchange(self) -> None:
        """When exchange is better, go_decision should be False."""
        n = 100
        rng = np.random.default_rng(42)
        outcomes = np.zeros((n, 3))
        for i in range(n):
            outcomes[i, rng.integers(0, 3)] = 1.0

        # Exchange: closer to outcomes
        exchange_preds = 0.7 * outcomes + 0.1 * rng.dirichlet([1, 1, 1], size=n)
        exchange_preds /= exchange_preds.sum(axis=1, keepdims=True)

        # Model: noisier
        model_preds = 0.4 * outcomes + 0.2 * rng.dirichlet([1, 1, 1], size=n)
        model_preds /= model_preds.sum(axis=1, keepdims=True)

        result = run_validation(model_preds, exchange_preds, outcomes)
        assert result.delta_bs > 0
        assert result.go_decision is False
