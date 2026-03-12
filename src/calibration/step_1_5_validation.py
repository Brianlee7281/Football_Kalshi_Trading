"""Step 1.5 — Time-Series Cross-Validation and Model Diagnostics.

Validates the MMPP model against Betfair Exchange closing lines using
Brier Score, calibrates Phase 2 sanity check thresholds, and determines
Go/No-Go for production deployment.

Reference: docs/phase1.md Step 1.5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# League tiers (for stratified Brier Score)
# ---------------------------------------------------------------------------

LEAGUE_TIERS: dict[str, set[int]] = {
    "tier_1": {1204, 1399, 1229, 1269, 1221, 1440, 1141, 1081},
    "tier_2": {1007, 1352},
}


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def brier_score(
    predictions: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    """Compute the multi-class Brier Score.

    BS = (1/N) Σ_n Σ_c (P_n,c - O_n,c)²

    Args:
        predictions: (N, C) array of predicted probabilities per class.
        outcomes: (N, C) one-hot array of actual outcomes.

    Returns:
        Scalar Brier Score (lower is better, 0.0 = perfect).
    """
    if predictions.shape != outcomes.shape:
        msg = f"Shape mismatch: predictions {predictions.shape} vs outcomes {outcomes.shape}"
        raise ValueError(msg)
    n = predictions.shape[0]
    if n == 0:
        return 0.0
    return float(np.mean(np.sum((predictions - outcomes) ** 2, axis=1)))


def brier_score_binary(
    predictions: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
) -> float:
    """Compute binary Brier Score.

    BS = (1/N) Σ_n (P_n - O_n)²

    Args:
        predictions: (N,) predicted probabilities.
        outcomes: (N,) binary outcomes (0 or 1).

    Returns:
        Scalar Brier Score.
    """
    n = len(predictions)
    if n == 0:
        return 0.0
    return float(np.mean((predictions - outcomes) ** 2))


# ---------------------------------------------------------------------------
# Log Loss
# ---------------------------------------------------------------------------


def log_loss(
    predictions: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    *,
    eps: float = 1e-15,
) -> float:
    """Compute multi-class log loss.

    LogLoss = -(1/N) Σ_n Σ_c O_n,c · ln(P_n,c)

    Args:
        predictions: (N, C) predicted probabilities.
        outcomes: (N, C) one-hot outcomes.
        eps: Small constant to avoid log(0).

    Returns:
        Scalar log loss.
    """
    n = predictions.shape[0]
    if n == 0:
        return 0.0
    clipped = np.clip(predictions, eps, 1.0 - eps)
    return float(-np.mean(np.sum(outcomes * np.log(clipped), axis=1)))


# ---------------------------------------------------------------------------
# Outcome encoding
# ---------------------------------------------------------------------------


def encode_outcome_1x2(
    home_goals: int,
    away_goals: int,
) -> npt.NDArray[np.float64]:
    """Encode match result as a one-hot [H, D, A] vector.

    Args:
        home_goals: Final home goals.
        away_goals: Final away goals.

    Returns:
        Array of shape (3,) — one-hot [H, D, A].
    """
    out = np.zeros(3, dtype=np.float64)
    if home_goals > away_goals:
        out[0] = 1.0
    elif home_goals == away_goals:
        out[1] = 1.0
    else:
        out[2] = 1.0
    return out


# ---------------------------------------------------------------------------
# Model prediction → 1X2 probabilities (from Poisson μ_H, μ_A)
# ---------------------------------------------------------------------------


def poisson_1x2(
    mu_H: float,
    mu_A: float,
    *,
    max_goals: int = 10,
) -> npt.NDArray[np.float64]:
    """Compute Match Winner probabilities from Poisson intensities.

    Args:
        mu_H: Expected home goals.
        mu_A: Expected away goals.
        max_goals: Maximum goals to sum over.

    Returns:
        Array [P(Home), P(Draw), P(Away)].
    """
    from scipy.stats import poisson as poisson_dist

    p_h = 0.0
    p_d = 0.0
    p_a = 0.0

    for i in range(max_goals + 1):
        p_i = poisson_dist.pmf(i, mu_H)
        for j in range(max_goals + 1):
            p_j = poisson_dist.pmf(j, mu_A)
            p_ij = p_i * p_j
            if i > j:
                p_h += p_ij
            elif i == j:
                p_d += p_ij
            else:
                p_a += p_ij

    total = p_h + p_d + p_a
    if total > 0:
        return np.array([p_h / total, p_d / total, p_a / total])
    return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def poisson_over_under(
    mu_H: float,
    mu_A: float,
    line: float = 2.5,
    *,
    max_goals: int = 10,
) -> float:
    """Compute P(Over line) from Poisson intensities.

    Args:
        mu_H: Expected home goals.
        mu_A: Expected away goals.
        line: Goals threshold (e.g., 2.5).
        max_goals: Maximum goals to sum over.

    Returns:
        P(total goals > line).
    """
    from scipy.stats import poisson as poisson_dist

    p_under = 0.0
    threshold = int(line)
    for i in range(threshold + 1):
        for j in range(threshold + 1 - i):
            p_under += poisson_dist.pmf(i, mu_H) * poisson_dist.pmf(j, mu_A)

    return 1.0 - p_under


# ---------------------------------------------------------------------------
# Sanity check threshold calibration
# ---------------------------------------------------------------------------


@dataclass
class SanityThresholds:
    """Calibrated thresholds for Phase 2 Step 2.4 sanity check."""

    go_threshold: float  # 90th percentile — below = GO
    hold_threshold: float  # 99th percentile — above = SKIP, between = HOLD
    median_delta: float  # diagnostic
    n_matches: int


def calibrate_sanity_thresholds(
    model_preds: npt.NDArray[np.float64],
    exchange_preds: npt.NDArray[np.float64],
) -> SanityThresholds:
    """Calibrate Phase 2 sanity check thresholds from validation data.

    Computes per-match maximum discrepancy between model and exchange
    probabilities, then sets thresholds at empirical quantiles.

    Args:
        model_preds: (N, 3) model-predicted [H, D, A] probabilities.
        exchange_preds: (N, 3) Betfair Exchange close [H, D, A] probabilities.

    Returns:
        SanityThresholds with go/hold thresholds.
    """
    deltas = np.max(np.abs(model_preds - exchange_preds), axis=1)
    n = len(deltas)

    if n == 0:
        return SanityThresholds(
            go_threshold=0.0, hold_threshold=0.0, median_delta=0.0, n_matches=0,
        )

    return SanityThresholds(
        go_threshold=float(np.percentile(deltas, 90)),
        hold_threshold=float(np.percentile(deltas, 99)),
        median_delta=float(np.median(deltas)),
        n_matches=n,
    )


# ---------------------------------------------------------------------------
# League-stratified Brier Score
# ---------------------------------------------------------------------------


@dataclass
class LeagueTierResult:
    """Brier Score comparison for one league tier."""

    tier: str
    bs_model: float
    bs_baseline: float
    delta_bs: float  # model - baseline (negative = model better)
    n_matches: int


def compute_league_stratified_bs(
    league_ids: npt.NDArray[np.int64],
    model_preds: npt.NDArray[np.float64],
    baseline_preds: npt.NDArray[np.float64],
    outcomes: npt.NDArray[np.float64],
    *,
    min_matches: int = 10,
) -> list[LeagueTierResult]:
    """Compute ΔBS per league tier.

    Args:
        league_ids: (N,) league ID per match.
        model_preds: (N, 3) model predictions.
        baseline_preds: (N, 3) Betfair Exchange predictions.
        outcomes: (N, 3) one-hot outcomes.
        min_matches: Minimum matches required per tier.

    Returns:
        List of LeagueTierResult per qualifying tier.
    """
    results: list[LeagueTierResult] = []

    for tier_name, tier_ids in LEAGUE_TIERS.items():
        mask = np.array([lid in tier_ids for lid in league_ids])
        n = int(mask.sum())
        if n < min_matches:
            continue

        bs_model = brier_score(model_preds[mask], outcomes[mask])
        bs_base = brier_score(baseline_preds[mask], outcomes[mask])

        results.append(LeagueTierResult(
            tier=tier_name,
            bs_model=bs_model,
            bs_baseline=bs_base,
            delta_bs=bs_model - bs_base,
            n_matches=n,
        ))

    return results


# ---------------------------------------------------------------------------
# Walk-forward CV fold generation
# ---------------------------------------------------------------------------


@dataclass
class CVFold:
    """One cross-validation fold with train/validation indices."""

    fold_idx: int
    fold_type: str  # "in_season" or "cross_season"
    train_seasons: list[int]
    val_season: int


def generate_walk_forward_folds(
    n_seasons: int,
    *,
    min_train_seasons: int = 3,
) -> list[CVFold]:
    """Generate walk-forward and cross-season CV folds.

    Args:
        n_seasons: Total number of available seasons (numbered 1..n).
        min_train_seasons: Minimum seasons required for training.

    Returns:
        List of CVFold objects.
    """
    folds: list[CVFold] = []
    seasons = list(range(1, n_seasons + 1))
    fold_idx = 0

    # In-season walk-forward folds
    for val_start in range(min_train_seasons + 1, n_seasons + 1):
        train_start = max(1, val_start - min_train_seasons - (val_start - min_train_seasons - 1))
        train = list(range(train_start, val_start))
        if len(train) >= min_train_seasons:
            folds.append(CVFold(
                fold_idx=fold_idx,
                fold_type="in_season",
                train_seasons=train,
                val_season=val_start,
            ))
            fold_idx += 1

    # Cross-season folds (when 5+ seasons available)
    if n_seasons >= 5:
        for held_out in seasons:
            train = [s for s in seasons if s != held_out]
            if len(train) >= min_train_seasons:
                folds.append(CVFold(
                    fold_idx=fold_idx,
                    fold_type="cross_season",
                    train_seasons=train,
                    val_season=held_out,
                ))
                fold_idx += 1

    return folds


# ---------------------------------------------------------------------------
# Gamma / delta sign validation
# ---------------------------------------------------------------------------


@dataclass
class SignValidationResult:
    """Result of parameter sign validation."""

    param_name: str
    expected_sign: str  # "negative" or "positive"
    actual_value: float
    passes: bool


def validate_gamma_signs(
    gamma_H: npt.NDArray[np.float64],
    gamma_A: npt.NDArray[np.float64],
) -> list[SignValidationResult]:
    """Validate gamma parameter signs against football intuition.

    Expected:
        γ^H_1 < 0: home dismissal → home scoring down
        γ^H_2 > 0: away dismissal → home scoring up
        γ^A_1 > 0: home dismissal → away scoring up
        γ^A_2 < 0: away dismissal → away scoring down
    """
    return [
        SignValidationResult("gamma_H_1", "negative", float(gamma_H[1]), gamma_H[1] <= 0),
        SignValidationResult("gamma_H_2", "positive", float(gamma_H[2]), gamma_H[2] >= 0),
        SignValidationResult("gamma_A_1", "positive", float(gamma_A[1]), gamma_A[1] >= 0),
        SignValidationResult("gamma_A_2", "negative", float(gamma_A[2]), gamma_A[2] <= 0),
    ]


def validate_delta_signs(
    delta_H: npt.NDArray[np.float64],
    delta_A: npt.NDArray[np.float64],
) -> list[SignValidationResult]:
    """Validate delta lookup signs against football intuition.

    delta_H[-1] > 0: trailing home attacks more (ΔS=-1 → bin 1)
    delta_H[+1] < 0: leading home shifts defensive (ΔS=+1 → bin 3)
    delta_A[-1] < 0: leading away shifts defensive (ΔS=-1 → bin 1)
    delta_A[+1] > 0: trailing away attacks more (ΔS=+1 → bin 3)
    """
    return [
        SignValidationResult("delta_H(ΔS=-1)", "positive", float(delta_H[1]), delta_H[1] >= 0),
        SignValidationResult("delta_H(ΔS=+1)", "negative", float(delta_H[3]), delta_H[3] <= 0),
        SignValidationResult("delta_A(ΔS=-1)", "negative", float(delta_A[1]), delta_A[1] <= 0),
        SignValidationResult("delta_A(ΔS=+1)", "positive", float(delta_A[3]), delta_A[3] >= 0),
    ]


# ---------------------------------------------------------------------------
# Go/No-Go decision
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Complete validation output from Step 1.5."""

    bs_model: float
    bs_exchange: float
    delta_bs: float  # model - exchange (negative = model better)
    log_loss_model: float
    thresholds: SanityThresholds
    league_results: list[LeagueTierResult] = field(default_factory=list)
    gamma_sign_checks: list[SignValidationResult] = field(default_factory=list)
    delta_sign_checks: list[SignValidationResult] = field(default_factory=list)
    go_decision: bool = False
    reasons: list[str] = field(default_factory=list)


def run_validation(
    model_preds_1x2: npt.NDArray[np.float64],
    exchange_preds_1x2: npt.NDArray[np.float64],
    outcomes_1x2: npt.NDArray[np.float64],
    *,
    gamma_H: npt.NDArray[np.float64] | None = None,
    gamma_A: npt.NDArray[np.float64] | None = None,
    delta_H: npt.NDArray[np.float64] | None = None,
    delta_A: npt.NDArray[np.float64] | None = None,
    league_ids: npt.NDArray[np.int64] | None = None,
) -> ValidationResult:
    """Run full Step 1.5 validation.

    Args:
        model_preds_1x2: (N, 3) model [H, D, A] probabilities.
        exchange_preds_1x2: (N, 3) Betfair Exchange close [H, D, A].
        outcomes_1x2: (N, 3) one-hot actual outcomes.
        gamma_H: (4,) home red-card penalties (optional sign check).
        gamma_A: (4,) away red-card penalties (optional sign check).
        delta_H: (5,) home score-difference lookup (optional sign check).
        delta_A: (5,) away score-difference lookup (optional sign check).
        league_ids: (N,) league IDs for stratified BS (optional).

    Returns:
        ValidationResult with all metrics and Go/No-Go decision.
    """
    bs_model = brier_score(model_preds_1x2, outcomes_1x2)
    bs_exchange = brier_score(exchange_preds_1x2, outcomes_1x2)
    delta_bs = bs_model - bs_exchange
    ll_model = log_loss(model_preds_1x2, outcomes_1x2)

    thresholds = calibrate_sanity_thresholds(model_preds_1x2, exchange_preds_1x2)

    result = ValidationResult(
        bs_model=bs_model,
        bs_exchange=bs_exchange,
        delta_bs=delta_bs,
        log_loss_model=ll_model,
        thresholds=thresholds,
    )

    # Gamma sign validation
    if gamma_H is not None and gamma_A is not None:
        result.gamma_sign_checks = validate_gamma_signs(gamma_H, gamma_A)

    # Delta sign validation
    if delta_H is not None and delta_A is not None:
        result.delta_sign_checks = validate_delta_signs(delta_H, delta_A)

    # League-stratified BS
    if league_ids is not None:
        result.league_results = compute_league_stratified_bs(
            league_ids, model_preds_1x2, exchange_preds_1x2, outcomes_1x2,
        )

    # Go/No-Go
    reasons: list[str] = []
    go = True

    if delta_bs >= 0:
        reasons.append(f"ΔBS = {delta_bs:.4f} >= 0 (model does not beat exchange)")
        go = False
    else:
        reasons.append(f"ΔBS = {delta_bs:.4f} < 0 (model beats exchange)")

    if thresholds.go_threshold >= thresholds.hold_threshold:
        reasons.append("go_threshold >= hold_threshold (threshold inversion)")
        go = False

    if result.gamma_sign_checks:
        failed = [c for c in result.gamma_sign_checks if not c.passes]
        if failed:
            names = ", ".join(c.param_name for c in failed)
            reasons.append(f"gamma sign check failed: {names}")
            go = False

    result.go_decision = go
    result.reasons = reasons
    return result
