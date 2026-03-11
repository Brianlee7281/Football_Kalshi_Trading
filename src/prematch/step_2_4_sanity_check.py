"""Step 2.4 — Pre-Match Sanity Check.

Verifies that model-predicted probabilities do not deviate excessively
from market consensus (Betfair Exchange + market average).

Two checks:
  1. Primary: Match Winner vs Betfair Exchange + market average
  2. Secondary: Over/Under cross-validation

Verdicts: GO, GO_WITH_CAUTION, HOLD, SKIP

Reference: docs/phase2.md Step 2.4
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import poisson as poisson_dist

from src.calibration.step_1_5_validation import SanityThresholds, poisson_1x2
from src.common.logging import get_logger

logger = get_logger("step_2_4")


@dataclass
class SanityResult:
    """Output of Step 2.4 sanity check."""

    verdict: str  # GO | GO_WITH_CAUTION | HOLD | SKIP
    delta_match_winner: float = 0.0
    delta_over_under: float = 0.0
    mu_H: float = 0.0
    mu_A: float = 0.0
    model_probs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    exchange_probs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    warning: str | None = None


def primary_sanity_check(
    mu_H: float,
    mu_A: float,
    exchange_prob: tuple[float, float, float],
    market_avg: tuple[float, float, float],
    thresholds: SanityThresholds,
) -> tuple[str, float, float]:
    """Compare model probabilities vs Betfair Exchange + market average.

    Args:
        mu_H: Model predicted home expected goals.
        mu_A: Model predicted away expected goals.
        exchange_prob: (H, D, A) implied from Betfair Exchange.
        market_avg: (H, D, A) market average implied probabilities.
        thresholds: Calibrated thresholds from Phase 1 Step 1.5.

    Returns:
        (verdict, delta_exchange, delta_market)
    """
    P_model = poisson_1x2(mu_H, mu_A)

    # Max absolute discrepancy vs exchange
    delta_pin = max(
        abs(float(P_model[i]) - exchange_prob[i])
        for i in range(3)
    )

    # Max absolute discrepancy vs market average
    delta_mkt = max(
        abs(float(P_model[i]) - market_avg[i])
        for i in range(3)
    )

    go_thresh = thresholds.go_threshold
    hold_thresh = thresholds.hold_threshold

    if delta_pin < go_thresh:
        return "GO", delta_pin, delta_mkt

    if delta_pin < hold_thresh:
        # Deviates from Betfair Exchange but close to market average
        # → Betfair Exchange may be a temporary outlier
        if delta_mkt < go_thresh * 0.67:
            return "GO_WITH_CAUTION", delta_pin, delta_mkt
        return "HOLD", delta_pin, delta_mkt

    return "SKIP", delta_pin, delta_mkt


def secondary_sanity_check(
    mu_H: float,
    mu_A: float,
    ou_threshold: float,
    *,
    over_odds: float = 0.0,
    under_odds: float = 0.0,
) -> dict[str, float | bool]:
    """Cross-check model total goals vs Over/Under market.

    Detects cases where Match Winner alone misses
    "right total, wrong split" scenarios.

    Args:
        mu_H: Model home expected goals.
        mu_A: Model away expected goals.
        ou_threshold: Calibrated O/U threshold from Phase 1.
        over_odds: Over 2.5 decimal odds (0 if unavailable).
        under_odds: Under 2.5 decimal odds (0 if unavailable).

    Returns:
        Dict with P_model_over25, P_market_over25, delta_ou, ou_consistent.
    """
    mu_total = mu_H + mu_A

    # Model Over 2.5 probability
    P_model_over25 = 1.0 - float(poisson_dist.cdf(2, mu_total))

    # Market implied Over 2.5 probability
    P_market_over25 = 0.5  # default if odds unavailable
    if over_odds > 1.0 and under_odds > 1.0:
        ou_sum = 1.0 / over_odds + 1.0 / under_odds
        P_market_over25 = (1.0 / over_odds) / ou_sum

    delta_ou = abs(P_model_over25 - P_market_over25)
    ou_consistent = delta_ou < ou_threshold if ou_threshold > 0 else True

    return {
        "P_model_over25": P_model_over25,
        "P_market_over25": P_market_over25,
        "delta_ou": delta_ou,
        "ou_consistent": ou_consistent,
    }


def combined_sanity_check(
    mu_H: float,
    mu_A: float,
    thresholds: SanityThresholds,
    *,
    exchange_prob: tuple[float, float, float] | None = None,
    market_avg: tuple[float, float, float] | None = None,
    over_odds: float = 0.0,
    under_odds: float = 0.0,
    ou_threshold: float = 0.0,
) -> SanityResult:
    """Run combined primary + secondary sanity check.

    If odds data is unavailable, returns GO (optimistic fallback —
    the system can still run without odds-based sanity checking).

    Args:
        mu_H: Model home expected goals.
        mu_A: Model away expected goals.
        thresholds: Phase 1 calibrated thresholds.
        exchange_prob: Betfair Exchange implied (H, D, A).
        market_avg: Market average implied (H, D, A).
        over_odds: Over 2.5 decimal odds.
        under_odds: Under 2.5 decimal odds.
        ou_threshold: O/U discrepancy threshold.

    Returns:
        SanityResult with verdict and diagnostics.
    """
    P_model = poisson_1x2(mu_H, mu_A)
    model_probs = (float(P_model[0]), float(P_model[1]), float(P_model[2]))

    # If no odds available, return GO with warning
    if exchange_prob is None:
        logger.warning("no_exchange_odds", verdict="GO")
        return SanityResult(
            verdict="GO",
            mu_H=mu_H,
            mu_A=mu_A,
            model_probs=model_probs,
            warning="No exchange odds available — skipping sanity check",
        )

    if market_avg is None:
        market_avg = exchange_prob

    # Primary check
    primary_verdict, delta_pin, delta_mkt = primary_sanity_check(
        mu_H, mu_A, exchange_prob, market_avg, thresholds,
    )

    # Secondary check
    secondary = secondary_sanity_check(
        mu_H, mu_A, ou_threshold,
        over_odds=over_odds, under_odds=under_odds,
    )

    # Combined verdict
    ou_consistent = bool(secondary["ou_consistent"])
    delta_ou = float(secondary["delta_ou"])

    if primary_verdict == "SKIP":
        verdict = "SKIP"
        warning = f"Model-exchange delta {delta_pin:.3f} exceeds hold threshold"
    elif primary_verdict == "GO" and ou_consistent:
        verdict = "GO"
        warning = None
    elif primary_verdict == "GO" and not ou_consistent:
        verdict = "GO_WITH_CAUTION"
        warning = f"O/U mismatch (delta={delta_ou:.3f}) — mu ratio may be off"
    elif primary_verdict == "GO_WITH_CAUTION":
        verdict = "GO_WITH_CAUTION"
        warning = f"Exchange outlier (delta_pin={delta_pin:.3f}, delta_mkt={delta_mkt:.3f})"
    elif primary_verdict == "HOLD":
        verdict = "HOLD"
        warning = f"Model-exchange delta {delta_pin:.3f} in hold zone"
    else:
        verdict = "GO_WITH_CAUTION"
        warning = None

    result = SanityResult(
        verdict=verdict,
        delta_match_winner=delta_pin,
        delta_over_under=delta_ou,
        mu_H=mu_H,
        mu_A=mu_A,
        model_probs=model_probs,
        exchange_probs=exchange_prob,
        warning=warning,
    )

    logger.info(
        "sanity_check_complete",
        verdict=verdict,
        delta_pin=round(delta_pin, 4),
        delta_ou=round(delta_ou, 4),
        mu_H=round(mu_H, 3),
        mu_A=round(mu_A, 3),
    )

    return result
