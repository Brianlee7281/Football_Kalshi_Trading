"""Monte Carlo pricing layer — async wrapper + market aggregation.

Bridges the Numba JIT mc_core with the async tick loop:
  - step_3_4_async: analytical vs MC branching, executor decoupling
  - aggregate_markets: final_scores → dict of market probabilities
  - compute_mc_stderr: per-market σ_MC with analytical floor

Reference: docs/phase3.md §Logic B, §Market Probability Estimation
"""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import poisson as poisson_dist

from src.common.logging import get_logger
from src.engine.mc_core import mc_simulate_remaining
from src.engine.model import EVENT_PRELIMINARY

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("mc_pricing")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_MC = 50_000  # Monte Carlo paths per pricing call
_SIGMA_FLOOR = 0.005  # Analytical mode σ_MC floor

mc_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mc")

# MC version counter (module-level, keyed per model instance via model attr)
# Each model tracks its own _mc_version as an int attribute.


# ---------------------------------------------------------------------------
# Analytical pricing (X=0, ΔS=0, delta not significant)
# ---------------------------------------------------------------------------


def analytical_pricing(
    mu_H: float,
    mu_A: float,
    score: tuple[int, int],
) -> dict[str, float]:
    """Compute market probabilities analytically via independent Poisson.

    Valid when X=0 (no red cards), ΔS=0 (level score), and delta
    coefficients are not statistically significant.

    Args:
        mu_H: Remaining expected home goals.
        mu_A: Remaining expected away goals.
        score: Current (home, away) score.

    Returns:
        Dict of market → probability.
    """
    S_H, S_A = score
    mu_total = mu_H + mu_A

    # Match result probabilities via Poisson convolution
    max_goals = 15
    P_home_win = 0.0
    P_draw = 0.0
    P_away_win = 0.0
    P_btts = 0.0

    for h_rem in range(max_goals):
        p_h = float(poisson_dist.pmf(h_rem, mu_H))
        for a_rem in range(max_goals):
            p_a = float(poisson_dist.pmf(a_rem, mu_A))
            p_joint = p_h * p_a
            final_h = S_H + h_rem
            final_a = S_A + a_rem

            if final_h > final_a:
                P_home_win += p_joint
            elif final_h == final_a:
                P_draw += p_joint
            else:
                P_away_win += p_joint

            if final_h > 0 and final_a > 0:
                P_btts += p_joint

    # Over/Under thresholds (total goals)
    # P(total > k) = 1 - P(total_remaining <= k - current_total)
    current_total = S_H + S_A
    P_over_15 = 1.0 - float(poisson_dist.cdf(max(0, 1 - current_total), mu_total))
    P_over_25 = 1.0 - float(poisson_dist.cdf(max(0, 2 - current_total), mu_total))
    P_over_35 = 1.0 - float(poisson_dist.cdf(max(0, 3 - current_total), mu_total))

    return {
        "home_win": P_home_win,
        "draw": P_draw,
        "away_win": P_away_win,
        "over_15": P_over_15,
        "over_25": P_over_25,
        "over_35": P_over_35,
        "btts_yes": P_btts,
    }


# ---------------------------------------------------------------------------
# Market aggregation from MC results
# ---------------------------------------------------------------------------


def aggregate_markets(
    final_scores: np.ndarray,
    current_score: tuple[int, int],
) -> dict[str, float]:
    """Convert MC final scores into market probability estimates.

    Args:
        final_scores: Shape (N, 2) array of [home_goals, away_goals].
        current_score: Current (home, away) — not used since final_scores
            already includes current goals from the simulation.

    Returns:
        Dict of market → probability.
    """
    sh = final_scores[:, 0]
    sa = final_scores[:, 1]
    total = sh + sa

    return {
        "home_win": float(np.mean(sh > sa)),
        "draw": float(np.mean(sh == sa)),
        "away_win": float(np.mean(sh < sa)),
        "over_15": float(np.mean(total > 1)),
        "over_25": float(np.mean(total > 2)),
        "over_35": float(np.mean(total > 3)),
        "btts_yes": float(np.mean((sh > 0) & (sa > 0))),
    }


# ---------------------------------------------------------------------------
# Per-market MC standard error
# ---------------------------------------------------------------------------


def compute_mc_stderr(
    P_true: dict[str, float],
    N: int,
    *,
    analytical: bool = False,
) -> dict[str, float]:
    """Compute standard error for each market probability estimate.

    σ_MC = sqrt(p * (1-p) / N) per market (Bernoulli variance).

    In analytical mode, applies a floor of 0.005 so P_cons provides
    conservative protection even without MC sampling uncertainty.

    Args:
        P_true: Market → probability dict.
        N: Number of MC paths (or N_MC for analytical floor calc).
        analytical: Whether pricing was analytical (applies σ floor).

    Returns:
        Dict of market → σ_MC.
    """
    result: dict[str, float] = {}
    for market, p in P_true.items():
        sigma = math.sqrt(p * (1.0 - p) / N) if 0.0 < p < 1.0 else 0.0

        if analytical:
            sigma = max(sigma, _SIGMA_FLOOR)

        result[market] = sigma

    return result


# ---------------------------------------------------------------------------
# Async pricing entry point
# ---------------------------------------------------------------------------


async def step_3_4_async(
    model: LiveFootballQuantModel,
    mu_H: float,
    mu_A: float,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Non-blocking async pricing — analytical or MC.

    Dispatches to analytical pricing when X=0, ΔS=0, and delta is not
    statistically significant. Otherwise runs MC simulation in a thread
    pool executor to avoid blocking the async event loop.

    Implements stale-check: if model._mc_version changes or event_state
    becomes PRELIMINARY during MC execution, returns (None, None).

    Args:
        model: LiveFootballQuantModel with current match state.
        mu_H: Remaining expected home goals for this tick.
        mu_A: Remaining expected away goals for this tick.

    Returns:
        (P_true, σ_MC) dicts, or (None, None) if result is stale.
    """
    # Analytical path: no red cards, level score, delta not significant
    if (
        model.current_state_X == 0
        and model.delta_S == 0
        and not model.delta_significant
    ):
        P_true = analytical_pricing(mu_H, mu_A, model.score)
        sigma_MC = compute_mc_stderr(P_true, N_MC, analytical=True)
        return P_true, sigma_MC

    # MC path
    model._mc_version += 1
    my_version = model._mc_version

    # Deterministic seed from match state
    seed = (
        hash((model.match_id, model.t, model.score_home, model.score_away, model.current_state_X))
        % (2**31)
    )

    loop = asyncio.get_running_loop()

    final_scores = await loop.run_in_executor(
        mc_executor,
        mc_simulate_remaining,
        model.t,
        model.T_exp,
        model.score_home,
        model.score_away,
        model.current_state_X,
        model.delta_S,
        model.a_H,
        model.a_A,
        model.b,
        model.gamma_H,
        model.gamma_A,
        model.delta_H,
        model.delta_A,
        model.Q_diag,
        model.Q_off_normalized,
        model.basis_bounds,
        N_MC,
        seed,
    )

    # Stale check: model state changed during MC execution
    if my_version != model._mc_version:
        return None, None
    if model.event_state == EVENT_PRELIMINARY:
        return None, None

    P_true = aggregate_markets(final_scores, model.score)
    sigma_MC = compute_mc_stderr(P_true, N_MC)

    return P_true, sigma_MC
