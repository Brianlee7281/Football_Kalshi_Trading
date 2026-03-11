"""Step 2.5 — Live Engine Initialization and Connectivity.

Prepares the ``EngineState`` struct that Phase 3 needs to run:
  - Load Phase 1 parameters (b, gamma, delta, Q)
  - Precompute matrix exponential grids P_grid
  - Normalize Q off-diagonal for MC simulation
  - Validate connectivity (Goalserve REST, Odds-API WS, Kalshi WS)

Reference: docs/phase2.md Step 2.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg

from src.common.logging import get_logger
from src.prematch.step_2_3_backsolve import BacksolveResult
from src.prematch.step_2_4_sanity_check import SanityResult

logger = get_logger("step_2_5")


@dataclass
class EngineState:
    """All state needed by Phase 3 to start the live trading engine.

    This is the concrete realization of the ``LiveFootballQuantModel``
    described in docs/phase2.md Step 2.5.
    """

    # Metadata
    match_id: str = ""

    # Time state
    current_time: float = 0.0
    engine_phase: str = "WAITING_FOR_KICKOFF"
    T_exp: float = 96.0

    # Match state
    current_state_X: int = 0  # 11v11
    current_score: tuple[int, int] = (0, 0)
    delta_S: int = 0

    # Intensity parameters
    a_H: float = 0.0
    a_A: float = 0.0
    b: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gamma_H: np.ndarray = field(default_factory=lambda: np.zeros(4))
    gamma_A: np.ndarray = field(default_factory=lambda: np.zeros(4))
    delta_H: np.ndarray = field(default_factory=lambda: np.zeros(5))
    delta_A: np.ndarray = field(default_factory=lambda: np.zeros(5))
    beta_H: float = 0.0
    kappa_H: float = 0.0
    tau_H: float = 1.0
    beta_A: float = 0.0
    kappa_A: float = 0.0
    tau_A: float = 1.0
    C_time: float = 0.0

    # Markov model
    Q: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    Q_by_delta_S: dict[int, np.ndarray] = field(default_factory=dict)
    Q_off_normalized: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4)),
    )
    P_grid: dict[int, np.ndarray] = field(default_factory=dict)
    P_fine_grid: dict[int, np.ndarray] = field(default_factory=dict)

    # Phase 1 flags
    delta_significant: bool = False

    # Event state machine
    event_state: str = "IDLE"
    cooldown: bool = False
    ob_freeze: bool = False

    # Connectivity
    live_score_ready: bool = False
    live_odds_healthy: bool = False
    kalshi_healthy: bool = False

    # Sanity check result
    sanity_verdict: str = "GO"
    sanity_result: SanityResult | None = None

    # Risk parameters
    bankroll: float = 0.0
    f_order_cap: float = 0.03
    f_match_cap: float = 0.05
    f_total_cap: float = 0.20


def load_phase1_params(
    params: dict[str, Any],
    state: EngineState,
) -> None:
    """Load Phase 1 parameters from production_params JSON into EngineState.

    Args:
        params: The ``production_params.params`` JSONB dict.
        state: EngineState to populate.
    """
    state.b = np.array(params["b"], dtype=np.float64)
    state.gamma_H = np.array(params["gamma_H"], dtype=np.float64)
    state.gamma_A = np.array(params["gamma_A"], dtype=np.float64)
    state.delta_H = np.array(params["delta_H"], dtype=np.float64)
    state.delta_A = np.array(params["delta_A"], dtype=np.float64)
    state.beta_H = float(params.get("beta_H", 0.0))
    state.kappa_H = float(params.get("kappa_H", 0.0))
    state.tau_H = float(params.get("tau_H", 1.0))
    state.beta_A = float(params.get("beta_A", 0.0))
    state.kappa_A = float(params.get("kappa_A", 0.0))
    state.tau_A = float(params.get("tau_A", 1.0))

    Q_raw = params.get("Q_global", [[0] * 4] * 4)
    state.Q = np.array(Q_raw, dtype=np.float64)

    Q_ds_raw = params.get("Q_by_delta_S", {})
    state.Q_by_delta_S = {
        int(k): np.array(v, dtype=np.float64)
        for k, v in Q_ds_raw.items()
    }

    state.Q_off_normalized = normalize_Q_off(state.Q)

    logger.info(
        "phase1_params_loaded",
        b_shape=state.b.shape,
        gamma_H_shape=state.gamma_H.shape,
        Q_shape=state.Q.shape,
    )


def load_backsolve_result(
    backsolve: BacksolveResult,
    state: EngineState,
) -> None:
    """Load Step 2.3 back-solve results into EngineState."""
    state.a_H = backsolve.a_H
    state.a_A = backsolve.a_A
    state.C_time = backsolve.C_time
    state.T_exp = backsolve.T_exp


def load_sanity_result(
    sanity: SanityResult,
    state: EngineState,
) -> None:
    """Load Step 2.4 sanity result into EngineState."""
    state.sanity_verdict = sanity.verdict
    state.sanity_result = sanity


# ---------------------------------------------------------------------------
# Matrix exponential precomputation
# ---------------------------------------------------------------------------


def precompute_P_grid(
    Q: np.ndarray,
    *,
    max_minutes: int = 100,
) -> dict[int, np.ndarray]:
    """Precompute matrix exponentials P(dt) = expm(Q·dt) for dt=0..100.

    Used in Phase 3 Step 3.2 for O(1) Markov state lookups.

    Args:
        Q: (4, 4) transition rate matrix from Phase 1.
        max_minutes: Maximum time horizon.

    Returns:
        Dict mapping minute → 4×4 transition probability matrix.
    """
    P_grid: dict[int, np.ndarray] = {}
    for dt in range(max_minutes + 1):
        P_grid[dt] = scipy.linalg.expm(Q * dt)
    return P_grid


def precompute_P_fine_grid(
    Q: np.ndarray,
    *,
    max_steps: int = 30,
) -> dict[int, np.ndarray]:
    """Precompute fine-grained matrix exponentials for last 5 minutes.

    10-second resolution: dt_10sec / 6.0 minutes.

    Args:
        Q: (4, 4) transition rate matrix.
        max_steps: Number of 10-second steps (30 = 5 minutes).

    Returns:
        Dict mapping 10-sec step index → 4×4 matrix.
    """
    P_fine: dict[int, np.ndarray] = {}
    for step in range(max_steps + 1):
        dt_min = step / 6.0
        P_fine[step] = scipy.linalg.expm(Q * dt_min)
    return P_fine


# ---------------------------------------------------------------------------
# Q normalization for MC simulation
# ---------------------------------------------------------------------------


def normalize_Q_off(Q: np.ndarray) -> np.ndarray:
    """Normalize off-diagonal Q entries into transition probabilities.

    For MC simulation, we need P(transition i→j | transition occurs).
    This is Q[i,j] / (-Q[i,i]) for i ≠ j.

    Q_off is team-independent (Markov state X(t) transitions don't
    depend on which team has the red card).

    Args:
        Q: (4, 4) transition rate matrix.

    Returns:
        (4, 4) normalized transition probability matrix.
    """
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        total_off_diag = -Q[i, i]
        if total_off_diag > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = Q[i, j] / total_off_diag
    return Q_off


# ---------------------------------------------------------------------------
# Full initialization
# ---------------------------------------------------------------------------


def initialize_engine(
    match_id: str,
    params: dict[str, Any],
    backsolve: BacksolveResult,
    sanity: SanityResult,
    *,
    bankroll: float = 0.0,
    delta_significant: bool = False,
) -> EngineState:
    """Initialize the full EngineState for Phase 3.

    This is the main entry point for Step 2.5. It loads Phase 1
    parameters, back-solve results, precomputes matrix exponential
    grids, and sets initial state.

    Args:
        match_id: Goalserve match ID.
        params: Phase 1 production_params JSON.
        backsolve: Step 2.3 back-solve result.
        sanity: Step 2.4 sanity check result.
        bankroll: Current Kalshi account balance.
        delta_significant: Whether delta is statistically significant.

    Returns:
        Fully initialized EngineState.
    """
    state = EngineState(match_id=match_id)

    # Load Phase 1 params
    load_phase1_params(params, state)

    # Load back-solve results
    load_backsolve_result(backsolve, state)

    # Load sanity result
    load_sanity_result(sanity, state)

    # Set flags
    state.delta_significant = delta_significant
    state.bankroll = bankroll

    # Precompute matrix exponential grids
    state.P_grid = precompute_P_grid(state.Q)
    state.P_fine_grid = precompute_P_fine_grid(state.Q)

    logger.info(
        "engine_initialized",
        match_id=match_id,
        a_H=round(state.a_H, 4),
        a_A=round(state.a_A, 4),
        verdict=state.sanity_verdict,
        P_grid_size=len(state.P_grid),
        P_fine_grid_size=len(state.P_fine_grid),
    )

    return state
