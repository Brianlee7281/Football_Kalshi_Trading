"""LiveFootballQuantModel — Phase 3 live engine state container.

Holds all mutable state for a single match from kickoff to full time.
Initialized from Phase2Result + production_params (Step 2.5), then
mutated by the tick loop, event handlers, and Phase 4 execution layer.

Key responsibilities:
  - Mathematical state (t, S, X, ΔS, μ_H, μ_A)
  - Engine / event state machines
  - MMPP parameters (a, b, γ, δ, Q)
  - Precomputed grids (P_grid, P_fine_grid)
  - Phase 3→4 queue (asyncio.Queue maxsize=1)
  - ob_freeze / cooldown safety flags
  - Connectivity health flags

Reference: docs/phase2.md Step 2.5, docs/phase3.md
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.types import Phase2Result, TickData

logger = get_logger("model")


# ---------------------------------------------------------------------------
# Engine phase & event state constants
# ---------------------------------------------------------------------------

WAITING_FOR_KICKOFF = "WAITING_FOR_KICKOFF"
FIRST_HALF = "FIRST_HALF"
HALFTIME = "HALFTIME"
SECOND_HALF = "SECOND_HALF"
FINISHED = "FINISHED"

EVENT_IDLE = "IDLE"
EVENT_PRELIMINARY = "PRELIMINARY_DETECTED"
EVENT_CONFIRMED = "CONFIRMED"

# Default basis boundaries (minutes): 6 intervals × 15 min
# [0, 15, 30, 45+α₁, 60+α₁, 75+α₁, 90+α₁+α₂]
_DEFAULT_ALPHA_1 = 2.0
_DEFAULT_ALPHA_2 = 4.0


def _default_basis_bounds() -> np.ndarray:
    """Compute default basis interval boundaries (7 values for 6 intervals)."""
    a1 = _DEFAULT_ALPHA_1
    a2 = _DEFAULT_ALPHA_2
    return np.array([0.0, 15.0, 30.0, 45.0 + a1, 60.0 + a1, 75.0 + a1, 90.0 + a1 + a2])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class LiveFootballQuantModel:
    """All mutable state for a single live match.

    Phase 3 (tick_loop, event handlers) reads and mutates this object.
    Phase 4 (signal_generator) reads P_true from ``phase4_queue``.

    Attributes are grouped by function, matching the layout in
    docs/phase2.md Step 2.5.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    match_id: str = ""
    league_id: int = 0
    trading_mode: str = "paper"  # "paper" or "live"

    # ------------------------------------------------------------------
    # Time state
    # ------------------------------------------------------------------
    t: float = 0.0  # current effective play time (minutes, halftime excluded)
    engine_phase: str = WAITING_FOR_KICKOFF
    T_exp: float = 96.0  # expected match duration (90 + α₁ + α₂)
    kickoff_wall_clock: float = 0.0  # time.monotonic() at kickoff
    halftime_accumulated: float = 0.0  # total seconds spent in halftime
    halftime_start: float | None = None  # set when HALFTIME entered

    # ------------------------------------------------------------------
    # Match state
    # ------------------------------------------------------------------
    current_state_X: int = 0  # Markov state ∈ {0, 1, 2, 3}
    score_home: int = 0
    score_away: int = 0
    delta_S: int = 0  # score_home - score_away

    # ------------------------------------------------------------------
    # Remaining expected goals (recomputed each tick)
    # ------------------------------------------------------------------
    mu_H: float = 0.0
    mu_A: float = 0.0

    # ------------------------------------------------------------------
    # Intensity function parameters (from Phase 1 + Phase 2)
    # ------------------------------------------------------------------
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

    # Piecewise basis boundaries: shape (7,)
    basis_bounds: np.ndarray = field(default_factory=_default_basis_bounds)

    # ------------------------------------------------------------------
    # Markov model
    # ------------------------------------------------------------------
    Q: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    Q_by_delta_S: dict[int, np.ndarray] = field(default_factory=dict)
    Q_off_normalized: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4)),
    )
    P_grid: dict[int, np.ndarray] = field(default_factory=dict)
    P_fine_grid: dict[int, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Phase 1 flags
    # ------------------------------------------------------------------
    delta_significant: bool = False
    pricing_mode: str = "analytical"  # "analytical" or "mc"

    # ------------------------------------------------------------------
    # Phase 3 event state machine
    # ------------------------------------------------------------------
    event_state: str = EVENT_IDLE
    cooldown: bool = False
    ob_freeze: bool = False
    preliminary_cache: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # ob_freeze sensors (previous-tick values for spike detection)
    # ------------------------------------------------------------------
    bet365_odds_prev: dict[str, float] | None = None
    live_score_prev: dict[str, Any] | None = None
    P_kalshi_prev: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Connectivity health
    # ------------------------------------------------------------------
    live_score_ready: bool = False
    live_odds_healthy: bool = False
    kalshi_healthy: bool = False

    # ------------------------------------------------------------------
    # Sanity check result (from Phase 2 Step 2.4)
    # ------------------------------------------------------------------
    sanity_verdict: str = "GO"

    # ------------------------------------------------------------------
    # Risk parameters
    # ------------------------------------------------------------------
    bankroll: float = 0.0
    f_order_cap: float = 0.03
    f_match_cap: float = 0.05
    f_total_cap: float = 0.20

    # ------------------------------------------------------------------
    # Phase 3 → Phase 4 queue
    # ------------------------------------------------------------------
    phase4_queue: asyncio.Queue[TickData] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1),
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def score(self) -> tuple[int, int]:
        """Current match score as (home, away)."""
        return (self.score_home, self.score_away)

    @property
    def order_allowed(self) -> bool:
        """Whether new orders can be placed this tick."""
        return (
            not self.cooldown
            and not self.ob_freeze
            and self.event_state == EVENT_IDLE
        )

    @property
    def is_active(self) -> bool:
        """Whether the engine is in an active pricing phase."""
        return self.engine_phase in (FIRST_HALF, SECOND_HALF)

    @property
    def Q_diag(self) -> np.ndarray:
        """Diagonal of Q matrix (departure rates), shape (4,)."""
        return np.diag(self.Q)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def from_phase2(
        cls,
        phase2_result: Phase2Result,
        params: dict[str, Any],
        *,
        match_id: str = "",
        league_id: int = 0,
        trading_mode: str = "paper",
        bankroll: float = 0.0,
        delta_significant: bool = False,
        alpha_1_mean: float = _DEFAULT_ALPHA_1,
        alpha_2_mean: float = _DEFAULT_ALPHA_2,
    ) -> LiveFootballQuantModel:
        """Create a model from Phase 2 output and Phase 1 params.

        Args:
            phase2_result: Output of Phase 2 pipeline.
            params: Phase 1 production_params JSON dict.
            match_id: Goalserve match ID.
            league_id: Goalserve league ID.
            trading_mode: "paper" or "live".
            bankroll: Current account balance.
            delta_significant: Phase 1 LRT result for delta.
            alpha_1_mean: League mean 1st-half stoppage.
            alpha_2_mean: League mean 2nd-half stoppage.

        Returns:
            Initialized model ready for Phase 3.
        """
        model = cls(
            match_id=match_id,
            league_id=league_id,
            trading_mode=trading_mode,
            a_H=phase2_result.a_H,
            a_A=phase2_result.a_A,
            C_time=phase2_result.C_time,
            sanity_verdict=phase2_result.verdict,
            bankroll=bankroll,
            delta_significant=delta_significant,
            T_exp=90.0 + alpha_1_mean + alpha_2_mean,
        )

        # Load Phase 1 parameters
        model.b = np.array(params["b"], dtype=np.float64)
        model.gamma_H = np.array(params["gamma_H"], dtype=np.float64)
        model.gamma_A = np.array(params["gamma_A"], dtype=np.float64)
        model.delta_H = np.array(params["delta_H"], dtype=np.float64)
        model.delta_A = np.array(params["delta_A"], dtype=np.float64)
        model.beta_H = float(params.get("beta_H", 0.0))
        model.kappa_H = float(params.get("kappa_H", 0.0))
        model.tau_H = float(params.get("tau_H", 1.0))
        model.beta_A = float(params.get("beta_A", 0.0))
        model.kappa_A = float(params.get("kappa_A", 0.0))
        model.tau_A = float(params.get("tau_A", 1.0))

        model.Q = np.array(params["Q_global"], dtype=np.float64)
        Q_ds_raw = params.get("Q_by_delta_S", {})
        model.Q_by_delta_S = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in Q_ds_raw.items()
        }

        # Derived quantities
        model.Q_off_normalized = _normalize_Q_off(model.Q)
        model.basis_bounds = _compute_basis_bounds(
            alpha_1_mean, alpha_2_mean,
        )

        # Pricing mode selection
        model.pricing_mode = "mc" if delta_significant else "analytical"

        logger.info(
            "model_initialized",
            match_id=match_id,
            a_H=round(model.a_H, 4),
            a_A=round(model.a_A, 4),
            pricing_mode=model.pricing_mode,
            verdict=model.sanity_verdict,
        )

        return model

    # ------------------------------------------------------------------
    # State mutation helpers
    # ------------------------------------------------------------------

    def update_score(self, home: int, away: int) -> None:
        """Update the score and recompute ΔS."""
        self.score_home = home
        self.score_away = away
        self.delta_S = home - away

    def transition_state(self, new_state_X: int) -> None:
        """Update Markov state X (red card transition)."""
        old = self.current_state_X
        self.current_state_X = new_state_X
        logger.info(
            "state_transition",
            match_id=self.match_id,
            old_X=old,
            new_X=new_state_X,
        )

    def enter_halftime(self) -> None:
        """Transition to HALFTIME phase."""
        self.engine_phase = HALFTIME
        self.halftime_start = None  # will be set by tick loop wall clock
        logger.info("halftime_entered", match_id=self.match_id, t=self.t)

    def exit_halftime(self) -> None:
        """Transition from HALFTIME to SECOND_HALF."""
        self.engine_phase = SECOND_HALF
        logger.info("second_half_started", match_id=self.match_id, t=self.t)

    def finish(self) -> None:
        """Transition to FINISHED."""
        self.engine_phase = FINISHED
        logger.info(
            "match_finished",
            match_id=self.match_id,
            score=self.score,
            t=self.t,
        )

    def emit_tick(self, tick_data: TickData) -> None:
        """Push tick data to Phase 4 queue (replace stale if full)."""
        if self.phase4_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.phase4_queue.get_nowait()
        self.phase4_queue.put_nowait(tick_data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_Q_off(Q: np.ndarray) -> np.ndarray:
    """Normalize off-diagonal Q into transition probabilities for MC."""
    Q_off = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        total = -Q[i, i]
        if total > 0:
            for j in range(4):
                if i != j:
                    Q_off[i, j] = Q[i, j] / total
    return Q_off


def _compute_basis_bounds(
    alpha_1: float,
    alpha_2: float,
) -> np.ndarray:
    """Compute basis interval boundaries from stoppage times.

    6 intervals of 15 min each, with stoppage added to intervals 3 and 6:
        [0, 15, 30, 45+α₁, 60+α₁, 75+α₁, 90+α₁+α₂]
    """
    return np.array([
        0.0,
        15.0,
        30.0,
        45.0 + alpha_1,
        60.0 + alpha_1,
        75.0 + alpha_1,
        90.0 + alpha_1 + alpha_2,
    ])
