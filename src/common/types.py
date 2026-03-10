"""Shared dataclasses used across all phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase 1: Calibration
# ---------------------------------------------------------------------------


@dataclass
class RedCardTransition:
    """A single red-card dismissal event within a match."""

    minute: float
    team: str  # "localteam" or "visitorteam"
    from_state: int  # Markov state X before
    to_state: int  # Markov state X after


@dataclass
class IntervalRecord:
    """A continuous interval where lambda is constant (no state change)."""

    match_id: str
    t_start: float
    t_end: float
    state_X: int  # Markov state {0, 1, 2, 3}
    delta_S: int  # Score difference (home - away)
    home_goal_times: list[float] = field(default_factory=list)
    away_goal_times: list[float] = field(default_factory=list)
    goal_delta_before: list[int] = field(default_factory=list)
    T_m: float = 90.0  # Actual match end time
    is_halftime: bool = False
    alpha_1: float = 0.0  # First-half stoppage
    alpha_2: float = 0.0  # Second-half stoppage
    red_card_transitions: list[RedCardTransition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2: Pre-Match
# ---------------------------------------------------------------------------


@dataclass
class Phase2Result:
    """Output of Phase 2 pipeline, consumed by Phase 3 to initialize the model."""

    a_H: float
    a_A: float
    C_time: float
    verdict: str  # GO | GO_WITH_CAUTION | HOLD | SKIP
    pre_match_data: dict[str, object] = field(default_factory=dict)
    warning: str | None = None


# ---------------------------------------------------------------------------
# Phase 3: Live Engine
# ---------------------------------------------------------------------------


@dataclass
class NormalizedEvent:
    """Unified event representation from any data source."""

    type: str  # goal_detected, goal_confirmed, red_card, period_change, odds_spike, etc.
    source: str  # "live_odds" or "live_score"
    confidence: str  # "preliminary" or "confirmed"
    timestamp: float
    score: tuple[int, int] | None = None
    team: str | None = None
    minute: float | None = None
    period: str | None = None
    var_cancelled: bool | None = None
    scorer_id: str | None = None
    delta: float | None = None


@dataclass
class TickData:
    """Per-tick pricing output emitted from Phase 3 to Phase 4."""

    P_true: dict[str, float]  # {"home_win": 0.55, "over_25": 0.65, ...}
    sigma_MC: dict[str, float]  # {"home_win": 0.0022, ...}
    order_allowed: bool
    event_state: str = "IDLE"
    pricing_mode: str = "analytical"
    engine_phase: str = "FIRST_HALF"
    mu_H: float = 0.0
    mu_A: float = 0.0
    P_bet365: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 4: Execution
# ---------------------------------------------------------------------------


@dataclass
class MarketAlignment:
    """bet365 alignment check result for a single market."""

    status: str  # "ALIGNED", "DIVERGENT", "UNAVAILABLE"
    kelly_multiplier: float  # ALIGNED->0.8, DIVERGENT->0.5, UNAVAILABLE->0.6


@dataclass
class Signal:
    """Trading signal for a single market."""

    direction: str  # BUY_YES, BUY_NO, HOLD
    EV: float
    P_cons: float  # Directional conservative P
    P_kalshi: float  # VWAP effective price
    rough_qty: int
    alignment_status: str  # ALIGNED, DIVERGENT, UNAVAILABLE
    kelly_multiplier: float
    market_ticker: str


@dataclass
class Position:
    """An open or settled trading position."""

    match_id: str
    market_ticker: str
    direction: str  # BUY_YES | BUY_NO
    entry_price: float
    quantity: int
    status: str = "OPEN"  # PENDING | OPEN | AWAITING_SETTLEMENT | SETTLED | CLOSED
    is_paper: bool = False
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    settlement_price: float | None = None
    realized_pnl: float | None = None
    fill_delay: float | None = None


@dataclass
class PaperFill:
    """Result of a paper execution fill."""

    price: float
    quantity: int
    timestamp: float
    is_paper: bool = True
    slippage: float = 0.0
    partial: bool = False
    fill_delay: float = 0.0


@dataclass
class FillResult:
    """Unified fill result from either paper or live execution."""

    success: bool
    fill_price: float | None = None
    fill_quantity: int | None = None
    order_id: str | None = None
    error: str | None = None


@dataclass
class ExitSignal:
    """Signal to exit an existing position."""

    reason: str  # EDGE_DECAY, EDGE_REVERSAL, EXPIRY_EVAL, BET365_DIVERGENCE
    EV: float | None = None
    E_hold: float | None = None
    E_exit: float | None = None


@dataclass
class TradeLog:
    """Complete record of a trade for post-analysis."""

    timestamp: float
    match_id: str
    market_ticker: str
    direction: str  # BUY_YES | BUY_NO | SELL_YES | SELL_NO
    order_type: str  # ENTRY | EXIT_EDGE_DECAY | EXIT_EDGE_REVERSAL | ...
    quantity_ordered: int
    quantity_filled: int
    limit_price: float
    fill_price: float
    P_true_at_order: float
    P_true_cons_at_order: float
    P_kalshi_at_order: float
    P_kalshi_best_at_order: float
    P_bet365_at_order: float
    EV_adj: float
    sigma_MC: float
    pricing_mode: str
    f_kelly: float
    K_frac: float
    alignment_status: str
    kelly_multiplier: float
    cooldown_active: bool
    ob_freeze_active: bool
    event_state: str
    engine_phase: str
    bankroll_before: float
    bankroll_after: float
    is_paper: bool


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class MatchSchedule:
    """A scheduled match in the lifecycle pipeline."""

    match_id: str
    league_id: int
    kickoff_utc: datetime
    phase2_trigger: datetime
    phase3_trigger: datetime
    kalshi_tickers: list[str] = field(default_factory=list)
    odds_api_event_id: str | None = None
    status: str = "SCHEDULED"
    container_id: str | None = None
    trading_mode: str = "paper"
    param_version: int | None = None


@dataclass
class MatchConfig:
    """Runtime configuration for a single match container."""

    match_id: str
    trading_mode: str  # "paper" or "live"
    param_version: int
    kalshi_tickers: list[str] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
