"""Six exit triggers for open positions — Step 4.4.

Evaluates each open position every tick and returns an ExitSignal when
the position should be closed (fully or partially).

Six triggers evaluated in priority order:
  1. EDGE_DECAY        — EV below θ_exit (0.5¢); gradual erosion
  2. EDGE_REVERSAL     — model has crossed to opposite side of market
  3. EXPIRY_EVAL       — last 3 minutes: compare E_hold vs E_exit
  4. BET365_DIVERGENCE — bet365 moves against position; logging only (auto-exit disabled)
  5. POSITION_TRIM     — edge weakened but above θ_exit; trim to current optimal
  6. OPPORTUNITY_COST  — opposite direction has strong edge; deadlock resolution

Key v2 fixes (docs/phase4.md):
  - Trigger 2 BUY_NO: `P_cons > P_kalshi_bid + θ` (not `1 - P_kalshi_bid`)
  - Trigger 3 BUY_NO: direction-specific E_hold formula
  - Trigger 4 BUY_NO: `P_bet365 > entry_price + threshold` (not `1 - entry`)

All comparisons are in Yes-probability space.
P_kalshi_bid is the current bid (best price to sell Yes into).

Reference: docs/phase4.md Step 4.4
"""

from __future__ import annotations

from src.common.logging import get_logger
from src.common.types import ExitSignal, Position

logger = get_logger("exit_logic")

# ---------------------------------------------------------------------------
# Constants (docs/config_reference.md — Exit Parameters)
# ---------------------------------------------------------------------------

THETA_EXIT: float = 0.005    # Exit EV threshold: 0.5¢
THETA_ENTRY: float = 0.02    # Entry EV threshold: 2¢ (used for reversal check)
DIVERGENCE_THRESHOLD: float = 0.05  # bet365 divergence: 5pp

# Trigger 4: set True to auto-exit on bet365 divergence (off until calibrated)
BET365_DIVERGENCE_AUTO_EXIT: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _P_cons(P_true: float, sigma_MC: float, direction: str, z: float) -> float:
    """Directional conservative probability (lower for BUY_YES, upper for BUY_NO)."""
    if direction == "BUY_YES":
        return P_true - z * sigma_MC
    return P_true + z * sigma_MC


def _position_ev(
    P_cons: float,
    P_kalshi_bid: float,
    direction: str,
    c: float,
) -> float:
    """EV of holding position given current bid price.

    Uses bid (not ask) because we're evaluating the ongoing value of a
    position we'd exit by selling at the bid.

    BUY_YES: win prob P_cons, win net (1-c)*(1-P_bid); lose P_bid.
    BUY_NO:  win prob (1-P_cons), win net (1-c)*P_bid; lose (1-P_bid).
    """
    if direction == "BUY_YES":
        return (
            P_cons * (1.0 - c) * (1.0 - P_kalshi_bid)
            - (1.0 - P_cons) * P_kalshi_bid
        )
    return (
        (1.0 - P_cons) * (1.0 - c) * P_kalshi_bid
        - P_cons * (1.0 - P_kalshi_bid)
    )


# ---------------------------------------------------------------------------
# Trigger 1: Edge Decay
# ---------------------------------------------------------------------------


def check_edge_decay(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_bid: float,
    c: float,
    z: float,
) -> ExitSignal | None:
    """Exit if current position EV has fallen below θ_exit (0.5¢).

    Args:
        position: Open position to evaluate.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error for this market.
        P_kalshi_bid: Current best bid (probability space).
        c: Kalshi fee rate.
        z: Conservative z-score.

    Returns:
        ExitSignal(reason="EDGE_DECAY") or None.
    """
    P_c = _P_cons(P_true, sigma_MC, position.direction, z)
    ev = _position_ev(P_c, P_kalshi_bid, position.direction, c)
    if ev < THETA_EXIT:
        return ExitSignal(reason="EDGE_DECAY", EV=ev)
    return None


# ---------------------------------------------------------------------------
# Trigger 2: Edge Reversal
# ---------------------------------------------------------------------------


def check_edge_reversal(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_bid: float,
    z: float,
) -> ExitSignal | None:
    """Exit immediately if model has flipped to the opposite side.

    All comparisons in Yes-probability space (v2 fix: no 1-P conversion
    for BUY_NO, which fixed a ~20pp threshold error in the original).

    BUY_YES reversal: P_cons drops θ_entry below market bid.
    BUY_NO  reversal: P_cons rises θ_entry above market bid.

    Args:
        position: Open position to evaluate.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error.
        P_kalshi_bid: Current best bid (probability space).
        z: Conservative z-score.

    Returns:
        ExitSignal(reason="EDGE_REVERSAL") or None.
    """
    P_c = _P_cons(P_true, sigma_MC, position.direction, z)

    if position.direction == "BUY_YES" and P_c < P_kalshi_bid - THETA_ENTRY:
        return ExitSignal(reason="EDGE_REVERSAL")

    # v2 fix: use P_kalshi_bid directly (not 1 - P_kalshi_bid)
    if position.direction == "BUY_NO" and P_c > P_kalshi_bid + THETA_ENTRY:
        return ExitSignal(reason="EDGE_REVERSAL")

    return None


# ---------------------------------------------------------------------------
# Trigger 3: Time-Based Expiry Evaluation
# ---------------------------------------------------------------------------


def check_expiry_eval(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_bid: float,
    c: float,
    z: float,
    t: float,
    T: float,
) -> ExitSignal | None:
    """In the last 3 minutes: exit if selling now beats holding to settlement.

    Compares direction-specific E_hold vs E_exit (v2 fix: BUY_NO uses its
    own E_hold formula instead of reusing the BUY_YES one).

    Args:
        position: Open position.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error.
        P_kalshi_bid: Current best bid (probability space).
        c: Kalshi fee rate.
        z: Conservative z-score.
        t: Current effective play time (minutes).
        T: Expected match end time (minutes).

    Returns:
        ExitSignal(reason="EXPIRY_EVAL") or None.
    """
    if T - t >= 3.0:
        return None

    P_c = _P_cons(P_true, sigma_MC, position.direction, z)
    entry = position.entry_price

    # ── E_hold: expected value held to settlement ─────────────────────────
    if position.direction == "BUY_YES":
        # Win (prob P_c): net = (1-c) * (1 - entry)
        # Lose (prob 1-P_c): loss = entry
        E_hold = (
            P_c * (1.0 - c) * (1.0 - entry)
            - (1.0 - P_c) * entry
        )
    else:
        # BUY_NO — v2 fix: direction-specific formula
        # Win (prob 1-P_c): net = (1-c) * entry
        # Lose (prob P_c): loss = (1 - entry)
        E_hold = (
            (1.0 - P_c) * (1.0 - c) * entry
            - P_c * (1.0 - entry)
        )

    # ── E_exit: expected value selling now ───────────────────────────────
    if position.direction == "BUY_YES":
        # Sell Yes at bid
        profit_if_exit = P_kalshi_bid - entry
    else:
        # Close No = buy Yes at bid to offset
        profit_if_exit = entry - P_kalshi_bid

    fee_if_exit = c * max(0.0, profit_if_exit)
    E_exit = profit_if_exit - fee_if_exit

    if E_exit > E_hold:
        return ExitSignal(
            reason="EXPIRY_EVAL",
            E_hold=E_hold,
            E_exit=E_exit,
        )
    return None


# ---------------------------------------------------------------------------
# Trigger 4: bet365 Divergence Warning
# ---------------------------------------------------------------------------


def check_bet365_divergence(
    position: Position,
    P_bet365: float | None,
) -> ExitSignal | None:
    """Warn (and optionally auto-exit) when bet365 moves against position.

    All comparisons in Yes-probability space (v2 fix: BUY_NO uses
    `entry + threshold` not `(1 - entry) + threshold`).

    Currently logging-only (BET365_DIVERGENCE_AUTO_EXIT = False).
    Enable auto-exit in Step 4.6 after live calibration.

    Args:
        position: Open position.
        P_bet365: bet365 implied probability (Yes space), or None if stale.

    Returns:
        ExitSignal(reason="BET365_DIVERGENCE") if auto-exit enabled,
        None otherwise (divergence is logged).
    """
    if P_bet365 is None:
        return None

    entry = position.entry_price
    diverged = False

    if position.direction == "BUY_YES" and P_bet365 < entry - DIVERGENCE_THRESHOLD:
        diverged = True
    elif position.direction == "BUY_NO" and P_bet365 > entry + DIVERGENCE_THRESHOLD:
        # v2 fix: use entry directly (not 1 - entry)
        diverged = True

    if not diverged:
        return None

    logger.warning(
        "bet365_divergence_detected",
        market_ticker=position.market_ticker,
        direction=position.direction,
        entry_price=round(entry, 4),
        P_bet365=round(P_bet365, 4),
        threshold=DIVERGENCE_THRESHOLD,
        auto_exit=BET365_DIVERGENCE_AUTO_EXIT,
    )

    if BET365_DIVERGENCE_AUTO_EXIT:
        return ExitSignal(reason="BET365_DIVERGENCE")
    return None


# ---------------------------------------------------------------------------
# Trigger 5: Position Trimming
# ---------------------------------------------------------------------------


def check_position_trim(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_bid: float,
    c: float,
    z: float,
    K_frac: float,
    bankroll: float,
) -> ExitSignal | None:
    """Partial exit when edge has weakened but remains above θ_exit.

    Without trimming, positions stay oversized in the "dead zone" where
    EV > θ_exit but has materially shrunk from the original allocation.

    Trims when: f_optimal < 0.5 × f_existing.

    Args:
        position: Open position.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error.
        P_kalshi_bid: Current best bid (probability space, used for sizing).
        c: Kalshi fee rate.
        z: Conservative z-score.
        K_frac: Fractional Kelly coefficient.
        bankroll: Current bankroll in dollars.

    Returns:
        ExitSignal(reason="POSITION_TRIM") or None.
    """
    if P_kalshi_bid <= 0.0 or bankroll <= 0.0:
        return None

    P_c = _P_cons(P_true, sigma_MC, position.direction, z)

    # Current optimal f (same formula as compute_kelly, without existing exposure)
    if position.direction == "BUY_YES":
        W = (1.0 - c) * (1.0 - P_kalshi_bid)
        L = P_kalshi_bid
        ev = P_c * W - (1.0 - P_c) * L
    else:
        W = (1.0 - c) * P_kalshi_bid
        L = 1.0 - P_kalshi_bid
        ev = (1.0 - P_c) * W - P_c * L

    if ev <= 0.0 or W * L <= 0.0:
        return None  # Trigger 1 (edge_decay) will handle this

    f_optimal = K_frac * (ev / (W * L)) * position.kelly_multiplier
    existing_fraction = (position.entry_price * position.quantity) / bankroll

    if existing_fraction <= 0.0:
        return None

    if f_optimal < existing_fraction * 0.5:
        trim_qty = position.quantity - int(f_optimal * bankroll / P_kalshi_bid)
        trim_qty = max(0, min(trim_qty, position.quantity))
        if trim_qty < 1:
            return None
        return ExitSignal(
            reason="POSITION_TRIM",
            trim_quantity=trim_qty,
            f_optimal=f_optimal,
            f_existing=existing_fraction,
        )

    return None


# ---------------------------------------------------------------------------
# Trigger 6: Opportunity Cost Exit
# ---------------------------------------------------------------------------


def check_opportunity_cost_exit(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_ask: float,
    P_kalshi_bid: float,
    c: float,
    z: float,
) -> ExitSignal | None:
    """Exit if the opposite direction has a strong edge and current is weak.

    Resolves the deadlock where: current position EV is marginally positive
    (above θ_exit, so Trigger 1 won't fire) but the model strongly favours
    the opposite direction (which can't be entered while this position exists).

    Fires when: opposite_EV > θ_entry AND current_EV < 2 × θ_exit.

    Args:
        position: Open position.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error.
        P_kalshi_ask: Current best ask (probability space).
        P_kalshi_bid: Current best bid (probability space).
        c: Kalshi fee rate.
        z: Conservative z-score.

    Returns:
        ExitSignal(reason="OPPORTUNITY_COST") or None.
    """
    # Current position EV
    P_c_current = _P_cons(P_true, sigma_MC, position.direction, z)
    current_ev = _position_ev(P_c_current, P_kalshi_bid, position.direction, c)

    # Opposite direction EV
    if position.direction == "BUY_YES":
        # Opposite = BUY_NO, evaluated using bid (price for selling into)
        P_c_opp = P_true + z * sigma_MC
        opp_ev = (
            (1.0 - P_c_opp) * (1.0 - c) * P_kalshi_bid
            - P_c_opp * (1.0 - P_kalshi_bid)
        )
    else:
        # Opposite = BUY_YES, evaluated using ask (price for buying at)
        P_c_opp = P_true - z * sigma_MC
        opp_ev = (
            P_c_opp * (1.0 - c) * (1.0 - P_kalshi_ask)
            - (1.0 - P_c_opp) * P_kalshi_ask
        )

    if opp_ev > THETA_ENTRY and current_ev < 2.0 * THETA_EXIT:
        return ExitSignal(
            reason="OPPORTUNITY_COST",
            current_EV=current_ev,
            opposite_EV=opp_ev,
        )

    return None


# ---------------------------------------------------------------------------
# Full exit evaluation loop
# ---------------------------------------------------------------------------


def evaluate_exit(
    position: Position,
    P_true: float,
    sigma_MC: float,
    P_kalshi_bid: float,
    P_kalshi_ask: float,
    P_bet365: float | None,
    c: float,
    z: float,
    t: float,
    T: float,
    K_frac: float,
    bankroll: float,
) -> ExitSignal | None:
    """Evaluate all six exit triggers for a single position, in priority order.

    Call this each tick for every open position. Returns the first trigger
    that fires, or None if the position should be held.

    Args:
        position: Open position to evaluate.
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error for this market.
        P_kalshi_bid: Current best bid (probability space).
        P_kalshi_ask: Current best ask (probability space).
        P_bet365: bet365 implied probability, or None if stale.
        c: Kalshi fee rate.
        z: Conservative z-score.
        t: Current effective play time (minutes).
        T: Expected match end time (minutes).
        K_frac: Fractional Kelly coefficient.
        bankroll: Current bankroll in dollars.

    Returns:
        ExitSignal if a trigger fires, None otherwise.
    """
    # Trigger 1: edge decay (EV below 0.5¢)
    signal = check_edge_decay(position, P_true, sigma_MC, P_kalshi_bid, c, z)
    if signal:
        return signal

    # Trigger 2: edge reversal (model has flipped sides)
    signal = check_edge_reversal(position, P_true, sigma_MC, P_kalshi_bid, z)
    if signal:
        return signal

    # Trigger 3: expiry eval (last 3 minutes)
    signal = check_expiry_eval(
        position, P_true, sigma_MC, P_kalshi_bid, c, z, t, T
    )
    if signal:
        return signal

    # Trigger 4: bet365 divergence (logging only unless BET365_DIVERGENCE_AUTO_EXIT)
    signal = check_bet365_divergence(position, P_bet365)
    if signal:
        return signal

    # Trigger 5: position trimming (edge weakened but above θ_exit)
    signal = check_position_trim(
        position, P_true, sigma_MC, P_kalshi_bid, c, z, K_frac, bankroll
    )
    if signal:
        return signal

    # Trigger 6: opportunity cost (opposite direction has strong edge)
    signal = check_opportunity_cost_exit(
        position, P_true, sigma_MC, P_kalshi_ask, P_kalshi_bid, c, z
    )
    if signal:
        return signal

    return None
