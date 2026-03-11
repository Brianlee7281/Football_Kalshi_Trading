"""Fee-adjusted edge detection — Step 4.2.

Implements the full signal generation pipeline:
    1. compute_conservative_P   — directional lower/upper bound on P_true
    2. compute_signal_with_vwap — 2-pass VWAP EV computation
    3. check_market_alignment   — bet365 directional alignment check
    4. generate_signal          — combined entry point

Design notes:
  - All prices in Kalshi's order book are stored as integer cents (1–99) in
    OrderBookSync. This module normalises them to probability space (0.0–1.0)
    before applying any EV formula.
  - compute_signal_with_vwap returns a Signal with placeholder alignment fields
    ("HOLD", 0.0). generate_signal overwrites them with the real alignment result.
  - HOLD signals are created via _hold() to satisfy the fully-typed Signal dataclass.
  - THETA_ENTRY default = 0.02 (2¢ in probability space). Calibrate per Step 4.6.

Reference: docs/phase4.md Step 4.2
"""

from __future__ import annotations

from src.common.logging import get_logger
from src.common.types import MarketAlignment, Signal
from src.execution.order_book_sync import OrderBookSync

logger = get_logger("edge_detection")

# ---------------------------------------------------------------------------
# Constants (docs/config_reference.md — Trading Parameters)
# ---------------------------------------------------------------------------

THETA_ENTRY: float = 0.02       # Minimum EV (2¢) after VWAP slippage
ALIGNED_MULTIPLIER: float = 0.8
DIVERGENT_MULTIPLIER: float = 0.5
UNAVAILABLE_MULTIPLIER: float = 0.6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hold(ticker: str = "") -> Signal:
    """Return a HOLD signal with sentinel zero values."""
    return Signal(
        direction="HOLD",
        EV=0.0,
        P_cons=0.0,
        P_kalshi=0.0,
        rough_qty=0,
        alignment_status="HOLD",
        kelly_multiplier=0.0,
        market_ticker=ticker,
    )


def _rough_kelly(
    direction: str,
    P_cons: float,
    P_kalshi: float,
    c: float,
    K_frac: float,
    EV: float,
) -> float:
    """Rough fractional Kelly for Pass 1 quantity estimation.

    No alignment multiplier — that's not known until after Pass 2.

    Args:
        direction: "BUY_YES" or "BUY_NO".
        P_cons: Directional conservative probability.
        P_kalshi: Best ask (BUY_YES) or best bid (BUY_NO) in probability space.
        c: Kalshi fee rate (e.g. 0.07).
        K_frac: Fractional Kelly coefficient (e.g. 0.25).
        EV: Expected value from Pass 1.

    Returns:
        Fractional Kelly investment fraction (may be 0.0 if degenerate).
    """
    if direction == "BUY_YES":
        W = (1.0 - c) * (1.0 - P_kalshi)
        L = P_kalshi
    else:  # BUY_NO
        W = (1.0 - c) * P_kalshi
        L = 1.0 - P_kalshi

    denom = W * L
    if denom <= 0.0:
        return 0.0

    f_kelly = EV / denom
    return K_frac * f_kelly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_conservative_P(
    P_true: float,
    sigma_MC: float,
    direction: str,
    z: float = 1.645,
) -> float:
    """Directional conservative adjustment on P_true.

    Buy Yes: higher P_true is favorable → use lower confidence bound.
    Buy No:  lower P_true is favorable → use upper confidence bound.

    Using a symmetric bound for both directions would artificially inflate
    No-side EV (the larger σ_MC, the more the system overbets No).

    Args:
        P_true: Model true probability (Yes space, 0.0–1.0).
        sigma_MC: Monte Carlo standard error for this market.
        direction: "BUY_YES" or "BUY_NO".
        z: Confidence z-score (default 1.645 = 95th percentile one-tailed).

    Returns:
        Conservative probability in Yes space.
    """
    if direction == "BUY_YES":
        return P_true - z * sigma_MC
    elif direction == "BUY_NO":
        return P_true + z * sigma_MC
    return P_true


def compute_signal_with_vwap(
    P_true: float,
    sigma_MC: float,
    ob_sync: OrderBookSync,
    c: float,
    z: float,
    K_frac: float,
    bankroll: float,
    market_ticker: str,
    *,
    theta_entry: float = THETA_ENTRY,
) -> Signal:
    """2-pass VWAP EV computation (Phase 4, Step 4.2).

    Pass 1 — rough EV with best bid/ask:
        Compute EV for BUY_YES and BUY_NO using top-of-book prices.
        Select the higher-EV direction.
        Estimate rough Kelly quantity.

    Pass 2 — final EV with VWAP:
        Compute VWAP effective price for rough_qty.
        Re-compute EV with VWAP to account for market impact.
        If edge disappears (EV ≤ theta_entry), return HOLD.

    Prices in OrderBookSync are in cents (integers). This function converts
    them to probability space (divides by 100) before applying EV formulas.

    Args:
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error for this market.
        ob_sync: Order book state for this market.
        c: Kalshi fee rate (e.g. 0.07 for 7%).
        z: Conservative P z-score.
        K_frac: Fractional Kelly coefficient.
        bankroll: Current bankroll in dollars.
        market_ticker: Kalshi ticker (for Signal output).
        theta_entry: Minimum EV threshold.

    Returns:
        Signal with direction, EV, P_cons, P_kalshi, rough_qty.
        alignment_status and kelly_multiplier are set to placeholder values
        ("HOLD", 0.0) — generate_signal() fills these in.
    """
    best_ask_cents = ob_sync.kalshi_best_ask
    best_bid_cents = ob_sync.kalshi_best_bid

    if best_ask_cents is None or best_bid_cents is None:
        return _hold(market_ticker)

    # Convert to probability space
    P_best_ask = best_ask_cents / 100.0
    P_best_bid = best_bid_cents / 100.0

    # ── Pass 1: rough EV with best ask/bid ─────────────────────────────────

    P_cons_yes = compute_conservative_P(P_true, sigma_MC, "BUY_YES", z)
    rough_EV_yes = (
        P_cons_yes * (1.0 - c) * (1.0 - P_best_ask)
        - (1.0 - P_cons_yes) * P_best_ask
    )

    P_cons_no = compute_conservative_P(P_true, sigma_MC, "BUY_NO", z)
    rough_EV_no = (
        (1.0 - P_cons_no) * (1.0 - c) * P_best_bid
        - P_cons_no * (1.0 - P_best_bid)
    )

    if rough_EV_yes > rough_EV_no and rough_EV_yes > theta_entry:
        direction = "BUY_YES"
        rough_P_kalshi = P_best_ask
        P_cons = P_cons_yes
        rough_EV = rough_EV_yes
    elif rough_EV_no > theta_entry:
        direction = "BUY_NO"
        rough_P_kalshi = P_best_bid
        P_cons = P_cons_no
        rough_EV = rough_EV_no
    else:
        return _hold(market_ticker)

    # Rough Kelly quantity
    rough_f = _rough_kelly(direction, P_cons, rough_P_kalshi, c, K_frac, rough_EV)
    if rough_P_kalshi <= 0.0:
        return _hold(market_ticker)
    rough_qty = int(rough_f * bankroll / rough_P_kalshi)
    if rough_qty < 1:
        return _hold(market_ticker)

    # ── Pass 2: final EV with VWAP ─────────────────────────────────────────

    if direction == "BUY_YES":
        vwap_cents = ob_sync.compute_vwap_buy(rough_qty)
    else:
        vwap_cents = ob_sync.compute_vwap_sell(rough_qty)

    if vwap_cents is None:
        return _hold(market_ticker)  # insufficient depth

    P_effective = vwap_cents / 100.0

    if direction == "BUY_YES":
        final_EV = (
            P_cons * (1.0 - c) * (1.0 - P_effective)
            - (1.0 - P_cons) * P_effective
        )
    else:  # BUY_NO
        final_EV = (
            (1.0 - P_cons) * (1.0 - c) * P_effective
            - P_cons * (1.0 - P_effective)
        )

    if final_EV <= theta_entry:
        return _hold(market_ticker)  # edge disappears after VWAP slippage

    logger.debug(
        "signal_computed",
        ticker=market_ticker,
        direction=direction,
        EV=round(final_EV, 5),
        P_cons=round(P_cons, 4),
        P_effective=round(P_effective, 4),
        rough_qty=rough_qty,
    )

    return Signal(
        direction=direction,
        EV=final_EV,
        P_cons=P_cons,
        P_kalshi=P_effective,   # VWAP effective price (probability space)
        rough_qty=rough_qty,
        alignment_status="HOLD",   # placeholder — generate_signal() fills this
        kelly_multiplier=0.0,      # placeholder — generate_signal() fills this
        market_ticker=market_ticker,
    )


def check_market_alignment(
    P_true_cons: float,
    P_kalshi: float,
    P_bet365: float | None,
    direction: str,
) -> MarketAlignment:
    """Directional bet365 market alignment check.

    Compares whether model and bet365 agree on the direction of mispricing.
    This is NOT independent validation (both use the same Odds-API feed).
    Even when aligned, use kelly_multiplier=0.8 (not 1.0) to reflect the
    limited independence between the two signals.

    All comparisons are in Yes-probability space per the v2 convention.

    Args:
        P_true_cons: Directional conservative probability (from compute_conservative_P).
        P_kalshi: VWAP effective price in probability space.
        P_bet365: bet365 implied probability, or None if stale/unavailable.
        direction: "BUY_YES" or "BUY_NO".

    Returns:
        MarketAlignment with status and kelly_multiplier.
    """
    if P_bet365 is None:
        return MarketAlignment(status="UNAVAILABLE", kelly_multiplier=UNAVAILABLE_MULTIPLIER)

    if direction == "BUY_YES":
        model_says_high = P_true_cons > P_kalshi
        bet365_says_high = P_bet365 > P_kalshi
        aligned = model_says_high and bet365_says_high

    elif direction == "BUY_NO":
        model_says_low = P_true_cons < P_kalshi
        bet365_says_low = P_bet365 < P_kalshi
        aligned = model_says_low and bet365_says_low

    else:
        return MarketAlignment(status="UNAVAILABLE", kelly_multiplier=UNAVAILABLE_MULTIPLIER)

    if aligned:
        return MarketAlignment(status="ALIGNED", kelly_multiplier=ALIGNED_MULTIPLIER)
    return MarketAlignment(status="DIVERGENT", kelly_multiplier=DIVERGENT_MULTIPLIER)


def generate_signal(
    P_true: float,
    sigma_MC: float,
    ob_sync: OrderBookSync,
    P_bet365: float | None,
    c: float,
    z: float,
    K_frac: float,
    bankroll: float,
    market_ticker: str,
    *,
    theta_entry: float = THETA_ENTRY,
) -> Signal:
    """Generate a full trading signal for a single market.

    Combines the 2-pass VWAP EV computation (Step 4.2) with the bet365
    market alignment check. Returns a HOLD Signal if no actionable edge is
    found.

    The bet365 probability should be retrieved via
    ``ob_sync.get_bet365_for_alignment(market_key)`` which returns None
    automatically when data is stale (> BET365_STALE_THRESHOLD).

    Args:
        P_true: Model probability (Yes space).
        sigma_MC: MC standard error for this market.
        ob_sync: Order book state (Kalshi best bid/ask + depth).
        P_bet365: bet365 implied probability from Odds-API, or None if stale.
        c: Kalshi fee rate.
        z: Conservative P z-score.
        K_frac: Fractional Kelly coefficient.
        bankroll: Current bankroll in dollars.
        market_ticker: Kalshi ticker for this market.
        theta_entry: Minimum EV threshold.

    Returns:
        Fully-populated Signal (HOLD or BUY_YES/BUY_NO with alignment).
    """
    base = compute_signal_with_vwap(
        P_true, sigma_MC, ob_sync, c, z, K_frac, bankroll, market_ticker,
        theta_entry=theta_entry,
    )

    if base.direction == "HOLD":
        return base

    alignment = check_market_alignment(
        base.P_cons, base.P_kalshi, P_bet365, base.direction
    )

    logger.debug(
        "signal_generated",
        ticker=market_ticker,
        direction=base.direction,
        EV=round(base.EV, 5),
        alignment=alignment.status,
        kelly_multiplier=alignment.kelly_multiplier,
    )

    return Signal(
        direction=base.direction,
        EV=base.EV,
        P_cons=base.P_cons,
        P_kalshi=base.P_kalshi,
        rough_qty=base.rough_qty,
        alignment_status=alignment.status,
        kelly_multiplier=alignment.kelly_multiplier,
        market_ticker=market_ticker,
    )
