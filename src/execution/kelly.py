"""Incremental Kelly sizing, 3-layer risk limits, and liquidity gate — Step 4.3.

Pipeline:
    1. compute_kelly          — incremental allocation fraction
    2. apply_risk_limits      — 3-layer dollar caps (order/match/total)
    3. liquidity_gate         — depth-fraction cap + min-fill-ratio guard

Design notes:
  - compute_kelly returns a fraction (0.0–1.0) of bankroll, not dollars.
    The caller multiplies by bankroll to get the dollar amount, then passes
    that amount through apply_risk_limits.
  - apply_risk_limits takes pre-queried exposure values as parameters so the
    function is pure and unit-testable (no DB access). The caller queries
    current_match_exposure and total_exposure before calling.
  - liquidity_gate returns (gated_qty, proceed). proceed=False means skip
    the order entirely (edge preserved for deeper markets).
  - total_ask_depth and total_bid_depth on OrderBookSync are properties.

Reference: docs/phase4.md Step 4.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import Signal

if TYPE_CHECKING:
    from src.execution.order_book_sync import OrderBookSync

logger = get_logger("kelly")

# ---------------------------------------------------------------------------
# Constants (docs/config_reference.md — Risk Parameters)
# ---------------------------------------------------------------------------

F_ORDER_CAP: float = 0.03   # Single order ≤ 3% of bankroll
F_MATCH_CAP: float = 0.05   # Per-match exposure ≤ 5% of bankroll
F_TOTAL_CAP: float = 0.20   # Portfolio exposure ≤ 20% of bankroll

DEPTH_FRACTION: float = 0.30   # Consume at most 30% of visible depth
MIN_FILL_RATIO: float = 0.50   # Skip if gated qty / target qty < 50%


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_kelly(
    signal: Signal,
    c: float,
    K_frac: float,
    *,
    existing_exposure: float = 0.0,
    bankroll: float = 1.0,
) -> float:
    """Incremental Kelly fraction with market alignment multiplier.

    Uses signal.P_cons (directional conservative probability) and
    signal.P_kalshi (VWAP effective price from Step 4.2) to compute
    direction-specific win/loss payoffs, then applies the EV-over-WL
    formula to get f_kelly.

    Incremental sizing:
        existing_fraction = existing_exposure / bankroll
        f_incremental = max(0, f_optimal - existing_fraction)

    If the edge has shrunk and existing > optimal, returns 0.0 (exit
    logic in Step 4.4 handles reductions, not Kelly).

    Args:
        signal: Trading signal with direction, EV, P_cons, P_kalshi,
                and kelly_multiplier (from market alignment).
        c: Kalshi fee rate (e.g. 0.07 for 7%).
        K_frac: Fractional Kelly coefficient (e.g. 0.25).
        existing_exposure: Dollars already allocated to this
                           market + direction.
        bankroll: Current account balance in dollars.

    Returns:
        Incremental Kelly fraction (0.0–1.0) to multiply by bankroll.
        Returns 0.0 for HOLD direction or degenerate payoffs.
    """
    if signal.direction == "BUY_YES":
        W = (1.0 - c) * (1.0 - signal.P_kalshi)
        L = signal.P_kalshi
    elif signal.direction == "BUY_NO":
        W = (1.0 - c) * signal.P_kalshi
        L = 1.0 - signal.P_kalshi
    else:
        return 0.0

    if W * L <= 0.0:
        return 0.0

    f_kelly = signal.EV / (W * L)

    # Fractional Kelly + alignment multiplier
    f_optimal = K_frac * f_kelly * signal.kelly_multiplier

    # Incremental: only add what we don't already have
    existing_fraction = existing_exposure / bankroll if bankroll > 0.0 else 0.0
    return max(0.0, f_optimal - existing_fraction)


def apply_risk_limits(
    f_invest: float,
    bankroll: float,
    *,
    current_match_exposure: float = 0.0,
    total_exposure: float = 0.0,
    f_order_cap: float = F_ORDER_CAP,
    f_match_cap: float = F_MATCH_CAP,
    f_total_cap: float = F_TOTAL_CAP,
) -> float:
    """3-layer risk limit: order cap → match cap → total portfolio cap.

    Converts f_invest to a dollar amount then applies three successive
    caps, each potentially reducing the amount further.

    Layer 1 — Single order cap:
        amount ≤ bankroll × F_ORDER_CAP (default 3%)

    Layer 2 — Per-match exposure cap:
        amount ≤ max(0, bankroll × F_MATCH_CAP - current_match_exposure)

    Layer 3 — Total portfolio cap:
        amount ≤ max(0, bankroll × F_TOTAL_CAP - total_exposure)

    Args:
        f_invest: Kelly fraction (0.0–1.0) to invest.
        bankroll: Current account balance in dollars.
        current_match_exposure: Dollars currently allocated across
                                all markets in this match.
        total_exposure: Dollars currently allocated across all matches.
        f_order_cap: Single-order cap as a fraction of bankroll.
        f_match_cap: Per-match cap as a fraction of bankroll.
        f_total_cap: Total portfolio cap as a fraction of bankroll.

    Returns:
        Capped dollar amount to invest (≥ 0.0).
    """
    amount = f_invest * bankroll

    # Layer 1: single order cap
    amount = min(amount, bankroll * f_order_cap)

    # Layer 2: per-match exposure cap
    remaining_match = bankroll * f_match_cap - current_match_exposure
    amount = min(amount, max(0.0, remaining_match))

    # Layer 3: total portfolio cap
    remaining_total = bankroll * f_total_cap - total_exposure
    amount = min(amount, max(0.0, remaining_total))

    return max(0.0, amount)


def liquidity_gate(
    target_qty: int,
    ob_sync: OrderBookSync,
    direction: str,
    *,
    depth_fraction: float = DEPTH_FRACTION,
    min_fill_ratio: float = MIN_FILL_RATIO,
) -> tuple[int, bool]:
    """Gate position size against available order book depth.

    Caps the order at depth_fraction × visible depth to prevent slippage
    traps on thin markets. If the capped size is below min_fill_ratio of
    the Kelly-optimal size, the trade is skipped entirely (preserving edge
    rather than accepting a poor fill).

    Example:
        100 contracts on ask, Kelly wants 50:
          max_qty = int(0.30 × 100) = 30
          30 / 50 = 60% > 50% → proceed with 30 contracts

        100 contracts on ask, Kelly wants 80:
          max_qty = int(0.30 × 100) = 30
          30 / 80 = 37.5% < 50% → skip (return 0, False)

    Args:
        target_qty: Kelly-optimal quantity in contracts.
        ob_sync: Order book state for this market.
        direction: "BUY_YES" (consumes ask depth) or "BUY_NO"
                   (consumes bid depth).
        depth_fraction: Maximum fraction of visible depth to consume.
        min_fill_ratio: Minimum gated_qty / target_qty to proceed.

    Returns:
        (gated_qty, proceed) where proceed=False means skip the order.
    """
    if target_qty <= 0:
        return 0, False

    if direction == "BUY_YES":
        available = ob_sync.total_ask_depth
    elif direction == "BUY_NO":
        available = ob_sync.total_bid_depth
    else:
        return 0, False

    if available <= 0:
        return 0, False

    max_qty = int(depth_fraction * available)
    gated_qty = min(target_qty, max_qty)

    if gated_qty < target_qty * min_fill_ratio:
        return 0, False

    return max(gated_qty, 1), True
