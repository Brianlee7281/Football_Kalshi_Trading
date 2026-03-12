"""Gap-filling tests for exit_logic — BUY_NO position_trim, bankroll=0, W*L=0.

Supplements test_exit_logic.py with Trigger 5 BUY_NO and edge cases.

Reference: docs/phase4.md Step 4.4
"""

from __future__ import annotations

import pytest

from src.common.types import Position
from src.execution.exit_logic import (
    check_edge_decay,
    check_position_trim,
)

_C = 0.07
_Z = 1.645
_K = 0.25
_BANKROLL = 10_000.0


def _pos(
    *,
    direction: str = "BUY_YES",
    entry_price: float = 0.50,
    quantity: int = 100,
    kelly_multiplier: float = 0.8,
) -> Position:
    return Position(
        match_id="m1",
        market_ticker="T",
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        kelly_multiplier=kelly_multiplier,
    )


# ---------------------------------------------------------------------------
# Trigger 5 (position_trim) — BUY_NO direction
# ---------------------------------------------------------------------------


def test_position_trim_buy_no_fires() -> None:
    """BUY_NO position trim fires when f_optimal << f_existing."""
    # Large BUY_NO position: 2000 contracts at 0.55 → f_existing = 2000*0.55/10000 = 0.11
    # P_true=0.45 means No side: P_cons = 0.45 + z*0 = 0.45
    # W = (1-0.07)*0.50 = 0.465, L = 1-0.50 = 0.50
    # ev = (1-0.45)*0.465 - 0.45*0.50 = 0.25575 - 0.225 = 0.03075
    # f_optimal = 0.25 * (0.03075/(0.465*0.50)) * 0.8 ≈ 0.0265
    # 0.0265 < 0.5 * 0.11 → trim fires
    pos = _pos(direction="BUY_NO", entry_price=0.55, quantity=2000)
    result = check_position_trim(pos, P_true=0.45, sigma_MC=0.0,
                                  P_kalshi_bid=0.50, c=_C, z=_Z,
                                  K_frac=_K, bankroll=_BANKROLL)
    assert result is not None
    assert result.reason == "POSITION_TRIM"
    assert result.trim_quantity is not None
    assert result.trim_quantity > 0


def test_position_trim_buy_no_no_fire_small_position() -> None:
    """BUY_NO with small position → no trim needed."""
    pos = _pos(direction="BUY_NO", entry_price=0.55, quantity=10)
    result = check_position_trim(pos, P_true=0.40, sigma_MC=0.0,
                                  P_kalshi_bid=0.50, c=_C, z=_Z,
                                  K_frac=_K, bankroll=_BANKROLL)
    assert result is None


# ---------------------------------------------------------------------------
# Trigger 5 — bankroll=0 edge case
# ---------------------------------------------------------------------------


def test_position_trim_bankroll_zero() -> None:
    """bankroll=0 → None (no ZeroDivisionError)."""
    pos = _pos(quantity=100)
    result = check_position_trim(pos, P_true=0.55, sigma_MC=0.0,
                                  P_kalshi_bid=0.50, c=_C, z=_Z,
                                  K_frac=_K, bankroll=0.0)
    assert result is None


def test_position_trim_p_kalshi_bid_zero() -> None:
    """P_kalshi_bid=0 → None (no ZeroDivisionError)."""
    pos = _pos(quantity=100)
    result = check_position_trim(pos, P_true=0.55, sigma_MC=0.0,
                                  P_kalshi_bid=0.0, c=_C, z=_Z,
                                  K_frac=_K, bankroll=_BANKROLL)
    assert result is None


# ---------------------------------------------------------------------------
# Trigger 1 — σ_MC=0 edge case
# ---------------------------------------------------------------------------


def test_edge_decay_sigma_zero_buy_yes() -> None:
    """σ_MC=0 with positive edge → no decay (P_cons = P_true)."""
    pos = _pos(direction="BUY_YES", entry_price=0.40)
    result = check_edge_decay(pos, P_true=0.60, sigma_MC=0.0,
                               P_kalshi_bid=0.45, c=_C, z=_Z)
    # P_cons = 0.60, ev = 0.60*W - 0.40*L > 0 → no decay
    assert result is None


def test_edge_decay_sigma_zero_buy_no() -> None:
    """σ_MC=0 BUY_NO with positive edge → no decay."""
    pos = _pos(direction="BUY_NO", entry_price=0.55)
    result = check_edge_decay(pos, P_true=0.40, sigma_MC=0.0,
                               P_kalshi_bid=0.50, c=_C, z=_Z)
    # P_cons = 0.40, ev = (1-0.40)*W - 0.40*L
    # W = (1-0.07)*0.50 = 0.465, L = 0.50
    # ev = 0.60*0.465 - 0.40*0.50 = 0.279 - 0.20 = 0.079 > 0
    assert result is None
