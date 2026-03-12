"""Gap-filling tests for settlement — mixed directions, boundary settlement prices.

Supplements test_settlement.py with directional and edge case coverage.

Reference: docs/phase4.md Step 4.6
"""

from __future__ import annotations

import pytest

from src.common.types import Position
from src.execution.settlement import FEE_RATE, compute_realized_pnl


def _pos(
    *,
    direction: str = "BUY_YES",
    entry_price: float = 0.45,
    quantity: int = 10,
) -> Position:
    return Position(
        match_id="m1",
        market_ticker="T",
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
    )


# ---------------------------------------------------------------------------
# Directional: BUY_YES
# ---------------------------------------------------------------------------


def test_buy_yes_win() -> None:
    """BUY_YES wins (settlement=1.0): gross=+0.55*10, fee on profit."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=1.0)
    gross = (1.0 - 0.45) * 10
    fee = FEE_RATE * gross
    assert pnl == pytest.approx(gross - fee)


def test_buy_yes_lose() -> None:
    """BUY_YES loses (settlement=0.0): gross=-0.45*10, no fee."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.0)
    assert pnl == pytest.approx(-0.45 * 10)


# ---------------------------------------------------------------------------
# Directional: BUY_NO
# ---------------------------------------------------------------------------


def test_buy_no_win() -> None:
    """BUY_NO wins (settlement=0.0): gross=+entry*qty, fee on profit."""
    pos = _pos(direction="BUY_NO", entry_price=0.40, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.0)
    gross = 0.40 * 10
    fee = FEE_RATE * gross
    assert pnl == pytest.approx(gross - fee)


def test_buy_no_lose() -> None:
    """BUY_NO loses (settlement=1.0): gross=-(1-entry)*qty, no fee."""
    pos = _pos(direction="BUY_NO", entry_price=0.40, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=1.0)
    assert pnl == pytest.approx((0.40 - 1.0) * 10)


# ---------------------------------------------------------------------------
# Settlement price = 0.5 (non-standard edge case)
# ---------------------------------------------------------------------------


def test_buy_yes_settlement_half() -> None:
    """BUY_YES with settlement=0.5, entry=0.45 → small profit."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.5)
    gross = (0.5 - 0.45) * 10
    fee = FEE_RATE * max(0, gross)
    assert pnl == pytest.approx(gross - fee)


def test_buy_no_settlement_half() -> None:
    """BUY_NO with settlement=0.5, entry=0.40 → loss."""
    pos = _pos(direction="BUY_NO", entry_price=0.40, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.5)
    gross = (0.40 - 0.5) * 10
    assert pnl == pytest.approx(gross)  # negative → no fee


# ---------------------------------------------------------------------------
# fee_rate=0 and fee_rate=1.0
# ---------------------------------------------------------------------------


def test_zero_fee_rate() -> None:
    """fee_rate=0 → net = gross."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=1)
    pnl = compute_realized_pnl(pos, settlement_price=1.0, fee_rate=0.0)
    assert pnl == pytest.approx(0.55)


def test_full_fee_rate() -> None:
    """fee_rate=1.0 → all profit taken as fee (net = 0 for wins)."""
    pos = _pos(direction="BUY_YES", entry_price=0.45, quantity=1)
    pnl = compute_realized_pnl(pos, settlement_price=1.0, fee_rate=1.0)
    assert pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Breakeven (settlement = entry)
# ---------------------------------------------------------------------------


def test_breakeven_buy_yes() -> None:
    """BUY_YES: settlement = entry → gross = 0, net = 0."""
    pos = _pos(direction="BUY_YES", entry_price=0.50, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.50)
    assert pnl == pytest.approx(0.0)


def test_breakeven_buy_no() -> None:
    """BUY_NO: settlement = entry → gross = 0, net = 0."""
    pos = _pos(direction="BUY_NO", entry_price=0.50, quantity=10)
    pnl = compute_realized_pnl(pos, settlement_price=0.50)
    assert pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mixed directions in sequence (not simultaneous, but verifies sign)
# ---------------------------------------------------------------------------


def test_mixed_directions_opposite_signs() -> None:
    """BUY_YES win and BUY_NO win with same settlement → opposite underlying outcomes."""
    # settlement = 1.0: Yes wins → BUY_YES profits, BUY_NO loses
    yes_pnl = compute_realized_pnl(
        _pos(direction="BUY_YES", entry_price=0.50), settlement_price=1.0
    )
    no_pnl = compute_realized_pnl(
        _pos(direction="BUY_NO", entry_price=0.50), settlement_price=1.0
    )
    assert yes_pnl > 0
    assert no_pnl < 0


# ---------------------------------------------------------------------------
# Unknown direction
# ---------------------------------------------------------------------------


def test_unknown_direction_zero() -> None:
    """Unknown direction → gross=0, net=0."""
    pos = Position(
        match_id="m1", market_ticker="T",
        direction="SELL_YES", entry_price=0.5, quantity=10,
    )
    pnl = compute_realized_pnl(pos, settlement_price=1.0)
    assert pnl == 0.0
