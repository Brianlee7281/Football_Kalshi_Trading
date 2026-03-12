"""Property: P&L sign consistency with direction and outcome.

BUY_YES wins when settlement > entry; BUY_NO wins when settlement < entry.
Fee only on profits → net P&L sign matches gross P&L sign (fee never
flips a profit to a loss or makes a loss worse in sign).

Reference: docs/phase4.md Step 4.6
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.common.types import Position
from src.execution.settlement import compute_realized_pnl


def _pos(direction: str, entry_price: float, quantity: int) -> Position:
    return Position(
        match_id="m1",
        market_ticker="T",
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
    )


# ---------------------------------------------------------------------------
# BUY_YES sign consistency
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    settlement=st.floats(min_value=0.0, max_value=1.0),
    qty=st.integers(min_value=1, max_value=10_000),
    fee_rate=st.floats(min_value=0.0, max_value=0.5),
)
def test_buy_yes_pnl_sign(
    entry: float, settlement: float, qty: int, fee_rate: float
) -> None:
    """BUY_YES: net P&L sign matches (settlement - entry) sign."""
    pos = _pos("BUY_YES", entry, qty)
    pnl = compute_realized_pnl(pos, settlement, fee_rate)
    gross = (settlement - entry) * qty

    if gross > 0:
        assert pnl >= 0.0, f"Profitable gross={gross} but net={pnl}"
    elif gross < 0:
        assert pnl < 0.0, f"Loss gross={gross} but net={pnl}"
    else:
        assert pnl == 0.0


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    qty=st.integers(min_value=1, max_value=10_000),
    fee_rate=st.floats(min_value=0.0, max_value=0.5),
)
def test_buy_yes_win_positive(entry: float, qty: int, fee_rate: float) -> None:
    """BUY_YES with settlement=1.0 always has non-negative P&L."""
    pos = _pos("BUY_YES", entry, qty)
    pnl = compute_realized_pnl(pos, 1.0, fee_rate)
    assert pnl >= 0.0


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    qty=st.integers(min_value=1, max_value=10_000),
)
def test_buy_yes_loss_negative(entry: float, qty: int) -> None:
    """BUY_YES with settlement=0.0 always has negative P&L."""
    pos = _pos("BUY_YES", entry, qty)
    pnl = compute_realized_pnl(pos, 0.0)
    assert pnl < 0.0


# ---------------------------------------------------------------------------
# BUY_NO sign consistency
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    settlement=st.floats(min_value=0.0, max_value=1.0),
    qty=st.integers(min_value=1, max_value=10_000),
    fee_rate=st.floats(min_value=0.0, max_value=0.5),
)
def test_buy_no_pnl_sign(
    entry: float, settlement: float, qty: int, fee_rate: float
) -> None:
    """BUY_NO: net P&L sign matches (entry - settlement) sign."""
    pos = _pos("BUY_NO", entry, qty)
    pnl = compute_realized_pnl(pos, settlement, fee_rate)
    gross = (entry - settlement) * qty

    if gross > 0:
        assert pnl >= 0.0, f"Profitable gross={gross} but net={pnl}"
    elif gross < 0:
        assert pnl < 0.0, f"Loss gross={gross} but net={pnl}"
    else:
        assert pnl == 0.0


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    qty=st.integers(min_value=1, max_value=10_000),
    fee_rate=st.floats(min_value=0.0, max_value=0.5),
)
def test_buy_no_win_positive(entry: float, qty: int, fee_rate: float) -> None:
    """BUY_NO with settlement=0.0 always has non-negative P&L."""
    pos = _pos("BUY_NO", entry, qty)
    pnl = compute_realized_pnl(pos, 0.0, fee_rate)
    assert pnl >= 0.0


@settings(max_examples=1000)
@given(
    entry=st.floats(min_value=0.01, max_value=0.99),
    qty=st.integers(min_value=1, max_value=10_000),
)
def test_buy_no_loss_negative(entry: float, qty: int) -> None:
    """BUY_NO with settlement=1.0 always has negative P&L."""
    pos = _pos("BUY_NO", entry, qty)
    pnl = compute_realized_pnl(pos, 1.0)
    assert pnl < 0.0


# ---------------------------------------------------------------------------
# Fee never flips sign
# ---------------------------------------------------------------------------


@settings(max_examples=1000)
@given(
    direction=st.sampled_from(["BUY_YES", "BUY_NO"]),
    entry=st.floats(min_value=0.01, max_value=0.99),
    settlement=st.floats(min_value=0.0, max_value=1.0),
    qty=st.integers(min_value=1, max_value=10_000),
    fee_rate=st.floats(min_value=0.0, max_value=1.0),
)
def test_fee_never_makes_profit_negative(
    direction: str, entry: float, settlement: float, qty: int, fee_rate: float
) -> None:
    """Fee on profits can reduce them to zero but never below zero."""
    pos = _pos(direction, entry, qty)
    pnl_no_fee = compute_realized_pnl(pos, settlement, 0.0)
    pnl_with_fee = compute_realized_pnl(pos, settlement, fee_rate)

    if pnl_no_fee > 0:
        assert pnl_with_fee >= 0.0, "Fee flipped profit to loss"
    if pnl_no_fee < 0:
        assert pnl_with_fee == pnl_no_fee, "Fee applied to loss"
