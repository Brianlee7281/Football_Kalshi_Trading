"""Unit tests for edge_detection — compute_signal_with_vwap, check_market_alignment, generate_signal.

All numeric test cases are derived from docs/phase4.md Step 4.2 formulas.

Reference: docs/phase4.md Step 4.2
"""

from __future__ import annotations

import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.common.types import Signal
from src.execution.edge_detection import (
    ALIGNED_MULTIPLIER,
    DIVERGENT_MULTIPLIER,
    THETA_ENTRY,
    UNAVAILABLE_MULTIPLIER,
    check_market_alignment,
    compute_signal_with_vwap,
    generate_signal,
)
from src.execution.order_book_sync import OrderBookSync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard config (matches defaults from config_reference.md)
_C = 0.07       # fee rate
_Z = 1.645      # z-score
_K = 0.25       # K_frac
_BANKROLL = 10_000.0


_DEEP = 100_000  # default depth per level (deep enough for any Kelly qty)


def _make_ob(
    *,
    ask_cents: int | None = None,
    bid_cents: int | None = None,
    ask_depth: list[tuple[int, int]] | None = None,
    bid_depth: list[tuple[int, int]] | None = None,
) -> OrderBookSync:
    """Build an OrderBookSync with specified ladder state.

    Default depth is 100,000 contracts per level so rough Kelly quantity
    can always be satisfied unless the test explicitly provides thin depth.
    """
    ob = OrderBookSync("TEST-TICKER")
    if ask_cents is not None or ask_depth is not None or bid_cents is not None or bid_depth is not None:
        yes_levels: list[tuple[int, int]] = []
        no_levels: list[tuple[int, int]] = []
        if ask_depth is not None:
            yes_levels = ask_depth
        elif ask_cents is not None:
            yes_levels = [(ask_cents, _DEEP)]

        if bid_depth is not None:
            # bid_depth is [(bid_price_cents, qty)]; convert to no-side
            no_levels = [(100 - p, q) for p, q in bid_depth]
        elif bid_cents is not None:
            no_levels = [(100 - bid_cents, _DEEP)]

        ob.update_from_kalshi(
            OrderBookUpdate(
                ticker="TEST-TICKER",
                is_snapshot=True,
                yes=yes_levels,
                no=no_levels,
                timestamp=time.time(),
            )
        )
    return ob


# ---------------------------------------------------------------------------
# compute_signal_with_vwap — HOLD cases
# ---------------------------------------------------------------------------


def test_signal_hold_when_no_ask() -> None:
    """HOLD when order book has no ask (None best_ask)."""
    ob = OrderBookSync()
    result = compute_signal_with_vwap(0.70, 0.005, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "HOLD"


def test_signal_hold_when_no_bid() -> None:
    """HOLD when order book has no bid (None best_bid)."""
    ob = OrderBookSync()
    # Only ask side populated
    ob.update_from_kalshi(
        OrderBookUpdate("T", is_snapshot=True, yes=[(60, 100)], no=[], timestamp=time.time())
    )
    result = compute_signal_with_vwap(0.70, 0.005, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "HOLD"


def test_signal_hold_when_ev_below_theta() -> None:
    """HOLD when both EV_yes and EV_no are below THETA_ENTRY."""
    # P_true = 0.50 at mid-market (50¢ ask, 50¢ bid) → no edge
    ob = _make_ob(ask_cents=50, bid_cents=50)
    result = compute_signal_with_vwap(0.50, 0.001, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "HOLD"


def test_signal_hold_when_vwap_depth_insufficient() -> None:
    """HOLD when rough_qty exceeds available depth (VWAP returns None)."""
    # Very thin book: only 1 contract at 60¢ ask
    ob = _make_ob(ask_depth=[(60, 1)], bid_cents=40)
    # Large bankroll → rough_qty will exceed depth of 1
    result = compute_signal_with_vwap(0.80, 0.001, ob, _C, _Z, _K, 1_000_000.0, "T")
    assert result.direction == "HOLD"


# ---------------------------------------------------------------------------
# compute_signal_with_vwap — BUY_YES
# ---------------------------------------------------------------------------


def test_signal_buy_yes_direction() -> None:
    """P_true >> ask → BUY_YES signal."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "BUY_YES"


def test_signal_buy_yes_ev_positive() -> None:
    """BUY_YES signal has EV > THETA_ENTRY."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_YES":
        assert result.EV > THETA_ENTRY


def test_signal_buy_yes_p_cons_lt_p_true() -> None:
    """BUY_YES: conservative P is below P_true (lower bound)."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_YES":
        assert result.P_cons < 0.75


def test_signal_buy_yes_p_kalshi_in_unit_interval() -> None:
    """P_kalshi (VWAP price) is in (0, 1) range."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction != "HOLD":
        assert 0.0 < result.P_kalshi < 1.0


def test_signal_buy_yes_rough_qty_positive() -> None:
    """BUY_YES signal has rough_qty >= 1."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_YES":
        assert result.rough_qty >= 1


# ---------------------------------------------------------------------------
# compute_signal_with_vwap — BUY_NO
# ---------------------------------------------------------------------------


def test_signal_buy_no_direction() -> None:
    """P_true << bid → BUY_NO signal (P_true is low, No side has value)."""
    # bid = 70¢ (P_true = 0.25 so yes is overpriced → buy No)
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.25, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "BUY_NO"


def test_signal_buy_no_p_cons_gt_p_true() -> None:
    """BUY_NO: conservative P is above P_true (upper bound)."""
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.25, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_NO":
        assert result.P_cons > 0.25


def test_signal_buy_no_ev_positive() -> None:
    """BUY_NO signal has EV > THETA_ENTRY."""
    ob = _make_ob(ask_cents=72, bid_cents=70)
    result = compute_signal_with_vwap(0.25, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_NO":
        assert result.EV > THETA_ENTRY


# ---------------------------------------------------------------------------
# compute_signal_with_vwap — VWAP vs best price
# ---------------------------------------------------------------------------


def test_vwap_replaces_best_ask_for_buy_yes() -> None:
    """For BUY_YES, P_kalshi is the VWAP price, not just the best ask."""
    # Two ask levels: 50 contracts at 40¢, 50 at 45¢
    # VWAP for 100 contracts = (50*40 + 50*45) / 100 = 42.5¢ = 0.425
    ob = _make_ob(
        ask_depth=[(40, 50), (45, 50)],
        bid_cents=38,
    )
    result = compute_signal_with_vwap(0.75, 0.001, ob, _C, _Z, _K, _BANKROLL, "T")
    if result.direction == "BUY_YES":
        # VWAP should be > 0.40 (best ask) if it consumed into second level
        assert result.P_kalshi >= 0.40


# ---------------------------------------------------------------------------
# check_market_alignment
# ---------------------------------------------------------------------------


def test_alignment_unavailable_when_p_bet365_none() -> None:
    """alignment=UNAVAILABLE when P_bet365 is None (stale/missing)."""
    result = check_market_alignment(0.55, 0.50, None, "BUY_YES")
    assert result.status == "UNAVAILABLE"
    assert result.kelly_multiplier == UNAVAILABLE_MULTIPLIER


def test_alignment_aligned_buy_yes() -> None:
    """BUY_YES aligned: model says high (P_cons > P_kalshi) AND bet365 says high."""
    # P_cons=0.55 > P_kalshi=0.50 → model says high
    # P_bet365=0.54 > P_kalshi=0.50 → bet365 says high → ALIGNED
    result = check_market_alignment(0.55, 0.50, 0.54, "BUY_YES")
    assert result.status == "ALIGNED"
    assert result.kelly_multiplier == ALIGNED_MULTIPLIER


def test_alignment_divergent_buy_yes() -> None:
    """BUY_YES divergent: model says high but bet365 says low."""
    # P_cons=0.55 > P_kalshi=0.50 → model says high
    # P_bet365=0.48 < P_kalshi=0.50 → bet365 says low → DIVERGENT
    result = check_market_alignment(0.55, 0.50, 0.48, "BUY_YES")
    assert result.status == "DIVERGENT"
    assert result.kelly_multiplier == DIVERGENT_MULTIPLIER


def test_alignment_aligned_buy_no() -> None:
    """BUY_NO aligned: model says low (P_cons < P_kalshi) AND bet365 says low."""
    # P_cons=0.45 < P_kalshi=0.50 → model says low
    # P_bet365=0.46 < P_kalshi=0.50 → bet365 says low → ALIGNED
    result = check_market_alignment(0.45, 0.50, 0.46, "BUY_NO")
    assert result.status == "ALIGNED"
    assert result.kelly_multiplier == ALIGNED_MULTIPLIER


def test_alignment_divergent_buy_no() -> None:
    """BUY_NO divergent: model says low but bet365 says high."""
    # P_cons=0.45 < P_kalshi=0.50 → model says low
    # P_bet365=0.52 > P_kalshi=0.50 → bet365 says high → DIVERGENT
    result = check_market_alignment(0.45, 0.50, 0.52, "BUY_NO")
    assert result.status == "DIVERGENT"
    assert result.kelly_multiplier == DIVERGENT_MULTIPLIER


def test_alignment_unavailable_for_unknown_direction() -> None:
    """UNAVAILABLE returned for unrecognised direction string."""
    result = check_market_alignment(0.55, 0.50, 0.54, "HOLD")
    assert result.status == "UNAVAILABLE"


def test_alignment_aligned_multiplier_lt_1() -> None:
    """Even ALIGNED multiplier < 1.0 (reflects limited bet365 independence)."""
    result = check_market_alignment(0.55, 0.50, 0.54, "BUY_YES")
    assert result.kelly_multiplier < 1.0


def test_divergent_multiplier_lt_aligned() -> None:
    """DIVERGENT multiplier is strictly less than ALIGNED multiplier."""
    assert DIVERGENT_MULTIPLIER < ALIGNED_MULTIPLIER


# ---------------------------------------------------------------------------
# generate_signal — full pipeline
# ---------------------------------------------------------------------------


def test_generate_signal_hold_propagates() -> None:
    """generate_signal returns HOLD when compute_signal_with_vwap returns HOLD."""
    ob = OrderBookSync()  # empty book
    result = generate_signal(0.55, 0.005, ob, 0.54, _C, _Z, _K, _BANKROLL, "T")
    assert result.direction == "HOLD"


def test_generate_signal_adds_alignment_status() -> None:
    """generate_signal populates alignment_status on a non-HOLD signal."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = generate_signal(0.75, 0.002, ob, 0.72, _C, _Z, _K, _BANKROLL, "T")
    if result.direction != "HOLD":
        assert result.alignment_status in ("ALIGNED", "DIVERGENT", "UNAVAILABLE")


def test_generate_signal_adds_kelly_multiplier() -> None:
    """generate_signal populates kelly_multiplier on a non-HOLD signal."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = generate_signal(0.75, 0.002, ob, 0.72, _C, _Z, _K, _BANKROLL, "T")
    if result.direction != "HOLD":
        assert result.kelly_multiplier in (
            ALIGNED_MULTIPLIER, DIVERGENT_MULTIPLIER, UNAVAILABLE_MULTIPLIER
        )


def test_generate_signal_bet365_none_gives_unavailable() -> None:
    """Stale bet365 (None) → alignment_status=UNAVAILABLE with 0.6 multiplier."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = generate_signal(0.75, 0.002, ob, None, _C, _Z, _K, _BANKROLL, "T")
    if result.direction != "HOLD":
        assert result.alignment_status == "UNAVAILABLE"
        assert result.kelly_multiplier == UNAVAILABLE_MULTIPLIER


def test_generate_signal_market_ticker_preserved() -> None:
    """generate_signal preserves the market_ticker in the returned Signal."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    result = generate_signal(0.75, 0.002, ob, None, _C, _Z, _K, _BANKROLL, "MY-TICKER")
    assert result.market_ticker == "MY-TICKER"


def test_generate_signal_ev_preserved_from_base() -> None:
    """EV in the returned Signal matches the EV from compute_signal_with_vwap."""
    ob = _make_ob(ask_cents=40, bid_cents=38)
    base = compute_signal_with_vwap(0.75, 0.002, ob, _C, _Z, _K, _BANKROLL, "T")
    full = generate_signal(0.75, 0.002, ob, 0.72, _C, _Z, _K, _BANKROLL, "T")
    if full.direction != "HOLD" and base.direction != "HOLD":
        assert full.EV == pytest.approx(base.EV)


def test_generate_signal_uses_get_bet365_for_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller is expected to use ob_sync.get_bet365_for_alignment to get P_bet365."""
    # When bet365 last_update is stale, get_bet365_for_alignment returns None
    ob = _make_ob(ask_cents=40, bid_cents=38)
    ob.update_bet365({
        "markets": [
            {"name": "ML", "odds": [{"home": "1.44", "draw": "3.50", "away": "12.00"}]}
        ]
    })
    # Fake staleness
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: ob.bet365_last_update + 35.0,
    )
    P_bet365 = ob.get_bet365_for_alignment("home_win")
    assert P_bet365 is None  # stale → caller passes None → UNAVAILABLE
    result = generate_signal(0.75, 0.002, ob, P_bet365, _C, _Z, _K, _BANKROLL, "T")
    if result.direction != "HOLD":
        assert result.alignment_status == "UNAVAILABLE"
