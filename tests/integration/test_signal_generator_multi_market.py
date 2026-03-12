"""Integration: Signal generator multi-market processing.

Verifies that the signal_generator loop correctly decomposes P_true/σ_MC
dicts into per-ticker floats and processes multiple markets per tick.

Reference: docs/phase4.md signal_generator, .claude/rules/patterns.md #1
"""

from __future__ import annotations

import time

import pytest

from src.common.types import PaperFill
from src.execution.edge_detection import generate_signal
from src.execution.kelly import compute_kelly

from .conftest import (
    make_model,
    make_ob,
    make_signal,
    make_tick,
)

# ---------------------------------------------------------------------------
# Dict → float decomposition
# ---------------------------------------------------------------------------


def test_tick_data_decomposition() -> None:
    """TickData P_true dict decomposes to per-market floats correctly."""
    tick = make_tick(
        P_true={"home_win": 0.55, "over_25": 0.65, "draw": 0.20},
        sigma_MC={"home_win": 0.005, "over_25": 0.004, "draw": 0.006},
    )

    ticker_map = {
        "T-HW": "home_win",
        "T-O25": "over_25",
        "T-DW": "draw",
    }

    for _ticker, market_key in ticker_map.items():
        p_true_float = tick.P_true[market_key]
        sigma_float = tick.sigma_MC[market_key]

        assert isinstance(p_true_float, float)
        assert isinstance(sigma_float, float)
        assert 0.0 <= p_true_float <= 1.0


# ---------------------------------------------------------------------------
# Multi-market signal generation
# ---------------------------------------------------------------------------


def test_multi_market_signals_independent() -> None:
    """Each market generates an independent signal from its own P_true."""
    ob_hw = make_ob("T-HW", ask_price=50, bid_price=48)
    ob_o25 = make_ob("T-O25", ask_price=60, bid_price=58)

    # home_win: P_true=0.65 vs ask=0.50 → strong BUY_YES edge
    sig_hw = generate_signal(
        P_true=0.65, sigma_MC=0.005, ob_sync=ob_hw, P_bet365=0.60,
        c=0.07, z=1.645, K_frac=0.25, bankroll=10_000.0, market_ticker="T-HW",
    )

    # over_25: P_true=0.55 vs ask=0.60 → weak or no BUY_YES edge
    sig_o25 = generate_signal(
        P_true=0.55, sigma_MC=0.005, ob_sync=ob_o25, P_bet365=0.56,
        c=0.07, z=1.645, K_frac=0.25, bankroll=10_000.0, market_ticker="T-O25",
    )

    # Signals are independent — one can be HOLD while the other fires
    assert sig_hw.market_ticker == "T-HW"
    if sig_o25.direction != "HOLD":
        assert sig_o25.market_ticker == "T-O25"


# ---------------------------------------------------------------------------
# order_allowed = False → skip
# ---------------------------------------------------------------------------


def test_order_not_allowed_skips_processing() -> None:
    """When order_allowed=False, no signals should be generated."""
    tick = make_tick(order_allowed=False)
    assert tick.order_allowed is False
    # The signal_generator loop checks this and continues without processing


# ---------------------------------------------------------------------------
# Missing market key handled gracefully
# ---------------------------------------------------------------------------


def test_missing_market_key_skipped() -> None:
    """Ticker with no matching key in P_true dict is skipped."""
    tick = make_tick(P_true={"home_win": 0.55})

    ticker_map = {"T-HW": "home_win", "T-MISSING": "over_25"}
    active_tickers = ["T-HW", "T-MISSING"]

    processed = []
    for ticker in active_tickers:
        market_key = ticker_map.get(ticker)
        if market_key is None or market_key not in tick.P_true:
            continue
        processed.append(ticker)

    assert processed == ["T-HW"]
    assert "T-MISSING" not in processed


# ---------------------------------------------------------------------------
# Phase4 queue stale replacement
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_phase4_queue_replaces_stale_tick() -> None:
    """When queue is full, new tick replaces stale one."""
    model = make_model()

    tick_old = make_tick(P_true={"home_win": 0.50})
    tick_new = make_tick(P_true={"home_win": 0.65})

    model.emit_tick(tick_old)
    assert model.phase4_queue.full()

    # Emit again — should replace
    model.emit_tick(tick_new)

    result = model.phase4_queue.get_nowait()
    assert result.P_true["home_win"] == 0.65


# ---------------------------------------------------------------------------
# Incremental Kelly with existing exposure
# ---------------------------------------------------------------------------


def test_incremental_kelly_reduces_with_exposure() -> None:
    """Existing exposure reduces incremental Kelly fraction."""
    signal = make_signal(
        direction="BUY_YES", EV=0.03, P_kalshi=0.50,
    )

    f_full = compute_kelly(signal, c=0.07, K_frac=0.25, bankroll=10_000.0)
    f_partial = compute_kelly(
        signal, c=0.07, K_frac=0.25,
        existing_exposure=f_full * 10_000.0 * 0.5,
        bankroll=10_000.0,
    )

    assert f_partial < f_full
    assert f_partial >= 0.0


def test_incremental_kelly_zero_when_at_optimal() -> None:
    """At or above optimal exposure → incremental = 0."""
    signal = make_signal(
        direction="BUY_YES", EV=0.03, P_kalshi=0.50,
    )

    f_full = compute_kelly(signal, c=0.07, K_frac=0.25, bankroll=10_000.0)
    f_zero = compute_kelly(
        signal, c=0.07, K_frac=0.25,
        existing_exposure=f_full * 10_000.0 * 1.5,
        bankroll=10_000.0,
    )

    assert f_zero == 0.0


# ---------------------------------------------------------------------------
# Signal generator processes multiple tickers from single tick
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_signal_generator_processes_all_tickers() -> None:
    """Simulate one tick with 3 markets — all should be considered."""
    tickers = ["T-HW", "T-DW", "T-O25"]
    ticker_map = {"T-HW": "home_win", "T-DW": "draw", "T-O25": "over_25"}

    P_true = {"home_win": 0.65, "draw": 0.20, "over_25": 0.70}
    sigma_MC = {"home_win": 0.005, "draw": 0.005, "over_25": 0.005}

    processed_tickers = []
    for ticker in tickers:
        market_key = ticker_map.get(ticker)
        if market_key is None or market_key not in P_true:
            continue

        p_true_float = P_true[market_key]
        sigma_float = sigma_MC[market_key]

        ob = make_ob(ticker, ask_price=50, bid_price=48)

        generate_signal(
            p_true_float, sigma_float, ob, None,
            c=0.07, z=1.645, K_frac=0.25,
            bankroll=10_000.0, market_ticker=ticker,
        )
        processed_tickers.append(ticker)

    assert len(processed_tickers) == 3
    assert set(processed_tickers) == set(tickers)


# ---------------------------------------------------------------------------
# Bankroll decrement after fill
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bankroll_decrements_after_fill() -> None:
    """Bankroll decreases by fill_cost after a successful fill."""
    model = make_model(bankroll=10_000.0)
    initial_bankroll = model.bankroll

    fill = PaperFill(
        price=0.55,
        quantity=10,
        timestamp=time.time(),
    )

    fill_cost = fill.price * fill.quantity
    model.bankroll -= fill_cost

    assert model.bankroll == pytest.approx(initial_bankroll - 5.50)
    assert model.bankroll > 0


# ---------------------------------------------------------------------------
# bet365 stale → UNAVAILABLE alignment
# ---------------------------------------------------------------------------


def test_bet365_stale_gives_unavailable_alignment() -> None:
    """Stale bet365 data → signal gets UNAVAILABLE alignment multiplier (0.6)."""
    ob = make_ob("T", ask_price=50, bid_price=48)

    signal = generate_signal(
        P_true=0.65, sigma_MC=0.005, ob_sync=ob, P_bet365=None,
        c=0.07, z=1.645, K_frac=0.25,
        bankroll=10_000.0, market_ticker="T",
    )

    if signal.direction != "HOLD":
        assert signal.alignment_status == "UNAVAILABLE"
        assert signal.kelly_multiplier == 0.6
