"""Unit tests for OrderBookSync — staleness, VWAP, bet365 parsing.

Covers:
  - kalshi_is_stale: True when never updated or updated > 5s ago
  - bet365_is_stale: True when never updated or updated > 30s ago
  - update_from_kalshi: snapshot replaces ladders, delta mutates them
  - compute_vwap_buy / compute_vwap_sell: exact VWAP math
  - update_bet365: ML + Totals parsing with vig normalisation
  - get_bet365_for_alignment: returns None when stale
  - liquidity_ok: threshold check

Reference: docs/phase4.md Step 4.1
"""

from __future__ import annotations

import time

import pytest

from src.clients.kalshi import OrderBookUpdate
from src.execution.order_book_sync import OrderBookSync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(yes: list[tuple[int, int]], no: list[tuple[int, int]]) -> OrderBookUpdate:
    return OrderBookUpdate(
        ticker="TEST-TICK",
        is_snapshot=True,
        yes=yes,
        no=no,
        timestamp=time.time(),
    )


def _delta(yes: list[tuple[int, int]], no: list[tuple[int, int]]) -> OrderBookUpdate:
    return OrderBookUpdate(
        ticker="TEST-TICK",
        is_snapshot=False,
        yes=yes,
        no=no,
        timestamp=time.time(),
    )


def _ml_msg(home: float, draw: float, away: float) -> dict:  # type: ignore[type-arg]
    return {
        "type": "updated",
        "bookie": "Bet365",
        "markets": [
            {"name": "ML", "odds": [{"home": str(home), "draw": str(draw), "away": str(away)}]},
        ],
    }


def _totals_msg(over: float, under: float, hdp: float = 2.5) -> dict:  # type: ignore[type-arg]
    return {
        "type": "updated",
        "bookie": "Bet365",
        "markets": [
            {"name": "Totals", "odds": [{"over": str(over), "under": str(under), "hdp": hdp}]},
        ],
    }


# ---------------------------------------------------------------------------
# Staleness — Kalshi (5s threshold)
# ---------------------------------------------------------------------------


def test_kalshi_stale_never_updated() -> None:
    """kalshi_is_stale=True when no update has ever been received."""
    ob = OrderBookSync("TICKER")
    assert ob.kalshi_is_stale is True


def test_kalshi_fresh_after_update() -> None:
    """kalshi_is_stale=False immediately after a snapshot update."""
    ob = OrderBookSync("TICKER")
    ob.update_from_kalshi(_snapshot([(60, 50)], [(40, 50)]))
    assert ob.kalshi_is_stale is False


def test_kalshi_stale_after_6s(monkeypatch: pytest.MonkeyPatch) -> None:
    """kalshi_is_stale=True when last update was 6 seconds ago."""
    ob = OrderBookSync("TICKER")
    ob.update_from_kalshi(_snapshot([(60, 50)], [(40, 50)]))
    # Fake monotonic clock to 6 seconds later
    original_time = ob.kalshi_last_update
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: original_time + 6.0,
    )
    assert ob.kalshi_is_stale is True


def test_kalshi_fresh_after_4s(monkeypatch: pytest.MonkeyPatch) -> None:
    """kalshi_is_stale=False when last update was only 4 seconds ago."""
    ob = OrderBookSync("TICKER")
    ob.update_from_kalshi(_snapshot([(60, 50)], [(40, 50)]))
    original_time = ob.kalshi_last_update
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: original_time + 4.0,
    )
    assert ob.kalshi_is_stale is False


# ---------------------------------------------------------------------------
# Staleness — bet365 (30s threshold)
# ---------------------------------------------------------------------------


def test_bet365_stale_never_updated() -> None:
    """bet365_is_stale=True when no bet365 message has been received."""
    ob = OrderBookSync("TICKER")
    assert ob.bet365_is_stale is True


def test_bet365_fresh_after_update() -> None:
    """bet365_is_stale=False immediately after a bet365 message."""
    ob = OrderBookSync("TICKER")
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    assert ob.bet365_is_stale is False


def test_bet365_stale_after_35s(monkeypatch: pytest.MonkeyPatch) -> None:
    """bet365_is_stale=True when last bet365 update was 35 seconds ago."""
    ob = OrderBookSync("TICKER")
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    original_time = ob.bet365_last_update
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: original_time + 35.0,
    )
    assert ob.bet365_is_stale is True


def test_bet365_fresh_after_25s(monkeypatch: pytest.MonkeyPatch) -> None:
    """bet365_is_stale=False when last bet365 update was 25 seconds ago."""
    ob = OrderBookSync("TICKER")
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    original_time = ob.bet365_last_update
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: original_time + 25.0,
    )
    assert ob.bet365_is_stale is False


# ---------------------------------------------------------------------------
# update_from_kalshi — snapshot
# ---------------------------------------------------------------------------


def test_snapshot_sets_best_bid_ask() -> None:
    """After snapshot: best_bid and best_ask derived from ladders."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot(yes=[(60, 100), (61, 50)], no=[(39, 80), (38, 60)]))
    # Ask lowest = 60; Bid highest = 100 - 38 = 62 (but no side: 39 → bid = 100-39=61, 38 → 62)
    assert ob.kalshi_best_ask == 60
    # no side: 39→61, 38→62 → descending: [(62,60),(61,80)] → best_bid = 62
    assert ob.kalshi_best_bid == 62


def test_snapshot_replaces_previous_ladders() -> None:
    """Second snapshot fully replaces prior ladder state."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(55, 200)], [(45, 200)]))
    ob.update_from_kalshi(_snapshot([(62, 50)], [(38, 50)]))
    assert ob.kalshi_depth_ask == [(62, 50)]
    # no: 38 → bid_price 62
    assert ob.kalshi_depth_bid == [(62, 50)]


def test_snapshot_empty_ladders() -> None:
    """Snapshot with empty yes/no produces None bid/ask and empty ladders."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([], []))
    assert ob.kalshi_best_bid is None
    assert ob.kalshi_best_ask is None
    assert ob.kalshi_depth_ask == []
    assert ob.kalshi_depth_bid == []


# ---------------------------------------------------------------------------
# update_from_kalshi — delta
# ---------------------------------------------------------------------------


def test_delta_adds_new_level() -> None:
    """Delta with new price level inserts it into the ask ladder."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 100)], []))
    ob.update_from_kalshi(_delta([(61, 50)], []))
    prices = [p for p, _ in ob.kalshi_depth_ask]
    assert 60 in prices
    assert 61 in prices


def test_delta_removes_level_on_zero_qty() -> None:
    """Delta with qty=0 removes the price level from the ladder."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 100), (61, 50)], []))
    ob.update_from_kalshi(_delta([(60, 0)], []))
    prices = [p for p, _ in ob.kalshi_depth_ask]
    assert 60 not in prices
    assert 61 in prices


def test_delta_updates_existing_level() -> None:
    """Delta with existing price and new qty updates that level."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 100)], []))
    ob.update_from_kalshi(_delta([(60, 200)], []))
    assert ob.kalshi_depth_ask == [(60, 200)]


# ---------------------------------------------------------------------------
# VWAP computation
# ---------------------------------------------------------------------------


def test_vwap_buy_single_level_exact() -> None:
    """VWAP buy: single ask level, exact fill quantity."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 100)], []))
    vwap = ob.compute_vwap_buy(50)
    assert vwap == pytest.approx(60.0)


def test_vwap_buy_multi_level() -> None:
    """VWAP buy: spans two price levels, weighted average."""
    ob = OrderBookSync()
    # 50 contracts at 60¢, 50 at 62¢
    ob.update_from_kalshi(_snapshot([(60, 50), (62, 50)], []))
    vwap = ob.compute_vwap_buy(100)
    assert vwap == pytest.approx(61.0)  # (50*60 + 50*62) / 100


def test_vwap_buy_partial_second_level() -> None:
    """VWAP buy: takes only part of second level."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 50), (65, 200)], []))
    vwap = ob.compute_vwap_buy(75)
    # 50@60 + 25@65 = 3000 + 1625 = 4625 / 75
    assert vwap == pytest.approx(4625 / 75)


def test_vwap_buy_insufficient_depth_returns_none() -> None:
    """VWAP buy returns None when total ask depth < target qty."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 10)], []))
    assert ob.compute_vwap_buy(50) is None


def test_vwap_buy_empty_ladder_returns_none() -> None:
    """VWAP buy returns None when ask ladder is empty."""
    ob = OrderBookSync()
    assert ob.compute_vwap_buy(10) is None


def test_vwap_sell_single_level() -> None:
    """VWAP sell: single bid level, exact fill quantity."""
    ob = OrderBookSync()
    # no price 40 → bid price 60
    ob.update_from_kalshi(_snapshot([], [(40, 100)]))
    vwap = ob.compute_vwap_sell(50)
    assert vwap == pytest.approx(60.0)


def test_vwap_sell_insufficient_depth_returns_none() -> None:
    """VWAP sell returns None when total bid depth < target qty."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([], [(40, 5)]))
    assert ob.compute_vwap_sell(50) is None


# ---------------------------------------------------------------------------
# bet365 update — ML market
# ---------------------------------------------------------------------------


def test_bet365_ml_vig_normalised() -> None:
    """ML odds are vig-normalised: home_win + draw + away_win ≈ 1.0."""
    ob = OrderBookSync()
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    total = (
        ob.bet365_implied["home_win"]
        + ob.bet365_implied["draw"]
        + ob.bet365_implied["away_win"]
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_bet365_ml_home_prob() -> None:
    """home_win implied prob matches hand-calculated value."""
    ob = OrderBookSync()
    # home=1.44, draw=3.50, away=12.00
    # raw_sum = 1/1.44 + 1/3.50 + 1/12.00
    raw_sum = 1 / 1.44 + 1 / 3.50 + 1 / 12.00
    expected = (1 / 1.44) / raw_sum
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    assert ob.bet365_implied["home_win"] == pytest.approx(expected, rel=1e-6)


def test_bet365_ml_updates_last_update_timestamp() -> None:
    """update_bet365 sets bet365_last_update to nonzero."""
    ob = OrderBookSync()
    assert ob.bet365_last_update == 0.0
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    assert ob.bet365_last_update > 0.0


# ---------------------------------------------------------------------------
# bet365 update — Totals market
# ---------------------------------------------------------------------------


def test_bet365_totals_default_hdp() -> None:
    """Totals with hdp=2.5 stores over_2.5 key."""
    ob = OrderBookSync()
    ob.update_bet365(_totals_msg(1.90, 1.90))
    assert "over_2.5" in ob.bet365_implied
    assert "under_2.5" in ob.bet365_implied


def test_bet365_totals_vig_normalised() -> None:
    """Totals over + under sum to 1.0."""
    ob = OrderBookSync()
    ob.update_bet365(_totals_msg(1.90, 1.90))
    total = ob.bet365_implied["over_2.5"] + ob.bet365_implied["under_2.5"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_bet365_totals_equal_odds() -> None:
    """Equal over/under odds → implied prob = 0.5 each."""
    ob = OrderBookSync()
    ob.update_bet365(_totals_msg(2.00, 2.00))
    assert ob.bet365_implied["over_2.5"] == pytest.approx(0.5)
    assert ob.bet365_implied["under_2.5"] == pytest.approx(0.5)


def test_bet365_invalid_odds_does_not_crash() -> None:
    """Malformed odds message is silently skipped; no crash."""
    ob = OrderBookSync()
    ob.update_bet365({"type": "updated", "markets": [{"name": "ML", "odds": [{"bad": "data"}]}]})
    assert ob.bet365_implied == {}


# ---------------------------------------------------------------------------
# get_bet365_for_alignment
# ---------------------------------------------------------------------------


def test_get_bet365_alignment_returns_none_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_bet365_for_alignment returns None when bet365 data is stale."""
    ob = OrderBookSync()
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    original_time = ob.bet365_last_update
    monkeypatch.setattr(
        "src.execution.order_book_sync.time.monotonic",
        lambda: original_time + 35.0,
    )
    assert ob.get_bet365_for_alignment("home_win") is None


def test_get_bet365_alignment_returns_prob_when_fresh() -> None:
    """get_bet365_for_alignment returns float when data is fresh."""
    ob = OrderBookSync()
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    result = ob.get_bet365_for_alignment("home_win")
    assert result is not None
    assert 0.0 < result < 1.0


def test_get_bet365_alignment_missing_key_returns_none() -> None:
    """get_bet365_for_alignment returns None for unknown market key."""
    ob = OrderBookSync()
    ob.update_bet365(_ml_msg(1.44, 3.50, 12.00))
    assert ob.get_bet365_for_alignment("btts_yes") is None


# ---------------------------------------------------------------------------
# Liquidity check
# ---------------------------------------------------------------------------


def test_liquidity_ok_true_when_sufficient_depth() -> None:
    """liquidity_ok returns True when ask depth >= min_qty."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 100)], []))
    assert ob.liquidity_ok(min_qty=20) is True


def test_liquidity_ok_false_when_insufficient_depth() -> None:
    """liquidity_ok returns False when ask depth < min_qty."""
    ob = OrderBookSync()
    ob.update_from_kalshi(_snapshot([(60, 10)], []))
    assert ob.liquidity_ok(min_qty=20) is False


def test_liquidity_ok_empty_book() -> None:
    """liquidity_ok returns False on empty order book."""
    ob = OrderBookSync()
    assert ob.liquidity_ok() is False
