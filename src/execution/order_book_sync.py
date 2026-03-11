"""Live order book synchronization — Kalshi WS + bet365 Odds-API reference.

Implements Step 4.1 from docs/phase4.md:
    - Maintains Kalshi best bid/ask and depth arrays
    - VWAP computation for buy/sell sides (2-pass EV support)
    - bet365 implied probability tracking from Odds-API live WS messages
    - Staleness checks: Kalshi (5s threshold), bet365 (30s threshold)

Usage:
    ob = OrderBookSync()
    # Feed from Kalshi WS:
    ob.update_from_kalshi(ob_update)
    # Feed from Odds-API WS:
    ob.update_bet365(odds_api_msg)
    # In signal generator:
    if ob.kalshi_is_stale:
        skip_order()
    P_effective = ob.compute_vwap_buy(qty)

Reference: docs/phase4.md Step 4.1
"""

from __future__ import annotations

import time
from typing import Any

from src.clients.kalshi import OrderBookUpdate
from src.common.logging import get_logger

logger = get_logger("order_book_sync")


class OrderBookSync:
    """Synchronized order-book state for a single Kalshi market.

    Holds Kalshi bid/ask ladder and bet365 reference prices, with staleness
    guards to prevent trading on stale data.

    Attributes:
        ticker: Kalshi market ticker this sync is tracking.
        kalshi_best_bid: Highest bid price in cents (1–99), or None.
        kalshi_best_ask: Lowest ask price in cents (1–99), or None.
        kalshi_depth_ask: Ask ladder — [(price_cents, qty), ...] ascending.
        kalshi_depth_bid: Bid ladder — [(price_cents, qty), ...] descending.
        kalshi_last_update: monotonic timestamp of last Kalshi WS message (0 = never).
        bet365_implied: Normalised implied probs keyed by model market key.
        bet365_last_update: monotonic timestamp of last bet365 push (0 = never).
    """

    KALSHI_STALE_THRESHOLD: float = 5.0  # seconds
    BET365_STALE_THRESHOLD: float = 30.0  # seconds

    def __init__(self, ticker: str = "") -> None:
        self.ticker = ticker

        # Kalshi quotes (in cents, 1–99)
        self.kalshi_best_bid: int | None = None
        self.kalshi_best_ask: int | None = None
        self.kalshi_depth_ask: list[tuple[int, int]] = []  # ascending
        self.kalshi_depth_bid: list[tuple[int, int]] = []  # descending
        self.kalshi_last_update: float = 0.0

        # bet365 reference prices (in Yes-probability space, 0.0–1.0)
        self.bet365_implied: dict[str, float] = {}
        self.bet365_last_update: float = 0.0

        self._log = logger.bind(ticker=ticker)

    # ------------------------------------------------------------------
    # Staleness checks (docs/phase4.md §KALSHI_STALE_THRESHOLD)
    # ------------------------------------------------------------------

    @property
    def kalshi_is_stale(self) -> bool:
        """True if Kalshi order book data is too old to trade on."""
        if self.kalshi_last_update == 0.0:
            return True
        return (time.monotonic() - self.kalshi_last_update) > self.KALSHI_STALE_THRESHOLD

    @property
    def bet365_is_stale(self) -> bool:
        """True if bet365 data is too old for alignment check."""
        if self.bet365_last_update == 0.0:
            return True
        return (time.monotonic() - self.bet365_last_update) > self.BET365_STALE_THRESHOLD

    # ------------------------------------------------------------------
    # Kalshi WS update
    # ------------------------------------------------------------------

    def update_from_kalshi(self, update: OrderBookUpdate) -> None:
        """Apply a Kalshi order-book snapshot or delta.

        For a snapshot: replaces entire ladder.
        For a delta: applies incremental changes (qty=0 removes level).

        Prices from the WS are in integer cents (1–99).
        Bid ladder is stored descending; ask ladder ascending.

        Args:
            update: OrderBookUpdate from KalshiClient.stream_orderbook.
        """
        if update.is_snapshot:
            self._apply_snapshot(update)
        else:
            self._apply_delta(update)

        # Derive best bid/ask from updated ladders
        self.kalshi_best_bid = self.kalshi_depth_bid[0][0] if self.kalshi_depth_bid else None
        self.kalshi_best_ask = self.kalshi_depth_ask[0][0] if self.kalshi_depth_ask else None
        self.kalshi_last_update = time.monotonic()

        self._log.debug(
            "kalshi_ob_updated",
            best_bid=self.kalshi_best_bid,
            best_ask=self.kalshi_best_ask,
            ask_levels=len(self.kalshi_depth_ask),
            bid_levels=len(self.kalshi_depth_bid),
            is_snapshot=update.is_snapshot,
        )

    def _apply_snapshot(self, update: OrderBookUpdate) -> None:
        """Replace ladders from snapshot."""
        # Ask (yes side): ascending by price
        self.kalshi_depth_ask = sorted(update.yes, key=lambda x: x[0])
        # Bid (no side converted to yes space): no price p → yes bid = 100 - p
        # The WS delta sends no levels as (no_price, qty); bid = 100 - no_price
        # But OrderBookUpdate already stores .no as raw no-side levels.
        # We convert: bid_price_cents = 100 - no_price
        self.kalshi_depth_bid = sorted(
            [(100 - p, q) for p, q in update.no],
            key=lambda x: x[0],
            reverse=True,
        )

    def _apply_delta(self, update: OrderBookUpdate) -> None:
        """Apply incremental delta to existing ladders.

        qty == 0 means level removed; qty > 0 means level set/added.
        """
        # Apply yes-side (ask) deltas
        ask_dict = dict(self.kalshi_depth_ask)
        for price, qty in update.yes:
            if qty == 0:
                ask_dict.pop(price, None)
            else:
                ask_dict[price] = qty
        self.kalshi_depth_ask = sorted(ask_dict.items(), key=lambda x: x[0])

        # Apply no-side → bid deltas
        bid_dict = {p: q for p, q in self.kalshi_depth_bid}
        for no_price, qty in update.no:
            bid_price = 100 - no_price
            if qty == 0:
                bid_dict.pop(bid_price, None)
            else:
                bid_dict[bid_price] = qty
        self.kalshi_depth_bid = sorted(bid_dict.items(), key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # VWAP computation (docs/phase4.md §P_effective(Q))
    # ------------------------------------------------------------------

    def compute_vwap_buy(self, target_qty: int) -> float | None:
        """Effective buy price (VWAP) for target_qty contracts, in cents.

        Consumes ask levels from lowest price upward.
        Returns None if total depth is insufficient.

        Args:
            target_qty: Number of contracts to fill.

        Returns:
            VWAP price in cents (1–99), or None if depth < target_qty.
        """
        if not self.kalshi_depth_ask or target_qty <= 0:
            return None

        filled = 0
        cost = 0.0
        for price, qty in self.kalshi_depth_ask:
            take = min(qty, target_qty - filled)
            cost += price * take
            filled += take
            if filled >= target_qty:
                break

        if filled < target_qty:
            return None  # insufficient depth

        return cost / filled

    def compute_vwap_sell(self, target_qty: int) -> float | None:
        """Effective sell price (VWAP) for target_qty contracts, in cents.

        Consumes bid levels from highest price downward.
        Returns None if total depth is insufficient.

        Args:
            target_qty: Number of contracts to fill.

        Returns:
            VWAP price in cents (1–99), or None if depth < target_qty.
        """
        if not self.kalshi_depth_bid or target_qty <= 0:
            return None

        filled = 0
        revenue = 0.0
        for price, qty in self.kalshi_depth_bid:
            take = min(qty, target_qty - filled)
            revenue += price * take
            filled += take
            if filled >= target_qty:
                break

        if filled < target_qty:
            return None

        return revenue / filled

    # ------------------------------------------------------------------
    # bet365 update (Odds-API live WS)
    # ------------------------------------------------------------------

    def update_bet365(self, odds_api_msg: dict[str, Any]) -> None:
        """Parse Odds-API live WS message → bet365 implied probabilities.

        Handles "ML" (moneyline: home_win, draw, away_win) and "Totals"
        (over_{hdp}, under_{hdp}) markets. Vig-normalized to sum to 1.0.

        Expected message format (filtered to Bet365 bookmaker):
        {
            "type": "updated",
            "bookie": "Bet365",
            "markets": [
                {"name": "ML", "odds": [{"home": "1.44", "draw": "3.50", "away": "12.00"}]},
                {"name": "Totals", "odds": [{"hdp": 2.5, "over": "1.90", "under": "1.90"}]}
            ]
        }

        Args:
            odds_api_msg: Raw message dict from Odds-API WebSocket.
        """
        markets: list[dict[str, Any]] = odds_api_msg.get("markets", [])
        updated = False

        for market in markets:
            name: str = market.get("name", "")
            odds_list: list[dict[str, Any]] = market.get("odds", [])
            if not odds_list:
                continue

            if name == "ML":
                updated |= self._parse_ml(odds_list[0])
            elif name == "Totals":
                updated |= self._parse_totals(odds_list[0])

        if updated:
            self.bet365_last_update = time.monotonic()
            self._log.debug(
                "bet365_updated",
                markets=list(self.bet365_implied.keys()),
            )

    def _parse_ml(self, odds: dict[str, Any]) -> bool:
        """Parse moneyline odds into home_win / draw / away_win implied probs."""
        try:
            home_odds = float(odds["home"])
            draw_odds = float(odds["draw"])
            away_odds = float(odds["away"])
            raw_sum = 1 / home_odds + 1 / draw_odds + 1 / away_odds
            if raw_sum == 0.0:
                return False
            self.bet365_implied["home_win"] = (1 / home_odds) / raw_sum
            self.bet365_implied["draw"] = (1 / draw_odds) / raw_sum
            self.bet365_implied["away_win"] = (1 / away_odds) / raw_sum
            return True
        except (KeyError, ValueError, ZeroDivisionError):
            return False

    def _parse_totals(self, odds: dict[str, Any]) -> bool:
        """Parse totals odds into over_{hdp} / under_{hdp} implied probs."""
        try:
            over_odds = float(odds["over"])
            under_odds = float(odds["under"])
            hdp = float(odds.get("hdp", 2.5))
            raw_sum = 1 / over_odds + 1 / under_odds
            if raw_sum == 0.0:
                return False
            self.bet365_implied[f"over_{hdp}"] = (1 / over_odds) / raw_sum
            self.bet365_implied[f"under_{hdp}"] = (1 / under_odds) / raw_sum
            return True
        except (KeyError, ValueError, ZeroDivisionError):
            return False

    # ------------------------------------------------------------------
    # Bet365 accessor for alignment check
    # ------------------------------------------------------------------

    def get_bet365_for_alignment(self, market_key: str) -> float | None:
        """Return bet365 implied prob for a market, or None if stale.

        Stale data yields UNAVAILABLE alignment (kelly_multiplier 0.6)
        instead of false ALIGNED/DIVERGENT on outdated information.

        Args:
            market_key: Model market key, e.g. "home_win", "over_2.5".

        Returns:
            Float probability in [0, 1], or None if bet365 data is stale.
        """
        if self.bet365_is_stale:
            return None
        return self.bet365_implied.get(market_key)

    # ------------------------------------------------------------------
    # Liquidity check
    # ------------------------------------------------------------------

    @property
    def total_ask_depth(self) -> int:
        """Total contracts available on the ask side."""
        return sum(q for _, q in self.kalshi_depth_ask)

    @property
    def total_bid_depth(self) -> int:
        """Total contracts available on the bid side."""
        return sum(q for _, q in self.kalshi_depth_bid)

    def liquidity_ok(self, min_qty: int = 20) -> bool:
        """True if ask depth meets the minimum liquidity threshold.

        Args:
            min_qty: Minimum contracts required on the ask side.
        """
        return self.total_ask_depth >= min_qty

    # ------------------------------------------------------------------
    # Debug repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        bid = f"{self.kalshi_best_bid}¢" if self.kalshi_best_bid is not None else "—"
        ask = f"{self.kalshi_best_ask}¢" if self.kalshi_best_ask is not None else "—"
        stale = "STALE" if self.kalshi_is_stale else "fresh"
        return (
            f"OrderBookSync({self.ticker!r} bid={bid} ask={ask} "
            f"ask_levels={len(self.kalshi_depth_ask)} "
            f"bid_levels={len(self.kalshi_depth_bid)} [{stale}])"
        )
