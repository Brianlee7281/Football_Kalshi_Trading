"""Paper execution layer — realistic order fill simulation for paper trading.

Simulates live Kalshi fills without touching real money:
  1. Convert dollar amount → contract quantity via signal.P_kalshi
  2. Simulate fill delay (1–3s), abort if model state changes during wait
  3. Re-snapshot VWAP after delay (price may have moved)
  4. Apply directional slippage (BUY_YES: price higher; BUY_NO: price lower)
  5. Partial fill based on available depth at the fill price

Directional slippage rationale:
  BUY_YES: we pay the ask, adverse slippage → price moves up (worse for buyer)
  BUY_NO: we sell Yes (buy No), adverse slippage → price moves down (worse)

ob_freeze abort:
  If ob_freeze or event_state != IDLE occurs during the fill delay, the order
  is cancelled. This reflects real execution risk: our limit may fill right as
  a goal is being confirmed, at a stale price.

Reference: docs/phase4.md Step 4.5 (PaperExecutionLayer)
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import PaperFill, Signal
from src.engine.model import EVENT_IDLE

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel
    from src.execution.order_book_sync import OrderBookSync

logger = get_logger("paper_executor")


class PaperExecutionLayer:
    """Simulates realistic Kalshi order fills for paper trading.

    Shares the same interface as LiveExecutionLayer so Phase 4 logic
    is identical in both modes — only the fill mechanism differs.

    Attributes:
        slippage_ticks: Adverse tick slippage per fill (1 tick = 1¢ = 0.01
                        in probability space).
        fill_delay_range: (min_s, max_s) uniform random fill delay.
    """

    def __init__(
        self,
        slippage_ticks: int = 1,
        fill_delay_range: tuple[float, float] = (1.0, 3.0),
    ) -> None:
        self.slippage_ticks = slippage_ticks
        self.fill_delay_range = fill_delay_range

    async def execute_order(
        self,
        signal: Signal,
        amount: float,
        ob_sync: OrderBookSync,
        model: LiveFootballQuantModel,
        urgent: bool = False,
    ) -> PaperFill | None:
        """Simulate a paper fill for a single market.

        Args:
            signal: Trading signal with direction, P_kalshi, EV, etc.
            amount: Dollar amount to invest (after Kelly + risk limits).
            ob_sync: Current order book state for this market.
            model: Live match model (provides ob_freeze, event_state).
            urgent: If True, skips the fill delay (rapid-entry mode).

        Returns:
            PaperFill on success, None if the order is aborted or unfillable.
        """
        if signal.P_kalshi <= 0.0:
            return None

        target_qty = int(amount / signal.P_kalshi)
        if target_qty < 1:
            return None

        # ── Fill delay simulation ─────────────────────────────────────────
        # Real limit orders sit in the book 1–3s before confirmation.
        # During this window, events can change the market state.
        if not urgent:
            delay = random.uniform(*self.fill_delay_range)
            await asyncio.sleep(delay)
        else:
            delay = 0.0

        # Abort if state changed during the delay (ob_freeze or event)
        if model.ob_freeze or model.event_state != EVENT_IDLE:
            logger.info(
                "paper_order_cancelled_state_change",
                match_id=model.match_id,
                ob_freeze=model.ob_freeze,
                event_state=model.event_state,
                delay_s=round(delay, 2),
                ticker=signal.market_ticker,
            )
            return None

        # ── Re-snapshot VWAP after delay ──────────────────────────────────
        if signal.direction == "BUY_YES":
            vwap_cents = ob_sync.compute_vwap_buy(target_qty)
        else:
            vwap_cents = ob_sync.compute_vwap_sell(target_qty)

        if vwap_cents is None:
            return None  # depth dried up during wait

        P_effective = vwap_cents / 100.0

        # ── Directional slippage ──────────────────────────────────────────
        # BUY_YES: adverse = price higher (we pay more)
        # BUY_NO:  adverse = price lower (we receive less)
        slip = self.slippage_ticks * 0.01
        fill_price = P_effective + slip if signal.direction == "BUY_YES" else P_effective - slip

        # ── Partial fill based on post-delay depth ────────────────────────
        fill_price_cents = round(fill_price * 100)
        if signal.direction == "BUY_YES":
            available_depth = sum(
                qty for price, qty in ob_sync.kalshi_depth_ask
                if price <= fill_price_cents
            )
        else:
            available_depth = sum(
                qty for price, qty in ob_sync.kalshi_depth_bid
                if price >= fill_price_cents
            )

        filled_qty = min(target_qty, available_depth)
        if filled_qty < 1:
            return None

        logger.info(
            "paper_fill",
            match_id=model.match_id,
            ticker=signal.market_ticker,
            direction=signal.direction,
            fill_price=round(fill_price, 4),
            filled_qty=filled_qty,
            target_qty=target_qty,
            partial=(filled_qty < target_qty),
            delay_s=round(delay, 2),
        )

        return PaperFill(
            price=fill_price,
            quantity=filled_qty,
            timestamp=time.time(),
            is_paper=True,
            slippage=abs(fill_price - P_effective),
            partial=(filled_qty < target_qty),
            fill_delay=delay,
        )
