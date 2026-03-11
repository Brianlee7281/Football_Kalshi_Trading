"""Live execution layer — real Kalshi order submission for live trading.

Submits actual limit orders to Kalshi and polls for fill status:
  1. Staleness gate: skip if order book data is stale
  2. Build order (BUY_YES → yes side; BUY_NO → no side, yes_price = 100 - no_cents)
  3. Submit limit order via KalshiClient.submit_order
  4. Poll for fill status (0.5s intervals, up to timeout seconds)
  5. Cancel remainder on partial fill or timeout
  6. Handle Kalshi error codes (market_closed, insufficient_balance, etc.)

Price conversion (Yes-probability space → Kalshi cents):
  BUY_YES: yes_price = round(P_kalshi * 100), buy Yes side
  BUY_NO:  no_price  = round((1 - P_kalshi) * 100)
           yes_price = 100 - no_price  (Kalshi stores all prices as Yes price)

Reference: docs/phase4.md Step 4.5 (execute_order, wait_for_fill)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from src.clients.kalshi import KalshiApiError
from src.common.logging import get_logger
from src.common.types import FillResult, Signal

if TYPE_CHECKING:
    from src.clients.kalshi import KalshiClient
    from src.execution.order_book_sync import OrderBookSync

logger = get_logger("live_executor")

# Seconds between fill-status polls
_POLL_INTERVAL: float = 0.5

# Maximum seconds to wait for a fill before cancelling
_FILL_TIMEOUT: float = 5.0


class LiveExecutionLayer:
    """Submits real limit orders to Kalshi and waits for fills.

    Shares the same interface as PaperExecutionLayer so Phase 4 logic
    is identical in both modes — only the fill mechanism differs.

    Attributes:
        kalshi_client: Authenticated KalshiClient for REST API calls.
    """

    def __init__(self, kalshi_client: KalshiClient) -> None:
        self.kalshi_client = kalshi_client

    async def execute_order(
        self,
        signal: Signal,
        amount: float,
        ob_sync: OrderBookSync,
        urgent: bool = False,
    ) -> FillResult | None:
        """Submit a limit order and wait for fill confirmation.

        Args:
            signal: Trading signal with direction, P_kalshi, market_ticker.
            amount: Dollar amount to invest (after Kelly + risk limits).
            ob_sync: Current order book state (for staleness check + pricing).
            urgent: If True, price 1¢ above ask for immediate fill probability.

        Returns:
            FillResult on success (full or partial), None on skip/error.
        """
        # ── Staleness gate ────────────────────────────────────────────────
        if ob_sync.kalshi_is_stale:
            logger.warning(
                "live_order_skipped_stale_book",
                ticker=signal.market_ticker,
            )
            return None

        if signal.P_kalshi <= 0.0:
            return None

        contracts = int(amount / signal.P_kalshi)
        if contracts < 1:
            return None

        # ── Build order ───────────────────────────────────────────────────
        best_ask = ob_sync.kalshi_best_ask
        if best_ask is None:
            return None

        price_cents = best_ask + 1 if urgent else best_ask

        price_cents = max(1, min(99, price_cents))

        if signal.direction == "BUY_YES":
            side = "yes"
            yes_price = price_cents
        else:
            side = "no"
            # Kalshi stores No orders as their Yes-equivalent price
            yes_price = 100 - price_cents

        # ── Submit order ──────────────────────────────────────────────────
        try:
            resp = await self.kalshi_client.submit_order(
                ticker=signal.market_ticker,
                action="buy",
                side=side,
                count=contracts,
                yes_price=yes_price,
            )
        except KalshiApiError as e:
            if e.code == "market_closed":
                logger.warning(
                    "live_order_market_closed",
                    ticker=signal.market_ticker,
                )
            elif e.code == "insufficient_balance":
                logger.error(
                    "live_order_insufficient_balance",
                    ticker=signal.market_ticker,
                    amount=amount,
                )
            elif e.code == "price_out_of_range":
                logger.warning(
                    "live_order_price_out_of_range",
                    ticker=signal.market_ticker,
                    yes_price=yes_price,
                )
            else:
                logger.error(
                    "live_order_api_error",
                    ticker=signal.market_ticker,
                    code=e.code,
                )
            return None
        except Exception:
            logger.exception(
                "live_order_unexpected_error",
                ticker=signal.market_ticker,
            )
            return None

        order_id: str = resp["order"]["id"]

        # ── Poll for fill ─────────────────────────────────────────────────
        fill: dict[str, Any] | None = await self._wait_for_fill(
            order_id, timeout=_FILL_TIMEOUT
        )

        if fill is None:
            # Timeout: cancel remainder
            with contextlib.suppress(KalshiApiError):
                await self.kalshi_client.cancel_order(order_id)
            return None

        status: str = str(fill.get("status", ""))
        filled_count: int = int(fill.get("filled_count", 0))
        fill_price: float = float(fill.get("yes_price", yes_price)) / 100.0

        if status == "filled" and filled_count > 0:
            logger.info(
                "live_order_filled",
                ticker=signal.market_ticker,
                order_id=order_id,
                filled_count=filled_count,
                fill_price=round(fill_price, 4),
            )
            return FillResult(
                success=True,
                fill_price=fill_price,
                fill_quantity=filled_count,
                order_id=order_id,
            )

        if status == "resting" and filled_count > 0:
            # Partial fill: cancel remainder, keep what filled
            with contextlib.suppress(KalshiApiError):
                await self.kalshi_client.cancel_order(order_id)
            logger.info(
                "live_order_partial_fill",
                ticker=signal.market_ticker,
                order_id=order_id,
                filled_count=filled_count,
            )
            return FillResult(
                success=True,
                fill_price=fill_price,
                fill_quantity=filled_count,
                order_id=order_id,
            )

        # No fill: cancel and return None
        with contextlib.suppress(KalshiApiError):
            await self.kalshi_client.cancel_order(order_id)
        return None

    async def _wait_for_fill(
        self,
        order_id: str,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Poll for order fill status until timeout.

        Args:
            order_id: Kalshi order UUID.
            timeout: Maximum seconds to wait.

        Returns:
            Order dict on fill (full or partial), None on timeout/error.
        """
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            try:
                resp = await self.kalshi_client.get_order(order_id)
                order: dict[str, Any] = resp.get("order", {})
                status: str = str(order.get("status", ""))
                if status == "filled":
                    return order
                if status == "resting":
                    filled_count = int(order.get("filled_count", 0))
                    if filled_count > 0:
                        return order
                if status in ("canceled", "expired"):
                    return None
            except KalshiApiError:
                logger.warning(
                    "live_order_poll_error",
                    order_id=order_id,
                    elapsed=round(elapsed, 1),
                )
        return None
