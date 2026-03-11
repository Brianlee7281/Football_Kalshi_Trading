"""ExecutionRouter — mode-switched order execution for Phase 4.

Unified entry point for order submission. All upstream Phase 4 logic
(signal generation, Kelly sizing, risk limits) calls this router; the
router delegates to PaperExecutionLayer or LiveExecutionLayer based on
the trading_mode injected at container startup.

This keeps Phase 4 code completely mode-invariant — only the fill
mechanism differs between paper and live.

Usage:
    router = ExecutionRouter("paper", model)
    fill = await router.submit_order(signal, amount, ob_sync)

Reference: docs/phase4.md Step 4.5 (ExecutionRouter)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.common.types import FillResult, PaperFill, Signal
from src.execution.paper_executor import PaperExecutionLayer

if TYPE_CHECKING:
    from src.clients.kalshi import KalshiClient
    from src.engine.model import LiveFootballQuantModel
    from src.execution.live_executor import LiveExecutionLayer
    from src.execution.order_book_sync import OrderBookSync

logger = get_logger("execution_router")


class ExecutionRouter:
    """Mode-switched execution router for Phase 4.

    Delegates to PaperExecutionLayer (paper mode) or LiveExecutionLayer
    (live mode) based on trading_mode. Both layers share the same
    ``execute_order`` interface so all upstream logic is identical.

    Attributes:
        mode: "paper" or "live".
        model: Live match state (provides ob_freeze, event_state for paper).
        paper: PaperExecutionLayer (paper mode only).
        live: LiveExecutionLayer (live mode only).
    """

    def __init__(
        self,
        trading_mode: str,
        model: LiveFootballQuantModel,
        *,
        kalshi_client: KalshiClient | None = None,
        slippage_ticks: int = 1,
        fill_delay_range: tuple[float, float] = (1.0, 3.0),
    ) -> None:
        """Initialize the router for the given trading mode.

        Args:
            trading_mode: "paper" or "live".
            model: Live match state container.
            kalshi_client: Required for live mode; None for paper mode.
            slippage_ticks: Paper-mode slippage per fill (1¢ = 0.01).
            fill_delay_range: Paper-mode (min_s, max_s) fill delay.

        Raises:
            ValueError: If live mode is requested but kalshi_client is None.
        """
        self.mode = trading_mode
        self.model = model
        self.paper: PaperExecutionLayer | None = None
        self.live: LiveExecutionLayer | None = None

        if trading_mode == "paper":
            self.paper = PaperExecutionLayer(
                slippage_ticks=slippage_ticks,
                fill_delay_range=fill_delay_range,
            )
        elif trading_mode == "live":
            if kalshi_client is None:
                raise ValueError(
                    "kalshi_client is required for live trading mode"
                )
            from src.execution.live_executor import LiveExecutionLayer
            self.live = LiveExecutionLayer(kalshi_client)
        else:
            raise ValueError(
                f"Unknown trading_mode: {trading_mode!r}. "
                "Expected 'paper' or 'live'."
            )

        logger.info(
            "execution_router_initialized",
            match_id=model.match_id,
            mode=trading_mode,
        )

    async def submit_order(
        self,
        signal: Signal,
        amount: float,
        ob_sync: OrderBookSync,
        urgent: bool = False,
    ) -> PaperFill | FillResult | None:
        """Submit an order via the active execution layer.

        Args:
            signal: Trading signal (direction, EV, P_kalshi, etc.).
            amount: Dollar amount to invest (post Kelly + risk limits).
            ob_sync: Current order book state for this market.
            urgent: If True, request fast fill (rapid-entry mode).

        Returns:
            PaperFill (paper mode), FillResult (live mode), or None on skip.
        """
        if signal.direction == "HOLD":
            return None

        if self.mode == "paper" and self.paper is not None:
            return await self.paper.execute_order(
                signal, amount, ob_sync, self.model, urgent
            )

        if self.mode == "live" and self.live is not None:
            return await self.live.execute_order(
                signal, amount, ob_sync, urgent
            )

        return None
