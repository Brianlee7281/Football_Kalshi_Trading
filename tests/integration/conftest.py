"""Shared fixtures for integration tests.

Provides mock DB pool, mock Redis, mock model, and order book helpers
so integration tests can exercise multi-component flows without real
infrastructure.
"""

from __future__ import annotations

import time
from typing import Any

from src.clients.kalshi import OrderBookUpdate
from src.common.types import PaperFill, Signal, TickData
from src.engine.model import EVENT_IDLE, LiveFootballQuantModel, Phase4Config
from src.execution.order_book_sync import OrderBookSync

# ---------------------------------------------------------------------------
# Mock DB pool
# ---------------------------------------------------------------------------


class MockRecord(dict[str, Any]):
    """dict subclass that supports attribute access like asyncpg.Record."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


class MockConnection:
    """Simulates an asyncpg connection with fetchrow/fetch/execute."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    async def fetchrow(self, query: str, *args: Any) -> MockRecord | None:
        # Exposure reservation INSERT
        if "INSERT INTO exposure_reservation" in query:
            self._store.setdefault("_reservation_counter", 0)
            self._store["_reservation_counter"] += 1
            rid = self._store["_reservation_counter"]
            self._store.setdefault("reservations", {})[rid] = {
                "id": rid,
                "status": "RESERVED",
                "amount": args[2] if len(args) > 2 else 0.0,
            }
            return MockRecord({"id": rid})

        # get_total_exposure
        if "get_total_exposure" in query:
            total = sum(
                r["amount"]
                for r in self._store.get("reservations", {}).values()
                if r["status"] == "RESERVED"
            )
            return MockRecord({"total": total})

        # Existing exposure query
        if "SUM(entry_price * quantity)" in query:
            return MockRecord({"total": self._store.get("existing_exposure", 0.0)})

        return MockRecord({"total": 0.0})

    async def fetch(self, query: str, *args: Any) -> list[MockRecord]:
        return []

    async def execute(self, query: str, *args: Any) -> None:
        # Handle reservation status updates
        if "UPDATE exposure_reservation" in query:
            if "CONFIRMED" in query:
                rid = args[1] if len(args) > 1 else args[0]
                if rid in self._store.get("reservations", {}):
                    self._store["reservations"][rid]["status"] = "CONFIRMED"
            elif "RELEASED" in query:
                rid = args[0]
                if rid in self._store.get("reservations", {}):
                    self._store["reservations"][rid]["status"] = "RELEASED"


class MockPool:
    """Simulates asyncpg.Pool with context-managed acquire()."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._conn = MockConnection(self._store)

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


class _AcquireCtx:
    def __init__(self, conn: MockConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> MockConnection:
        return self._conn

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock Redis
# ---------------------------------------------------------------------------


class MockRedisLock:
    """Simulates a Redis lock context manager."""

    async def __aenter__(self) -> MockRedisLock:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class MockRedis:
    """Simulates redis.asyncio.Redis for testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    def lock(self, name: str, timeout: float = 2.0) -> MockRedisLock:
        return MockRedisLock()

    async def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))


# ---------------------------------------------------------------------------
# Mock ExecutionRouter
# ---------------------------------------------------------------------------


class MockExecutionRouter:
    """Simulates ExecutionRouter for testing signal_generator flow."""

    def __init__(self, *, fill_price: float = 0.55, fill_qty: int = 10) -> None:
        self.fill_price = fill_price
        self.fill_qty = fill_qty
        self.calls: list[dict[str, Any]] = []

    async def submit_order(
        self,
        signal: Signal,
        amount: float,
        ob_sync: OrderBookSync,
        urgent: bool = False,
    ) -> PaperFill | None:
        self.calls.append({
            "signal": signal,
            "amount": amount,
            "ticker": signal.market_ticker,
            "direction": signal.direction,
        })
        return PaperFill(
            price=self.fill_price,
            quantity=self.fill_qty,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Order book factory
# ---------------------------------------------------------------------------


def make_ob(
    ticker: str = "T",
    *,
    ask_price: int = 55,
    ask_depth: int = 1000,
    bid_price: int = 45,
    bid_depth: int = 1000,
) -> OrderBookSync:
    """Create an OrderBookSync with a populated snapshot.

    Uses update_from_kalshi (not _apply_snapshot) so that
    kalshi_best_bid/ask and kalshi_last_update are set correctly.
    """
    ob = OrderBookSync(ticker=ticker)
    ob.update_from_kalshi(OrderBookUpdate(
        ticker=ticker,
        is_snapshot=True,
        yes=[(ask_price, ask_depth)],
        no=[(100 - bid_price, bid_depth)],
        timestamp=time.time(),
    ))
    return ob


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def make_model(
    *,
    match_id: str = "test_match",
    bankroll: float = 10_000.0,
    tickers: list[str] | None = None,
    ticker_map: dict[str, str] | None = None,
    ob_syncs: dict[str, OrderBookSync] | None = None,
) -> LiveFootballQuantModel:
    """Create a LiveFootballQuantModel with mock infrastructure."""
    if tickers is None:
        tickers = ["T-HW"]
    if ticker_map is None:
        ticker_map = {"T-HW": "home_win"}

    model = LiveFootballQuantModel(
        match_id=match_id,
        trading_mode="paper",
        bankroll=bankroll,
        engine_phase="FIRST_HALF",
        event_state=EVENT_IDLE,
        active_tickers=tickers,
        ticker_to_model_key=ticker_map,
        config=Phase4Config(),
    )

    model.db_pool = MockPool()  # type: ignore[assignment]
    model.redis = MockRedis()  # type: ignore[assignment]

    if ob_syncs:
        model.ob_syncs = ob_syncs  # type: ignore[assignment]
    else:
        model.ob_syncs = {t: make_ob(t) for t in tickers}  # type: ignore[assignment]

    return model


# ---------------------------------------------------------------------------
# Signal factory
# ---------------------------------------------------------------------------


def make_signal(
    *,
    direction: str = "BUY_YES",
    EV: float = 0.03,
    P_cons: float = 0.55,
    P_kalshi: float = 0.50,
    rough_qty: int = 100,
    market_ticker: str = "T-HW",
) -> Signal:
    return Signal(
        direction=direction,
        EV=EV,
        P_cons=P_cons,
        P_kalshi=P_kalshi,
        rough_qty=rough_qty,
        alignment_status="ALIGNED",
        kelly_multiplier=0.8,
        market_ticker=market_ticker,
    )


# ---------------------------------------------------------------------------
# TickData factory
# ---------------------------------------------------------------------------


def make_tick(
    *,
    P_true: dict[str, float] | None = None,
    sigma_MC: dict[str, float] | None = None,
    order_allowed: bool = True,
) -> TickData:
    if P_true is None:
        P_true = {"home_win": 0.55}
    if sigma_MC is None:
        sigma_MC = {"home_win": 0.005}
    return TickData(
        P_true=P_true,
        sigma_MC=sigma_MC,
        order_allowed=order_allowed,
    )
