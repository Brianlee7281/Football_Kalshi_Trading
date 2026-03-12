"""Kalshi prediction market client — REST + WebSocket order book stream.

Authentication:
    RSA-PSS (SHA-256) via three request headers:
        KALSHI-ACCESS-KEY:       <uuid>
        KALSHI-ACCESS-TIMESTAMP: <unix-ms-string>
        KALSHI-ACCESS-SIGNATURE: base64(rsa_pss_sign(ts + METHOD + full_path))

    CRITICAL: full_path must include /trade-api/v2 prefix.
    trading-api.kalshi.com has been migrated to api.elections.kalshi.com.

REST endpoints (all via /trade-api/v2):
    GET  /markets/{ticker}           — market detail (public, no auth)
    GET  /markets/{ticker}/orderbook — OB snapshot (public, no auth)
    GET  /portfolio/balance          — account balance (auth required)
    GET  /portfolio/positions        — open positions (auth required)
    POST /portfolio/orders           — submit order (auth required)
    DEL  /portfolio/orders/{id}      — cancel order (auth required)

WebSocket:
    wss://api.elections.kalshi.com/trade-api/ws/v2
    Auth via same RSA-PSS headers in the WS handshake.
    Subscribe to "orderbook_delta" channel per market ticker.
    Yields OrderBookUpdate (snapshot or delta) as they arrive.

Reference:
    docs/kalshi_api_exploration.md §1 (auth), §4 (OB format), §7.4 (WS)
    docs/phase4.md Step 4.1 (OrderBookSync integration)
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets
import websockets.exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from src.clients.base_client import BaseClient
from src.common.logging import get_logger

logger = get_logger("kalshi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.elections.kalshi.com"
_API_PREFIX = "/trade-api/v2"
_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# Number of times to retry a WebSocket reconnect before giving up
_WS_MAX_RECONNECTS = 5
_WS_RECONNECT_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class KalshiApiError(Exception):
    """Raised when Kalshi returns a non-2xx HTTP response.

    Attributes:
        code: Short error code from Kalshi (e.g. "market_closed",
              "insufficient_balance", "price_out_of_range").
        message: Human-readable error message.
        status_code: HTTP status code.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 0,
    ) -> None:
        super().__init__(f"KalshiApiError({code}): {message}")
        self.code = code
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrderBook:
    """Parsed order book snapshot for a single market.

    Prices are in integer cents (1–99).
    yes side = buyers of Yes (bids). no side = buyers of No (bids).

    Best bid  = max(yes prices)
    Best ask  = 100 - max(no prices)    ← note: max, not min
    Spread    = best_ask - best_bid
    """

    ticker: str
    yes: list[tuple[int, int]]  # [(price_cents, qty)] sorted ascending
    no: list[tuple[int, int]]   # [(price_cents, qty)] sorted ascending
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> int | None:
        """Highest Yes bid price in cents, or None if empty."""
        return max((p for p, _ in self.yes), default=None)

    @property
    def best_ask(self) -> int | None:
        """Lowest Yes ask price in cents (= 100 - highest No bid), or None."""
        max_no = max((p for p, _ in self.no), default=None)
        return (100 - max_no) if max_no is not None else None

    @property
    def spread_cents(self) -> int | None:
        """Bid-ask spread in cents, or None if either side is empty."""
        bid, ask = self.best_bid, self.best_ask
        return (ask - bid) if (bid is not None and ask is not None) else None

    @property
    def depth_ask(self) -> list[tuple[int, int]]:
        """Ask levels from lowest price upward (for VWAP buy computation).

        Derived from the No side: no-buyer at price P_no offers Yes at (100-P_no).
        Sorted lowest ask (best ask) first.
        """
        # Convert no side: [(p_no, qty)] → [(100-p_no, qty)] sorted ascending by ask price
        return sorted(
            ((100 - p, q) for p, q in self.no),
            key=lambda x: x[0],
        )

    @property
    def depth_bid(self) -> list[tuple[int, int]]:
        """Bid levels from highest price downward (for VWAP sell computation)."""
        return sorted(self.yes, key=lambda x: x[0], reverse=True)


@dataclass
class OrderBookUpdate:
    """A single order book update event from the WebSocket.

    For snapshots (is_snapshot=True): yes/no contain the full ladder.
    For deltas (is_snapshot=False): yes/no contain only changed levels.
      A delta with delta=0 means the level was removed.
    """

    ticker: str
    is_snapshot: bool
    yes: list[tuple[int, int]]  # (price_cents, delta_qty or full_qty)
    no: list[tuple[int, int]]   # (price_cents, delta_qty or full_qty)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# RSA-PSS auth helpers
# ---------------------------------------------------------------------------


def _load_private_key(pem_path: str | Path) -> RSAPrivateKey:
    """Load RSA private key from PEM file."""
    pem = Path(pem_path).read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, RSAPrivateKey):
        raise TypeError(f"Expected RSAPrivateKey, got {type(key)}")
    return key


def _sign_request(
    private_key: RSAPrivateKey,
    timestamp_ms: str,
    method: str,
    full_path: str,
) -> str:
    """Compute Kalshi RSA-PSS-SHA256 signature.

    Message = timestamp_ms + METHOD.upper() + full_path
    full_path must include /trade-api/v2 prefix.
    """
    msg = (timestamp_ms + method.upper() + full_path).encode()
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def _auth_headers(
    api_key: str,
    private_key: RSAPrivateKey,
    method: str,
    endpoint: str,
) -> dict[str, str]:
    """Build Kalshi auth headers for a request.

    Args:
        api_key: Kalshi API key UUID.
        private_key: RSA private key loaded from PEM.
        method: HTTP method (GET, POST, DELETE).
        endpoint: Endpoint path starting with slash, e.g. "/portfolio/balance".
                  The /trade-api/v2 prefix is prepended automatically.
    """
    full_path = f"{_API_PREFIX}{endpoint}"
    ts = str(int(time.time() * 1000))
    sig = _sign_request(private_key, ts, method, full_path)
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Kalshi REST client
# ---------------------------------------------------------------------------


class KalshiClient:
    """Async Kalshi API client — REST + WebSocket order book stream.

    Args:
        api_key: Kalshi API key (UUID string from .env KALSHI_API_KEY).
        private_key_path: Path to RSA private key PEM file.
        timeout: HTTP request timeout in seconds.

    Usage::

        async with KalshiClient(api_key, key_path) as client:
            ob = await client.get_orderbook("KXUCLGAME-26MAR10PSGCFC-PSG")
            print(ob.best_bid, ob.best_ask)

            async with asyncio.TaskGroup() as tg:
                tg.create_task(client.stream_orderbook(tickers, on_update))
    """

    def __init__(
        self,
        api_key: str,
        private_key_path: str | Path,
        *,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._private_key: RSAPrivateKey = _load_private_key(private_key_path)
        self._http = BaseClient(
            _BASE_URL,
            timeout=timeout,
            max_retries=3,
            backoff_base=1.0,
            headers={"Content-Type": "application/json"},
        )
        self._ws: websockets.ClientConnection | None = None
        self._ws_subscribed: set[str] = set()

    async def __aenter__(self) -> KalshiClient:
        await self._http.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._http.__aexit__(*args)
        if self._ws is not None and self._ws.state != websockets.State.CLOSED:
            await self._ws.close()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _auth(self, method: str, endpoint: str) -> dict[str, str]:
        return _auth_headers(self._api_key, self._private_key, method, endpoint)

    # ------------------------------------------------------------------
    # Public REST — no auth required
    # ------------------------------------------------------------------

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Fetch market detail for a single ticker.

        Returns the ``market`` object with fields: ticker, yes_bid, yes_ask,
        no_bid, no_ask, status, volume, open_interest, close_time, etc.

        Prices are in integer cents (1–99). Status: "active", "paused", "resolved".

        Args:
            ticker: Kalshi market ticker (e.g. "KXUCLGAME-26MAR10PSGCFC-PSG").

        Returns:
            Market dict. Raises KalshiApiError on non-2xx.
        """
        resp = await self._http.get(f"{_API_PREFIX}/markets/{ticker}")
        _raise_for_kalshi_error(resp)
        data: dict[str, Any] = resp.json()
        result: dict[str, Any] = data.get("market", data)
        return result

    async def get_orderbook(self, ticker: str) -> OrderBook:
        """Fetch order book snapshot for a market.

        Returns an OrderBook with yes/no ladders as (price_cents, qty) tuples.
        Prices are in integer cents (1–99).

        Args:
            ticker: Kalshi market ticker.

        Returns:
            OrderBook dataclass.
        """
        resp = await self._http.get(f"{_API_PREFIX}/markets/{ticker}/orderbook")
        _raise_for_kalshi_error(resp)
        data: dict[str, Any] = resp.json()
        return _parse_orderbook(ticker, data)

    # ------------------------------------------------------------------
    # Authenticated REST
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        """Fetch account balance.

        Returns:
            Dict with keys: balance (cents), portfolio_value (cents), updated_ts.
        """
        endpoint = "/portfolio/balance"
        resp = await self._http.get(
            f"{_API_PREFIX}{endpoint}",
            headers=self._auth("GET", endpoint),
        )
        _raise_for_kalshi_error(resp)
        return resp.json()  # type: ignore[no-any-return]

    async def get_positions(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch open (or all) portfolio positions.

        Args:
            ticker: Filter to a single market ticker.
            status: "open", "closed", or None for all.
            limit: Max positions to return (default 100).

        Returns:
            List of position dicts with fields: ticker, side, quantity,
            remaining_value, average_price, unrealized_pnl, etc.
        """
        endpoint = "/portfolio/positions"
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        resp = await self._http.get(
            f"{_API_PREFIX}{endpoint}",
            params=params,
            headers=self._auth("GET", endpoint),
        )
        _raise_for_kalshi_error(resp)
        data: dict[str, Any] = resp.json()
        positions: list[dict[str, Any]] = data.get("positions", [])
        return positions

    async def submit_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: int,
        *,
        order_type: str = "limit",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a limit order.

        Args:
            ticker: Market ticker.
            action: "buy" or "sell".
            side: "yes" or "no".
            count: Number of contracts.
            yes_price: Limit price in cents for the Yes side (1–99).
                       For a No order, yes_price = 100 - no_price.
            order_type: "limit" (default) or "market".
            client_order_id: Optional idempotency key (UUID string).

        Returns:
            Response dict with "order" key containing order_id, status, etc.

        Raises:
            KalshiApiError: With code "market_closed", "insufficient_balance",
                            "price_out_of_range", or other Kalshi error codes.
        """
        endpoint = "/portfolio/orders"
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "yes_price": yes_price,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id

        resp = await self._http.post(
            f"{_API_PREFIX}{endpoint}",
            json=body,
            headers=self._auth("POST", endpoint),
        )
        _raise_for_kalshi_error(resp)
        return resp.json()  # type: ignore[no-any-return]

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch the current state of an order by ID.

        Args:
            order_id: Kalshi order UUID.

        Returns:
            Response dict with "order" key containing status, filled_count, etc.

        Raises:
            KalshiApiError: If the order is not found or another API error occurs.
        """
        endpoint = f"/portfolio/orders/{order_id}"
        resp = await self._http.get(
            f"{_API_PREFIX}{endpoint}",
            headers=self._auth("GET", endpoint),
        )
        _raise_for_kalshi_error(resp)
        return resp.json()  # type: ignore[no-any-return]

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order by ID.

        Args:
            order_id: Kalshi order UUID (from submit_order response).

        Raises:
            KalshiApiError: If the order cannot be cancelled (already filled, etc.).
        """
        endpoint = f"/portfolio/orders/{order_id}"
        resp = await self._http.request(
            "DELETE",
            f"{_API_PREFIX}{endpoint}",
            headers=self._auth("DELETE", endpoint),
        )
        _raise_for_kalshi_error(resp)
        logger.info("order_cancelled", order_id=order_id)

    # ------------------------------------------------------------------
    # Market Discovery (Scheduler — match discovery)
    # ------------------------------------------------------------------

    async def get_active_soccer_events(self) -> list[dict[str, Any]]:
        """Fetch active soccer events from Kalshi for match discovery.

        Calls ``GET /trade-api/v2/events`` with ``series_ticker=SOCCER`` and
        ``status=open``, paginating until all results are collected.

        Each event dict from Kalshi contains:
          ``event_ticker``, ``title``, ``series_ticker``, ``markets``
          (list of market tickers for this event), ``end_date``.

        Returns:
            List of event dicts for active soccer markets.  Empty list if
            no soccer events are found or the API returns an error.
        """
        events: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "series_ticker": "SOCCER",
                "status": "open",
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await self._http.get(
                    f"{_API_PREFIX}/events",
                    params=params,
                )
                _raise_for_kalshi_error(resp)
                data: dict[str, Any] = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_active_soccer_events_failed", error=str(exc))
                break

            page: list[dict[str, Any]] = data.get("events", [])
            events.extend(page)

            cursor = data.get("cursor") or ""
            if not cursor or not page:
                break

        logger.info("kalshi_soccer_events_fetched", count=len(events))
        return events

    # ------------------------------------------------------------------
    # WebSocket — order book stream
    # ------------------------------------------------------------------

    async def stream_orderbook(
        self,
        tickers: list[str],
        *,
        reconnect: bool = True,
    ) -> AsyncIterator[OrderBookUpdate]:
        """Stream real-time order book updates for the given tickers.

        Connects to Kalshi's WS endpoint and subscribes to the
        ``orderbook_delta`` channel. Yields OrderBookUpdate objects:
          - First message per ticker: is_snapshot=True (full ladder)
          - Subsequent messages: is_snapshot=False (price-level deltas)
            A delta with qty=0 means the level was removed from the book.

        The generator handles reconnects automatically when ``reconnect=True``.
        Stop iteration by cancelling the enclosing Task.

        Args:
            tickers: List of Kalshi market tickers to subscribe to.
            reconnect: If True, reconnect on disconnect (up to 5 times).

        Yields:
            OrderBookUpdate for each incoming message.
        """
        attempt = 0
        while True:
            try:
                async for update in self._ws_connect_and_stream(tickers):
                    attempt = 0  # reset reconnect counter on successful stream
                    yield update
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                if not reconnect:
                    raise
                attempt += 1
                if attempt > _WS_MAX_RECONNECTS:
                    logger.error(
                        "ws_max_reconnects_exceeded",
                        tickers=tickers,
                        attempts=attempt,
                    )
                    raise
                delay = _WS_RECONNECT_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "ws_reconnecting",
                    error=str(exc),
                    attempt=attempt,
                    delay=delay,
                )
                await asyncio.sleep(delay)

    async def _ws_connect_and_stream(
        self,
        tickers: list[str],
    ) -> AsyncIterator[OrderBookUpdate]:
        """Single WS connection lifecycle — connect, subscribe, yield updates."""
        # Build auth headers for the WS handshake
        ts = str(int(time.time() * 1000))
        full_path = "/trade-api/ws/v2"
        sig = _sign_request(self._private_key, ts, "GET", full_path)
        extra_headers = {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

        async with websockets.connect(
            _WS_URL,
            extra_headers=extra_headers,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            self._ws = ws
            logger.info("ws_connected", url=_WS_URL, tickers=tickers)

            # Subscribe to orderbook_delta channel
            sub_msg = json.dumps({
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": tickers,
                },
            })
            await ws.send(sub_msg)
            self._ws_subscribed = set(tickers)

            async for raw in ws:
                msg = json.loads(raw)
                update = _parse_ws_message(msg)
                if update is not None:
                    yield update

    async def subscribe_tickers(self, new_tickers: list[str]) -> None:
        """Add more tickers to an existing WS subscription (if connected).

        Sends an incremental subscribe message. Safe to call during a live
        stream_orderbook iteration — the WS connection handles this via
        the shared self._ws reference.

        Args:
            new_tickers: Tickers not yet in the current subscription.
        """
        if self._ws is None or self._ws.state == websockets.State.CLOSED:
            logger.warning("ws_subscribe_not_connected")
            return

        fresh = [t for t in new_tickers if t not in self._ws_subscribed]
        if not fresh:
            return

        msg = json.dumps({
            "id": 2,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": fresh,
            },
        })
        await self._ws.send(msg)
        self._ws_subscribed.update(fresh)
        logger.info("ws_subscribed_additional", tickers=fresh)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_orderbook(ticker: str, data: dict[str, Any]) -> OrderBook:
    """Parse a /markets/{ticker}/orderbook REST response into OrderBook."""
    ob = data.get("orderbook", {})
    yes_raw: list[list[int]] = ob.get("yes") or []
    no_raw:  list[list[int]] = ob.get("no") or []

    yes = [(int(p), int(q)) for p, q in yes_raw]
    no  = [(int(p), int(q)) for p, q in no_raw]

    return OrderBook(ticker=ticker, yes=yes, no=no, timestamp=time.time())


def _parse_ws_message(msg: dict[str, Any]) -> OrderBookUpdate | None:
    """Parse a raw WS message dict into an OrderBookUpdate, or None to skip."""
    msg_type = msg.get("type", "")

    if msg_type == "subscribed":
        logger.debug("ws_subscribed", msg=msg)
        return None

    if msg_type == "error":
        logger.error("ws_error", msg=msg)
        return None

    if msg_type not in ("orderbook_snapshot", "orderbook_delta"):
        return None

    payload = msg.get("msg", {})
    ticker  = payload.get("market_ticker", "")
    ts      = float(msg.get("ts", time.time() * 1_000_000)) / 1_000_000

    if msg_type == "orderbook_snapshot":
        yes_raw: list[list[int]] = payload.get("yes") or []
        no_raw:  list[list[int]] = payload.get("no") or []
        return OrderBookUpdate(
            ticker=ticker,
            is_snapshot=True,
            yes=[(int(p), int(q)) for p, q in yes_raw],
            no=[(int(p), int(q)) for p, q in no_raw],
            timestamp=ts,
        )

    # orderbook_delta — single price level update
    # Format: {"market_ticker": str, "price": int, "delta": int, "side": "yes"|"no"}
    price = int(payload.get("price", 0))
    delta = int(payload.get("delta", 0))
    side  = payload.get("side", "yes")

    if side == "yes":
        return OrderBookUpdate(
            ticker=ticker,
            is_snapshot=False,
            yes=[(price, delta)],
            no=[],
            timestamp=ts,
        )
    return OrderBookUpdate(
        ticker=ticker,
        is_snapshot=False,
        yes=[],
        no=[(price, delta)],
        timestamp=ts,
    )


def _raise_for_kalshi_error(resp: Any) -> None:
    """Raise KalshiApiError if the response indicates a Kalshi API error.

    Kalshi error responses have HTTP 4xx/5xx status and JSON body:
        {"code": "market_closed", "message": "..."}  or
        {"error": {"code": "...", "message": "..."}}
    """
    if resp.status_code < 400:
        return

    try:
        body: dict[str, Any] = resp.json()
    except Exception:
        body = {}

    # Try both response formats
    error_obj = body.get("error", body)
    code    = str(error_obj.get("code", f"http_{resp.status_code}"))
    message = str(error_obj.get("message", resp.text[:200]))

    raise KalshiApiError(code, message, status_code=resp.status_code)
