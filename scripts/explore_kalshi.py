"""Kalshi API exploration script — read-only, no orders placed.

Runs:
  1. Auth check (GET /trade-api/v2/portfolio/balance)
  2. List all soccer/football markets (active, public endpoint)
  3. Fetch full order book for one active soccer market (public)
  4. Count markets + order book depth stats (sample of 30)

Key findings from initial investigation:
  - Correct base URL: https://api.elections.kalshi.com/trade-api/v2
  - (trading-api.kalshi.com has been migrated to api.elections.kalshi.com)
  - RSA signature path must include /trade-api/v2 prefix
  - Markets list + order books are public (no auth required)
  - Auth (RSA-PSS) required only for /portfolio and /orders endpoints

Usage:
    python scripts/explore_kalshi.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).parent.parent / ".env"
_KEY_PATH = Path(__file__).parent.parent / "keys" / "kalshi_private.pem"

# Correct base URL (trading-api.kalshi.com migrated here)
KALSHI_BASE = "https://api.elections.kalshi.com"
API_V2 = "/trade-api/v2"


def _load_env() -> None:
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _load_private_key() -> RSAPrivateKey:
    pem = _KEY_PATH.read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    assert isinstance(key, RSAPrivateKey)
    return key


def _kalshi_signature(
    private_key: RSAPrivateKey,
    timestamp_ms: str,
    method: str,
    path: str,
) -> str:
    """Compute Kalshi RSA-PSS signature.

    Message = timestamp_ms + method.upper() + path
    path must be the FULL path including /trade-api/v2 prefix.
    """
    msg = (timestamp_ms + method.upper() + path).encode()
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
    full_path: str,
) -> dict[str, str]:
    """Build Kalshi auth headers. full_path = /trade-api/v2/..."""
    ts = str(int(time.time() * 1000))
    sig = _kalshi_signature(private_key, ts, method, full_path)
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


async def public_get(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET public endpoint (no auth required)."""
    url = f"{KALSHI_BASE}{API_V2}{endpoint}"
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


async def auth_get(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    pk: RSAPrivateKey,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET authenticated endpoint."""
    full_path = f"{API_V2}{endpoint}"
    headers = _auth_headers(api_key, pk, "GET", full_path)
    url = f"{KALSHI_BASE}{full_path}"
    resp = await client.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Exploration steps
# ---------------------------------------------------------------------------


async def step1_auth(
    client: httpx.AsyncClient,
    api_key: str,
    pk: RSAPrivateKey,
) -> dict[str, Any]:
    """Step 1: Verify auth by hitting /portfolio/balance."""
    print("\n── Step 1: Auth check ──────────────────────────────────────────────")
    print(f"  Base URL: {KALSHI_BASE}{API_V2}")
    data = await auth_get(client, "/portfolio/balance", api_key, pk)
    balance_cents = data.get("balance", 0)
    portfolio_cents = data.get("portfolio_value", 0)
    print(f"  ✓ Auth OK — balance: ${balance_cents/100:.2f}, portfolio: ${portfolio_cents/100:.2f}")
    print(f"  Raw: {json.dumps(data)}")
    return data


async def step2_list_soccer_markets(
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Step 2: List all open soccer/football markets (public endpoint)."""
    print("\n── Step 2: Soccer/football markets ────────────────────────────────")

    soccer_markets: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0
    total_scanned = 0

    # Soccer/football series ticker prefixes and title keywords
    soccer_series_prefixes = {
        "KXUCL", "KXEPL", "KXMLS", "KXLALIGA", "KXBUND", "KXSERIA",
        "KXLIGUE", "KXUECL", "KXUEEL", "KXWC", "KXEURO", "KXCOPA",
        "KXCL", "KXEL", "KXSOC", "KXFOOT",
    }
    soccer_title_keywords = {
        "soccer", "football", "epl", "premier league", "mls", "la liga",
        "bundesliga", "serie a", "ligue 1", "champions league", "europa league",
        "world cup", "copa", "ucl", "uel", "uecl", "ueel",
        " fc ", "united", "city", "real madrid", "barcelona", "psg",
        "chelsea", "arsenal", "liverpool", "atletico", "inter ", "milan",
        "juventus", "dortmund", "ajax", "porto", "benfica", "celtic",
        "rangers", "juventus", "napoli", "roma", "lazio", "sevilla",
        "valencia", "villarreal", "sociedad",
    }

    while True:
        params: dict[str, Any] = {"limit": 1000, "status": "open"}
        if cursor:
            params["cursor"] = cursor

        page += 1
        data = await public_get(client, "/markets", params=params)
        markets: list[dict[str, Any]] = data.get("markets", [])
        total_scanned += len(markets)
        if page == 1:
            print(f"  Fetching all open markets (limit=1000/page)...")

        for m in markets:
            title = (m.get("title") or "").lower()
            subtitle = (m.get("subtitle") or "").lower()
            series = (m.get("series_ticker") or "").upper()
            event  = (m.get("event_ticker") or "").upper()
            tags   = " ".join((m.get("tags") or [])).lower()

            text = f"{title} {subtitle} {tags}"

            # Match on series prefix OR title keywords
            series_match = any(series.startswith(p) or event.startswith(p)
                               for p in soccer_series_prefixes)
            title_match  = any(kw in text for kw in soccer_title_keywords)

            if series_match or title_match:
                soccer_markets.append(m)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

    print(f"  Scanned {total_scanned} total open markets across {page} page(s)")
    print(f"  Found {len(soccer_markets)} soccer/football markets\n")

    # Deduplicate series
    series_set: dict[str, int] = {}
    for m in soccer_markets:
        s = m.get("series_ticker") or m.get("event_ticker") or "UNKNOWN"
        series_set[s] = series_set.get(s, 0) + 1

    print(f"  Active soccer series/events ({len(series_set)} unique):")
    for s, cnt in sorted(series_set.items(), key=lambda x: -x[1])[:25]:
        print(f"    {s:<45} {cnt:>3} markets")

    if soccer_markets:
        print(f"\n  First 20 market tickers:")
        print(f"  {'Ticker':<50} {'Yes Ask':>8} {'Title':<55}")
        print(f"  {'-'*50} {'-'*8} {'-'*55}")
        for m in soccer_markets[:20]:
            ticker   = m.get("ticker", "")
            title    = (m.get("title") or "")[:54]
            yes_ask  = m.get("yes_ask", "-")
            print(f"  {ticker:<50} {str(yes_ask):>8} {title:<55}")

    return soccer_markets


def _depth_contracts(ob_data: dict[str, Any]) -> int:
    """Total contracts on both sides (yes + no ladder)."""
    ob = ob_data.get("orderbook", {})
    yes_levels: list[list[int]] = ob.get("yes") or []
    no_levels:  list[list[int]] = ob.get("no") or []
    return sum(q for _, q in yes_levels) + sum(q for _, q in no_levels)


def _best_bid_ask_cents(ob_data: dict[str, Any]) -> tuple[int | None, int | None]:
    """Best bid (highest yes price) and best ask (lowest yes price from no side)."""
    ob = ob_data.get("orderbook", {})
    yes_levels: list[list[int]] = ob.get("yes") or []
    no_levels:  list[list[int]] = ob.get("no") or []
    # yes side: sorted ascending by price → best bid is the highest
    best_bid = max((p for p, _ in yes_levels), default=None)
    # no side: lowest no price p_no → best ask = 100 - p_no (in cents)
    best_ask_no = min((p for p, _ in no_levels), default=None)
    best_ask = (100 - best_ask_no) if best_ask_no is not None else None
    return best_bid, best_ask


async def step3_fetch_order_book(
    client: httpx.AsyncClient,
    ticker: str,
) -> dict[str, Any]:
    """Step 3: Fetch full order book for one market + print raw JSON."""
    print(f"\n── Step 3: Full order book — {ticker} ──────────────────────────────")
    data = await public_get(client, f"/markets/{ticker}/orderbook")
    print(json.dumps(data, indent=2))
    depth = _depth_contracts(data)
    bid, ask = _best_bid_ask_cents(data)
    bid_str = f"{bid}¢" if bid is not None else "—"
    ask_str = f"{ask}¢" if ask is not None else "—"
    print(f"\n  Total depth: {depth:,} contracts | Best bid: {bid_str} | Best ask: {ask_str}")
    return data


async def step4_depth_stats(
    client: httpx.AsyncClient,
    soccer_markets: list[dict[str, Any]],
    *,
    sample: int = 40,
) -> list[dict[str, Any]]:
    """Step 4: Fetch OB depth for up to `sample` soccer markets."""
    print(f"\n── Step 4: Order book depth stats (sample={sample}) ───────────────")

    # Prioritize markets that have some bid/ask activity
    active = sorted(
        soccer_markets,
        key=lambda m: (m.get("yes_ask") or 0) + (m.get("yes_bid") or 0),
        reverse=True,
    )
    sample_markets = active[:sample]

    results: list[dict[str, Any]] = []
    for m in sample_markets:
        ticker = m.get("ticker", "")
        try:
            ob = await public_get(client, f"/markets/{ticker}/orderbook")
            d = _depth_contracts(ob)
            bid, ask = _best_bid_ask_cents(ob)
            spread = (ask - bid) if (bid is not None and ask is not None) else None
            results.append({
                "ticker": ticker,
                "title": (m.get("title") or "")[:60],
                "depth": d,
                "best_bid": bid,
                "best_ask": ask,
                "spread": spread,
                "yes_ask_market": m.get("yes_ask"),
            })
            await asyncio.sleep(0.08)
        except Exception as exc:
            results.append({"ticker": ticker, "title": (m.get("title") or "")[:60],
                             "depth": -1, "error": str(exc)})

    results.sort(key=lambda x: x.get("depth", -1), reverse=True)

    total = len(soccer_markets)
    sampled = len(results)
    deep = sum(1 for r in results if r.get("depth", 0) > 20)
    deep_pct = 100 * deep / sampled if sampled else 0
    extrapolated = int(deep / sampled * total) if sampled > 0 else 0
    any_bid = sum(1 for r in results if r.get("best_bid") is not None)

    print(f"\n  Total soccer markets: {total}")
    print(f"  Sampled for depth: {sampled}")
    print(f"  Depth > 20 contracts (in sample): {deep} / {sampled} ({deep_pct:.0f}%)")
    print(f"  Estimated depth>20 across ALL soccer markets: ~{extrapolated}")
    print(f"  Markets with any bid/ask (in sample): {any_bid}")

    print(f"\n  {'Ticker':<50} {'Depth':>8} {'Bid':>5} {'Ask':>5} {'Spread':>7}")
    print(f"  {'-'*50} {'-'*8} {'-'*5} {'-'*5} {'-'*7}")
    for r in results[:30]:
        if "error" in r:
            print(f"  {r['ticker']:<50} {'ERR':>8}  {r.get('error','')[:30]}")
        else:
            bid  = f"{r['best_bid']}¢"  if r.get("best_bid")  is not None else " — "
            ask  = f"{r['best_ask']}¢"  if r.get("best_ask")  is not None else " — "
            sprd = f"{r['spread']}¢"    if r.get("spread")     is not None else " — "
            print(f"  {r['ticker']:<50} {r['depth']:>8} {bid:>5} {ask:>5} {sprd:>7}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> dict[str, Any]:
    _load_env()
    api_key = os.environ.get("KALSHI_API_KEY", "")
    if not api_key:
        print("ERROR: KALSHI_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    pk = _load_private_key()

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Step 1: Auth
        balance = await step1_auth(client, api_key, pk)

        # Step 2: Soccer markets
        soccer_markets = await step2_list_soccer_markets(client)

        # Step 3: Order book for first active market
        raw_ob: dict[str, Any] = {}
        ob_ticker = ""
        if soccer_markets:
            # Pick the market with highest yes_ask (most active)
            by_activity = sorted(soccer_markets,
                key=lambda m: (m.get("yes_ask") or 0) + (m.get("yes_bid") or 0),
                reverse=True)
            ob_ticker = by_activity[0].get("ticker", soccer_markets[0].get("ticker", ""))
            raw_ob = await step3_fetch_order_book(client, ob_ticker)

        # Step 4: Depth stats
        depth_results = await step4_depth_stats(client, soccer_markets, sample=40)

    result: dict[str, Any] = {
        "balance": balance,
        "soccer_market_count": len(soccer_markets),
        "soccer_markets_sample": soccer_markets[:50],
        "ob_ticker": ob_ticker,
        "raw_ob": raw_ob,
        "depth_results": depth_results,
    }
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    out = Path(__file__).parent.parent / "outputs" / "kalshi_exploration_raw.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[Raw data saved → {out}]")
