# Kalshi API Exploration

**Date:** 2026-03-11
**Environment:** Production (paper account, balance = $0.00)
**Scope:** Read-only. No orders placed.

---

## 1. Authentication

### Endpoint Migration

> **CRITICAL:** `trading-api.kalshi.com` has been **permanently migrated** to
> `https://api.elections.kalshi.com`
> All new code must use the new base URL.

| Parameter | Value |
|-----------|-------|
| Base URL | `https://api.elections.kalshi.com` |
| API prefix | `/trade-api/v2` |
| Auth scheme | RSA-PSS (SHA-256) |
| Key format | PEM (`-----BEGIN RSA PRIVATE KEY-----`) |

### RSA-PSS Signature

Kalshi v2 uses three request headers for authentication:

```
KALSHI-ACCESS-KEY:       <api-key-uuid>
KALSHI-ACCESS-TIMESTAMP: <unix-millis-as-string>
KALSHI-ACCESS-SIGNATURE: <base64(rsa_pss_sign(message))>
```

**Message to sign** = `timestamp_ms + METHOD.upper() + full_path`
**Full path** must include the `/trade-api/v2` prefix:

```python
path = "/trade-api/v2/portfolio/balance"   # ← CORRECT
path = "/portfolio/balance"                 # ← WRONG (returns 401)
```

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def kalshi_headers(api_key: str, private_key, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }
```

### Auth Verification

```
GET /trade-api/v2/portfolio/balance
→ {"balance": 0, "portfolio_value": 0, "updated_ts": 1773261356}
```

Auth confirmed working on `api.elections.kalshi.com`.

### Public vs. Authenticated Endpoints

| Endpoint | Auth Required |
|----------|---------------|
| `GET /markets` | **No** — fully public |
| `GET /markets/{ticker}/orderbook` | **No** — fully public |
| `GET /markets/{ticker}` | **No** — fully public |
| `GET /portfolio/balance` | **Yes** |
| `POST /orders` | **Yes** |
| `GET /portfolio/positions` | **Yes** |

---

## 2. Soccer/Football Markets

### Market Discovery

Full scan: **435,780 total open markets** across 436 paginated requests (1,000/page).

Kalshi does not expose a `category` filter that cleanly isolates soccer. Markets are
identified by their `series_ticker` prefix:

| Series Prefix | Competition |
|---------------|-------------|
| `KXUCL` | UEFA Champions League |
| `KXEPL` | English Premier League |
| `KXBUND` / `KXBUNDESLIGA` | Bundesliga |
| `KXLALIGA` / `KXLALIGATOP` | La Liga |
| `KXSERIA` | Serie A |
| `KXLIGUE` | Ligue 1 |
| `KXUECL` / `KXUEEL` | UEFA Europa / Conference League |
| `KXWC` / `KXMENWORLDCUP` / `KXWCROUND` / `KXWCSQUAD` / `KXWCGROUPQUAL` | FIFA World Cup 2026 |
| `KXTEAMSINUCL` | UCL round advancement markets |
| `KXMLS` | Major League Soccer |
| `KXCOPA` | Copa América / Copa del Rey |
| `KXEURO` | UEFA Euro |

### Active Soccer Series (March 2026)

Top series by market count found during scan:

| Series Ticker | Markets | Description |
|---------------|---------|-------------|
| `KXTEAMSINUCL-26` | 64 | Which teams reach each UCL round |
| `KXWCROUND-26SEMI` | 64 | WC 2026 — who reaches semi-final |
| `KXWCROUND-26QUAR` | 64 | WC 2026 — who reaches quarter-final |
| `KXWCROUND-26FINAL` | 64 | WC 2026 — who reaches final |
| `KXWCROUND-26RO16` | 64 | WC 2026 — who reaches round of 16 |
| `KXWCSQUAD-26BRA` | 61 | Brazil WC 2026 squad markets |
| `KXMENWORLDCUP-26` | 52 | WC 2026 outright winner / group winner |
| `KXEPLTOP4-26` | ~10 | EPL top-4 finish (one per club) |
| `KXEPLRELEGATION-26` | ~6 | EPL relegation (per club) |
| `KXEPLTOP6-26` | ~10 | EPL top-6 finish |
| `KXBUNDESLIGA-26` | ~5 | Bundesliga champion / runner-up |
| `KXLALIGATOP4-26` | ~10 | La Liga top-4 finish |
| `KXUCLGAME-26MAR10PSGCFC` | 3 | UCL PSG vs Chelsea — winner |
| `KXUCLSPREAD-26MAR11PSGCFC` | 4 | UCL PSG vs Chelsea — spread |
| `KXUCLTOTAL-26MAR11PSGCFC` | 4 | UCL PSG vs Chelsea — total goals |
| `KXUCLBTTS-26MAR11PSGCFC` | 1 | UCL PSG vs Chelsea — BTTS |
| `KXUCLGAME-26MAR10RMAMCI` | 3 | UCL Real Madrid vs Man City — winner |
| `KXUCLSPREAD-26MAR11RMAMCI` | 4 | UCL Real Madrid vs Man City — spread |
| `KXUCLTOTAL-26MAR11RMAMCI` | 4 | UCL Real Madrid vs Man City — total goals |
| `KXUCLBTTS-26MAR11RMAMCI` | 1 | UCL Real Madrid vs Man City — BTTS |

---

## 3. UCL Match Markets — Full Taxonomy

For each UCL fixture, Kalshi creates **~12 markets** across 4 market types.
Using Real Madrid vs Man City (March 10-11, 2026) as the reference fixture:

### 3.1 Winner Markets (`KXUCLGAME-*`)

Three binary markets for the 90-minute result:

| Ticker | Bid | Ask | Volume | Description |
|--------|-----|-----|--------|-------------|
| `KXUCLGAME-26MAR10RMAMCI-RMA` | 76¢ | 77¢ | 1,641,801 | Real Madrid win |
| `KXUCLGAME-26MAR10RMAMCI-TIE` | 14¢ | 16¢ | 247,054 | Draw |
| `KXUCLGAME-26MAR10RMAMCI-MCI` | 8¢ | 9¢ | 1,916,452 | Man City win |
| `KXUCLGAME-26MAR10PSGCFC-PSG` | 46¢ | 47¢ | 755,746 | PSG win |
| `KXUCLGAME-26MAR10PSGCFC-TIE` | 30¢ | 31¢ | 126,435 | Draw |
| `KXUCLGAME-26MAR10PSGCFC-CFC` | 23¢ | 24¢ | 631,604 | Chelsea win |

> **Note:** All three winner markets for a fixture sum to ~100¢ (enforced by no-arbitrage).
> PSG: 46+30+23 = 99¢. RMA: 76+14+8 = 98¢ (1-2¢ rounding).

### 3.2 Spread Markets (`KXUCLSPREAD-*`)

Four markets for handicap (goals spread):

| Ticker | Bid | Ask | Volume | Description |
|--------|-----|-----|--------|-------------|
| `KXUCLSPREAD-26MAR11RMAMCI-RMA2` | 19¢ | 44¢ | 2,259 | RMA wins by 2+ |
| `KXUCLSPREAD-26MAR11RMAMCI-RMA1` | 50¢ | 51¢ | 14,834 | RMA wins by 1+ |
| `KXUCLSPREAD-26MAR11RMAMCI-MCI2` | 0¢ | 1¢ | 10,110 | MCI wins by 2+ |
| `KXUCLSPREAD-26MAR11RMAMCI-MCI1` | 0¢ | 6¢ | 42,117 | MCI wins by 1+ |

### 3.3 Total Goals Markets (`KXUCLTOTAL-*`)

Four markets for total goals O/U:

| Ticker | Bid | Ask | Volume | Description |
|--------|-----|-----|--------|-------------|
| `KXUCLTOTAL-26MAR11RMAMCI-1` | 99¢ | 100¢ | 18,258 | Over 1 goal (≥2) |
| `KXUCLTOTAL-26MAR11RMAMCI-2` | 73¢ | 97¢ | 50,703 | Over 2 goals (≥3) |
| `KXUCLTOTAL-26MAR11RMAMCI-3` | 64¢ | 65¢ | 19,934 | Over 3 goals (≥4) |
| `KXUCLTOTAL-26MAR11RMAMCI-4` | 32¢ | 36¢ | 16,393 | Over 4 goals (≥5) |

> **Mapping to MMPP model:** `KXUCLTOTAL-*-3` ≈ over_35; `KXUCLTOTAL-*-2` ≈ over_25.

### 3.4 Both Teams To Score (`KXUCLBTTS-*`)

| Ticker | Bid | Ask | Volume | Description |
|--------|-----|-----|--------|-------------|
| `KXUCLBTTS-26MAR11RMAMCI` | 72¢ | 83¢ | 54,231 | BTTS — Yes |
| `KXUCLBTTS-26MAR11PSGCFC` | 99¢ | — | 15,819 | BTTS — Yes |

---

## 4. Order Book Structure

### Raw JSON (KXUCLGAME-26MAR10PSGCFC-PSG)

```json
{
  "orderbook": {
    "yes": [
      [1, 9001],   [2, 15000],  [3, 2000],
      [5, 3488],   [10, 18074], [17, 2000],
      [18, 2000],  [19, 2500],  [43, 6749],
      [44, 17082], [45, 12106], [46, 559]
    ],
    "no": [
      [1, 5000],   [2, 1861],   [5, 4361],
      [6, 14021],  [10, 18030], [13, 2000],
      [18, 2500],  [49, 6163],  [50, 5623],
      [51, 6041],  [52, 13016], [53, 23570]
    ],
    "yes_dollars": [ ["0.0100", 9001], ... ],  ← dollar-formatted alias
    "no_dollars":  [ ["0.0100", 5000], ... ]
  },
  "orderbook_fp": { ... }  ← fractional precision alias (ignore)
}
```

### Price Encoding

| Field | Unit | Range | Example |
|-------|------|-------|---------|
| `yes[i][0]` | cents | 1–99 | `46` = 46¢ |
| `yes[i][1]` | contracts | ≥ 1 | `12106` |
| `yes_dollars[i][0]` | dollars (string) | "0.01"–"0.99" | `"0.4600"` |

**Prices are in integer cents (1–99)**, not decimals.

### Computing Best Bid / Best Ask

```python
# yes side = bids (buyers of Yes, paying price P for Yes)
# no side  = bids (buyers of No, paying P_no for No = sellers of Yes at 100-P_no)

yes: list[list[int]]  # [[price_cents, qty], ...]
no:  list[list[int]]  # [[price_cents, qty], ...]

best_bid = max(p for p, _ in yes)               # highest Yes bid
best_ask = 100 - max(p for p, _ in no)          # ← NO buyers at max → Yes sellers at (100-max_no)
spread   = best_ask - best_bid                   # in cents
```

> **Common error:** `best_ask = 100 - min(no)` is **wrong** — it gives the lowest no order,
> not the best ask. Use `100 - max(no)` instead.

### Depth for Active Soccer Match Markets

| Ticker | Yes Levels | No Levels | Total Contracts | Spread |
|--------|-----------|-----------|-----------------|--------|
| `KXUCLGAME-26MAR10RMAMCI-RMA` | 28 | 22 | **330,147** | 1¢ |
| `KXUCLGAME-26MAR10RMAMCI-TIE` | 14 | 50 | 256,251 | 2¢ |
| `KXUCLGAME-26MAR10RMAMCI-MCI` | 8 | 53 | 258,194 | 1¢ |
| `KXUCLGAME-26MAR10PSGCFC-PSG` | 32 | 28 | **152,516** | 1¢ |
| `KXUCLGAME-26MAR10PSGCFC-TIE` | 15 | 48 | 139,302 | — |
| `KXUCLGAME-26MAR10PSGCFC-CFC` | 8 | 50 | 161,467 | — |
| `KXUCLTOTAL-26MAR11RMAMCI-3` | 17 | 11 | 11,406 | 6¢ |
| `KXUCLTOTAL-26MAR11RMAMCI-4` | 12 | 10 | 11,235 | 4¢ |
| `KXUCLBTTS-26MAR11RMAMCI` | 10 | 10 | 7,632 | 11¢ |
| `KXUCLTOTAL-26MAR11PSGCFC-3` | 19 | 4 | 4,989 | — |

### Outright/Futures Soccer Market Depth

| Ticker | Bid | Ask | Spread | Depth |
|--------|-----|-----|--------|-------|
| `KXEPLTOP4-26-MCI` | 97¢ | 98¢ | **1¢** | 45,739 |
| `KXEPLTOP4-26-ARS` | 98¢ | 100¢ | — | 32,231 |
| `KXWCGROUPQUAL-26C-BRA` | 94¢ | 98¢ | 4¢ | 23,292 |
| `KXBUNDESLIGA-26-BMU` | 98¢ | 99¢ | 1¢ | 21,023 |
| `KXLALIGATOP4-26-VIL` | 96¢ | 98¢ | 2¢ | 20,804 |
| `KXUCLRO8-26-BMU` | 99¢ | 100¢ | — | 14,323 |

---

## 5. Market Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| Total open markets (all sports) | **440,664** | 441 pages × 1,000 (full scan) |
| Soccer by loose keyword | ~71,000 | 16.3% — includes false positives (NBA "City", etc.) |
| **Pure soccer (series prefix match)** | **2,412** | 0.55% of all markets; exact count |
| UCL R16 2nd-leg markets (2 fixtures) | 24 | 12 per fixture × 2 |
| **Markets with OB depth > 20 contracts** | **All sampled** | 40/40 = 100% in activity-sorted sample |
| Average depth (UCL winner markets) | ~250,000 | contracts; 5–55K for O/U and BTTS |

> **Why 2,412 pure soccer?** Markets cluster at pages 100 (+73) and 400 (+619), confirming
> Kalshi's pagination returns markets in creation-time order — soccer markets were created in
> batches. The 2,412 count covers all currently open soccer fixtures + outrights identifiable
> by known series prefixes (`KXUCL`, `KXEPL`, `KXBUND`, `KXWC`, etc.).

---

## 6. Ticker Format Conventions

```
KXUCLGAME-26MAR10PSGCFC-PSG
│  │      │        │     │
│  │      │        │     └── Outcome: PSG=home win, TIE=draw, CFC=away win
│  │      │        └──────── Home+Away abbreviation: PSGCFC
│  │      └───────────────── Date: YY-MON-DD → 26MAR10 = March 10, 2026
│  └──────────────────────── Market type: GAME/SPREAD/TOTAL/BTTS
└─────────────────────────── KX prefix + competition: UCL=Champions League

KXUCLTOTAL-26MAR11RMAMCI-3   → "Over 3 goals" (UCL RMA vs MCI, March 11)
KXUCLSPREAD-26MAR11PSGCFC-PSG1 → "PSG to win by 1+" (UCL PSG vs CFC, March 11)
KXEPLTOP4-26-ARS             → "Arsenal to finish top 4" (EPL 2026)
KXWCGROUPQUAL-26C-BRA        → "Brazil to qualify from WC Group C"
```

---

## 7. Key Findings for Client Implementation

### 7.1 Market Pausing During Live Play

- `close_time` on UCL R16 markets is the **settlement date** (2026-03-24), not the match date.
- Kalshi **pauses/suspends** markets during live events — the API still returns bids but
  `POST /orders` will reject with status=`closed` or `paused`.
- During live trading, track `market.status` field: `active` → `paused` → `settled`.

### 7.2 Price Mapping to Our Markets

| MMPP Market Key | Kalshi Ticker Pattern | Notes |
|-----------------|----------------------|-------|
| `home_win` | `KXUCLGAME-{date}{HOME}{AWAY}-{HOME}` | e.g., `-PSG` suffix |
| `draw` | `KXUCLGAME-{date}{HOME}{AWAY}-TIE` | |
| `away_win` | `KXUCLGAME-{date}{HOME}{AWAY}-{AWAY}` | e.g., `-MCI` suffix |
| `over_25` | `KXUCLTOTAL-{date}{HOME}{AWAY}-2` | "Over 2 goals" = ≥3 goals |
| `over_35` | `KXUCLTOTAL-{date}{HOME}{AWAY}-3` | "Over 3 goals" = ≥4 goals |
| `btts_yes` | `KXUCLBTTS-{date}{HOME}{AWAY}` | |

> **Warning on `over_25`:** Kalshi's "Over 2 goals" market resolves Yes if ≥3 goals scored
> (i.e., the line is 2.5). Double-check resolution rules per fixture — some use exact lines.

### 7.3 Volumes and Liquidity

UCL winner markets are the most liquid soccer markets on Kalshi:
- **Real Madrid win** (`KXUCLGAME-26MAR10RMAMCI-RMA`): 1.6M contracts, 1¢ spread
- **Man City win** (`KXUCLGAME-26MAR10RMAMCI-MCI`): 1.9M contracts, 1¢ spread
- Total goals and BTTS are thinner: 5K–55K contracts, 4–11¢ spread

For our Kelly sizing (f_order_cap = 3%, bankroll = $X), expected fill sizes will be
well within the available depth for winner markets. O/U and BTTS markets have enough
depth for moderate position sizes (hundreds of contracts at a time).

### 7.4 WebSocket API

The REST order book is a snapshot. For live intra-match pricing, Kalshi provides a
WebSocket feed. Endpoint and auth: same key/signature scheme, ws:// connection.

- Subscribe to `orderbook_delta` channel per ticker for real-time OB updates.
- Message types: `orderbook_snapshot`, `orderbook_delta`, `trade`.
- **This is the source for `P_kalshi_buy` / `P_kalshi_sell` in Phase 4 `ob_sync`.**

---

## 8. API Client Design Implications

### Key decisions from exploration:

1. **Base URL:** `https://api.elections.kalshi.com/trade-api/v2` (hardcode, no env fallback)
2. **Auth path:** Always prepend `/trade-api/v2` to the endpoint path in the RSA message
3. **Public polling (3s):** `GET /markets/{ticker}` for bid/ask/volume (no auth)
4. **Authenticated:** `POST /orders`, `GET /portfolio/positions`, `GET /portfolio/balance`
5. **Order book WS:** Subscribe per ticker; maintain local OB state with delta updates
6. **Price units:** Integer cents (1–99). Convert to/from probability: `P = price_cents / 100`
7. **Order type for live:** Use limit orders with slippage budget (see phase4.md Step 4.3)
8. **Market suspension check:** Query `market.status` before each order placement

### Ticker mapping design:

```python
# In KalshiClient or model initialization:
ticker_to_model_key = {
    "KXUCLGAME-26MAR10PSGCFC-PSG": "home_win",
    "KXUCLGAME-26MAR10PSGCFC-TIE": "draw",
    "KXUCLGAME-26MAR10PSGCFC-CFC": "away_win",
    "KXUCLTOTAL-26MAR11PSGCFC-2":  "over_25",   # ≥3 goals
    "KXUCLTOTAL-26MAR11PSGCFC-3":  "over_35",   # ≥4 goals
    "KXUCLBTTS-26MAR11PSGCFC":     "btts_yes",
}
```

---

## 9. Open Questions for Client Build

1. **Live market suspension:** Does Kalshi suspend the market at kickoff, or leave it open?
   Need to test with a live match. If it suspends, `ob_freeze` = True is the right response.

2. **WebSocket auth:** Is the WS `connect` message authenticated with the same RSA-PSS
   headers, or via a separate login frame? Need to test against the WS endpoint.

3. **Order rejection codes:** What status/code does a rejected order return during a paused
   market? Need to handle gracefully (not retry, just log + skip).

4. **Settlement timing:** `close_time` is the settlement deadline, not the match end.
   How soon after a UCL match ends does Kalshi settle the market?

5. **Partial fills:** Does Kalshi support partial fills on limit orders? The phase4.md
   VWAP model assumes partial fills are possible — needs verification.

---

*Exploration script: `scripts/explore_kalshi.py`*
*Raw data: `outputs/kalshi_exploration_raw.json`*
