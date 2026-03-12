# Dashboard Decomposition

Claude Code implementation spec for Sprint 7. Read this BEFORE coding any dashboard file.
Parent architecture: `docs/dashboard.md`. This doc covers what dashboard.md doesn't:
type contracts, component specs, edge cases, formatting rules, and test assertions.

---

## Part 1: Type Contracts

Every field name, type, and shape must match exactly between Python → JSON → TypeScript.
Claude Code must use these definitions, not invent its own.

### 1.1 Python Pydantic Models (dashboard/api/models.py)

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# ─── Shared value types ───

class MarketProbs(BaseModel):
    """Per-market probability dict. Used for P_true, P_kalshi, P_bet365, sigma_MC."""
    home_win: float
    draw: float
    away_win: float
    over_25: Optional[float] = None
    under_25: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None

class Score(BaseModel):
    home: int
    away: int

# ─── REST API response models ───

class MatchSummary(BaseModel):
    """GET /api/matches — one item per match."""
    match_id: str
    league_id: int
    league_name: str                    # "EPL", "La Liga", etc.
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str                         # SCHEDULED|PHASE2_RUNNING|PHASE2_DONE|PHASE3_RUNNING|SETTLING|FINISHED|SKIPPED|FAILED
    trading_mode: str                   # "paper" | "live"
    score: Optional[Score] = None       # null if not started
    engine_phase: Optional[str] = None  # FIRST_HALF|HALFTIME|SECOND_HALF|FINISHED
    t: Optional[float] = None           # effective play minutes
    position_count: int = 0
    unrealized_pnl: Optional[float] = None

class MatchDetail(BaseModel):
    """GET /api/match/{id} — full match detail."""
    match_id: str
    league_id: int
    league_name: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str
    trading_mode: str
    param_version: Optional[int] = None
    score: Optional[Score] = None
    engine_phase: Optional[str] = None
    event_state: Optional[str] = None   # IDLE|PRELIMINARY_DETECTED|CONFIRMED
    t: Optional[float] = None
    cooldown: bool = False
    ob_freeze: bool = False
    order_allowed: bool = True
    pricing_mode: Optional[str] = None  # "analytical" | "mc"
    latest_P_true: Optional[MarketProbs] = None
    latest_sigma_MC: Optional[MarketProbs] = None
    positions: list["PositionItem"] = []
    recent_signals: list["SignalItem"] = []
    recent_events: list["EventItem"] = []

class TickSnapshot(BaseModel):
    """GET /api/match/{id}/ticks — one item per tick."""
    t: float
    mu_H: Optional[float] = None
    mu_A: Optional[float] = None
    P_true: Optional[MarketProbs] = None
    P_kalshi: Optional[dict] = None     # {home_win_ask: 0.52, home_win_bid: 0.50, ...}
    P_bet365: Optional[MarketProbs] = None
    sigma_MC: Optional[MarketProbs] = None
    engine_phase: Optional[str] = None
    event_state: Optional[str] = None
    cooldown: bool = False
    ob_freeze: bool = False
    order_allowed: bool = True

class PositionItem(BaseModel):
    """Used in MatchDetail.positions and GET /api/positions."""
    id: int
    match_id: str
    market_ticker: str
    direction: str                      # "BUY_YES" | "BUY_NO"
    entry_price: float                  # Yes-space, 0.01-0.99
    quantity: int
    status: str                         # PENDING|OPEN|CLOSED|AWAITING_SETTLEMENT|SETTLED
    is_paper: bool
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    settlement_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None  # computed at query time

class SignalItem(BaseModel):
    """Used in MatchDetail.recent_signals."""
    t: float
    ticker: str
    direction: str                      # "BUY_YES" | "BUY_NO" | "HOLD"
    EV: float                           # in dollars (e.g., 0.032)
    P_cons: float
    P_kalshi: float
    alignment: str                      # "ALIGNED" | "DIVERGENT" | "UNAVAILABLE"
    kelly_fraction: float
    outcome: Optional[str] = None       # "FILLED" | "SKIPPED" | "TRIM" | "OPP_COST_EXIT"
    fill_qty: Optional[int] = None
    fill_price: Optional[float] = None
    timestamp: datetime

class EventItem(BaseModel):
    """Used in MatchDetail.recent_events and GET /api/match/{id}/events."""
    event_type: str                     # goal_confirmed, red_card, period_change, etc.
    source: str                         # odds_api, live_score, system
    t: Optional[float] = None
    payload: dict                       # {score: [1,0], team: "localteam", minute: 23, ...}
    created_at: datetime

class PnLReport(BaseModel):
    """GET /api/analytics/pnl."""
    total_trades: int
    win_rate: float                     # 0.0-1.0
    total_pnl: float                    # dollars
    avg_ev: float                       # dollars per trade
    edge_realization: float             # actual_return / predicted_ev
    max_drawdown_pct: float             # 0.0-1.0
    sharpe: Optional[float] = None
    avg_slippage: float                 # cents
    by_league: dict[str, float]         # {"EPL": 145.0, "La Liga": 62.0, ...}
    by_market: dict[str, float]         # {"home_win": 180.0, "over_25": 40.0, ...}
    by_direction: dict[str, float]      # {"BUY_YES": 198.0, "BUY_NO": 36.0}
    by_alignment: dict[str, float]      # {"ALIGNED": 210.0, "DIVERGENT": 24.0}
    daily_pnl: list[dict]               # [{date: "2026-01-15", pnl: 12.50}, ...]

class ModelHealthReport(BaseModel):
    """GET /api/analytics/model-health."""
    brier_score_rolling20: float
    exchange_brier_rolling20: float
    delta_bs: float
    edge_realization_rolling20: float
    direction_realization: dict[str, float]  # {"BUY_YES": 0.85, "BUY_NO": 0.92}
    alignment_value: float              # ALIGNED avg return - DIVERGENT avg return
    param_version: int
    param_trained_at: datetime
    matches_since_retrain: int

class GraduationChecklist(BaseModel):
    """GET /api/analytics/paper-graduation."""
    trade_count: int                    # target: >= 50
    trade_count_ok: bool
    edge_realization: float             # target: 0.6-1.5
    edge_realization_ok: bool
    brier_within_range: bool            # target: Phase 1.5 ± 0.03
    max_drawdown_pct: float             # target: < 15%
    max_drawdown_ok: bool
    directional_correctness: bool       # target: 100%
    paper_realism_score: float          # target: > 0.85
    paper_realism_ok: bool
    no_system_crashes: bool
    theta_entry_calibrated: bool
    all_passed: bool                    # all 8 criteria

class SystemStatus(BaseModel):
    """GET /api/system/status."""
    containers: list["ContainerStatus"]
    connections: dict[str, "ConnectionHealth"]  # {"odds_api_ws": ..., "goalserve_rest": ..., ...}
    param_version: int
    param_trained_at: datetime
    matches_since_retrain: int
    recent_alerts: list[dict]

class ContainerStatus(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    status: str
    uptime_minutes: float
    last_heartbeat_ago: float           # seconds since last heartbeat
    healthy: bool                       # heartbeat < 60s

class ConnectionHealth(BaseModel):
    name: str                           # "Odds-API WS", "Goalserve REST", etc.
    connected: bool
    last_message_ago: Optional[float]   # seconds
    status: str                         # "connected" | "reconnecting" | "failed"
```

### 1.2 WebSocket Message Types (shared between Python publish and TypeScript consume)

```typescript
// dashboard/ui/src/lib/types.ts

// ─── Inbound WebSocket messages (server → client) ───

interface TickMessage {
  type: "tick";
  match_id: string;
  t: number;                            // effective play minutes
  engine_phase: string;
  P_true: MarketProbs;
  σ_MC: MarketProbs;                    // JSON key is "σ_MC" — use bracket notation
  P_bet365: MarketProbs | null;
  order_allowed: boolean;
  cooldown: boolean;
  ob_freeze: boolean;
  event_state: string;
  mu_H: number;
  mu_A: number;
  score: [number, number];              // [home, away]
}

interface EventMessage {
  type: "event";
  match_id: string;
  event_type: string;
  t: number;
  payload: {
    score: [number, number];
    team: string | null;
    minute: number;
    var_cancelled: boolean;
  };
}

interface SignalMessage {
  type: "signal";
  match_id: string;
  ticker: string;
  direction: string;
  EV: number;
  P_cons: number;
  P_kalshi: number;
  alignment: string;
  kelly_fraction: number;
  fill_qty: number;
  fill_price: number;
  timestamp: number;                    // unix seconds
}

interface PositionUpdateMessage {
  type: "new_fill" | "exit" | "settled";
  match_id: string;
  ticker: string;
  direction: string;
  quantity: number;
  price: number;
}

interface SystemAlertMessage {
  type: "alert";
  severity: "critical" | "warning" | "info";
  title: string;
  details: Record<string, string>;
  timestamp: number;
}

type WSMessage = TickMessage | EventMessage | SignalMessage
              | PositionUpdateMessage | SystemAlertMessage;

// ─── Outbound WebSocket messages (client → server) ───

interface SubscribeMessage {
  subscribe: string[];                  // match_ids to subscribe to
}

// ─── Shared types ───

interface MarketProbs {
  home_win: number;
  draw: number;
  away_win: number;
  over_25?: number;
  under_25?: number;
  btts_yes?: number;
  btts_no?: number;
}

interface Score {
  home: number;
  away: number;
}
```

### 1.3 JSON Key Warning

The Phase 3 Redis publish uses `σ_MC` (Greek sigma) as a JSON key.
TypeScript cannot use `data.σ_MC` — must use `data["σ_MC"]`.
**Alternative:** rename to `sigma_MC` in the Redis publish to avoid this.
Decision: use `sigma_MC` everywhere in JSON. Python models keep `σ_MC` internally.

---

## Part 2: WebSocket Protocol

### 2.1 Server-Side (dashboard/api/routes/websocket.py)

The redis_listener in dashboard.md has a bug: re-subscribing inside the while loop
causes duplicate subscriptions. Fixed version:

```python
async def live_updates(ws: WebSocket):
    await ws.accept()
    subscriptions: set[str] = set()
    pubsub = redis.pubsub()
    # Always subscribe to global channels once
    await pubsub.subscribe("position_update", "system_alert")

    async def redis_listener():
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    await ws.send_json(json.loads(message["data"]))
                except WebSocketDisconnect:
                    break

    listener_task = asyncio.create_task(redis_listener())

    try:
        async for data in ws.iter_json():
            if "subscribe" in data:
                new_subs = set(data["subscribe"])
                # Unsubscribe removed matches
                for match_id in subscriptions - new_subs:
                    await pubsub.unsubscribe(
                        f"tick:{match_id}", f"event:{match_id}", f"signal:{match_id}")
                # Subscribe new matches
                for match_id in new_subs - subscriptions:
                    await pubsub.subscribe(
                        f"tick:{match_id}", f"event:{match_id}", f"signal:{match_id}")
                subscriptions = new_subs
    except WebSocketDisconnect:
        pass
    finally:
        listener_task.cancel()
        await pubsub.unsubscribe()
        await pubsub.close()
```

### 2.2 Client-Side Reconnection (dashboard/ui/src/hooks/useWebSocket.ts)

```typescript
// Exponential backoff reconnection
const BACKOFF_BASE = 1000;   // 1s
const BACKOFF_MAX = 30000;   // 30s
const MAX_RETRIES = 10;

// State exposed to components:
// status: "connecting" | "connected" | "reconnecting" | "disconnected"
// lastMessageAge: number (ms since last message — for staleness UI)

// On disconnect:
// 1. Set status = "reconnecting"
// 2. Backoff: delay = min(BACKOFF_BASE * 2^attempt, BACKOFF_MAX)
// 3. Reconnect, re-send subscribe message with current match_ids
// 4. If MAX_RETRIES exceeded: status = "disconnected", show AlertBanner

// On message:
// 1. Reset retry counter
// 2. Update lastMessageAge = 0
// 3. Parse WSMessage, dispatch to correct handler by type
```

---

## Part 3: Component Specs

### 3.1 Component Tree and Data Flow

```
App (layout.tsx)
├── StatusBar                        ← REST: /api/system/status (poll 10s)
│   props: bankroll, exposure_pct, drawdown_pct, trading_mode, ws_status
│
├── AlertBanner                      ← WS: system_alert channel
│   props: alerts[]
│   state: visible (auto-dismiss info after 10s, critical stays)
│
├── Page: Command Center (/)         ← REST: /api/matches (poll 5s) + WS: tick per match
│   ├── MatchCard[]                  ← WS: tick:{match_id}
│   │   props: match: MatchSummary, latestTick: TickMessage | null
│   │
│   └── UpcomingList                 ← filtered from /api/matches where status=SCHEDULED
│       props: matches: MatchSummary[]
│
├── Page: Match Deep Dive (/match/[id])  ← REST: /api/match/{id} + WS: tick+event+signal
│   ├── MatchHeader                  ← props from MatchDetail
│   │   props: teams, score, engine_phase, t, status_flags
│   │
│   ├── PriceChart                   ← REST: /api/match/{id}/ticks + WS: tick (append)
│   │   props: ticks: TickSnapshot[], events: EventItem[] (for annotations)
│   │   state: selectedMarket ("home_win" | "over_25" | ...)
│   │
│   ├── OrderBookViz                 ← WS: tick (bid/ask from P_kalshi)
│   │   props: P_kalshi, spread
│   │
│   ├── SignalLog                    ← WS: signal:{match_id} (append)
│   │   props: signals: SignalItem[]
│   │
│   ├── PositionTable                ← REST: /api/positions?match_id={id}
│   │   props: positions: PositionItem[]
│   │
│   └── EventTimeline                ← REST: /api/match/{id}/events + WS: event (append)
│       props: events: EventItem[]
│
├── Page: P&L Analytics (/analytics) ← REST: /api/analytics/pnl
│   props: report: PnLReport
│   state: filters (date_range, league, market, direction, paper/live)
│
└── Page: System Operations (/system) ← REST: /api/system/status (poll 10s)
    ├── ContainerTable               ← props: containers[]
    ├── ConnectionPanel              ← props: connections{}
    ├── AlertHistory                 ← REST or props from SystemStatus
    └── ParamVersionInfo             ← props: param_version, trained_at, matches_since
```

### 3.2 Per-Component Specs

**StatusBar**
- Data: REST `/api/system/status` polled every 10s
- Display: `Bankroll: $9,847.32 | Exposure: 12.4% / 20% | Drawdown: 3.2% | [Paper Mode 🟡]`
- WS status indicator: 🟢 connected, 🟡 reconnecting, 🔴 disconnected
- Trading mode badge: 🟡 Paper, 🟢 Live

**MatchCard**
- Data: MatchSummary from REST + TickMessage from WS
- Layout: teams, score, time, per-market edge summary, position count
- Edge indicator: ▲ (positive edge > 2¢), ▬ (hold), ▼ (negative)
- Click → navigates to `/match/{match_id}`
- Pulse animation on new goal (event.type === "goal_confirmed")

**PriceChart**
- Library: Recharts LineChart
- Lines: P_true (solid blue), P_kalshi (solid red), P_bet365 (dashed gray)
- σ_MC band: shaded area around P_true (P_true ± 1.96 × σ_MC)
- Event annotations: vertical dashed lines at goal/red card t values
- X axis: effective play time 0-90+
- Y axis: probability 0.0-1.0
- Market selector tabs: home_win | draw | away_win | over_25
- Initial load: REST `/api/match/{id}/ticks`, then WS appends

**OrderBookViz**
- Horizontal bar chart: asks (red, right) + bids (green, left)
- Center: spread in cents
- Data: extracted from latest TickMessage P_kalshi field
- Stale indicator if kalshi data > 5s old

**SignalLog**
- Scrollable table, newest at top, max 50 visible
- Columns: Time | Ticker | Direction | EV | Kelly% | Alignment | Outcome
- Color: BUY_YES rows green tint, BUY_NO rows red tint, HOLD rows gray
- New signal rows flash briefly on arrival

**PositionTable**
- Columns: Market | Dir | Entry | Qty | Current | Unreal P&L | Status
- Current price: from latest tick P_kalshi
- Unrealized P&L: computed client-side from entry_price, current, direction
- Color: positive P&L green, negative red
- Status badges: OPEN 🟢, AWAITING_SETTLEMENT 🟡, SETTLED ⚪

**EventTimeline**
- Vertical timeline, newest at bottom (chronological)
- Icons: ⚽ goal, 🟥 red card, 🔄 substitution, ⏸ period change, ❄️ ob_freeze, 🧊 cooldown
- Each event shows: icon + time + description + score after

**GraduationChecklist** (on P&L Analytics page)
- 8 criteria, each with: label, current value, target, pass/fail badge
- Overall: "Ready for Live" banner when all_passed = true

---

## Part 4: Edge Cases and Empty States

Every component must handle these. Claude Code must implement all of them.

| Component | Condition | Display |
|-----------|-----------|---------|
| Command Center | 0 running matches | "No active matches. Next: {upcoming[0].home} vs {upcoming[0].away} in {time_until}" |
| Command Center | 0 upcoming matches | "No matches scheduled in the next 48 hours." |
| MatchCard | WS not connected | Show REST data only, gray "Live data unavailable" badge |
| MatchCard | match status = SCHEDULED | Show kickoff time, "Starting in {time}" |
| PriceChart | 0 tick_snapshots | "Waiting for match to start..." with empty axes |
| PriceChart | < 10 ticks | Render chart but show "Data collecting..." label |
| OrderBookViz | P_kalshi is null | "Order book unavailable" placeholder |
| OrderBookViz | kalshi stale > 5s | Red "STALE" badge on top |
| SignalLog | 0 signals | "No signals generated yet" |
| PositionTable | 0 positions | "No open positions for this match" |
| EventTimeline | 0 events | "No events yet — match in progress" |
| StatusBar | WS disconnected | 🔴 indicator + "Connection lost — retrying..." |
| StatusBar | bankroll = null | "Loading..." |
| AlertBanner | critical alert | Red banner, sticky, requires manual dismiss |
| AlertBanner | warning alert | Yellow banner, auto-dismiss after 30s |
| P&L Analytics | 0 settled trades | "No completed trades yet. Paper trading in progress." |
| GraduationChecklist | < 50 trades | Show progress bar: "{count}/50 trades" |
| System Operations | container heartbeat > 60s | Red row, "UNRESPONSIVE" badge |

---

## Part 5: Formatting Rules

All formatting in one place. Implement in `dashboard/ui/src/lib/format.ts`.

| Data Type | Format | Example | Function |
|-----------|--------|---------|----------|
| Probability | 2 decimal places | `0.55` | `formatProb(0.5523)` → `"0.55"` |
| Probability (percent) | 1 decimal + % | `55.2%` | `formatProbPct(0.5523)` → `"55.2%"` |
| Edge (EV) | 1 decimal + ¢ | `+4.2¢` | `formatEdge(0.042)` → `"+4.2¢"` |
| P&L | $ + 2 decimal, red/green | `+$12.50` or `-$3.20` | `formatPnL(12.5)` → `"+$12.50"` |
| Bankroll | $ + 2 decimal, comma sep | `$9,847.32` | `formatBankroll(9847.32)` → `"$9,847.32"` |
| Exposure % | 1 decimal + % | `12.4%` | `formatPct(0.124)` → `"12.4%"` |
| Kelly fraction | 1 decimal + % | `2.6%` | `formatPct(0.0265)` → `"2.7%"` |
| Match time | integer + ' | `67'` | `formatTime(67.3)` → `"67'"` |
| Match time (stoppage) | 45+2' or 90+3' | `90+3'` | `formatTime(93, {period: 2})` → `"90+3'"` |
| Sigma (σ_MC) | 4 decimal places | `0.0022` | `formatSigma(0.00224)` → `"0.0022"` |
| Price (cents) | integer ¢ | `65¢` | `formatCents(0.65)` → `"65¢"` |
| Latency | 1 decimal + unit | `7.6ms` or `1.2s` | `formatLatency(0.0076)` → `"7.6ms"` |
| Timestamp | HH:MM:SS local | `14:32:05` | `formatTimestamp(unix)` → `"14:32:05"` |
| Date | YYYY-MM-DD | `2026-03-11` | `formatDate(date)` → `"2026-03-11"` |
| Duration | Xm Ys | `23m 15s` | `formatDuration(1395)` → `"23m 15s"` |
| Count | comma separated | `2,412` | `formatCount(2412)` → `"2,412"` |

**Color rules:**

| Condition | Color | Tailwind Class |
|-----------|-------|----------------|
| P&L positive | Green | `text-green-600` |
| P&L negative | Red | `text-red-600` |
| P&L zero | Gray | `text-gray-500` |
| Exposure 0-15% | Green | `text-green-600` |
| Exposure 15-20% | Yellow | `text-yellow-600` |
| Exposure > 20% | Red | `text-red-600` |
| BUY_YES row | Green tint | `bg-green-50` |
| BUY_NO row | Red tint | `bg-red-50` |
| HOLD row | Gray | `bg-gray-50` |
| Status: connected | Green | `text-green-500` |
| Status: reconnecting | Yellow | `text-yellow-500` |
| Status: disconnected | Red | `text-red-500` |
| Heartbeat healthy | Green dot | `bg-green-500 rounded-full` |
| Heartbeat stale | Red dot | `bg-red-500 rounded-full` |

---

## Part 6: Test Assertions

Concrete test cases for dashboard components.

### API Tests (dashboard/api/tests/)

```
Test: GET /api/matches returns MatchSummary[] with all required fields
  → response[0] has keys: match_id, league_id, home_team, away_team, status, trading_mode
  → score is null for SCHEDULED matches, {home: int, away: int} for running matches

Test: GET /api/match/{id}/ticks returns TickSnapshot[] sorted by t ascending
  → all t values are monotonically increasing
  → P_true.home_win + P_true.draw + P_true.away_win ≈ 1.0 (within 0.01)

Test: GET /api/positions?status=OPEN returns only OPEN positions
  → all items have status = "OPEN"
  → unrealized_pnl is computed (not null)

Test: GET /api/analytics/pnl with 0 trades returns total_trades=0, total_pnl=0.0
  → does not crash, returns valid PnLReport

Test: GET /api/analytics/paper-graduation with < 50 trades
  → trade_count_ok = false, all_passed = false

Test: WebSocket subscribe → tick messages received
  → send {"subscribe": ["match_123"]}
  → receive messages with type="tick" and match_id="match_123"

Test: WebSocket duplicate subscribe does not duplicate messages
  → subscribe to same match twice → still receive 1 tick per publish
```

### Formatting Tests (dashboard/ui/src/lib/format.test.ts)

```
formatProb(0.5523) === "0.55"
formatProb(0.0) === "0.00"
formatProb(1.0) === "1.00"

formatEdge(0.042) === "+4.2¢"
formatEdge(-0.013) === "-1.3¢"
formatEdge(0.0) === "+0.0¢"

formatPnL(12.5) === "+$12.50"    // green
formatPnL(-3.2) === "-$3.20"    // red
formatPnL(0.0) === "$0.00"       // gray

formatBankroll(9847.32) === "$9,847.32"
formatBankroll(10000) === "$10,000.00"

formatTime(67.3) === "67'"
formatTime(93, {period: 2, regular: 90}) === "90+3'"
formatTime(47, {period: 1, regular: 45}) === "45+2'"

formatCents(0.65) === "65¢"
formatCents(0.05) === "5¢"
```

### Component Tests (React Testing Library)

```
Test: MatchCard with SCHEDULED match shows "Starting in {time}"
  → render MatchCard with status="SCHEDULED", kickoff_utc=future
  → screen.getByText(/starting in/i)

Test: MatchCard with no WS data shows "Live data unavailable"
  → render MatchCard with latestTick=null
  → screen.getByText(/live data unavailable/i)

Test: PriceChart with 0 ticks shows "Waiting for match to start"
  → render PriceChart with ticks=[]
  → screen.getByText(/waiting/i)

Test: StatusBar exposure > 15% shows yellow
  → render StatusBar with exposure_pct=0.16
  → expect element to have class text-yellow-600

Test: StatusBar exposure > 20% shows red
  → render StatusBar with exposure_pct=0.21
  → expect element to have class text-red-600

Test: AlertBanner critical alert stays visible
  → render AlertBanner with severity="critical"
  → wait 15 seconds → element still visible

Test: PositionTable unrealized P&L computed correctly
  → render with position: direction=BUY_YES, entry=0.45, qty=100, current_price=0.55
  → unrealized = 100 * (0.55 - 0.45) = $10.00
  → screen.getByText("+$10.00")

Test: PositionTable BUY_NO unrealized P&L
  → render with position: direction=BUY_NO, entry=0.40, qty=100, current_price=0.35
  → unrealized = 100 * (0.40 - 0.35) = $5.00
  → screen.getByText("+$5.00")
```

---

## Part 7: Sprint 7 Task Breakdown

| Task | Files | Depends On |
|------|-------|-----------|
| T7.1 | `dashboard/api/models.py` | Part 1 Pydantic models above |
| T7.2 | `dashboard/api/routes/matches.py`, `positions.py`, `analytics.py`, `system.py` | T7.1 + `src/common/db.py` (S6) |
| T7.3 | `dashboard/api/routes/websocket.py` | T7.1 + Part 2 protocol above |
| T7.4 | `dashboard/ui/src/lib/types.ts`, `format.ts` | Part 1 TypeScript types + Part 5 formatting |
| T7.5 | `dashboard/ui/src/hooks/useWebSocket.ts`, `useApi.ts`, `useLiveTick.ts` | T7.4 |
| T7.6 | `dashboard/ui/src/components/StatusBar.tsx`, `AlertBanner.tsx` | T7.5 |
| T7.7 | `dashboard/ui/src/app/page.tsx` (Command Center) + `MatchCard.tsx` | T7.5 + T7.6 |
| T7.8 | `dashboard/ui/src/app/match/[id]/page.tsx` + `PriceChart.tsx`, `OrderBookViz.tsx`, `SignalLog.tsx`, `PositionTable.tsx`, `EventTimeline.tsx` | T7.5 + T7.6 |
| T7.9 | `dashboard/ui/src/app/analytics/page.tsx` + `GraduationChecklist.tsx` | T7.5 |
| T7.10 | `dashboard/ui/src/app/system/page.tsx` | T7.5 |
| T7.11 | `src/common/metrics.py` (Prometheus) + `monitoring/grafana/dashboards/*.json` | T7.2 (needs running data) |
| T7.12 | `src/common/alerts.py` (Slack + SMS) | `src/common/redis_client.py` (S6) |
| T7.13 | All tests: API tests, format tests, component tests | T7.1-T7.12 |
