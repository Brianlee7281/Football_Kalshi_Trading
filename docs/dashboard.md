# Dashboard & Monitoring Design

## Overview

Two monitoring layers serve different purposes:

| Layer | Tool | Purpose | Users |
|-------|------|---------|-------|
| **Operational** | Grafana + Prometheus | System health, latency, alerts | Always-on, automated alerting |
| **Trading** | Custom React app | Live match view, trading decisions, P&L analysis | Active monitoring during matches |

Grafana handles infrastructure metrics (container health, latency, uptime).
The custom trading dashboard handles domain-specific views (P_true curves,
order book visualization, position tracking) that require richer interactivity
than Grafana provides.

### Data Flow

```
Match Container (Phase 3+4)
    │
    ├── Prometheus metrics ──► Grafana (operational)
    │     (counters, gauges, histograms)
    │
    ├── PostgreSQL writes ──► Trading Dashboard (custom React)
    │     (tick_snapshots, positions, event_log, bankroll_snapshot)
    │
    └── Redis pubsub ──► Trading Dashboard (live updates)
          (heartbeats, events, signals)
```

---

## Part 1: Grafana Operational Dashboards

Six dashboards, each with specific panels and alert rules.

### Dashboard 1: System Overview

**Purpose:** at-a-glance system health. This is the "home page" — if everything is green, no action needed.

| Panel | Type | Query/Source | Thresholds |
|-------|------|-------------|------------|
| Active Containers | Stat | `active_match_containers` gauge | 0=gray, 1-5=green, >5=yellow |
| Bankroll (Live) | Stat | `bankroll_balance{mode="live"}` | — |
| Bankroll (Paper) | Stat | `bankroll_balance{mode="paper"}` | — |
| Total Exposure % | Gauge | `total_exposure_ratio` | 0-15%=green, 15-20%=yellow, >20%=red |
| Max Drawdown % | Gauge | `max_drawdown_pct` | 0-10%=green, 10-15%=yellow, >15%=red |
| Match Pipeline | Table | `match_schedule` DB query | Color by status |
| Today's P&L | Stat | SUM(realized_pnl) WHERE date=today | negative=red, positive=green |
| Cumulative P&L (30d) | Time series | Daily P&L rolling sum | — |
| Open Positions | Table | `positions WHERE status='OPEN'` | — |
| Alerts Firing | Alert list | Prometheus alertmanager | — |

**SQL for Match Pipeline:**
```sql
SELECT match_id, league_id, kickoff_utc, status, trading_mode,
       EXTRACT(EPOCH FROM (NOW() - updated_at)) AS seconds_since_update
FROM match_schedule
WHERE kickoff_utc > NOW() - INTERVAL '24 hours'
ORDER BY kickoff_utc DESC;
```

### Dashboard 2: Live Match Detail

**Purpose:** deep dive into a single running match. Select match via dropdown.

**Variable:** `$match_id` (dropdown populated from `match_schedule WHERE status='PHASE3_RUNNING'`)

| Panel | Type | Query/Source | Notes |
|-------|------|-------------|-------|
| Match Header | Stat row | match_schedule JOIN | Teams, league, kickoff, score, engine_phase, trading_mode |
| P_true vs P_kalshi (Home Win) | Time series | tick_snapshots | Two lines: model vs market. Entry/exit points as annotations |
| P_true vs P_kalshi (Over 2.5) | Time series | tick_snapshots | Same format, second market |
| P_bet365 Overlay | Time series | tick_snapshots | Overlaid on P_true charts as dashed line |
| σ_MC | Time series | tick_snapshots | Per-market MC uncertainty band |
| μ_H, μ_A | Time series | tick_snapshots | Expected remaining goals |
| Event Timeline | Annotations | event_log | Goals, red cards, substitutions, ob_freeze, cooldown on/off |
| Order Book Depth | Bar chart | Latest depth snapshot | Ask levels (red) + Bid levels (green) |
| Positions | Table | positions WHERE match_id=$match_id | Direction, entry price, current EV, P&L |
| Signals Log | Table | Last 50 signals | Direction, EV, alignment, Kelly fraction, outcome |
| Engine State | Stat row | Latest tick_snapshot | engine_phase, event_state, cooldown, ob_freeze, pricing_mode |

**SQL for P_true time series:**
```sql
SELECT t AS time,
       (P_true->>'home_win')::float AS p_model,
       (P_kalshi->>'home_win_ask')::float AS p_market,
       (P_bet365->>'home_win')::float AS p_bet365
FROM tick_snapshots
WHERE match_id = $match_id
ORDER BY t;
```

**Event annotations:**
```sql
SELECT created_at AS time, event_type AS title,
       payload::text AS text
FROM event_log
WHERE match_id = $match_id
ORDER BY created_at;
```

### Dashboard 3: Latency & Performance

**Purpose:** detect performance degradation before it impacts trading.

| Panel | Type | Metric | Alert |
|-------|------|--------|-------|
| Tick Duration (p50/p95/p99) | Time series | `tick_latency` histogram | p99 > 1s = warning, > 3s = critical |
| MC Compute Time | Time series | `mc_compute_latency_seconds` histogram | p99 > 10ms = warning |
| Odds-API WS Latency | Time series | `odds_api_ws_latency_seconds` | p99 > 500ms = warning |
| Live Score Poll Latency | Time series | `live_score_poll_latency_seconds` | p99 > 5s = warning |
| Order Fill Time | Time series | `order_fill_latency_seconds` | p99 > 5s = warning |
| Heartbeat Age | Time series | `heartbeat_age_seconds` per match | > 30s = warning, > 60s = critical |
| Tick Overruns/min | Counter rate | `tick_overrun_total` | > 5/min = warning |
| Queue Depth (Phase 3→4) | Gauge | `phase4_queue_depth` | should be 0 or 1 |

### Dashboard 4: Risk & Exposure

**Purpose:** portfolio-level risk monitoring.

| Panel | Type | Query | Notes |
|-------|------|-------|-------|
| Total Exposure (live) | Gauge | `get_total_exposure(false)` | Max 20% |
| Total Exposure (paper) | Gauge | `get_total_exposure(true)` | Max 20% |
| Exposure by Match | Stacked bar | exposure_cache GROUP BY match_id | Per-match contribution |
| Exposure by League | Pie chart | exposure_cache JOIN match_schedule | Concentration risk |
| Drawdown Curve (30d) | Time series | bankroll_snapshot | Highwater mark overlay |
| Active Reservations | Table | exposure_reservation WHERE status='RESERVED' | Should be near-empty |
| Stale Reservations | Stat | COUNT WHERE status='RESERVED' AND age > 60s | > 0 = investigate |
| Position Heat Map | Table | All open positions | Color by unrealized P&L |
| Bankroll History | Time series | bankroll_snapshot | Live + Paper separate lines |

### Dashboard 5: Phase 1 Model Health

**Purpose:** detect model drift before it costs money.

| Panel | Type | Query | Alert |
|-------|------|-------|-------|
| Brier Score (rolling 20 matches) | Time series | Post-match analytics | Outside Phase 1.5 ± 0.03 = warning |
| Edge Realization (rolling 20) | Time series | Post-match analytics | < 0.5 or > 1.5 = warning |
| Brier by League | Grouped bar | Post-match analytics per league | Tier 1 must be < baseline |
| Direction Realization | Two lines | Yes vs No edge realization | Divergence > 0.3 = investigate |
| Market Alignment Value | Time series | ALIGNED avg return - DIVERGENT avg return | < 0 = alignment worthless |
| Parameter Version History | Event log | production_params table | Version + validation scores |
| Param Change Log | Table | parameter_change_log | Recent adaptive tuning changes |
| Brier Score vs Betfair Exchange | Scatter | Model BS vs Exchange BS per match | Points below diagonal = winning |

### Dashboard 6: Paper Trading Validation (Phase 0)

**Purpose:** evaluate paper trading realism and readiness for live.

| Panel | Type | Query | Threshold |
|-------|------|-------|-----------|
| Paper Trades Count | Stat | COUNT WHERE is_paper=true | Target: ≥ 50 |
| Paper P&L Curve | Time series | Cumulative paper P&L | — |
| Paper Edge Realization | Stat | paper avg return / avg EV | 0.6-1.5 = healthy |
| Paper Max Drawdown | Stat | Paper bankroll drawdown | < 15% to graduate |
| Paper Realism Score | Stat | compute_paper_realism_score() | > 0.85 to graduate |
| Fill Delay Distribution | Histogram | positions.fill_delay WHERE is_paper | Should cluster 1-3s |
| Paper vs Actual Price | Scatter | Paper fill price vs market mid at fill time | Tight cluster = realistic |
| Partial Fill Rate | Stat | % of paper fills that were partial | — |
| Graduation Checklist | Stat grid | 8 criteria from Phase 0 | All green = ready for Phase A |
| Slippage Distribution | Histogram | paper_slippage field | Should be ~1-2 ticks |

**Graduation Checklist SQL:**
```sql
SELECT
    (SELECT COUNT(*) FROM positions WHERE is_paper) AS trade_count,
    (SELECT COUNT(*) >= 50 FROM positions WHERE is_paper) AS trades_ok,
    -- ... each of the 8 criteria as boolean columns
```

---

## Part 2: Custom Trading Dashboard (React)

Grafana is excellent for time-series and operational metrics but limited for:
- Real-time order book visualization
- Interactive position management
- Live signal decision display
- Multi-match simultaneous monitoring

### Architecture

```
┌──────────────────────────────────────┐
│  React Frontend (Trading Dashboard)   │
│  • Next.js + TailwindCSS + Recharts  │
│  • WebSocket for live updates         │
│  • REST for historical queries        │
└──────────┬───────────────────────────┘
           │
┌──────────┴───────────────────────────┐
│  Dashboard API Server (FastAPI)       │
│  • /ws/live — WebSocket push          │
│  • /api/matches — match list          │
│  • /api/match/{id}/ticks — tick data  │
│  • /api/positions — positions         │
│  • /api/analytics — post-match        │
│  Subscribes to Redis pubsub           │
│  Reads from PostgreSQL                │
└──────────┬───────────────────────────┘
           │
    ┌──────┴──────┐
    │ PostgreSQL  │  Redis
    └─────────────┘
```

### View 1: Command Center (Multi-Match Overview)

The primary view during active trading sessions. Shows all running matches simultaneously.

```
┌─────────────────────────────────────────────────────────────────┐
│  MMPP Trading System — Command Center           [Paper Mode] 🟡 │
│  Bankroll: $9,847.32  │  Exposure: 12.4% / 20%  │  Drawdown: 3.2% │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Arsenal vs Chelsea (EPL) ──── SECOND_HALF 67' ──── LIVE ─┐ │
│  │  Score: 1-0  │  Events: ⚽ 34' Saka                        │ │
│  │                                                             │ │
│  │  Home Win    Model: 0.72  Market: 0.68  Edge: +4.2¢  ▲    │ │
│  │  Over 2.5    Model: 0.58  Market: 0.61  Edge: -2.8¢  —    │ │
│  │                                                             │ │
│  │  Positions: BUY_YES home_win @ 0.65, qty=25, P&L: +$1.75  │ │
│  │  Last Signal: HOLD (edge below θ for over_25)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Barcelona vs Real Madrid (La Liga) ──── FIRST_HALF 23' ──┐ │
│  │  Score: 0-0  │  Events: (none)                              │ │
│  │                                                             │ │
│  │  Home Win    Model: 0.45  Market: 0.43  Edge: +1.8¢  —    │ │
│  │  Over 2.5    Model: 0.71  Market: 0.66  Edge: +4.9¢  ▲    │ │
│  │                                                             │ │
│  │  Positions: BUY_YES over_25 @ 0.64, qty=15, P&L: +$0.30  │ │
│  │  Last Signal: HOLD (already at optimal allocation)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Upcoming ─────────────────────────────────────────────────┐ │
│  │  Bayern vs Dortmund (Bundesliga) — Phase 2 in 42 min       │ │
│  │  Liverpool vs Man City (EPL) — Phase 2 in 1h 15min         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Data refresh:** WebSocket push every 1 second for running matches.

### View 2: Match Deep Dive (Single Match)

Click any match from Command Center to enter detailed view.

**Top Section — Status Bar:**
```
Arsenal vs Chelsea │ EPL │ SECOND_HALF 67' │ Score: 1-0
Param v12 │ Mode: Paper │ Pricing: Analytical │ Cooldown: OFF │ ob_freeze: OFF
```

**Middle Section — Three Columns:**

```
┌─ Price Chart ─────────────────┐ ┌─ Order Book ──────┐ ┌─ Signals ──────────┐
│                               │ │                    │ │                     │
│  0.80 ┤                      │ │  Ask               │ │  67:32 HOLD         │
│       │    ╭─ P_true         │ │  0.72  ████ 45     │ │  67:31 HOLD         │
│  0.70 ┤───╯   ╭─ P_kalshi   │ │  0.71  ██████ 80   │ │  67:30 BUY_YES      │
│       │  ╭────╯             │ │  0.70  ████████ 120 │ │    EV: 0.032        │
│  0.60 ┤──╯   ⚽ 34'          │ │  ── spread: 2¢ ──  │ │    Kelly: 0.018     │
│       │  ┈┈ P_bet365        │ │  0.68  ██████ 90    │ │    Aligned: YES     │
│  0.50 ┤                      │ │  0.67  ████ 50     │ │    Fill: 25 @ 0.70  │
│       ├──┬──┬──┬──┬──┬──┬── │ │  0.66  ██ 20       │ │  67:15 HOLD         │
│       0  15 30 45 60 75 90   │ │  Bid               │ │  67:14 HOLD         │
│           minute              │ │                    │ │  ...                │
└───────────────────────────────┘ └────────────────────┘ └─────────────────────┘
```

**Bottom Section — Positions and Events:**

```
┌─ Open Positions ────────────────────────────────────────────────────────────┐
│  Market      Dir      Entry   Qty   Current   Unreal P&L   EV@Entry  Align │
│  home_win    BUY_YES  0.65    25    0.70      +$1.25       0.032    ALIGN  │
│  over_25     —        —       —     —         —            —         —      │
├─ Event Timeline ────────────────────────────────────────────────────────────┤
│  ⚽ 34' Saka (Arsenal) │ ob_freeze 34:00-34:08 │ cooldown 34:08-34:23     │
│  🟢 P_true jumped 0.55→0.72 │ Signal: BUY_YES home_win                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### View 3: P&L Analytics (Post-Match & Cumulative)

**Filters:** date range, league, market type, paper/live, direction

```
┌─ Summary Stats ─────────────────────────────────────────────────┐
│  Total Trades: 47  │  Win Rate: 59.6%  │  Avg EV: 2.8¢         │
│  Total P&L: +$234  │  Edge Real: 0.87  │  Avg Slippage: 0.8¢   │
│  Max DD: 8.2%      │  Sharpe: 1.4      │  Brier vs Exchange: -0.008 │
├─ P&L Curve ─────────────────────────────────────────────────────┤
│  [Cumulative P&L line chart with drawdown shading]               │
├─ Breakdown ─────────────────────────────────────────────────────┤
│  By League: EPL +$145, La Liga +$62, MLS +$34, Brasileirão +$18, ...  │
│  By Market: home_win +$180, over_25 +$40, btts +$14             │
│  By Direction: BUY_YES +$198, BUY_NO +$36                       │
│  By Alignment: ALIGNED +$210, DIVERGENT +$24                     │
├─ Trade Table ───────────────────────────────────────────────────┤
│  [Sortable, filterable table of all trades]                      │
└──────────────────────────────────────────────────────────────────┘
```

### View 4: System Operations

```
┌─ Container Status ──────────────────────────────────────────────┐
│  match-ARS-CHE    │ PHASE3_RUNNING │ 67min │ ❤ 3s ago │ 🟢    │
│  match-BAR-RMA    │ PHASE3_RUNNING │ 23min │ ❤ 1s ago │ 🟢    │
│  match-BAY-DOR    │ SCHEDULED      │ -42min │ —       │ ⏳    │
├─ Connection Health ─────────────────────────────────────────────┤
│  Odds-API WS:    🟢 connected (last msg 0.3s ago)              │
│  Goalserve REST:  🟢 polling (last poll 1.2s ago)               │
│  Kalshi WS:       🟢 connected (last msg 0.8s ago)              │
│  PostgreSQL:      🟢 pool 2/5 active                            │
│  Redis:           🟢 connected                                  │
├─ Recent Alerts ─────────────────────────────────────────────────┤
│  (none)                                                          │
├─ Param Version ─────────────────────────────────────────────────┤
│  Active: v12 (trained 2025-03-08, Brier: 0.198)                │
│  Matches since retrain: 23                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Dashboard API Server

### API Endpoints

```python
# FastAPI server — bridges PostgreSQL/Redis to the React frontend

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MMPP Trading Dashboard API")

# ─── REST Endpoints ───

@app.get("/api/matches")
async def list_matches(status: Optional[str] = None,
                       date: Optional[str] = None) -> List[MatchSummary]:
    """List matches with current status and basic stats."""

@app.get("/api/match/{match_id}")
async def match_detail(match_id: str) -> MatchDetail:
    """Full match detail including latest tick, positions, events."""

@app.get("/api/match/{match_id}/ticks")
async def match_ticks(match_id: str,
                      market: Optional[str] = None,
                      downsample: int = 1) -> List[TickSnapshot]:
    """Time series of tick snapshots for charting.
    downsample=10 returns every 10th tick (for long matches)."""

@app.get("/api/match/{match_id}/events")
async def match_events(match_id: str) -> List[EventLogEntry]:
    """Event log for annotations."""

@app.get("/api/positions")
async def list_positions(status: Optional[str] = "OPEN",
                         is_paper: Optional[bool] = None) -> List[Position]:
    """All positions, filterable by status and paper/live."""

@app.get("/api/analytics/pnl")
async def pnl_analytics(days: int = 30,
                         league: Optional[str] = None,
                         is_paper: Optional[bool] = None) -> PnLReport:
    """Aggregated P&L analytics with breakdowns."""

@app.get("/api/analytics/model-health")
async def model_health() -> ModelHealthReport:
    """Phase 1 model health metrics (Brier, edge realization, etc.)."""

@app.get("/api/analytics/paper-graduation")
async def paper_graduation() -> GraduationChecklist:
    """Phase 0 graduation criteria check."""

@app.get("/api/system/status")
async def system_status() -> SystemStatus:
    """Container status, connection health, param version."""

# ─── WebSocket for Live Updates ───

@app.websocket("/ws/live")
async def live_updates(ws: WebSocket):
    """Push real-time updates to frontend.

    Subscribes to Redis pubsub channels:
    - tick:{match_id} — every-second tick data
    - event:{match_id} — goal, card, period change
    - signal:{match_id} — Phase 4 signals
    - position_update — new fills, exits
    - system_alert — critical alerts

    Client sends: {"subscribe": ["match_id_1", "match_id_2"]}
    Server pushes: {"type": "tick", "match_id": "...", "data": {...}}
    """
    await ws.accept()
    subscriptions = set()

    # Background task: listen to Redis and forward to WebSocket
    async def redis_listener():
        pubsub = redis.pubsub()
        while True:
            # Dynamic subscription based on client requests
            for match_id in subscriptions:
                await pubsub.subscribe(
                    f"tick:{match_id}",
                    f"event:{match_id}",
                    f"signal:{match_id}",
                )
            await pubsub.subscribe("position_update", "system_alert")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    await ws.send_json(json.loads(message["data"]))

    listener_task = asyncio.create_task(redis_listener())

    try:
        async for data in ws.iter_json():
            if "subscribe" in data:
                subscriptions = set(data["subscribe"])
    except WebSocketDisconnect:
        listener_task.cancel()
```

### Container → Redis Publishing

Match containers must publish to Redis for the dashboard to receive live data.
Add to Phase 3/4 tick loop:

```python
# In emit_to_phase4 — also publish to Redis for dashboard
async def publish_tick_to_dashboard(model, P_true, σ_MC, order_allowed):
    """Publish tick data to Redis for live dashboard consumption.
    Throttled to every 1s (same as tick frequency)."""
    await redis.publish(f"tick:{model.match_id}", json.dumps({
        "type": "tick",
        "match_id": model.match_id,
        "t": model.t,
        "engine_phase": model.engine_phase,
        "P_true": P_true,
        "σ_MC": σ_MC,
        "P_bet365": model.bet365_implied,
        "order_allowed": order_allowed,
        "cooldown": model.cooldown,
        "ob_freeze": model.ob_freeze,
        "event_state": model.event_state,
        "mu_H": model.μ_H,
        "mu_A": model.μ_A,
        "score": list(model.S),
    }))

# In signal_generator — publish signals
async def publish_signal_to_dashboard(model, ticker, signal):
    await redis.publish(f"signal:{model.match_id}", json.dumps({
        "type": "signal",
        "match_id": model.match_id,
        "ticker": ticker,
        "direction": signal.direction,
        "EV": signal.EV,
        "P_cons": signal.P_cons,
        "P_kalshi": signal.P_kalshi,
        "alignment": signal.alignment_status,
        "kelly_multiplier": signal.kelly_multiplier,
        "timestamp": time.time(),
    }))
```

---

## Part 4: Alert Integration

### Alert Channels

| Severity | Channel | Response Time |
|----------|---------|---------------|
| Critical | SMS + Slack + Dashboard banner | < 1 min |
| Warning | Slack + Dashboard notification | < 15 min |
| Info | Dashboard log only | Next review |

### Alert Definitions

**Critical (immediate action required):**

| Alert | Condition | Action |
|-------|-----------|--------|
| Container crash | exit code ≠ 0 | Freeze positions, check logs |
| Drawdown > 15% | `max_drawdown_pct > 15` | Stop all new entries |
| DB unreachable | 3 consecutive query failures | All containers freeze |
| Stale PENDING position | PENDING > 5 min | Manual reconciliation with Kalshi |
| Bankroll below minimum | balance < $500 | Stop all new entries |
| Heartbeat dead | > 60s for any running match | Investigate container |

**Warning (investigate soon):**

| Alert | Condition | Action |
|-------|-----------|--------|
| Exposure > 15% | `total_exposure_ratio > 0.15` | Review positions |
| Tick overrun | p99 > 3s for 5 min | Check MC performance |
| Odds-API WS disconnected | > 30s | Fallback mode active |
| Brier Score drifting | outside ± 0.03 for 10 matches | Consider retrain |
| Edge realization < 0.5 | rolling 20 matches | Review model assumptions |
| Stale reservation | RESERVED > 60s | Check execution pipeline |
| Settlement timeout | AWAITING > 6h | Manual check on Kalshi |

**Info (review at convenience):**

| Alert | Condition | Notes |
|-------|-----------|-------|
| New param version | Phase 1 retrain complete | Check validation scores |
| Match skipped | Phase 2 sanity = SKIP | Normal — check reason |
| Paper graduation ready | All 8 criteria met | Consider Phase A transition |
| Adaptive param change | Step 4.6 tuning | Review parameter_change_log |

### Slack Integration

```python
async def send_alert(severity: str, title: str, details: dict):
    """Route alert to appropriate channels."""
    message = {
        "text": f"{'🔴' if severity == 'critical' else '🟡' if severity == 'warning' else 'ℹ️'} {title}",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": title}},
            {"type": "section", "text": {"type": "mrkdwn",
                "text": "\n".join(f"*{k}:* {v}" for k, v in details.items())}},
        ]
    }

    # Slack
    await httpx.post(config.slack_webhook, json=message)

    # SMS for critical
    if severity == "critical":
        await twilio.send_sms(config.alert_phone, f"MMPP CRITICAL: {title}")

    # Dashboard banner (via Redis)
    await redis.publish("system_alert", json.dumps({
        "severity": severity,
        "title": title,
        "details": details,
        "timestamp": time.time(),
    }))
```

---

## Part 5: Deployment

### Docker Compose Addition

```yaml
  dashboard-api:
    build: ./dashboard-api
    depends_on: [postgres, redis]
    environment:
      DB_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/soccer_trading
      REDIS_URL: redis://redis:6379
      SLACK_WEBHOOK: ${SLACK_WEBHOOK}
    ports:
      - "8000:8000"  # FastAPI

  dashboard-ui:
    build: ./dashboard-ui
    depends_on: [dashboard-api]
    environment:
      NEXT_PUBLIC_API_URL: http://dashboard-api:8000
      NEXT_PUBLIC_WS_URL: ws://dashboard-api:8000/ws/live
    ports:
      - "3001:3000"  # Next.js
```

### Key Libraries

| Component | Library | Purpose |
|-----------|---------|---------|
| Frontend framework | Next.js 14 | React SSR + routing |
| Charting | Recharts + lightweight-charts | Time series + order book |
| Styling | TailwindCSS | Rapid UI development |
| WebSocket client | native WebSocket API | Live data streaming |
| Backend framework | FastAPI | REST + WebSocket API |
| DB driver | asyncpg | PostgreSQL async access |
| Redis client | redis-py (async) | Pubsub subscription |

### DB Indexes for Dashboard Queries

```sql
-- Fast tick queries for price chart (most frequent dashboard query)
CREATE INDEX idx_ticks_match_time ON tick_snapshots(match_id, t);

-- Fast event queries for annotations
CREATE INDEX idx_events_match_time ON event_log(match_id, created_at);

-- Fast position queries
CREATE INDEX idx_positions_match_status ON positions(match_id, status);
CREATE INDEX idx_positions_paper ON positions(is_paper, status);

-- P&L analytics (date-range queries)
CREATE INDEX idx_positions_settled ON positions(status, created_at)
    WHERE status = 'SETTLED';
```

---

## Dashboard Development Priority

| Priority | View | Reason |
|----------|------|--------|
| **P0** | Grafana: System Overview | Must have from day 1 for health monitoring |
| **P0** | Grafana: Latency | Must catch performance issues early |
| **P0** | Alerts (Slack + SMS) | Must know when things break |
| **P1** | Custom: Command Center | Essential for active match monitoring |
| **P1** | Grafana: Risk & Exposure | Essential once paper trading starts |
| **P1** | Grafana: Paper Validation | Needed for Phase 0 graduation decision |
| **P2** | Custom: Match Deep Dive | Helpful for debugging but not critical |
| **P2** | Custom: P&L Analytics | Needed for Phase A performance review |
| **P2** | Grafana: Phase 1 Health | Needed once 20+ matches accumulated |
| **P3** | Custom: System Operations | Nice-to-have, Grafana covers basics |