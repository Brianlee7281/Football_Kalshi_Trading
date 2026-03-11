# Configuration Reference

All system parameters in one place. Each parameter specifies its source document,
which phase uses it, whether it's tunable at runtime, and its adaptive tuning range.

---

## Trading Parameters

| Parameter | Default | Range | Source | Phase | Adaptive (Step 4.6) |
|-----------|---------|-------|--------|-------|---------------------|
| `THETA_ENTRY` | 0.02 (2¢) | [0.01, 0.05] | phase4.md Step 4.2 | 4 | Yes — calibrated from breakeven edge + margin |
| `THETA_EXIT` | 0.005 (0.5¢) | [0.003, 0.01] | phase4.md Step 4.4 Trigger 1 | 4 | No |
| `K_frac` | 0.25 | [0.10, 0.50] | phase4.md Step 4.3 | 4 | Yes — increase if edge_realization ≥ 0.8, decrease if < 0.5 |
| `z` | 1.645 | [1.0, 2.5] | phase4.md Step 4.2 | 4 | Yes — directional tuning based on No-side realization |
| `c` (fee_rate) | 0.07 (7%) | fixed | phase4.md Step 4.2 | 4 | No — set by Kalshi |
| `DIVERGENCE_THRESHOLD` | 0.05 (5pp) | [0.03, 0.10] | phase4.md Step 4.4 Trigger 4 | 4 | No |
| `TRIM_THRESHOLD` | 0.5 | [0.3, 0.7] | phase4.md Step 4.4 Trigger 5 | 4 | No — trim when f_optimal < existing × this |
| `OPP_COST_CURRENT_MAX` | 2× THETA_EXIT | — | phase4.md Step 4.4 Trigger 6 | 4 | No — exit if current EV below this |
| `σ_MC_FLOOR` | 0.005 | [0.003, 0.01] | phase3.md Step 3.4 | 3 | No — minimum σ_MC in analytical mode |
| `KALSHI_STALE_THRESHOLD` | 5.0s | [3, 10] | phase4.md Step 4.1 | 4 | No — skip trading if Kalshi WS older than this |
| `BET365_STALE_THRESHOLD` | 30.0s | [15, 60] | phase4.md Step 4.1 | 4 | No — treat as UNAVAILABLE if older than this |

## Risk Limits

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `F_ORDER_CAP` | 0.03 (3%) | [0.01, 0.05] | phase4.md Step 4.3 | 4 | Single order cannot exceed this % of bankroll |
| `F_MATCH_CAP` | 0.05 (5%) | [0.03, 0.10] | phase4.md Step 4.3 | 4 | Total exposure per match |
| `F_TOTAL_CAP` | 0.20 (20%) | [0.10, 0.30] | phase4.md Step 4.3 | 4 | Total portfolio exposure across all matches |

## Market Alignment

| Parameter | Default | Range | Source | Phase | Adaptive |
|-----------|---------|-------|--------|-------|----------|
| `ALIGNED_MULTIPLIER` | 0.8 | [0.7, 1.0] | phase4.md Step 4.2 | 4 | No |
| `DIVERGENT_MULTIPLIER` | 0.5 | [0.3, 0.65] | phase4.md Step 4.2 | 4 | Yes — tuned by alignment_value metric |
| `UNAVAILABLE_MULTIPLIER` | 0.6 | [0.4, 0.8] | phase4.md Step 4.2 | 4 | No |

## Liquidity Gate

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `Q_min` | 20 contracts | [10, 50] | phase4.md Step 4.1 | 4 | Minimum order book depth to enter |
| `depth_fraction` | 0.30 | [0.20, 0.50] | phase4.md Step 4.3 | 4 | Max fraction of visible depth to consume |
| `min_fill_ratio` | 0.50 | [0.30, 0.70] | phase4.md Step 4.3 | 4 | Skip if gated qty / target qty below this |

## Cooldown & Event Handling

| Parameter | Default | Range | Source | Phase | Adaptive |
|-----------|---------|-------|--------|-------|----------|
| `COOLDOWN_SECONDS` | 15 | [8, 20] | phase3.md Step 3.5 | 3 | Yes — decrease if cooldown_suppressed_profitable_rate > 0.6 |
| `VAR_SAFETY_WAIT` | 5s | [3, 10] | phase4.md Step 4.5 Rapid Entry | 4 | No |
| `OB_FREEZE_ODDS_THRESHOLD` | varies | — | phase3.md Step 3.1 | 3 | No — derived from odds delta detection |

## Monte Carlo

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `N_MC` | 50,000 | [10000, 200000] | phase3.md Step 3.4 | 3 | Yields σ_MC ≈ 0.2pp at typical probabilities |
| `MC_EXECUTOR_WORKERS` | 4 | [2, 8] | phase3.md Step 3.4 | 3 | ThreadPoolExecutor max_workers |

## Phase 1 Calibration

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `σ_a` | grid-search | {0.1, 0.3, 0.5, 1.0} | phase1.md Step 1.4 | 1 | ML prior regularization strength |
| `τ_H, τ_A` | exp(0) = 1.0 | [0.1, 5.0] | phase1.md Step 1.4 | 1 | Parametric delta saturation rate (log-space optimized) |
| `go_threshold` | 90th percentile | calibrated | phase1.md Step 1.5 | 1→2 | From model-Betfair Exchange discrepancy distribution |
| `hold_threshold` | 99th percentile | calibrated | phase1.md Step 1.5 | 1→2 | From model-Betfair Exchange discrepancy distribution |
| `ou_threshold` | calibrated | calibrated | phase1.md Step 1.5 | 1→2 | Over/Under sanity threshold |

## Paper Trading

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `PAPER_BANKROLL` | $10,000 | — | orchestration.md | — | Virtual starting balance |
| `slippage_ticks` | 1 | [1, 3] | phase4.md Step 4.5 | 4 | Paper slippage simulation |
| `fill_delay_range` | (1.0, 3.0)s | — | phase4.md Step 4.5 | 4 | Paper fill delay simulation |

## Rapid Entry (Phase C only)

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `RAPID_ENTRY_ENABLED` | False | — | phase4.md Step 4.5 | 4 | Enabled after 200+ trades, conditions met |
| `cumulative_trades_min` | 200 | — | phase4.md Step 4.5 | 4 | Minimum trades before enabling |
| `edge_realization_min` | 0.8 | — | phase4.md Step 4.5 | 4 | Minimum edge realization |
| `preliminary_accuracy_min` | 0.95 | — | phase4.md Step 4.5 | 4 | Minimum preliminary event accuracy |
| `var_cancellation_rate_max` | 0.03 | — | phase4.md Step 4.5 | 4 | Maximum VAR cancellation rate |

## WebSocket Resilience

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `WS_BACKOFF_BASE` | 1.0s | — | phase3.md Step 3.1 | 3 | Initial reconnect delay |
| `WS_BACKOFF_MAX` | 30.0s | — | phase3.md Step 3.1 | 3 | Maximum reconnect delay |
| `WS_MAX_RETRIES` | 5 | [3, 10] | phase3.md Step 3.1 | 3 | Consecutive failures before fallback mode |

## Infrastructure

| Parameter | Default | Range | Source | Phase | Notes |
|-----------|---------|-------|--------|-------|-------|
| `MAX_CONCURRENT_MATCHES` | 8 | [5, 15] | orchestration.md | — | Maximum simultaneously running match containers |
| `MAX_MATCH_DURATION` | 9 hours | — | orchestration.md | — | Safety timeout (match ~2h + settlement ~6h + buffer) |
| `DB_POOL_MIN` | 2 | — | orchestration.md | — | asyncpg minimum pool size |
| `DB_POOL_MAX` | 5 | — | orchestration.md | — | asyncpg maximum pool size |
| `HEARTBEAT_INTERVAL` | 10s | — | orchestration.md | — | Container → Redis heartbeat frequency |
| `HEARTBEAT_STALE_THRESHOLD` | 60s | — | orchestration.md | — | Alert if heartbeat older than this |
| `RESERVATION_STALE_TIMEOUT` | 60s | — | orchestration.md | — | Auto-release stale RESERVED exposure |

## Scheduling

| Parameter | Default | Source | Notes |
|-----------|---------|--------|-------|
| `MATCH_DISCOVERY_INTERVAL` | 6 hours | orchestration.md | Scan upcoming 48h |
| `PHASE2_TRIGGER_OFFSET` | -65 min | orchestration.md | Before kickoff |
| `PHASE3_TRIGGER_OFFSET` | -2 min | orchestration.md | Before kickoff |
| `GOALSERVE_POLL_INTERVAL` | 3s | phase3.md | Live Score REST polling |
| `TICK_INTERVAL` | 1s | phase3.md | Phase 3 pricing frequency |

## Data API Keys (from Secret Store)

| Key | Source | Provider |
|-----|--------|----------|
| `ODDS_API_KEY` | orchestration.md | The Odds API |
| `GOALSERVE_API_KEY` | orchestration.md | Goalserve |
| `KALSHI_API_KEY` | orchestration.md | Kalshi |
| `DB_PASSWORD` | orchestration.md | PostgreSQL |
| `SLACK_WEBHOOK` | dashboard.md | Slack (alerts) |

## Bookmakers (Odds-API, 5 selected)

> **Historical data limitation:** Odds-API historical events are available from **December 2025 onwards** only.
> Pre-Dec-2025 seasons train XGBoost without odds features (NaN → XGBoost handles natively).
> `scripts/odds_backfill.py` accumulates settled events daily for future retraining.

| Bookmaker | Odds-API Key | Role |
|-----------|-------------|------|
| Bet365 | `bet365` | Primary in-play odds, ob_freeze early warning |
| Betfair Exchange | `betfair_exchange` | Brier Score baseline, most efficient close line |
| Sbobet | `sbobet` | Asian market sharp line, market diversity |
| 1xBet | `1xbet` | Wide league coverage, aggressive lines |
| DraftKings | `draftkings` | US market perspective, Kalshi trader correlation |

---

## Tradable Leagues (8 leagues)

| League | Region | Goalserve ID | Season | Weekly Matches | Notes |
|--------|--------|-------------|--------|---------------|-------|
| EPL | Europe | 1204 | Aug-May | ~10 | Highest Kalshi liquidity |
| La Liga | Europe | 1399 | Aug-May | ~10 | |
| Bundesliga | Europe | 1229 | Aug-May | ~9 | |
| Serie A | Europe | 1269 | Aug-May | ~10 | |
| Ligue 1 | Europe | 1221 | Aug-May | ~10 | |
| MLS | Americas | 1005 | Mar-Oct | ~14 | Different season window from Europe |
| Brasileirão | Americas | 1572 | Apr-Nov | ~10 | Verify Goalserve league_id |
| Liga Argentina | Americas | 1300 | Feb-Dec | ~7 | Verify Goalserve league_id |

> **Timezone note:** Europe matches typically 12:00-21:00 UTC, Americas 00:00-04:00 UTC.
> Peak concurrent load is Saturday afternoon UTC (Europe) overlapping with Americas evening.
> MLS + Brasileirão season overlaps with European season from Aug-Oct and Mar-May.

> **Data quality note:** Betfair Exchange coverage may be thinner for Brasileirão and Liga Argentina.
> If Betfair Exchange odds are unavailable for a match, fall back to market average of remaining 4 bookmakers.

---

## Phase Evolution — Parameter Changes by Phase

```
Phase 0 (Paper):
  TRADING_MODE = "paper"
  K_frac = 0.25
  z = 1.645
  DIVERGENT entries = BLOCKED
  RAPID_ENTRY_ENABLED = False
  THETA_ENTRY = calibrate from paper data

Phase A (Conservative Live):
  TRADING_MODE = "live"
  K_frac = 0.25
  DIVERGENT entries = BLOCKED
  RAPID_ENTRY_ENABLED = False
  Bankroll allocation = 50%

Phase B (Adaptive Live):
  K_frac = 0.25 → 0.50 (data-driven)
  DIVERGENT entries = ALLOWED (multiplier 0.5)
  z = directional tuning
  Bankroll allocation = 100%

Phase C (Mature Live):
  RAPID_ENTRY_ENABLED = True (if conditions met)
  BET365_DIVERGENCE_AUTO_EXIT = True (if data supports)
  Full adaptive tuning loop enabled
```

---

## Config File Format (YAML)

```yaml
# config/system.yaml — loaded by MatchConfig.from_env()

trading:
  mode: "paper"                    # "paper" or "live"
  fee_rate: 0.07                   # Kalshi fee
  theta_entry: 0.02
  theta_exit: 0.005
  K_frac: 0.25
  z: 1.645

risk:
  f_order_cap: 0.03
  f_match_cap: 0.05
  f_total_cap: 0.20

alignment:
  aligned_multiplier: 0.8
  divergent_multiplier: 0.5
  unavailable_multiplier: 0.6

liquidity:
  q_min: 20
  depth_fraction: 0.30
  min_fill_ratio: 0.50

cooldown:
  duration_seconds: 15
  var_safety_wait: 5

mc:
  n_mc: 50000
  executor_workers: 4

paper:
  bankroll: 10000
  slippage_ticks: 1
  fill_delay_min: 1.0
  fill_delay_max: 3.0

rapid_entry:
  enabled: false
  cumulative_trades_min: 200
  edge_realization_min: 0.8
  preliminary_accuracy_min: 0.95
  var_cancellation_rate_max: 0.03

websocket:
  backoff_base: 1.0
  backoff_max: 30.0
  max_retries: 5

infrastructure:
  max_concurrent_matches: 8
  max_match_duration_hours: 9
  db_pool_min: 2
  db_pool_max: 5
  heartbeat_interval: 10
  heartbeat_stale_threshold: 60
  reservation_stale_timeout: 60

scheduling:
  match_discovery_interval_hours: 6
  phase2_trigger_offset_minutes: -65
  phase3_trigger_offset_minutes: -2
  goalserve_poll_interval: 3
  tick_interval: 1

bookmakers:
  - key: "bet365"
    role: "primary_inplay"
  - key: "betfair_exchange"
    role: "baseline_benchmark"
  - key: "sbobet"
    role: "asian_sharp"
  - key: "1xbet"
    role: "wide_coverage"
  - key: "draftkings"
    role: "us_market"

# Phase 1 calibration params are NOT in this config —
# they are versioned in the production_params DB table.
# Sanity thresholds (go/hold/ou) are also in production_params.
```