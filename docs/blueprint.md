# Project Blueprint — MMPP Soccer Live Trading System

## Repository Structure

```
mmpp-soccer/
│
├── README.md                           # Project overview, quickstart, architecture diagram
├── LICENSE
├── .gitignore
├── .env.example                        # Template for local dev secrets
├── Makefile                            # Common commands (make test, make up, make lint)
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  DESIGN DOCS (this session's output)                        ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── docs/
│   ├── phase1.md                       # Phase 1: Offline Calibration
│   ├── phase2.md                       # Phase 2: Pre-Match Initialization
│   ├── phase3.md                       # Phase 3: Live Trading Engine
│   ├── phase4.md                       # Phase 4: Arbitrage & Execution
│   ├── orchestration.md                # System Orchestration & Scheduling
│   ├── dashboard.md                    # Dashboard & Monitoring Design
│   ├── config_reference.md             # Configuration Reference (60+ params)
│   └── schema.sql                      # Consolidated DB Schema (12 tables)
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  CONFIGURATION                                              ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── config/
│   ├── system.yaml                     # Main config (from config_reference.md YAML section)
│   ├── system.paper.yaml               # Phase 0 overrides (K_frac=0.25, DIVERGENT=blocked)
│   ├── system.live.yaml                # Phase A+ overrides
│   └── leagues.yaml                    # Tradable leagues + Goalserve IDs + Kalshi market types
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  SOURCE CODE                                                ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── src/
│   ├── __init__.py
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Phase 1: Offline Calibration (GPU worker)          │
│   │   │  Docs: phase1.md                                    │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── step_1_1_intervals.py       # IntervalRecord parsing, RedCardTransition
│   │   ├── step_1_2_Q_estimation.py    # Transition matrix Q estimation
│   │   ├── step_1_3_ml_prior.py        # XGBoost Poisson regression, build_odds_features
│   │   ├── step_1_4_nll_optimize.py    # Joint NLL optimization (PyTorch), log-tau, σ_a
│   │   ├── step_1_5_validation.py      # Brier Score, league-stratified BS, sanity thresholds
│   │   ├── step_3_6_backtest.py        # In-play backtest (Step 3.6, runs after Phase 1)
│   │   ├── phase1_worker.py            # Entry point for Phase 1 GPU container
│   │   └── odds_backfill.py            # Odds-API historical odds daily backfill
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Phase 2: Pre-Match Initialization                  │
│   │   │  Docs: phase2.md                                    │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── prematch/
│   │   ├── __init__.py
│   │   ├── step_2_1_data_collection.py # Goalserve fixtures + Odds-API odds + lineups
│   │   ├── step_2_2_feature_select.py  # 4-tier feature selection, extract_odds_features
│   │   ├── step_2_3_backsolve.py       # a_H, a_A backsolve from XGBoost prior
│   │   ├── step_2_4_sanity_check.py    # Primary + secondary sanity checks
│   │   ├── step_2_5_system_init.py     # P_grid precompute, WS connect, Numba warm-up
│   │   └── pipeline.py                 # Phase2Pipeline: orchestrates Steps 2.1→2.5
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Phase 3: Live Trading Engine                       │
│   │   │  Docs: phase3.md                                    │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── model.py                    # LiveFootballQuantModel: state container
│   │   ├── tick_loop.py                # Wall-clock tick loop, halftime tracking, backpressure
│   │   ├── event_sources.py            # OddsApiLiveOddsSource, GoalserveLiveScoreSource
│   │   ├── event_handlers.py           # handle_confirmed_goal_v2, handle_red_card, EventQueue
│   │   ├── period_handler.py           # handle_period_change, halftime_accumulated tracking
│   │   ├── mu_computation.py           # compute_remaining_mu (analytic + MC paths)
│   │   ├── mc_core.py                  # Numba JIT MC simulator, mc_simulate_remaining
│   │   ├── mc_pricing.py               # step_3_4_async, aggregate_markets, compute_mc_stderr
│   │   ├── analytical_pricing.py       # Poisson-based analytical P_true (X=0, ΔS=0)
│   │   ├── resilient_ws.py             # ResilientWebSocket: exponential backoff reconnect
│   │   ├── emit.py                     # emit_to_phase4: asyncio.Queue push + Redis publish
│   │   └── run_engine.py               # run_engine(): asyncio.gather(tick, odds, score)
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Phase 4: Arbitrage & Execution                     │
│   │   │  Docs: phase4.md                                    │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_book_sync.py          # OrderBookSync: Kalshi WS, VWAP, bet365 update
│   │   ├── signal_generator.py         # Multi-market loop: dict→float decomposition
│   │   ├── edge_detection.py           # compute_signal_with_vwap, 2-pass EV, market alignment
│   │   ├── kelly.py                    # compute_kelly (incremental), risk_limits
│   │   ├── exit_logic.py               # 6 triggers: edge_decay, reversal, expiry, bet365, trim, opportunity_cost
│   │   ├── execution_router.py         # ExecutionRouter: paper/live mode switch
│   │   ├── live_executor.py            # execute_order: Kalshi API submission
│   │   ├── paper_executor.py           # PaperExecutionLayer: directional slippage, fill delay
│   │   ├── rapid_entry.py              # post_event_rapid_entry: VAR safety wait
│   │   ├── settlement.py               # await_settlement: Kalshi resolution polling
│   │   ├── post_analysis.py            # 12 post-match metrics, adaptive_parameter_update
│   │   ├── exit_monitor.py             # exit_monitor(): per-position exit evaluation loop
│   │   └── market_mapping.py           # MODEL_TO_KALSHI_TYPE, ticker_to_model_key
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Match Container Entry Point                        │
│   │   │  Docs: orchestration.md Component 3                 │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── match_engine/
│   │   ├── __init__.py
│   │   ├── main.py                     # Container entry: init → Phase 3+4 gather → settle
│   │   ├── config.py                   # MatchConfig.from_env(): load env vars + YAML
│   │   └── heartbeat.py                # heartbeat_emitter: Redis SET every 10s
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Orchestrator + Scheduler                           │
│   │   │  Docs: orchestration.md Components 1-2              │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── main.py                     # Orchestrator entry point
│   │   ├── scheduler.py                # MatchDiscovery: Goalserve fixtures + Kalshi markets
│   │   ├── trigger_executor.py         # TriggerExecutor: check triggers every 30s
│   │   ├── lifecycle.py                # MatchLifecycleManager: Phase 2 → container launch
│   │   ├── container_manager.py        # Docker container launch, monitor, cleanup
│   │   ├── concurrency.py              # ConcurrencyManager: MAX_CONCURRENT, exposure budget
│   │   ├── recovery.py                 # recover_orchestrator_state: resume from DB
│   │   └── emergency.py                # emergency_freeze, emergency_cleanup
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  Shared Modules (used across phases)                │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   ├── common/
│   │   ├── __init__.py
│   │   ├── db.py                       # asyncpg pool, safe_submit_order, get_bankroll
│   │   ├── redis_client.py             # Redis connection, pubsub helpers
│   │   ├── logging.py                  # structlog setup, bind(match_id, component)
│   │   ├── metrics.py                  # Prometheus counters, gauges, histograms
│   │   ├── alerts.py                   # send_alert: Slack + SMS + Redis publish
│   │   ├── config_loader.py            # Load system.yaml + env overrides
│   │   └── types.py                    # Shared dataclasses: Signal, Position, TickData, etc.
│   │
│   │   ┌─────────────────────────────────────────────────────┐
│   │   │  API Clients (external services)                    │
│   │   └─────────────────────────────────────────────────────┘
│   │
│   └── clients/
│       ├── __init__.py
│       ├── goalserve.py                # Goalserve REST client: fixtures, live score, stats
│       ├── odds_api.py                 # Odds-API: historical odds REST + live odds WebSocket
│       ├── kalshi.py                   # Kalshi: REST orders/positions + WebSocket order book
│       └── base_client.py              # Shared HTTP client (httpx), retry logic, rate tracking
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  DASHBOARD                                                  ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── dashboard/
│   │
│   ├── api/                            # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app, CORS, lifespan
│   │   ├── routes/
│   │   │   ├── matches.py              # /api/matches, /api/match/{id}, /api/match/{id}/ticks
│   │   │   ├── positions.py            # /api/positions
│   │   │   ├── analytics.py            # /api/analytics/pnl, /api/analytics/model-health
│   │   │   ├── system.py               # /api/system/status, /api/analytics/paper-graduation
│   │   │   └── websocket.py            # /ws/live: Redis pubsub → WebSocket push
│   │   ├── models.py                   # Pydantic response models
│   │   └── requirements.txt            # fastapi, uvicorn, asyncpg, redis
│   │
│   └── ui/                             # Next.js frontend
│       ├── package.json
│       ├── next.config.js
│       ├── tailwind.config.js
│       ├── src/
│       │   ├── app/
│       │   │   ├── layout.tsx          # Root layout, global providers
│       │   │   ├── page.tsx            # Command Center (multi-match overview)
│       │   │   ├── match/[id]/
│       │   │   │   └── page.tsx        # Match Deep Dive
│       │   │   ├── analytics/
│       │   │   │   └── page.tsx        # P&L Analytics
│       │   │   └── system/
│       │   │       └── page.tsx        # System Operations
│       │   ├── components/
│       │   │   ├── MatchCard.tsx        # Single match summary card (Command Center)
│       │   │   ├── PriceChart.tsx       # P_true vs P_kalshi time series (Recharts)
│       │   │   ├── OrderBookViz.tsx     # Bid/Ask depth visualization
│       │   │   ├── SignalLog.tsx        # Real-time signal table
│       │   │   ├── PositionTable.tsx    # Open positions with live P&L
│       │   │   ├── EventTimeline.tsx    # Goals, cards, cooldown annotations
│       │   │   ├── GraduationChecklist.tsx  # Phase 0 → A criteria
│       │   │   ├── StatusBar.tsx        # Top bar: bankroll, exposure, drawdown, mode
│       │   │   └── AlertBanner.tsx      # Critical alert overlay
│       │   ├── hooks/
│       │   │   ├── useWebSocket.ts      # WebSocket connection + subscription management
│       │   │   ├── useLiveTick.ts       # Subscribe to tick:{match_id}
│       │   │   └── useApi.ts            # REST API fetch hooks
│       │   └── lib/
│       │       ├── types.ts             # TypeScript types matching API models
│       │       └── format.ts            # Price/probability/P&L formatters
│       └── public/
│           └── favicon.ico
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  TESTS                                                      ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── tests/
│   │
│   ├── unit/                           # Tier 1: Pure function tests (<1s each)
│   │   ├── test_kelly.py               # compute_kelly: directional, incremental, edge cases
│   │   ├── test_ev.py                  # EV computation: Buy Yes/No, VWAP, 2-pass
│   │   ├── test_p_cons.py              # compute_conservative_P: direction invariant
│   │   ├── test_settlement.py          # compute_realized_pnl: all 4 validation cases
│   │   ├── test_exit_triggers.py       # Edge decay, reversal, expiry, bet365 divergence
│   │   ├── test_event_handlers.py      # Goal/red card/sub → state transitions
│   │   ├── test_cooldown.py            # Rapid sequential events, timer reset, queue drain
│   │   ├── test_halftime.py            # halftime_accumulated tracking, model.t correctness
│   │   ├── test_mc_stderr.py           # compute_mc_stderr: per-market values
│   │   ├── test_market_mapping.py      # ticker_to_model_key: known mappings
│   │   └── test_paper_slippage.py      # Directional slippage: BUY_YES +, BUY_NO −
│   │
│   ├── integration/                    # Tier 2: Pipeline tests (1-10s, mock APIs)
│   │   ├── test_phase2_to_3.py         # Phase 2 output → Phase 3 init → valid P_true
│   │   ├── test_paper_fill.py          # PaperExecutionLayer: delay, slippage, ob_freeze abort
│   │   ├── test_db_lifecycle.py        # Position: PENDING → OPEN → SETTLED
│   │   ├── test_reserve_release.py     # Exposure reservation: reserve → confirm/release
│   │   ├── test_signal_generator.py    # dict → float decomposition, multi-market loop
│   │   └── conftest.py                 # Shared fixtures: mock DB, mock order book, mock model
│   │
│   ├── property/                       # Tier 3: Hypothesis property-based tests
│   │   ├── test_p_true_valid.py        # P_true ∈ [0,1] for any valid state
│   │   ├── test_settlement_signs.py    # P&L sign consistency with direction + outcome
│   │   ├── test_p_cons_direction.py    # P_cons_yes ≤ P_true ≤ P_cons_no
│   │   └── test_goal_monotonicity.py   # Home goal → P(home_win) increases
│   │
│   └── fixtures/                       # Test data
│       ├── epl_match_2024.json         # Mock Goalserve fixture
│       ├── odds_api_historical.json    # Mock Odds-API response
│       ├── kalshi_orderbook.json       # Mock Kalshi order book
│       └── goalserve_live_score.json   # Mock live score polling response
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  INFRASTRUCTURE                                             ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── docker/
│   ├── match-engine/
│   │   ├── Dockerfile                  # Python 3.11 + Numba + asyncpg (Phase 3+4)
│   │   └── requirements.txt
│   ├── orchestrator/
│   │   ├── Dockerfile                  # Python 3.11 + Docker SDK (Scheduler + Orchestrator)
│   │   └── requirements.txt
│   ├── phase1-worker/
│   │   ├── Dockerfile                  # CUDA 12.2 + PyTorch + XGBoost (GPU)
│   │   └── requirements.txt
│   ├── dashboard-api/
│   │   ├── Dockerfile                  # Python 3.11 + FastAPI + uvicorn
│   │   └── requirements.txt
│   ├── dashboard-ui/
│   │   ├── Dockerfile                  # Node 20 + Next.js
│   │   └── .dockerignore
│   └── docker-compose.yml              # Full local dev stack (all services)
│
├── sql/
│   ├── schema.sql                      # From docs/schema.sql — run on fresh DB
│   └── seed.sql                        # Dev seed data (sample match, test positions)
│
├── monitoring/
│   ├── prometheus.yml                  # Prometheus scrape config
│   ├── alerts.yml                      # Alert rules (critical, warning, info)
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources.yaml        # PostgreSQL + Prometheus datasources
│       │   └── dashboards.yaml         # Auto-provision dashboards
│       └── dashboards/
│           ├── system_overview.json     # Grafana Dashboard 1
│           ├── live_match.json          # Grafana Dashboard 2
│           ├── latency.json            # Grafana Dashboard 3
│           ├── risk_exposure.json      # Grafana Dashboard 4
│           ├── model_health.json       # Grafana Dashboard 5
│           └── paper_validation.json   # Grafana Dashboard 6
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  CI/CD                                                      ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── .github/
│   └── workflows/
│       ├── test.yml                    # On push: unit + integration + property tests
│       ├── lint.yml                    # On push: mypy + ruff
│       └── backtest.yml                # Nightly: Step 3.6 full backtest (advisory)
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  SCRIPTS                                                    ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── scripts/
│   ├── setup_dev.sh                    # Local dev setup: docker-compose up, run migrations
│   ├── run_backtest.sh                 # Run Step 3.6 backtest with config
│   ├── run_phase1.sh                   # Trigger Phase 1 recalibration manually
│   ├── observation_mode.py             # Data-only collection (no trading) for API validation
│   └── reconcile_pending.py            # Find and resolve stale PENDING positions
│
│
│   ╔══════════════════════════════════════════════════════════════╗
│   ║  PROJECT CONFIG                                             ║
│   ╚══════════════════════════════════════════════════════════════╝
│
├── pyproject.toml                      # Python project config (ruff, mypy, pytest)
├── requirements.txt                    # Top-level dev dependencies
└── requirements-lock.txt               # Pinned versions for reproducibility
```

---

## File Count Summary

| Category | Files | Purpose |
|----------|-------|---------|
| Design docs | 8 | Architecture & specification |
| Config | 4 | Runtime configuration |
| Source — calibration | 8 | Phase 1 offline training |
| Source — prematch | 6 | Phase 2 pre-match init |
| Source — engine | 12 | Phase 3 live pricing |
| Source — execution | 14 | Phase 4 trading logic |
| Source — match_engine | 3 | Container entry point |
| Source — orchestrator | 8 | Scheduler + lifecycle |
| Source — common | 7 | Shared utilities |
| Source — clients | 4 | External API clients |
| Dashboard — API | 7 | FastAPI backend |
| Dashboard — UI | 16 | Next.js frontend |
| Tests — unit | 11 | Tier 1 pure function |
| Tests — integration | 6 | Tier 2 pipeline |
| Tests — property | 4 | Tier 3 hypothesis |
| Tests — fixtures | 4 | Mock data |
| Docker | 7 | Container definitions |
| Monitoring | 9 | Prometheus + Grafana |
| CI/CD | 3 | GitHub Actions |
| Scripts | 5 | Dev utilities |
| Project config | 4 | pyproject, requirements |
| **Total** | **~145** | |

---

## Module Dependency Graph

```
clients/
  goalserve.py ──────────────────────────────────┐
  odds_api.py ───────────────────────────────────┤
  kalshi.py ─────────────────────────────────────┤
                                                  │
common/                                           │
  db.py ─────────────────────────────────────────┤
  redis_client.py ───────────────────────────────┤
  logging.py ────────────────────────────────────┤
  metrics.py ────────────────────────────────────┤
  config_loader.py ──────────────────────────────┤
  types.py ──────────────────────────────────────┤
                                                  │
calibration/ (Phase 1)                            │
  step_1_1 → step_1_2 → step_1_3 → step_1_4    ├── uses common/ + clients/
  → step_1_5 → phase1_worker.py                  │
                                                  │
prematch/ (Phase 2)                               │
  step_2_1 → step_2_2 → step_2_3 → step_2_4    ├── uses common/ + clients/
  → step_2_5 → pipeline.py                       │   + calibration/ output
                                                  │
engine/ (Phase 3)                                 │
  model.py ← tick_loop.py                        │
           ← event_sources.py                    ├── uses common/ + clients/
           ← event_handlers.py                   │
           ← mc_core.py → mc_pricing.py          │
           ← emit.py (→ Phase 4 queue)           │
                                                  │
execution/ (Phase 4)                              │
  signal_generator.py                             │
    ← edge_detection.py                          ├── uses common/ + clients/
    ← kelly.py                                   │   + reads from engine/ queue
    ← execution_router.py                        │
       ← live_executor.py | paper_executor.py    │
    ← exit_logic.py                              │
    ← settlement.py                              │
    ← post_analysis.py                           │
                                                  │
match_engine/ (Container)                         │
  main.py ── imports engine/ + execution/        ├── entry point
           ── heartbeat.py                        │
                                                  │
orchestrator/                                     │
  main.py ── scheduler.py                         │
           ── trigger_executor.py                ├── launches match_engine/
           ── lifecycle.py                        │   via Docker
           ── container_manager.py                │
           ── recovery.py                         │
                                                  │
dashboard/                                        │
  api/ ── reads from PostgreSQL + Redis          ├── independent service
  ui/  ── reads from api/                         │
```

---

## Implementation Sprint Mapping

| Sprint | Files to Implement | Design Doc | Deliverable |
|--------|--------------------|------------|-------------|
| **S0** | `pyproject.toml`, `docker-compose.yml`, `sql/schema.sql`, `common/types.py`, `common/logging.py`, `config/system.yaml` | config_reference.md + schema.sql | Project scaffolding: dirs, deps, DB, shared types |
| **S1** | `clients/goalserve.py`, `clients/odds_api.py`, `calibration/step_1_1_intervals.py`, `common/types.py` | phase1.md Steps 1.1 | Parse real API data into IntervalRecords |
| **S2** | `calibration/step_1_2_*.py`, `step_1_3_*.py` | phase1.md Steps 1.2-1.3 | Q estimation + XGBoost prior |
| **S3** | `calibration/step_1_4_*.py`, `step_1_5_*.py` | phase1.md Steps 1.4-1.5 | NLL optimization + validation (Brier < baseline) |
| **S4** | `prematch/*.py`, `engine/model.py`, `engine/tick_loop.py`, `engine/mc_core.py` | phase2.md + phase3.md | Single match replay: P_true time series |
| **S5** | `execution/*.py`, `match_engine/main.py`, `clients/kalshi.py` | phase4.md | Paper signals on live Kalshi order book |
| **S6** | `orchestrator/*.py`, `docker/*`, `sql/schema.sql` | orchestration.md | Docker end-to-end: one paper match |
| **S7** | `dashboard/api/*`, `dashboard/ui/*`, `monitoring/*` | dashboard.md | Grafana + React dashboard |
| **S8** | `tests/**`, `.github/workflows/*` | orchestration.md (Testing) | Full test suite + CI |

---

## Key Design Decisions Embedded in Structure

**1. Phase 3 and Phase 4 are separate packages (`engine/` and `execution/`):**
Phase 3 (pricing) is mode-invariant. Phase 4 (execution) has paper/live branching.
Separation ensures no execution code leaks into pricing.

**2. `clients/` is isolated from business logic:**
API clients handle HTTP/WebSocket transport only. No pricing, no Kelly, no signals.
This makes it easy to mock for testing and swap providers.

**3. `common/types.py` defines shared dataclasses:**
`Signal`, `Position`, `TickData`, `NormalizedEvent`, `Phase2Result` etc.
Used by all phases — prevents circular imports.

**4. `match_engine/main.py` is the only container entry point:**
Everything a container needs is imported from `engine/` and `execution/`.
The orchestrator never imports from `match_engine/` — it launches it via Docker.

**5. Config is layered: YAML + env overrides:**
`system.yaml` has defaults. `system.paper.yaml` overlays Phase 0 settings.
Environment variables (from Docker) override everything. This means the same
Docker image works for both paper and live — only config differs.

**6. Dashboard is a separate service, not embedded:**
The trading dashboard reads from the same PostgreSQL/Redis as the match containers
but has no write access to positions or bankroll. Read-only by design.