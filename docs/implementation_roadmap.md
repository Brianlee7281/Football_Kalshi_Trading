# Implementation Roadmap

Reference document for Claude Code. Read the relevant sprint section before starting work.
Each sprint has: prerequisites, files to create (in order), per-file specs, tests, and done criteria.

---

## Sprint 0: Project Scaffolding

**Goal:** Empty project structure, dependencies installed, DB running.

### Tasks (in order)

1. Create directory structure from `docs/blueprint.md`
2. Create `pyproject.toml`:
   - Dependencies: numpy, numba, scipy, xgboost, torch, httpx, websockets, asyncpg, redis, structlog, prometheus-client, hypothesis, pytest, mypy, ruff
   - Python 3.11+
3. Create `requirements.txt` (pinned versions)
4. Copy `docs/schema.sql` → `sql/schema.sql`
5. Create `docker/docker-compose.yml` with: postgres, redis (services only, no app containers yet)
6. Create `Makefile`:
   ```
   up: docker-compose up -d postgres redis
   down: docker-compose down
   migrate: psql -f sql/schema.sql
   test: pytest tests/ -v
   lint: mypy src/ --strict && ruff check src/
   ```
7. Create `src/common/types.py` — shared dataclasses:
   ```python
   IntervalRecord, RedCardTransition, NormalizedEvent, Signal, Position,
   Phase2Result, TickData, MatchSchedule, MatchConfig, PaperFill, FillResult,
   MarketAlignment, ExitSignal, TradeLog
   ```
   Reference: field definitions scattered across phase1.md (IntervalRecord), phase3.md (NormalizedEvent), phase4.md (Signal, Position, TradeLog)
8. Create `src/common/logging.py` — structlog setup
9. Create `src/common/config_loader.py` — load `config/system.yaml` + env overrides
10. Create `config/system.yaml` from `docs/config_reference.md` YAML section

### Done Criteria

- [ ] `make up` starts postgres + redis
- [ ] `make migrate` creates all 12 tables
- [ ] `python -c "from src.common.types import Signal"` works
- [ ] `make lint` passes (empty src is fine)

---

## Sprint 1: API Clients + Step 1.1

**Goal:** Parse real Goalserve + Odds-API data into IntervalRecords.

**Read first:** `docs/phase1.md` — focus on Step 1.1 (lines ~70-400) and Input Data section.

### Task 1.1: Base HTTP Client

**File:** `src/clients/base_client.py`
**Depends on:** `src/common/logging.py` (S0)
- httpx.AsyncClient wrapper with retry logic (3 retries, exponential backoff)
- Rate limit tracking (requests/minute counter)
- Structured logging on request/response

### Task 1.2: Goalserve Client

**File:** `src/clients/goalserve.py`
**Depends on:** `src/clients/base_client.py` (Task 1.1), `src/common/types.py` (S0)
- Reference: `docs/phase1.md` Input Data section (Goalserve endpoints)
- Methods:
  - `get_fixtures(league_id, season)` → match list (used for date discovery)
  - `get_commentaries_by_league(league_id, date)` → minute-level events, red cards, lineups (primary data source for Phase 1)
  - `get_match_stats(match_id)` → detailed stats including player stats, xG
  - `get_live_score(match_id)` → current score, minute, status (for Phase 3)
  - Helper exports: `parse_minute`, `resolve_scoring_team`, `extract_goals`, `extract_red_cards`, `extract_stoppage_time`
- Parse Goalserve XML/JSON into typed dicts
- Test with real API call → save response as `tests/fixtures/goalserve_*.json`

### Task 1.3: Odds-API Client

**File:** `src/clients/odds_api.py`
**Depends on:** `src/clients/base_client.py` (Task 1.1), `src/common/types.py` (S0)
- Reference: `docs/phase1.md` Input Data section (Odds-API endpoints), `docs/api_reference_odds_api.md`
- Base URL: `https://api.odds-api.io/v3`
- Methods:
  - `get_events(sport, league, status)` → event list (pending/live/settled)
  - `get_odds(event_id, bookmakers)` → odds for single event
  - `get_odds_multi(event_ids, bookmakers)` → odds for up to 10 events (1 API call)
  - `connect_live_ws(markets, sport, status)` → async iterator (for Phase 3)
- Filter to 5 bookmakers: Bet365, Betfair Exchange, Sbobet, 1xBet, DraftKings
- **Historical data available from Dec 2025 onwards only.** Return NaN for unavailable matches.
- Test with real API call → save response as `tests/fixtures/odds_api_*.json`

### Task 1.4: Step 1.1 — Interval Segmentation

**File:** `src/calibration/step_1_1_intervals.py`
**Depends on:** `src/clients/goalserve.py` (Task 1.2), `src/common/types.py` (S0 — IntervalRecord, RedCardTransition)
- Reference: `docs/phase1.md` Step 1.1 (entire section, ~200 lines)
- Input: Goalserve match stats
- Output: `List[IntervalRecord]` per match
- Key logic:
  - Split 90-min match into intervals at event boundaries (goals, red cards)
  - Each interval: `(t_start, t_end, lambda_observed, score, red_cards, team)`
  - Handle stoppage time (intervals beyond 45/90 min)
  - `RedCardTransition` dataclass for state changes
- Validate: run against the synthetic fixture below and verify exact interval table

### Tests

Two layers: synthetic fixtures (from design docs, exact expected values) + real API fixtures (saved in Sprint 1 Tasks 1.2-1.3).

**File:** `tests/fixtures/synthetic_match_phase1_example.json`
- Create from `docs/phase1.md` Step 1.1 example (lines ~230-258): the 3-3 match with goals at 23', 36', 80', 81', 118'
- This JSON is hand-crafted to match the doc's Goalserve format exactly

**File:** `tests/unit/test_step_1_1.py`
- Test `build_intervals_from_goalserve` with synthetic fixture:
  - Assert: 6+ intervals (exact count from doc's table: intervals 1-6 + extra time)
  - Assert: interval 1 = `t_start=0, t_end=23, delta_S=0, X=0`
  - Assert: interval 2 = `t_start=23, t_end=36, delta_S=+1, scoring_team="localteam"`
  - Assert: interval 3 `t_end = 45 + 7` (stoppage: `alpha_1=7`)
  - Assert: halftime excluded (no interval with t_start inside HT)
  - Assert: interval 5 = `t_start=80, t_end=81, delta_S=+1` (away goal, ΔS drops)
  - Assert: `T_m = 90 + 7 + 8 = 105`
- Test 0-0 match (construct minimal JSON): exactly 2 intervals (one per half)
- Test own goal: `owngoal="True"` → `scoring_team` flipped
- Test VAR cancelled goal: `var_cancelled="True"` → no interval split, no ΔS change

**File:** `tests/unit/test_clients.py`
- Test: Goalserve response parsing (from `tests/fixtures/goalserve_*.json` — saved by Task 1.2)
- Test: Odds-API response parsing (from `tests/fixtures/odds_api_*.json` — saved by Task 1.3)
- Test: `build_odds_features` with Betfair Exchange present → `exchange_home_prob` populated
- Test: `build_odds_features` with Betfair Exchange missing → falls back to market average of remaining 4

### Done Criteria

- [ ] `python -m src.clients.goalserve` fetches real fixtures
- [ ] `python -m src.clients.odds_api` fetches real historical odds
- [ ] Step 1.1 produces IntervalRecords from Goalserve data
- [ ] Manually verified: 5 matches have correct interval counts
- [ ] All unit tests pass

---

## Sprint 2: Steps 1.2-1.3

**Goal:** Estimate Q matrix + train XGBoost prior for a_H, a_A.

**Read first:** `docs/phase1.md` Steps 1.2 and 1.3.

### Task 2.1: Step 1.2 — Q Estimation

**File:** `src/calibration/step_1_2_Q_estimation.py`
**Depends on:** `src/calibration/step_1_1_intervals.py` (Task 1.4), `src/common/types.py` (S0 — IntervalRecord)
- Reference: `docs/phase1.md` Step 1.2
- Input: `List[IntervalRecord]` from Step 1.1
- Output: Q matrix (4x4), Q_off_normalized
- Key logic:
  - Count state transitions (X=0→1, etc.) across all intervals
  - Apply shrinkage (league-stratified or pooled)
  - Normalize off-diagonal for MC simulation
- Validate: Q diagonal should be negative, rows sum to ~0

### Task 2.2: Step 1.3 — XGBoost Prior

**File:** `src/calibration/step_1_3_ml_prior.py`
**Depends on:** `src/calibration/step_1_1_intervals.py` (Task 1.4), `src/clients/odds_api.py` (Task 1.3), `src/common/types.py` (S0)
- Reference: `docs/phase1.md` Step 1.3
- Input: IntervalRecords + Odds-API historical odds
- Output: XGBoost model predicting a_H, a_A per match
- Key logic:
  - `build_odds_features()` — extract from Odds-API JSON (Betfair Exchange primary, market average fallback)
  - **Missing odds (pre-Dec 2025):** return NaN, not 0.0. XGBoost handles NaN natively as missing values.
  - 4-tier feature architecture (phase1.md Step 1.3)
  - Poisson regression objective in XGBoost
  - Feature importance via Poisson deviance
- Validate: predicted a values in reasonable range (0.0 - 2.0)

### Tests

**File:** `tests/unit/test_step_1_2.py`
- Test: Q matrix (4×4) — every row sums to 0.0 (within 1e-10)
- Test: Q diagonal entries are all negative
- Test: Q off-diagonal entries are all non-negative
- Test: Q_off_normalized — each row sums to 1.0 (transition probabilities)
- Test: with zero red card transitions → Q off-diagonal = 0 (no state changes)

**File:** `tests/unit/test_step_1_3.py`
- Test: `build_odds_features` with synthetic Odds-API response (from `docs/phase1.md` Step 1.3 JSON example):
  - Input: `"betfair_exchange"` bookmaker with `home=1.44, draw=3.50, away=12.00`
  - Assert: `exchange_home_prob ≈ 1/1.44 / (1/1.44 + 1/3.50 + 1/12.00) ≈ 0.684`
- Test: XGBoost predicts `a_H > 0` and `a_A > 0` for all matches
- Test: XGBoost predictions in range [0.0, 2.5] (no extreme values)
- Test: with all-zero features → predictions near league average (~0.6)

### Done Criteria

- [ ] Q matrix estimated from 1+ season of data
- [ ] XGBoost trained, feature importance plotted
- [ ] a_H, a_A predictions in [0, 2.0] range for all matches
- [ ] Saved Q matrix and XGBoost model to files

---

## Sprint 3: Steps 1.4-1.5

**Goal:** Joint NLL optimization + validation. This is the mathematical core.

**Read first:** `docs/phase1.md` Steps 1.4 and 1.5 carefully. The pseudocode is detailed.

### Task 3.1: Step 1.4 — NLL Optimization

**File:** `src/calibration/step_1_4_nll_optimize.py`
**Depends on:** `step_1_1_intervals.py` (Task 1.4), `step_1_2_Q_estimation.py` (Task 2.1), `step_1_3_ml_prior.py` (Task 2.2)
- Reference: `docs/phase1.md` Step 1.4 (entire section, ~200 lines)
- Input: IntervalRecords, XGBoost a_H/a_A prior, Q matrix
- Output: b[], gamma_H/A, delta_H/A, tau_H/A (all MMPP parameters)
- Key logic:
  - PyTorch optimizer (Adam)
  - log_tau parameterization for tau ∈ [0.1, 5.0]
  - σ_a regularization (grid-search: {0.1, 0.3, 0.5, 1.0})
  - NLL = Σ (λΔt - n·log(λ)) + regularization
- **Critical math:** verify gradient flows correctly through log_tau → tau → lambda
- Validate: NLL decreases monotonically during training

### Task 3.2: Step 1.5 — Validation

**File:** `src/calibration/step_1_5_validation.py`
**Depends on:** `step_1_4_nll_optimize.py` (Task 3.1), `src/clients/odds_api.py` (Task 1.3 — Betfair Exchange close line)
- Reference: `docs/phase1.md` Step 1.5
- Input: trained parameters, validation matches
- Output: Brier Score, league-stratified BS, sanity thresholds
- Key logic:
  - Time-series 5-fold CV (chronological, no leakage)
  - Brier Score vs Betfair Exchange close line
  - Calibrate go_threshold (90th pct), hold_threshold (99th pct)
  - Go/No-Go criteria: ΔBS < 0 (model beats Betfair Exchange)
- Validate: model BS < Betfair Exchange BS on validation set

### Task 3.3: Phase 1 Worker Entry Point

**File:** `src/calibration/phase1_worker.py`
**Depends on:** all Steps 1.1→1.5 (Tasks 1.4, 2.1, 2.2, 3.1, 3.2), `sql/schema.sql` (S0 — production_params table)
- Orchestrates Step 1.1 → 1.2 → 1.3 → 1.4 → 1.5
- Writes results to `production_params` DB table
- Handles Go/No-Go: only activate if validation passes

### Tests

**File:** `tests/unit/test_step_1_4.py`
- Test: NLL decreases monotonically over 100 optimization steps (assert `nll[i] <= nll[i-1] + 1e-6`)
- Test: `log_tau=0.0` → `tau = exp(0) = 1.0`; `log_tau=-2.3` → `tau ≈ 0.1` (clamped); `log_tau=1.6` → `tau ≈ 5.0` (clamped)
- Test: `σ_a → ∞` (e.g., 1e6) → `a_H, a_A ≈ XGBoost prediction` (regularization dominates)
- Test: `σ_a = 0` would make regularization infinite → handled gracefully (no NaN)
- Test: output `b` array has exactly 6 elements (6 basis periods × 15 min)
- Test: `gamma_H` has 4 elements (states X=0,1,2,3), `gamma_H[0] = 0.0` (reference state)

**File:** `tests/unit/test_step_1_5.py`
- Test: Brier Score known case: if `P_model = [0.7, 0.2, 0.1]` and outcome = home_win → `BS = (0.7-1)² + (0.2-0)² + (0.1-0)² = 0.14`
- Test: perfect prediction `P = [1,0,0]` outcome = home_win → `BS = 0.0`
- Test: `go_threshold < hold_threshold` always (90th < 99th percentile by definition)
- Test: with uniform predictions (`P = [0.33, 0.33, 0.33]`) → `BS ≈ 0.67` (baseline)

### Done Criteria

- [ ] Phase 1 runs end-to-end on 1 season of data
- [ ] ΔBS < 0 (model beats Betfair Exchange) on validation set
- [ ] Parameters saved to `production_params` table
- [ ] Sanity thresholds calibrated and saved
- [ ] GPU training completes in < 2 hours

---

## Sprint 4: Phase 2 + Phase 3

**Goal:** Initialize a single match and produce P_true time series by replaying historical data.

**Read first:** `docs/phase2.md` (all steps) and `docs/phase3.md` (Steps 3.1-3.4).

### Task 4.1: Phase 2 Pipeline

**Files:** `src/prematch/step_2_1_data_collection.py` through `pipeline.py`
**Depends on:** `src/clients/goalserve.py` (Task 1.2), `src/clients/odds_api.py` (Task 1.3), `src/common/types.py` (S0 — Phase2Result), `production_params` table output from Sprint 3
- Reference: `docs/phase2.md` Steps 2.1-2.5
- Pipeline: collect data → select features → backsolve a → sanity check → init
- Output: Phase2Result (a_H, a_A, C_time, verdict)
- Validate: a_H, a_A within 2σ of XGBoost prior

### Task 4.2: Live Model State

**File:** `src/engine/model.py`
**Depends on:** `src/common/types.py` (S0 — Phase2Result, NormalizedEvent, MatchConfig), `src/common/config_loader.py` (S0)
- `LiveFootballQuantModel` class: holds all state
- Fields: t, S, X, delta_S, mu_H, mu_A, engine_phase, event_state, cooldown, ob_freeze, etc.
- Initialize from Phase2Result + production_params
- `phase4_queue = asyncio.Queue(maxsize=1)`

### Task 4.3: MC Core

**File:** `src/engine/mc_core.py`
**Depends on:** `src/engine/model.py` (Task 4.2 — state fields used as MC input)
- Reference: `docs/phase3.md` Step 3.4 (Numba JIT section)
- `mc_simulate_remaining()` — Numba @njit compiled
- Input: current state (t, T, S, X, ΔS, params)
- Output: final_scores array (N_MC × 2)
- **Must pre-compile Numba cache** on first import

**File:** `src/engine/mc_pricing.py`
**Depends on:** `src/engine/mc_core.py` (above), `src/engine/model.py` (Task 4.2)
- `step_3_4_async()` — run MC in executor, stale check
- `aggregate_markets()` — final_scores → P_true dict
- `compute_mc_stderr()` — per-market σ_MC dict

### Task 4.4: Tick Loop

**File:** `src/engine/tick_loop.py`
**Depends on:** `src/engine/model.py` (Task 4.2), `src/engine/mc_pricing.py` (Task 4.3), `src/common/metrics.py` (S0)
- Reference: `docs/phase3.md` tick_loop code
- Wall-clock model.t with halftime exclusion
- Backpressure monitoring
- `emit_to_phase4()` → queue push

### Task 4.5: Event Sources + Handlers

**Files:** `src/engine/event_sources.py`, `event_handlers.py`, `period_handler.py`
**Depends on:** `src/engine/model.py` (Task 4.2), `src/clients/odds_api.py` (Task 1.3), `src/clients/goalserve.py` (Task 1.2), `src/common/types.py` (S0 — NormalizedEvent)
- Reference: `docs/phase3.md` Steps 3.1, 3.3
- NormalizedEvent abstraction
- Goal/red card/substitution handlers
- EventQueue for rapid sequential events
- Halftime tracking in period_handler

### Task 4.6: Historical Replay Script

**File:** `scripts/replay_match.py` (temporary dev tool)
**Depends on:** `src/prematch/pipeline.py` (Task 4.1), `src/engine/model.py` (Task 4.2), `src/engine/tick_loop.py` (Task 4.4), `src/engine/event_handlers.py` (Task 4.5), Sprint 3 output (`production_params`)
- Load a historical match from Goalserve stats
- Simulate event sequence at real timestamps
- Run Phase 2 → Phase 3 tick loop
- Plot P_true(t) for each market
- **This is the first end-to-end validation.**

### Tests

**File:** `tests/unit/test_model.py`
- Test: `LiveFootballQuantModel` init from Phase2Result → `model.t = 0.0`, `model.S = (0,0)`, `model.X = 0`
- Test: `phase4_queue` created with maxsize=1

**File:** `tests/unit/test_mc_pricing.py`
- Test: `aggregate_markets` with known final_scores `[[2,1],[1,0],[0,0],[3,2]]`:
  - `home_win = 3/4 = 0.75`, `draw = 1/4 = 0.25`, `away_win = 0/4 = 0.0`
  - `over_25 = 2/4 = 0.5` (only 2-1 and 3-2 have total > 2)
  - `btts_yes = 2/4 = 0.5` (2-1 and 3-2)
- Test: `compute_mc_stderr({"home_win": 0.5, "over_25": 0.9}, N=50000)`:
  - `home_win: sqrt(0.5*0.5/50000) ≈ 0.00224`
  - `over_25: sqrt(0.9*0.1/50000) ≈ 0.00134`
- Test: all P_true values ∈ [0, 1]
- Test: `home_win + draw + away_win ≈ 1.0` (within 0.01)

**File:** `tests/unit/test_halftime.py`
- Test: model.t before HT = 47.0 min (45 + 2 min stoppage)
- Test: HT entered → `halftime_start` set, HT lasts 15 min wall clock
- Test: second half start → `halftime_accumulated = 900s` (15 min)
- Test: model.t after HT resume = 47.0 min (not 62.0 — halftime excluded)
- Test: model.t at 80th minute play = 80.0 min (halftime fully subtracted)

**File:** `tests/unit/test_event_handlers.py`
- Test: home goal at S=(0,0) → S=(1,0), delta_S=+1, cooldown=True
- Test: away goal at S=(1,0) → S=(1,1), delta_S=0
- Test: red card for home → X incremented, mu_H decreases
- Test: VAR cancelled goal → S unchanged, event_state=IDLE, ob_freeze=False
- Test: rapid sequential (goal during cooldown):
  - S updates immediately (not blocked)
  - cooldown timer resets (new 15s)
  - EventQueue drains after confirmation

**File:** `tests/unit/test_emit.py`
- Test: `emit_to_phase4` puts dict on queue with keys `P_true`, `σ_MC`, `order_allowed`
- Test: queue full → old tick replaced (maxsize=1 behavior)

### Done Criteria

- [ ] `scripts/replay_match.py` produces P_true chart for 1 historical match
- [ ] P_true(home_win) increases after home goal
- [ ] P_true is frozen during HALFTIME
- [ ] MC σ_MC ≈ 0.002 at N_MC=50,000
- [ ] Numba warm-up completes in < 5s

---

## Sprint 5: Phase 4 (Paper Execution)

**Goal:** Generate trading signals from live Kalshi order book, paper-execute, track positions.

**Read first:** `docs/phase4.md` (all steps) and `docs/schema.sql` (positions, exposure tables).

### Task 5.1: Kalshi Client

**File:** `src/clients/kalshi.py`
**Depends on:** `src/clients/base_client.py` (Task 1.1), `src/common/types.py` (S0)
- REST: get_market(), submit_order(), cancel_order(), get_positions(), get_balance()
- WebSocket: order book stream (bid/ask levels + depth)
- Save sample responses to `tests/fixtures/kalshi_*.json`

### Task 5.2: Order Book Sync

**File:** `src/execution/order_book_sync.py`
**Depends on:** `src/clients/kalshi.py` (Task 5.1), `src/clients/odds_api.py` (Task 1.3 — live WS for bet365)
- Reference: `docs/phase4.md` Step 4.1
- `OrderBookSync` class: kalshi_best_bid/ask, depth arrays, VWAP computation
- `update_bet365()` from Odds-API live odds

### Task 5.3: Market Mapping

**File:** `src/execution/market_mapping.py`
**Depends on:** `src/common/types.py` (S0), `sql/schema.sql` (S0 — ticker_mapping table)
- `MODEL_TO_KALSHI_TYPE` dict
- `build_ticker_mapping(match_id, kalshi_tickers)` → DB insert

### Task 5.4: Edge Detection

**File:** `src/execution/edge_detection.py`
**Depends on:** `src/common/types.py` (S0 — Signal, MarketAlignment), `src/execution/order_book_sync.py` (Task 5.2)
- Reference: `docs/phase4.md` Step 4.2
- `compute_conservative_P()` — directional P_cons
- `compute_signal_with_vwap()` — 2-pass EV
- `check_market_alignment()` — bet365 alignment check
- `generate_signal()` — full signal generation

### Task 5.5: Kelly + Risk Limits

**File:** `src/execution/kelly.py`
**Depends on:** `src/common/types.py` (S0 — Signal), `src/execution/edge_detection.py` (Task 5.4 — Signal output)
- Reference: `docs/phase4.md` Step 4.3
- `compute_kelly()` — incremental Kelly with existing_exposure
- `apply_risk_limits()` — 3-layer caps (order/match/total)
- Liquidity gate

### Task 5.6: Execution Router + Paper Executor

**Files:** `src/execution/execution_router.py`, `paper_executor.py`, `live_executor.py`
**Depends on:** `src/execution/order_book_sync.py` (Task 5.2), `src/clients/kalshi.py` (Task 5.1), `src/engine/model.py` (Task 4.2 — ob_freeze, event_state checks in paper fill)
- Reference: `docs/phase4.md` Step 4.5
- ExecutionRouter: mode switch
- PaperExecutionLayer: directional slippage, fill delay, ob_freeze check

### Task 5.7: Exit Logic

**File:** `src/execution/exit_logic.py`
**Depends on:** `src/common/types.py` (S0 — Position, ExitSignal), `src/execution/edge_detection.py` (Task 5.4 — compute_conservative_P)
- Reference: `docs/phase4.md` Step 4.4
- 6 triggers: edge_decay, edge_reversal, expiry_eval, bet365_divergence, position_trim, opportunity_cost
- `evaluate_exit()` — per-position evaluation loop

### Task 5.8: Signal Generator

**File:** `src/execution/signal_generator.py`
**Depends on:** `src/execution/edge_detection.py` (Task 5.4), `src/execution/kelly.py` (Task 5.5), `src/execution/execution_router.py` (Task 5.6), `src/execution/exit_logic.py` (Task 5.7), `src/execution/order_book_sync.py` (Task 5.2), `src/execution/market_mapping.py` (Task 5.3), `src/engine/model.py` (Task 4.2 — phase4_queue)
- Reference: `docs/phase4.md` signal_generator section
- Consume from phase4_queue
- Per-ticker: dict→float decomposition → generate_signal → kelly → execute
- This is the multi-market orchestration loop

### Task 5.9: Settlement

**File:** `src/execution/settlement.py`
**Depends on:** `src/clients/kalshi.py` (Task 5.1 — get_market for resolution polling), `src/common/types.py` (S0 — Position)
- Reference: `docs/phase4.md` Step 4.6 (auto-settlement)
- `await_settlement()` — poll Kalshi for resolution
- `compute_realized_pnl()` — directional settlement
- `settle_all_positions()` — end-of-match settlement

### Task 5.10: Match Engine Entry Point

**File:** `src/match_engine/main.py`
**Depends on:** all `src/engine/` (Sprint 4), all `src/execution/` (Tasks 5.1-5.9), `src/common/config_loader.py` (S0), `src/common/logging.py` (S0)
- Reference: `docs/orchestration.md` Container Entry Point
- Init: load params, inject ExecutionRouter, connect WebSockets
- asyncio.gather: Phase 3 coroutines + Phase 4 coroutines + heartbeat

### Tests

All concrete values below come from `docs/phase4.md` validation examples.

**File:** `tests/unit/test_p_cons.py`
- Test: `compute_conservative_P(P_true=0.55, σ=0.01, direction="BUY_YES", z=1.645)` → `0.55 - 1.645*0.01 = 0.5336`
- Test: `compute_conservative_P(P_true=0.55, σ=0.01, direction="BUY_NO", z=1.645)` → `0.55 + 1.645*0.01 = 0.5664`
- Test: BUY_YES result always ≤ P_true; BUY_NO result always ≥ P_true

**File:** `tests/unit/test_kelly.py`
- Test: known inputs `P_cons=0.55, P_kalshi=0.50, c=0.07, direction=BUY_YES`:
  - `W = 0.93 * 0.50 = 0.465`, `L = 0.50`, `EV = 0.55*0.465 - 0.45*0.50 = 0.03075`
  - `f_kelly = 0.03075 / (0.465*0.50) = 0.1323`, `f_invest = 0.25 * 0.1323 * 0.8 = 0.0265`
- Test: incremental — existing_exposure=265 (= f_invest*10000), bankroll=10000 → `f_incremental = 0.0`
- Test: incremental — existing_exposure=100, bankroll=10000 → `f_incremental = 0.0265 - 0.01 = 0.0165`

**File:** `tests/unit/test_settlement.py` (from phase4.md v2 validation table)
- Test: Buy Yes, entry=0.45, settlement=1.00, qty=100, fee=0.07 → `pnl = 100*(1.00-0.45) - 0.07*100*0.55 = 51.15`
- Test: Buy Yes, entry=0.45, settlement=0.00, qty=100, fee=0.07 → `pnl = 100*(0.00-0.45) = -45.00` (no fee on loss)
- Test: Buy No, entry=0.40, settlement=0.00, qty=100, fee=0.07 → `pnl = 100*(0.40-0.00) - 0.07*100*0.40 = 37.20`
- Test: Buy No, entry=0.40, settlement=1.00, qty=100, fee=0.07 → `pnl = 100*(0.40-1.00) = -60.00`

**File:** `tests/unit/test_exit_triggers.py` (from phase4.md v2 fix validation)
- Test: edge_reversal Buy No — `P_cons=0.42, P_kalshi_bid=0.40, θ=0.02`:
  - v2: `0.42 > 0.40 + 0.02 = 0.42` → borderline (no trigger at exactly equal)
  - v2: `P_cons=0.43 > 0.42` → trigger fires
  - v1 (wrong): would need `P_cons > 0.62` → verify v2 logic rejects v1 threshold
- Test: bet365_divergence Buy No — `entry=0.40, P_bet365=0.46, threshold=0.05`:
  - v2: `0.46 > 0.40 + 0.05 = 0.45` → trigger fires
  - v1 (wrong): would need `0.46 > 0.65` → no trigger (broken)
- Test: expiry_eval Buy No — `entry=0.40, P_cons=0.35, c=0.07`:
  - `E_hold = 0.65*0.93*0.40 - 0.35*0.60 = 0.2418 - 0.21 = +0.0318` (hold is better)
- Test: position_trim (Trigger 5) — `K_frac=0.25, bankroll=10000, existing=500 (5%)`:
  - If f_optimal=0.018 (1.8%): `0.018 < 0.05 * 0.5 = 0.025` → trim fires, trim_qty > 0
  - If f_optimal=0.035 (3.5%): `0.035 > 0.025` → no trim
- Test: opportunity_cost (Trigger 6) — `current_EV=0.008 (0.8¢), opposite_EV=0.035 (3.5¢)`:
  - `opposite_EV > THETA_ENTRY (0.02)` AND `current_EV < 2 * THETA_EXIT (0.01)` → fires
  - `current_EV=0.015` → `0.015 > 0.01` → no trigger (current still strong enough)
- Test: Kalshi staleness — `kalshi_last_update = 6s ago` → `kalshi_is_stale = True` → execute_order returns None
- Test: bet365 staleness — `bet365_last_update = 35s ago` → `get_bet365_for_alignment` returns None → UNAVAILABLE multiplier

**File:** `tests/unit/test_paper_executor.py`
- Test: BUY_YES slippage → `fill_price = P_effective + 0.01` (price goes up = worse for buyer)
- Test: BUY_NO slippage → `fill_price = P_effective - 0.01` (price goes down = worse for seller)
- Test: ob_freeze set during fill delay → fill returns None (cancelled)
- Test: fill_delay between 1.0 and 3.0 seconds (check with mock asyncio.sleep)

**File:** `tests/unit/test_signal_generator.py`
- Test: input `P_true_dict = {"home_win": 0.55, "over_25": 0.65}`, two active tickers:
  - Assert: `generate_signal` called twice, once with 0.55 and once with 0.65
  - Assert: σ_MC_float differs per market (not same value for both)

### Done Criteria

- [ ] Signal generator produces signals from live Kalshi order book
- [ ] Paper fills are recorded in `positions` table with `is_paper=true`
- [ ] Exit logic triggers on simulated edge decay
- [ ] Settlement correctly computes P&L for both directions
- [ ] `match_engine/main.py` runs end-to-end for one live match (paper mode)

---

## Sprint 6: Orchestration

**Goal:** Docker containers, scheduler, lifecycle management, DB resilience.

**Read first:** `docs/orchestration.md` (all components) and `docs/schema.sql`.

### Task 6.1: DB Module

**File:** `src/common/db.py`
**Depends on:** `sql/schema.sql` (S0), `src/common/types.py` (S0 — Position, MatchSchedule), `src/common/logging.py` (S0)
- asyncpg connection pool (min=2, max=5)
- `safe_submit_order()` — 2-phase write pattern (PENDING → OPEN)
- Helper queries: get_bankroll, get_match_exposure, get_existing_exposure
- Reconciliation: find stale PENDING positions

### Task 6.2: Redis Module

**File:** `src/common/redis_client.py`
**Depends on:** `src/common/logging.py` (S0)
- Connection, pubsub helpers
- `publish_tick_to_dashboard()`, `publish_signal_to_dashboard()`
- Lock wrapper for exposure reservation

### Task 6.3: Exposure Reservation

**Integrate into:** `src/common/db.py` or new `src/common/exposure.py`
**Depends on:** `src/common/db.py` (Task 6.1), `src/common/redis_client.py` (Task 6.2 — lock wrapper)
- `reserve_exposure()`, `confirm_reservation()`, `release_reservation()`
- `execute_with_reservation()` — full reserve→execute→confirm/release flow

### Task 6.4: Scheduler

**File:** `src/orchestrator/scheduler.py`
**Depends on:** `src/clients/goalserve.py` (Task 1.2), `src/clients/kalshi.py` (Task 5.1), `src/common/db.py` (Task 6.1 — match_schedule table)
- `MatchDiscovery`: scan Goalserve fixtures, cross-check Kalshi markets
- Compute trigger times (phase2_trigger, phase3_trigger)
- Write to match_schedule table

### Task 6.5: Lifecycle Manager

**File:** `src/orchestrator/lifecycle.py`
**Depends on:** `src/prematch/pipeline.py` (Task 4.1), `src/common/db.py` (Task 6.1), `src/orchestrator/container_manager.py` (Task 6.6)
- `MatchLifecycleManager`: Phase 2 in-process → container launch → monitor
- Status transitions: SCHEDULED → PHASE2_RUNNING → PHASE2_DONE → PHASE3_RUNNING → SETTLING → FINISHED

### Task 6.6: Container Manager

**File:** `src/orchestrator/container_manager.py`
**Depends on:** `src/common/logging.py` (S0), `src/common/config_loader.py` (S0 — API keys, TRADING_MODE, PARAM_VERSION)
- Docker SDK: run, inspect, stop, remove
- Environment injection (TRADING_MODE, PARAM_VERSION, API keys)
- Log archival

### Task 6.7: Recovery

**File:** `src/orchestrator/recovery.py`
**Depends on:** `src/common/db.py` (Task 6.1), `src/orchestrator/container_manager.py` (Task 6.6)
- `recover_orchestrator_state()`: resume from DB on restart
- Handle PHASE2_RUNNING (re-run), PHASE3_RUNNING (check container), SCHEDULED (re-enqueue)

### Task 6.8: Dockerfiles

**Files:** `docker/match-engine/Dockerfile`, `docker/orchestrator/Dockerfile`
**Depends on:** `src/match_engine/main.py` (Task 5.10), `src/orchestrator/main.py` (Task 6.5), `requirements.txt` (S0)
- Match engine: Python 3.11 + Numba pre-compile
- Orchestrator: Python 3.11 + Docker SDK

### Task 6.9: Full docker-compose.yml

**File:** `docker/docker-compose.yml`
**Depends on:** all Dockerfiles (Task 6.8), `monitoring/prometheus.yml` (will be created in Sprint 7)
- All services: postgres, redis, orchestrator, prometheus, grafana, dashboard-api, dashboard-ui
- TRADING_MODE default to paper

### Tests

**File:** `tests/integration/test_db_lifecycle.py`
- Test: `safe_submit_order` → position created with status='PENDING' before order submission
- Test: fill success → status updated to 'OPEN'
- Test: fill failure → PENDING position deleted
- Test: DB connection failure during Phase A → order never submitted (return None)
- Test: stale PENDING detection: position with status='PENDING' older than 5 min → found by reconciliation query

**File:** `tests/integration/test_reserve_release.py`
- Test: bankroll=10000, F_TOTAL_CAP=0.20 → max total exposure = 2000
  - Reserve 1500 for match_A → success (reservation_id returned)
  - Reserve 1500 for match_B → capped at 500 (2000 - 1500 reserved)
  - Release match_A → now match_C can reserve 1500 again
- Test: confirm_reservation with partial fill → amount updated to actual
- Test: stale reservation older than 60s → cleaned up by CRON query

**File:** `tests/integration/test_recovery.py`
- Test: SCHEDULED match with phase2_trigger in the past → `recover_orchestrator_state` picks it up
- Test: PHASE3_RUNNING match with dead container → status set to FAILED

### Done Criteria

- [ ] `docker-compose up` starts entire stack
- [ ] Orchestrator discovers a match, runs Phase 2, launches container
- [ ] Container runs Phase 3+4 in paper mode for one live match
- [ ] Container exits cleanly after match + settlement
- [ ] Orchestrator marks match as FINISHED

---

## Sprint 7: Dashboard

**Goal:** Grafana operational dashboards + React trading dashboard + alerts.

**Read first:** `docs/dashboard.md` (architecture) AND `docs/dashboard_decomposition.md` (type contracts, component specs, formatting, edge cases, tests).

### Task 7.1: Pydantic API Models

**File:** `dashboard/api/models.py`
**Depends on:** `docs/dashboard_decomposition.md` Part 1.1 (exact field definitions)
- All 15 Pydantic models from decomposition doc: MatchSummary, MatchDetail, TickSnapshot, PositionItem, SignalItem, EventItem, PnLReport, ModelHealthReport, GraduationChecklist, SystemStatus, etc.
- Must match TypeScript types exactly (Part 1.2)

### Task 7.2: REST API Routes

**Files:** `dashboard/api/routes/matches.py`, `positions.py`, `analytics.py`, `system.py`
**Depends on:** Task 7.1, `src/common/db.py` (S6)
- All endpoints from `docs/dashboard.md` Part 3
- Return types must use Pydantic models from Task 7.1

### Task 7.3: WebSocket Route

**File:** `dashboard/api/routes/websocket.py`
**Depends on:** Task 7.1, `src/common/redis_client.py` (S6)
- Use FIXED redis_listener from `docs/dashboard_decomposition.md` Part 2.1 (no duplicate subscribe bug)
- Subscribe/unsubscribe delta management

### Task 7.4: TypeScript Types + Formatters

**Files:** `dashboard/ui/src/lib/types.ts`, `format.ts`, `format.test.ts`
**Depends on:** `docs/dashboard_decomposition.md` Part 1.2 (TypeScript interfaces) + Part 5 (formatting rules)
- All WSMessage types, MarketProbs, Score
- All 15 format functions with exact test cases from Part 6
- Use `sigma_MC` (not `σ_MC`) in JSON keys

### Task 7.5: React Hooks

**Files:** `dashboard/ui/src/hooks/useWebSocket.ts`, `useApi.ts`, `useLiveTick.ts`
**Depends on:** Task 7.4
- Reconnection with exponential backoff from Part 2.2 (1s→30s, max 10 retries)
- Expose: status, lastMessageAge
- useLiveTick: subscribe to match_id, return latest TickMessage

### Task 7.6: StatusBar + AlertBanner

**Files:** `dashboard/ui/src/components/StatusBar.tsx`, `AlertBanner.tsx`
**Depends on:** Task 7.5
- StatusBar: bankroll, exposure%, drawdown%, mode badge, WS status indicator
- AlertBanner: critical=sticky, warning=30s dismiss, info=10s dismiss
- Edge cases: WS disconnected → 🔴, bankroll null → "Loading..."

### Task 7.7: Command Center Page

**Files:** `dashboard/ui/src/app/page.tsx`, `components/MatchCard.tsx`
**Depends on:** Tasks 7.5 + 7.6
- MatchCard: teams, score, time, per-market edge, position count, click → deep dive
- UpcomingList: SCHEDULED matches with countdown
- Edge cases: 0 matches → "No active matches" message

### Task 7.8: Match Deep Dive Page

**Files:** `dashboard/ui/src/app/match/[id]/page.tsx` + `PriceChart.tsx`, `OrderBookViz.tsx`, `SignalLog.tsx`, `PositionTable.tsx`, `EventTimeline.tsx`
**Depends on:** Tasks 7.5 + 7.6
- PriceChart: Recharts, P_true/P_kalshi/P_bet365 lines, σ_MC band, event annotations
- OrderBookViz: bid/ask bars, spread, stale indicator
- SignalLog: newest first, BUY_YES green / BUY_NO red / HOLD gray
- PositionTable: unrealized P&L computed client-side (directional)
- EventTimeline: icons (⚽🟥🔄⏸), chronological
- Edge cases: 0 ticks → "Waiting for match to start", stale order book → "STALE" badge

### Task 7.9: P&L Analytics Page

**Files:** `dashboard/ui/src/app/analytics/page.tsx`, `components/GraduationChecklist.tsx`
**Depends on:** Task 7.5
- Filters: date range, league, market, direction, paper/live
- Breakdowns: by_league, by_market, by_direction, by_alignment
- GraduationChecklist: 8 criteria with pass/fail badges
- Edge cases: 0 trades → "No completed trades yet"

### Task 7.10: System Operations Page

**File:** `dashboard/ui/src/app/system/page.tsx`
**Depends on:** Task 7.5
- ContainerTable, ConnectionPanel, AlertHistory, ParamVersionInfo
- Edge cases: heartbeat > 60s → red "UNRESPONSIVE"

### Task 7.11: Prometheus Metrics + Grafana

**Files:** `src/common/metrics.py`, `monitoring/grafana/dashboards/*.json`
**Depends on:** `docs/orchestration.md` Component 6, running system from Sprint 6
- All Prometheus counters/gauges/histograms
- 6 Grafana dashboard JSONs from `docs/dashboard.md` Part 1

### Task 7.12: Alert Integration

**File:** `src/common/alerts.py`
**Depends on:** `src/common/redis_client.py` (S6), `src/common/config_loader.py` (S0 — SLACK_WEBHOOK)
- Slack webhook + SMS for critical
- Alert definitions from `docs/dashboard.md` Part 4

### Task 7.13: Dashboard Tests

**Files:** API tests, format tests, component tests
**Depends on:** Tasks 7.1-7.12
- All test assertions from `docs/dashboard_decomposition.md` Part 6
- API: field presence, sort order, edge cases (0 trades, 0 ticks)
- Format: 15 functions × exact expected values
- Components: empty states, color rules, WS disconnect handling

### Done Criteria

- [ ] Grafana shows System Overview with real data from a paper match
- [ ] Command Center displays running match with live P_true updates via WebSocket
- [ ] Match Deep Dive chart renders P_true/P_kalshi with event annotations
- [ ] Critical alert fires on simulated container crash (test via `docker stop`)
- [ ] All 6 Grafana dashboards provisioned and functional
- [ ] All formatting tests pass
- [ ] All empty state edge cases handled (no blank screens)

---

## Sprint 8: Tests + CI

**Goal:** Full test suite, CI pipeline, pre-deployment checklist passing.

**Read first:** `docs/orchestration.md` Testing Strategy section.

### Task 8.1: Fill Test Gaps

**Depends on:** all `src/engine/` (Sprint 4), all `src/execution/` (Sprint 5)

Review all `tests/unit/` — every function in `src/execution/` and `src/engine/` needs coverage.
Pay special attention to:
- Directional correctness (Buy Yes vs Buy No in every formula)
- Edge cases: EV=0, W*L=0, σ_MC=0, empty order book

### Task 8.2: Property Tests

**Files:** `tests/property/test_*.py`
**Depends on:** `src/execution/edge_detection.py` (Task 5.4), `src/execution/kelly.py` (Task 5.5), `src/execution/settlement.py` (Task 5.9), `src/engine/mc_pricing.py` (Task 4.3)
- P_true ∈ [0,1] for any valid state
- Settlement sign consistency
- P_cons direction invariant
- Goal → scoring team win prob increases

### Task 8.3: Integration Tests

**Files:** `tests/integration/test_*.py`
**Depends on:** all `src/` (Sprints 1-6), `src/common/db.py` (Task 6.1), `src/common/redis_client.py` (Task 6.2)
- Phase 2→3 pipeline end-to-end (mock APIs)
- Reserve-confirm-release concurrency
- Paper fill lifecycle
- Signal generator multi-market

### Task 8.4: CI Pipeline

**Files:** `.github/workflows/*.yml`
**Depends on:** `tests/` (Tasks 8.1-8.3), `pyproject.toml` (S0 — mypy, ruff config)
- test.yml: unit + integration + property (merge blocking)
- lint.yml: mypy + ruff (merge blocking)
- backtest.yml: Step 3.6 nightly (advisory)

### Task 8.5: Pre-Deployment Verification

**Depends on:** all tests (Tasks 8.1-8.4), `src/calibration/step_3_6_backtest.py` (Sprint 3 backtest), running paper system (Sprint 6)

Run full checklist from `docs/orchestration.md`:
1. All unit tests pass
2. All integration tests pass
3. All property tests pass (1000+ examples each)
4. Step 3.6 backtest: Go/No-Go = GO
5. Paper trading graduation: all 8 criteria met

### Done Criteria

- [ ] `make test` passes all tiers
- [ ] `make lint` passes (mypy strict + ruff)
- [ ] CI pipeline triggers on push and blocks on failure
- [ ] Coverage: all math functions in execution/ and engine/ have tests
- [ ] Pre-deployment checklist: all 5 items green

---

## Post-Sprint: Phase 0 → A Transition

After Sprint 8, the system runs in paper mode. Monitor for 2-4 weeks until:

```
[x] Paper trades ≥ 50
[x] Edge realization ∈ [0.6, 1.5]
[x] Brier Score within Phase 1.5 ± 0.03
[x] Max drawdown < 15%
[x] Directional correctness = 100%
[x] Paper realism score > 0.85
[x] No system crashes
[x] THETA_ENTRY calibrated
```

When all 8 pass → change `TRADING_MODE=live` in config → Phase A begins.