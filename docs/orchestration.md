# System Orchestration & Scheduling — Docker + Cloud

## Overview

This document defines how Phases 1-4 are unified into a single automated system.

The orchestrator manages the **lifecycle of each match** through Phases 2-4,
the scheduler determines **when to start each lifecycle**,
and the infrastructure layer provides **isolation, scaling, and fault tolerance**
via Docker containers on cloud infrastructure.

### Design Principles

1. **One container per match** — each live match gets its own isolated Phase 3+4 container, preventing cross-match interference
2. **Shared state via database** — bankroll, positions, risk limits live in PostgreSQL, not in-memory
3. **Fail-safe defaults** — if any component dies, positions are frozen (not liquidated), and no new orders are placed
4. **Idempotent operations** — every phase can be re-run safely without side effects
5. **Observable everything** — every state transition, parameter change, and order is logged with timestamps

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Cloud Infrastructure                         │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  Scheduler    │   │  Orchestrator │   │  Phase 1 Worker      │    │
│  │  (CRON +      │──>│  (Lifecycle   │   │  (Weekly recalib)    │    │
│  │   Match       │   │   Manager)   │   │  GPU-enabled         │    │
│  │   Discovery)  │   │              │   └──────────────────────┘    │
│  └──────────────┘   └──────┬───────┘                               │
│                            │                                        │
│              ┌─────────────┼─────────────┐                          │
│              │             │             │                           │
│              ▼             ▼             ▼                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Match        │ │ Match        │ │ Match        │                │
│  │ Container 1  │ │ Container 2  │ │ Container 3  │                │
│  │ (Ph 2→3→4)  │ │ (Ph 2→3→4)  │ │ (Ph 2→3→4)  │                │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Shared Services                           │   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐│   │
│  │  │ PostgreSQL │  │  Redis   │  │ Prometheus│  │ Grafana  ││   │
│  │  │ (state DB) │  │ (pubsub  │  │ (metrics) │  │ (dashbd) ││   │
│  │  │            │  │  + locks)│  │           │  │          ││   │
│  │  └────────────┘  └──────────┘  └───────────┘  └──────────┘│   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Scheduler

### Role

Discover upcoming tradable matches, compute trigger times, and enqueue
match lifecycles to the Orchestrator.

### Match Discovery

```python
class MatchDiscovery:
    """
    Runs every 6 hours. Scans Goalserve fixtures for upcoming matches
    in tradable leagues, cross-checks Kalshi for active markets.
    """

    TRADABLE_LEAGUES = {
        # Europe (Tier 1)
        1204: "EPL",
        1399: "La Liga",
        1229: "Bundesliga",
        1269: "Serie A",
        1221: "Ligue 1",
        # Americas (Tier 2)
        1005: "MLS",
        1572: "Brasileirão",     # Verify Goalserve league_id
        1300: "Liga Argentina",  # Verify Goalserve league_id
    }

    async def discover(self) -> List[MatchSchedule]:
        # 1. Fetch upcoming fixtures from Goalserve (next 48 hours)
        fixtures = await self.goalserve.get_upcoming_fixtures(
            league_ids=list(self.TRADABLE_LEAGUES.keys()),
            hours_ahead=48
        )

        # 2. Cross-check Kalshi for active markets
        kalshi_events = await self.kalshi.get_active_soccer_events()

        # 3. Match Goalserve fixtures to Kalshi events
        matched = self.match_fixtures_to_markets(fixtures, kalshi_events)

        # 4. Compute trigger times
        schedules = []
        for match in matched:
            kickoff = match["kickoff_utc"]
            schedules.append(MatchSchedule(
                match_id=match["goalserve_match_id"],
                league_id=match["league_id"],
                kickoff_utc=kickoff,
                phase2_trigger=kickoff - timedelta(minutes=65),  # lineup + buffer
                phase3_trigger=kickoff - timedelta(minutes=2),   # final check
                kalshi_tickers=match["kalshi_tickers"],
                odds_api_event_id=match["odds_api_event_id"],
            ))

        return schedules
```

### CRON Schedule

| Job | Schedule | Description |
|-----|----------|-------------|
| `match_discovery` | Every 6h (00:00, 06:00, 12:00, 18:00 UTC) | Scan upcoming 48h, enqueue new matches |
| `phase1_recalibration` | Weekly (Sunday 02:00 UTC) | Retrain MMPP parameters |
| `phase1_recalibration` | On trigger from Step 4.6 drift detection | Emergency retrain |
| `odds_backfill` | Daily 04:00 UTC | Fetch yesterday's settled events from Odds-API → `data/odds_cache/{league}.jsonl`. Available Dec 2025+ only. |
| `stats_backfill` | Daily 05:00 UTC | Backfill yesterday's Goalserve match stats |
| `health_check` | Every 5 min | Verify all services alive |
| `stale_reservation_cleanup` | Every 5 min | Release RESERVED exposure entries older than 60s |
| `bankroll_snapshot` | Every 1h | Record bankroll for drawdown tracking |

### Match Schedule Table

```sql
CREATE TABLE match_schedule (
    match_id        TEXT PRIMARY KEY,   -- Goalserve match ID
    league_id       INTEGER NOT NULL,
    kickoff_utc     TIMESTAMPTZ NOT NULL,
    phase2_trigger  TIMESTAMPTZ NOT NULL,
    phase3_trigger  TIMESTAMPTZ NOT NULL,
    kalshi_tickers  JSONB NOT NULL,  -- ["ticker1", "ticker2"]
    odds_api_event_id   TEXT,
    status          TEXT DEFAULT 'SCHEDULED',
        -- SCHEDULED -> PHASE2_RUNNING -> PHASE2_DONE
        -- -> PHASE3_RUNNING -> FINISHED | SKIPPED | FAILED
    container_id    TEXT,                -- Docker container ID
    trading_mode    TEXT DEFAULT 'paper', -- 'paper' or 'live'
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_match_schedule_trigger ON match_schedule(phase2_trigger)
    WHERE status = 'SCHEDULED';
```

### Trigger Executor

```python
class TriggerExecutor:
    """
    Runs every 30 seconds. Checks for matches whose trigger time has arrived.
    """

    async def tick(self):
        now = datetime.utcnow()

        # Phase 2 triggers
        ready_for_phase2 = await db.fetch_all("""
            SELECT * FROM match_schedule
            WHERE status = 'SCHEDULED'
              AND phase2_trigger <= $1
            ORDER BY kickoff_utc ASC
        """, now)

        for match in ready_for_phase2:
            await self.orchestrator.start_match_lifecycle(match)

        # Phase 3 triggers (Phase 2 completed, waiting for kickoff)
        ready_for_phase3 = await db.fetch_all("""
            SELECT * FROM match_schedule
            WHERE status = 'PHASE2_DONE'
              AND phase3_trigger <= $1
        """, now)

        for match in ready_for_phase3:
            await self.orchestrator.start_live_engine(match)
```

---

## Component 2: Orchestrator

### Role

Manage the full lifecycle of each match: spin up containers,
run Phase 2→3→4 in sequence, handle failures, and clean up.

### Match Lifecycle State Machine

```
SCHEDULED
    │
    ▼ (phase2_trigger time reached)
PHASE2_RUNNING
    │
    ├──> SKIPPED (sanity check = SKIP)
    │
    ▼ (sanity check = GO or GO_WITH_CAUTION)
PHASE2_DONE
    │
    ▼ (phase3_trigger time reached)
PHASE3_RUNNING  (container launched: Phase 3 + 4 running)
    │
    ├──> FAILED (container crash, unrecoverable error)
    │
    ▼ (match finished)
SETTLING  (awaiting Kalshi market resolution — may take minutes to hours)
    │
    ▼ (all positions settled via Kalshi resolution polling, P&L computed)
FINISHED
    │
    ▼ (container removed, logs archived)
ARCHIVED
```

### Lifecycle Manager

```python
class MatchLifecycleManager:
    """
    Manages one match through its entire lifecycle.
    Runs inside the Orchestrator process.
    """

    async def start_match_lifecycle(self, match: MatchSchedule):
        match_id = match.match_id
        await db.update_status(match_id, "PHASE2_RUNNING")

        try:
            # ═══ Phase 2: Pre-Match Initialization ═══
            phase2_result = await self.run_phase2(match)

            if phase2_result.verdict == "SKIP":
                await db.update_status(match_id, "SKIPPED")
                log.info(f"Match {match_id} SKIPPED: {phase2_result.warning}")
                return

            await db.update_status(match_id, "PHASE2_DONE")
            await db.store_phase2_params(match_id, phase2_result)

            # Wait for phase3_trigger time
            wait_seconds = (match.phase3_trigger - datetime.utcnow()).total_seconds()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            # ═══ Phase 3+4: Launch Match Container ═══
            container = await self.launch_match_container(match, phase2_result)
            await db.update_status(match_id, "PHASE3_RUNNING",
                                   container_id=container.id)

            # Monitor container until completion
            await self.monitor_container(match_id, container)

        except Exception as e:
            await db.update_status(match_id, "FAILED", error=str(e))
            log.error(f"Match {match_id} lifecycle failed: {e}")
            await self.emergency_cleanup(match_id)

    async def run_phase2(self, match: MatchSchedule) -> Phase2Result:
        """
        Run Phase 2 inside the orchestrator process (lightweight).
        No separate container needed — Phase 2 is fast (<10s).
        """
        from phase2.pipeline import Phase2Pipeline

        params = await db.load_production_params()
        pipeline = Phase2Pipeline(params, match)

        # Steps 2.1 → 2.2 → 2.3 → 2.4
        pre_match_data = await pipeline.collect_data()         # Step 2.1
        X_match = pipeline.select_features(pre_match_data)     # Step 2.2
        a_H, a_A, C_time = pipeline.backsolve_a(X_match)      # Step 2.3
        sanity = pipeline.sanity_check(a_H, a_A, pre_match_data)  # Step 2.4

        return Phase2Result(
            a_H=a_H, a_A=a_A, C_time=C_time,
            verdict=sanity.verdict,
            pre_match_data=pre_match_data,
            warning=sanity.warning,
        )

    async def launch_match_container(self, match, phase2_result) -> Container:
        """
        Launch a Docker container for Phase 3+4.
        One container per match for isolation.
        """
        container = await docker.run(
            image="soccer-live-engine:latest",
            name=f"match-{match.match_id}",
            environment={
                "MATCH_ID": match.match_id,
                "LEAGUE_ID": str(match.league_id),
                "TRADING_MODE": config.trading_mode,  # "paper" or "live"
                "PARAM_VERSION": str(await db.get_active_param_version()),  # pinned at launch
                "A_H": str(phase2_result.a_H),
                "A_A": str(phase2_result.a_A),
                "C_TIME": str(phase2_result.C_time),
                "KALSHI_TICKERS": json.dumps(match.kalshi_tickers),
                "ODDS_API_EVENT_ID": match.odds_api_event_id,
                "DB_URL": config.db_url,
                "REDIS_URL": config.redis_url,
                "ODDS_API_KEY": config.odds_api_key,
                "GOALSERVE_API_KEY": config.goalserve_api_key,
                "KALSHI_API_KEY": config.kalshi_api_key,
            },
            mem_limit="512m",
            cpu_quota=50000,  # 0.5 CPU
            restart_policy="no",  # do NOT auto-restart trading containers
            labels={
                "service": "match-engine",
                "match_id": match.match_id,
                "league": str(match.league_id),
            },
        )
        return container
```

### Concurrent Match Management

```python
class ConcurrencyManager:
    """
    Manages shared resources across concurrent matches.
    Uses Redis for distributed locking and PostgreSQL for state.
    """

    MAX_CONCURRENT_MATCHES = 8
    F_TOTAL_CAP = 0.20  # 20% total portfolio exposure

    async def can_start_match(self, match_id: str) -> bool:
        """Check if we can start another match container."""
        active = await db.count_active_matches()
        if active >= self.MAX_CONCURRENT_MATCHES:
            log.warning(f"Max concurrent matches ({self.MAX_CONCURRENT_MATCHES}) reached")
            return False
        return True

    async def get_available_exposure(self, match_id: str) -> float:
        """
        Compute remaining exposure budget for a new match.
        Called by Phase 4 risk limits (via DB query from match container).
        """
        bankroll = await db.get_current_bankroll()
        total_exposure = await db.get_total_exposure_all_matches()
        remaining = bankroll * self.F_TOTAL_CAP - total_exposure
        return max(0, remaining)
```

### Bankroll State (Shared via PostgreSQL)

```sql
CREATE TABLE bankroll (
    id              SERIAL PRIMARY KEY,
    mode            TEXT NOT NULL DEFAULT 'live',  -- 'paper' or 'live'
    balance         NUMERIC(12, 4) NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(mode)    -- one row per mode
);

-- Initialize both bankrolls
INSERT INTO bankroll (mode, balance) VALUES ('live', 0);      -- real deposit
INSERT INTO bankroll (mode, balance) VALUES ('paper', 10000); -- virtual $10,000

CREATE TABLE positions (
    id              SERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL,
    market_ticker   TEXT NOT NULL,
    direction       TEXT NOT NULL,      -- BUY_YES | BUY_NO
    entry_price     NUMERIC(6, 4) NOT NULL,
    quantity        INTEGER NOT NULL,
    status          TEXT DEFAULT 'OPEN', -- OPEN | CLOSED | SETTLED
    is_paper        BOOLEAN NOT NULL DEFAULT FALSE,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    exit_price      NUMERIC(6, 4),
    settlement_price NUMERIC(6, 4),
    realized_pnl    NUMERIC(12, 4),
    fill_delay      NUMERIC(6, 3),     -- paper only: simulated fill delay (seconds)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE exposure_cache (
    match_id        TEXT PRIMARY KEY,
    is_paper        BOOLEAN NOT NULL DEFAULT FALSE,
    total_exposure  NUMERIC(12, 4) NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Phase 4 queries this from inside the container.
-- Paper exposure and live exposure are tracked independently:
-- paper positions never affect live risk limits and vice versa.
CREATE OR REPLACE FUNCTION get_total_exposure(p_is_paper BOOLEAN)
RETURNS NUMERIC AS $$
    SELECT COALESCE(SUM(total_exposure), 0)
    FROM exposure_cache
    WHERE is_paper = p_is_paper
      AND match_id IN (
        SELECT match_id FROM match_schedule WHERE status = 'PHASE3_RUNNING'
    );
$$ LANGUAGE SQL;
```

> **Paper bankroll isolation:** paper P&L updates the `bankroll WHERE mode='paper'` row only.
> Live bankroll is never affected by paper trades. The virtual $10,000 paper bankroll
> is reset at the start of each Phase 0 evaluation period.

### Container Monitoring

```python
async def monitor_container(self, match_id: str, container: Container):
    """
    Watch container until it exits.
    Handle crashes, timeouts, and graceful completion.
    """
    MAX_MATCH_DURATION = timedelta(hours=9)  # match (~2h) + settlement polling (up to 6h) + buffer
    start_time = datetime.utcnow()

    while True:
        status = await docker.inspect(container.id)

        if status["State"]["Status"] == "exited":
            exit_code = status["State"]["ExitCode"]
            if exit_code == 0:
                # Container already polled Kalshi for settlement
                # and computed P&L before exiting (see Phase 4 settle_all_positions).
                # Positions in DB are already SETTLED with realized_pnl.
                await db.update_status(match_id, "SETTLING")
                await self.settle_match(match_id)  # post-match analytics
                await db.update_status(match_id, "FINISHED")
            else:
                log.error(f"Match {match_id} container exited with code {exit_code}")
                await db.update_status(match_id, "FAILED")
                await self.emergency_freeze(match_id)
            break

        # Safety timeout
        if datetime.utcnow() - start_time > MAX_MATCH_DURATION:
            log.error(f"Match {match_id} exceeded max duration — killing")
            await docker.stop(container.id)
            await db.update_status(match_id, "FAILED")
            await self.emergency_freeze(match_id)
            break

        # Health check: container alive but not responding
        last_heartbeat = await redis.get(f"heartbeat:{match_id}")
        if last_heartbeat:
            age = time.time() - float(last_heartbeat)
            if age > 60:  # no heartbeat for 60s
                log.error(f"Match {match_id} heartbeat stale ({age:.0f}s)")
                await self.emergency_freeze(match_id)

        await asyncio.sleep(10)

    # Cleanup
    await docker.remove(container.id)
    await self.archive_logs(match_id, container.id)
    await db.update_status(match_id, "ARCHIVED")
```

### Emergency Procedures

```python
async def emergency_freeze(self, match_id: str):
    """
    Freeze all activity for a match without closing positions.
    Positions remain open — manual intervention required.
    """
    # Cancel all working orders
    positions = await db.get_open_positions(match_id)
    for pos in positions:
        try:
            await kalshi_api.cancel_all_orders(pos.market_ticker)
        except Exception:
            pass  # best-effort

    # Publish freeze event (other containers may need to know)
    await redis.publish("match_events", json.dumps({
        "type": "EMERGENCY_FREEZE",
        "match_id": match_id,
        "timestamp": time.time(),
    }))

    log.critical(f"EMERGENCY FREEZE: match {match_id}")

async def emergency_cleanup(self, match_id: str):
    """
    Full cleanup after unrecoverable failure.
    """
    await self.emergency_freeze(match_id)

    # Kill container if still running
    schedule = await db.get_schedule(match_id)
    if schedule.container_id:
        try:
            await docker.stop(schedule.container_id)
            await docker.remove(schedule.container_id)
        except Exception:
            pass
```

---

## Component 3: Match Container (Phase 3 + 4)

### Container Entry Point

```python
# src/match_engine/main.py
"""
Entry point for match container.
Runs Phase 2.5 (system init) + Phase 3 (pricing) + Phase 4 (execution).
TRADING_MODE env var controls paper vs live execution.
"""

async def main():
    config = MatchConfig.from_env()

    # ─── Phase 2.5: System Initialization ───
    model = LiveFootballQuantModel(config)

    # Pin parameter version at startup — never reload mid-match.
    # PARAM_VERSION is set by orchestrator at container launch time.
    # New Phase 1 params take effect on the NEXT match, not the current one.
    model.param_version = config.param_version
    await model.initialize(param_version=config.param_version)

    # Inject ExecutionRouter based on TRADING_MODE
    # Phase 3 (pricing) is identical in both modes.
    # Only Phase 4 (execution) differs: paper simulates fills, live submits real orders.
    model.execution = ExecutionRouter(
        trading_mode=config.trading_mode,  # "paper" or "live"
        model=model,
    )
    model.is_paper = (config.trading_mode == "paper")

    # Load mode-appropriate bankroll
    model.bankroll = await db.get_bankroll(mode=config.trading_mode)

    # Phase 3→4 communication queue (maxsize=1: always latest tick)
    model.phase4_queue = asyncio.Queue(maxsize=1)

    # Multi-market setup: map Kalshi tickers to model keys
    model.active_tickers = config.kalshi_tickers  # ["SOCCER-EPL-ARS-v-CHE-WINNER", "SOCCER-EPL-ARS-v-CHE-OU2.5", ...]
    model.ticker_to_model_key = await db.get_ticker_mapping(config.match_id)
    # e.g., {"SOCCER-EPL-ARS-v-CHE-WINNER": "home_win", "SOCCER-EPL-ARS-v-CHE-OU2.5": "over_25"}

    # Per-ticker order book sync objects
    model.ob_syncs = {ticker: OrderBookSync() for ticker in model.active_tickers}

    # Numba warm-up (both delta paths)
    model.warmup_numba()

    # Final pre-kickoff check
    ready = await pre_kickoff_final_check(model)
    if not ready:
        sys.exit(0)  # clean exit, orchestrator sees exit code 0

    log.info(f"Match engine starting: mode={config.trading_mode}, "
             f"bankroll={model.bankroll}, "
             f"markets={list(model.ticker_to_model_key.keys())}")

    # ─── Phase 3 + 4: Live Engine ───
    # Phase 3 coroutines (pricing) — identical in paper and live
    # Phase 4 coroutines (execution) — routed via ExecutionRouter
    try:
        await asyncio.gather(
            # Phase 3: pricing engine (mode-invariant)
            tick_loop(model),
            live_odds_listener(model),
            live_score_poller(model),

            # Phase 4: execution engine (paper/live via ExecutionRouter)
            order_book_sync(model),       # LIVE order book for all tickers
            signal_generator(model),       # multi-market loop (see Phase 4)
            exit_monitor(model),

            # Health
            heartbeat_emitter(model),
        )
    except Exception as e:
        log.error(f"Match engine crashed: {e}")
        sys.exit(1)  # non-zero exit, orchestrator handles

    # ─── Post-Match ───
    await settle_all_positions(model)
    await compute_post_match_analytics(model)
    await emit_final_heartbeat(model)
    sys.exit(0)


async def heartbeat_emitter(model):
    """Emit heartbeat to Redis every 10 seconds."""
    while model.engine_phase != "FINISHED":
        await redis.set(
            f"heartbeat:{model.match_id}",
            str(time.time()),
            ex=120  # expire after 2 min
        )
        await asyncio.sleep(10)
```

### Container-Orchestrator Communication

| Channel | Direction | Protocol | Purpose |
|---------|-----------|----------|---------|
| PostgreSQL | Container → DB | SQL | Position writes, bankroll reads, exposure queries |
| Redis heartbeat | Container → Orchestrator | SET/GET | Liveness monitoring |
| Redis pubsub | Bidirectional | PUBLISH/SUBSCRIBE | Emergency events, lineup changes |
| Container logs | Container → Orchestrator | Docker log driver | Structured logging for archival |

### Database Connection Resilience

PostgreSQL connections can drop due to network glitches, DB failover, or connection timeouts.
Without resilience, the worst case is: order submitted to Kalshi and filled,
but position not recorded in DB — creating an untracked live position.

**Connection Pool:**

```python
# Container initialization — use asyncpg pool, not single connection
db_pool = await asyncpg.create_pool(
    dsn=config.db_url,
    min_size=2,
    max_size=5,
    command_timeout=10,
    max_inactive_connection_lifetime=300,  # recycle idle connections
)
```

**2-Phase Write Pattern for Order Submission:**

```python
async def safe_submit_order(signal, amount, ob_sync, model):
    """
    Prevent ghost positions: record intent BEFORE submitting to Kalshi.

    Phase A: Write PENDING position to DB (with retry)
    Phase B: Submit order to Kalshi
    Phase C: Update position to OPEN (or cancel on failure)

    If DB is down at Phase A → order never submitted → safe
    If DB is down at Phase C → PENDING position exists → reconciliation catches it
    """
    # Phase A: record intent
    try:
        position_id = await db_pool.fetchval("""
            INSERT INTO positions (match_id, market_ticker, direction,
                                   entry_price, quantity, status, is_paper, entry_time)
            VALUES ($1, $2, $3, $4, $5, 'PENDING', $6, NOW())
            RETURNING id
        """, model.match_id, signal.market_ticker, signal.direction,
             signal.P_kalshi, int(amount / signal.P_kalshi), model.is_paper)
    except Exception as e:
        log.error(f"DB write failed — blocking order submission: {e}")
        return None  # safe: no order submitted

    # Phase B: submit to Kalshi (or paper simulate)
    fill = await model.execution.submit_order(signal, amount, ob_sync)

    # Phase C: update status
    try:
        if fill and fill.quantity > 0:
            await db_pool.execute("""
                UPDATE positions SET status = 'OPEN', quantity = $2,
                       entry_price = $3 WHERE id = $1
            """, position_id, fill.quantity, fill.price)
        else:
            await db_pool.execute(
                "DELETE FROM positions WHERE id = $1", position_id
            )
    except Exception as e:
        log.error(f"DB update failed for position {position_id} — "
                  f"PENDING position may need manual reconciliation: {e}")

    return fill
```

**Reconciliation:** on container startup, check for stale `PENDING` positions
(older than 5 minutes) and alert for manual review.

### Risk Limit Enforcement (Cross-Container)

> **Problem:** if a container holds a Redis lock during order submission (1-5s fill wait),
> other containers block. If the lock is released before fill, two containers can
> both see "15% remaining" and each allocate 15%, totaling 30%.
>
> **Solution: Reserve → Execute → Confirm/Release pattern.**
> Lock is held for <10ms (DB read + write only). Order execution happens outside the lock.

```sql
CREATE TABLE exposure_reservation (
    id          SERIAL PRIMARY KEY,
    match_id    TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    amount      NUMERIC(12, 4) NOT NULL,
    is_paper    BOOLEAN NOT NULL DEFAULT FALSE,
    status      TEXT DEFAULT 'RESERVED',  -- RESERVED → CONFIRMED | RELEASED
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- get_total_exposure now includes RESERVED amounts (pessimistic)
CREATE OR REPLACE FUNCTION get_total_exposure(p_is_paper BOOLEAN)
RETURNS NUMERIC AS $$
    SELECT COALESCE(SUM(e.total_exposure), 0)
           + COALESCE((
               SELECT SUM(amount) FROM exposure_reservation
               WHERE is_paper = p_is_paper AND status = 'RESERVED'
           ), 0)
    FROM exposure_cache e
    WHERE e.is_paper = p_is_paper
      AND e.match_id IN (
        SELECT match_id FROM match_schedule WHERE status = 'PHASE3_RUNNING'
    );
$$ LANGUAGE SQL;
```

```python
async def reserve_exposure(
    match_id: str,
    market_ticker: str,
    f_invest: float,
    bankroll: float,
    is_paper: bool,
) -> Optional[int]:
    """
    Phase 1: Reserve exposure under lock (<10ms).
    Returns reservation_id if successful, None if limit exceeded.
    """
    async with redis.lock("exposure_lock", timeout=2):
        # Layer 1: single order cap
        amount = min(f_invest * bankroll, bankroll * F_ORDER_CAP)

        # Layer 2: per match cap
        match_exposure = await db.get_match_exposure(match_id)
        remaining_match = bankroll * F_MATCH_CAP - match_exposure
        amount = min(amount, max(0, remaining_match))

        # Layer 3: total portfolio cap (includes existing RESERVED amounts)
        total_exposure = await db.execute_scalar(
            "SELECT get_total_exposure($1)", is_paper
        )
        remaining_total = bankroll * F_TOTAL_CAP - total_exposure
        amount = min(amount, max(0, remaining_total))

        if amount <= 0:
            return None

        # Reserve (visible to other containers immediately)
        reservation_id = await db.execute_scalar("""
            INSERT INTO exposure_reservation
                (match_id, market_ticker, amount, is_paper, status)
            VALUES ($1, $2, $3, $4, 'RESERVED')
            RETURNING id
        """, match_id, market_ticker, amount, is_paper)

        return reservation_id
    # Lock released here — other containers can now reserve concurrently


async def confirm_reservation(reservation_id: int, actual_amount: float):
    """
    Phase 3a: Order filled — confirm the reservation with actual fill amount.
    Release excess if partial fill.
    """
    await db.execute("""
        UPDATE exposure_reservation
        SET status = 'CONFIRMED', amount = $2
        WHERE id = $1
    """, reservation_id, actual_amount)


async def release_reservation(reservation_id: int):
    """
    Phase 3b: Order failed or cancelled — release the reservation.
    """
    await db.execute("""
        UPDATE exposure_reservation SET status = 'RELEASED' WHERE id = $1
    """, reservation_id)


async def execute_with_reservation(signal, amount, ob_sync, model) -> Optional:
    """
    Full reserve → execute → confirm/release flow.
    """
    # Phase 1: Reserve (<10ms lock)
    reservation_id = await reserve_exposure(
        model.match_id, signal.market_ticker,
        amount / model.bankroll, model.bankroll, model.is_paper
    )
    if reservation_id is None:
        return None  # limit exceeded

    # Phase 2: Execute order (no lock held, 1-5s)
    try:
        fill = await model.execution.submit_order(signal, amount, ob_sync)
    except Exception as e:
        await release_reservation(reservation_id)
        raise

    # Phase 3: Confirm or release
    if fill and fill.quantity > 0:
        actual_amount = fill.price * fill.quantity
        await confirm_reservation(reservation_id, actual_amount)
    else:
        await release_reservation(reservation_id)

    return fill
```

> **Stale reservation cleanup:** a CRON job runs every 5 minutes and releases
> any `RESERVED` entries older than 60 seconds (order should have filled or failed by then).
> This prevents leaked reservations from permanently reducing available exposure.

---

## Component 4: Phase 1 Worker

### Role

Periodic recalibration of MMPP parameters.
Runs as a separate, GPU-enabled container on a scheduled basis.

### Execution

```python
class Phase1Worker:
    """
    Heavy computation: XGBoost training + PyTorch NLL optimization.
    Runs weekly or on-demand (triggered by Step 4.6 drift detection).
    """

    async def run(self, trigger: str = "weekly"):
        log.info(f"Phase 1 recalibration started (trigger: {trigger})")

        # Step 1.1: Interval segmentation (parallel across leagues)
        intervals = await self.step_1_1_parallel()

        # Step 1.2: Q estimation
        Q_matrices = self.step_1_2(intervals)

        # Step 1.3: ML prior (XGBoost)
        xgb_model, feature_mask = self.step_1_3(intervals)

        # Step 1.4: Joint NLL optimization (PyTorch, GPU)
        params = self.step_1_4(intervals, xgb_model)

        # Step 1.5: Validation
        validation = self.step_1_5(params)

        if validation.go_no_go == "GO":
            # Atomic swap: write new params, update version
            version = await db.store_production_params(params, validation)
            log.info(f"Phase 1 complete: new param version {version}")

            # Notify system of new params.
            # Running match containers do NOT reload mid-match (version pinned at startup).
            # New version applies to the next match launched by orchestrator.
            await redis.publish("system_events", json.dumps({
                "type": "PARAMS_UPDATED",
                "version": version,
                "timestamp": time.time(),
            }))
        else:
            log.warning(f"Phase 1 validation FAILED: {validation.failures}")
            await self.alert("Phase 1 recalibration failed validation",
                           details=validation.failures)
```

### Parameter Versioning

```sql
CREATE TABLE production_params (
    version         SERIAL PRIMARY KEY,
    params          JSONB NOT NULL,     -- b, gamma, delta, Q, etc.
    xgb_model_path  TEXT NOT NULL,      -- S3 path to .xgb file
    feature_mask    JSONB NOT NULL,
    validation      JSONB NOT NULL,     -- Step 1.5 results
    sanity_thresholds JSONB NOT NULL,   -- go/hold/ou thresholds
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT FALSE
);

-- Only one version is active at a time
CREATE UNIQUE INDEX idx_active_params ON production_params(is_active)
    WHERE is_active = TRUE;
```

---

## Component 5: Docker Images

### Image Definitions

```
docker/
├── match-engine/
│   ├── Dockerfile          # Phase 3+4 runtime
│   └── requirements.txt
├── orchestrator/
│   ├── Dockerfile          # Scheduler + Orchestrator
│   └── requirements.txt
├── phase1-worker/
│   ├── Dockerfile          # Phase 1 recalibration (GPU)
│   └── requirements.txt
└── docker-compose.yml      # Local development
```

### Match Engine Dockerfile

```dockerfile
# docker/match-engine/Dockerfile
FROM python:3.11-slim

# Numba needs numpy at build time
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    numba==0.59.1 \
    scipy==1.13.0 \
    xgboost==2.0.3 \
    httpx==0.27.0 \
    websockets==12.0 \
    asyncpg==0.29.0 \
    redis==5.0.3 \
    prometheus-client==0.20.0

COPY src/ /app/src/
WORKDIR /app

# Pre-compile Numba cache
RUN python -c "from src.phase3.mc_core import mc_simulate_remaining; print('Numba AOT done')"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.match_engine.main"]
```

### Phase 1 Worker Dockerfile

```dockerfile
# docker/phase1-worker/Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    xgboost==2.0.3 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.0 \
    asyncpg==0.29.0

COPY src/ /app/src/
WORKDIR /app

ENTRYPOINT ["python", "-m", "src.calibration.phase1_worker"]
```

### Docker Compose (Development)

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: soccer_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  orchestrator:
    build: ./orchestrator
    depends_on: [postgres, redis]
    environment:
      DB_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/soccer_trading
      REDIS_URL: redis://redis:6379
      TRADING_MODE: ${TRADING_MODE:-paper}  # default to paper for safety
      ODDS_API_KEY: ${ODDS_API_KEY}
      GOALSERVE_API_KEY: ${GOALSERVE_API_KEY}
      KALSHI_API_KEY: ${KALSHI_API_KEY}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # Docker-in-Docker
    restart: always

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    depends_on: [prometheus]
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards

volumes:
  pgdata:
```

---

## Component 6: Monitoring & Alerting

### Metrics (Prometheus)

```python
from prometheus_client import Counter, Gauge, Histogram

# Match lifecycle
matches_started = Counter("matches_started_total", "Total matches started", ["league"])
matches_completed = Counter("matches_completed_total", "Completed matches", ["status"])
active_containers = Gauge("active_match_containers", "Currently running match containers")

# Trading
orders_submitted = Counter("orders_submitted_total", "Orders submitted", ["direction", "match_id"])
orders_filled = Counter("orders_filled_total", "Orders filled", ["direction"])
position_pnl = Histogram("position_pnl", "Realized P&L per position", buckets=[-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5])

# Risk
total_exposure = Gauge("total_exposure_ratio", "Total portfolio exposure / bankroll")
bankroll_balance = Gauge("bankroll_balance", "Current bankroll")
max_drawdown = Gauge("max_drawdown_pct", "Current max drawdown percentage")

# Latency
odds_api_latency = Histogram("odds_api_ws_latency_seconds", "Odds-API WS message latency")
live_score_latency = Histogram("live_score_poll_latency_seconds", "Goalserve REST poll latency")
mc_compute_latency = Histogram("mc_compute_latency_seconds", "MC pricing latency")

# Health
heartbeat_age = Gauge("heartbeat_age_seconds", "Age of last heartbeat", ["match_id"])
```

### Alerting Rules

```yaml
# monitoring/alerts.yml
groups:
  - name: trading_alerts
    rules:
      # Critical: container died
      - alert: MatchContainerDead
        expr: active_match_containers < count(match_schedule_phase3_running)
        for: 30s
        labels: { severity: critical }

      # Critical: bankroll drawdown
      - alert: DrawdownExceeded
        expr: max_drawdown_pct > 15
        for: 1m
        labels: { severity: critical }
        annotations:
          summary: "Max drawdown {{ $value }}% exceeds 15% threshold"

      # Warning: exposure approaching limit
      - alert: ExposureHigh
        expr: total_exposure_ratio > 0.15
        for: 5m
        labels: { severity: warning }

      # Warning: heartbeat stale
      - alert: HeartbeatStale
        expr: heartbeat_age_seconds > 60
        for: 30s
        labels: { severity: warning }

      # Warning: MC latency spike
      - alert: MCLatencyHigh
        expr: histogram_quantile(0.99, mc_compute_latency_seconds) > 0.01
        for: 5m
        labels: { severity: warning }
```

### Grafana Dashboards

> **Full dashboard design** including panel specifications, SQL queries, alert rules,
> custom React trading dashboard, and API server is in the dedicated
> **Dashboard & Monitoring Design** document (`dashboard.md`).

| Dashboard | Panels |
|-----------|--------|
| **System Overview** | Active containers, bankroll balance, total exposure, max drawdown, match pipeline status |
| **Live Match Detail** | Per-match P_true vs P_kalshi timeline, order fills, position P&L, event timeline |
| **Latency & Performance** | Tick duration, MC compute time, Odds-API WS latency, heartbeat age |
| **Risk & Exposure** | Exposure by match/league, drawdown curve, active reservations, position heat map |
| **Phase 1 Model Health** | Brier Score trend, edge realization, direction realization, alignment value |
| **Paper Trading Validation** | Paper P&L, realism score, graduation checklist, fill delay distribution |

---

## Component 7: Logging & Audit Trail

### Structured Logging

```python
import structlog

log = structlog.get_logger()

# Every log entry includes these fields automatically
log = log.bind(
    match_id=config.match_id,
    component="match_engine",
    version=config.param_version,
)

# Example log entries
log.info("tick", t=model.t, mu_H=mu_H, mu_A=mu_A, pricing_mode="MC")
log.info("order_submitted", direction="BUY_YES", qty=30, price=0.45, EV=0.031)
log.warning("ob_freeze", source="odds_api", delta=0.12)
log.error("ws_disconnected", source="odds_api", reconnect_attempt=3)
```

### Audit Tables

```sql
-- Every state transition
CREATE TABLE event_log (
    id              BIGSERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,      -- goal_confirmed, red_card, ob_freeze, etc.
    source          TEXT NOT NULL,      -- odds_api, live_score, system
    payload         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Every tick snapshot (sampled: every 10s during normal play, every 1s during events)
CREATE TABLE tick_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL,
    t               NUMERIC(6, 2) NOT NULL,
    mu_H            NUMERIC(8, 4),
    mu_A            NUMERIC(8, 4),
    P_true          JSONB,              -- {home_win: 0.55, draw: 0.25, ...}
    P_kalshi    JSONB,              -- {home_win_ask: 0.52, ...}
    P_bet365        JSONB,
    sigma_MC        JSONB,                           -- {"home_win": 0.0022, ...} (per-market since v3)
    engine_phase    TEXT,
    event_state     TEXT,
    cooldown        BOOLEAN,
    ob_freeze       BOOLEAN,
    order_allowed   BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Adaptive parameter changes
CREATE TABLE parameter_change_log (
    id              SERIAL PRIMARY KEY,
    parameter_name  TEXT NOT NULL,
    old_value       NUMERIC(10, 6),
    new_value       NUMERIC(10, 6),
    trigger_metric  TEXT,               -- edge_realization, brier_trend, etc.
    trigger_value   NUMERIC(10, 6),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Partition tick_snapshots by date for query performance
CREATE INDEX idx_tick_match_time ON tick_snapshots(match_id, t);
```

---

## Testing Strategy

Step 3.6 backtest validates model accuracy, but does not verify code correctness.
A delta sign error or Kelly formula typo can produce plausible-looking backtest metrics
while silently losing money in production.

### Tier 1: Unit Tests (< 1 second each, no external dependencies)

Pure function correctness with known inputs and expected outputs.

```python
# test_kelly.py — Example unit tests

def test_kelly_buy_yes_known_edge():
    """Known edge should produce known fraction."""
    signal = Signal(direction="BUY_YES", EV=0.03, P_cons=0.55,
                    P_kalshi=0.50, kelly_multiplier=0.8)
    f = compute_kelly(signal, c=0.07, K_frac=0.25, existing_exposure=0, bankroll=10000)
    assert 0.01 < f < 0.10  # reasonable range

def test_kelly_buy_no_symmetric():
    """Buy No with mirrored inputs should produce comparable sizing."""
    sig_yes = Signal(direction="BUY_YES", EV=0.03, P_cons=0.55, P_kalshi=0.50, kelly_multiplier=0.8)
    sig_no = Signal(direction="BUY_NO", EV=0.03, P_cons=0.45, P_kalshi=0.50, kelly_multiplier=0.8)
    f_yes = compute_kelly(sig_yes, c=0.07, K_frac=0.25, existing_exposure=0, bankroll=10000)
    f_no = compute_kelly(sig_no, c=0.07, K_frac=0.25, existing_exposure=0, bankroll=10000)
    assert abs(f_yes - f_no) < 0.02  # approximately symmetric

def test_kelly_incremental_no_add_when_at_optimal():
    """If existing exposure already at optimal, f_incremental should be 0."""
    signal = Signal(direction="BUY_YES", EV=0.03, P_cons=0.55, P_kalshi=0.50, kelly_multiplier=0.8)
    f_fresh = compute_kelly(signal, c=0.07, K_frac=0.25, existing_exposure=0, bankroll=10000)
    f_incr = compute_kelly(signal, c=0.07, K_frac=0.25,
                            existing_exposure=f_fresh * 10000, bankroll=10000)
    assert f_incr == 0.0

def test_settlement_buy_no_wins():
    """Buy No position when Yes settles at 0 → profit."""
    pos = Position(direction="BUY_NO", entry_price=0.40, quantity=100)
    pnl = compute_realized_pnl(pos, settlement_price=0.00, fee_rate=0.07)
    assert pnl > 0  # No wins

def test_settlement_buy_no_loses():
    """Buy No position when Yes settles at 1 → loss."""
    pos = Position(direction="BUY_NO", entry_price=0.40, quantity=100)
    pnl = compute_realized_pnl(pos, settlement_price=1.00, fee_rate=0.07)
    assert pnl < 0  # No loses

def test_ev_buy_yes_positive():
    """EV should be positive when P_cons > P_kalshi after fees."""
    EV = 0.60 * (1-0.07) * (1-0.45) - 0.40 * 0.45
    assert EV > 0

def test_ev_buy_no_positive():
    """EV should be positive when (1-P_cons) > (1-P_kalshi) after fees."""
    P_cons = 0.40  # model says Yes is low → No is favorable
    P_poly = 0.50  # selling Yes at 0.50
    EV = (1-P_cons) * (1-0.07) * P_poly - P_cons * (1-P_poly)
    assert EV > 0
```

**Coverage targets:**

| Component | Functions | Key Tests |
|-----------|-----------|-----------|
| Kelly (Step 4.3) | `compute_kelly` | directional correctness, incremental sizing, edge cases (EV=0, W*L=0) |
| EV (Step 4.2) | `compute_signal_with_vwap` | Buy Yes/No EV sign, VWAP > best ask, 2-pass convergence |
| P_cons (Step 4.2) | `compute_conservative_P` | Yes direction lowers P, No direction raises P |
| Exit triggers (Step 4.4) | all 4 triggers | v2 validation cases from Phase 4 doc, edge reversal symmetry |
| Settlement (Step 4.6) | `compute_realized_pnl` | all 4 rows of the v2 validation table |
| Event handlers (Step 3.3) | goal, red card, substitution | S update, ΔS update, X_H/X_A update, μ recomputation |
| Cooldown (Step 3.5) | `handle_confirmed_goal_v2` | timer reset on rapid events, queue drain, VAR cancel rollback |
| MC pricing (Step 3.4) | `mc_simulate_remaining` | P_true ∈ [0,1], sum of market probs ≈ 1.0, σ_MC decreases with N |

### Tier 2: Integration Tests (1-10 seconds each, mock external APIs)

Phase pipeline correctness with controlled inputs.

```python
# test_integration.py

async def test_phase2_to_phase3_pipeline():
    """Phase 2 output feeds Phase 3 correctly."""
    # Mock Goalserve fixture + Odds-API odds
    mock_data = load_fixture("epl_match_2024.json")
    phase2_result = await Phase2Pipeline(mock_params, mock_data).run()

    # Phase 3 init with Phase 2 output
    model = LiveFootballQuantModel(phase2_result)
    await model.initialize(param_version=1)

    # First tick should produce valid P_true
    μ_H, μ_A = compute_remaining_mu(model)
    P_true, σ_MC = step_3_4_sync(model, μ_H, μ_A)

    assert 0 < P_true["home_win"] < 1
    assert 0 < P_true["draw"] < 1
    assert 0 < P_true["away_win"] < 1
    assert abs(sum(P_true.values()) - 1.0) < 0.01

async def test_paper_fill_directional_slippage():
    """Paper fill slippage is adverse for both directions."""
    ob = mock_order_book(best_ask=0.50, best_bid=0.48, depth=100)
    paper = PaperExecutionLayer(slippage_ticks=1)
    model = mock_model(ob_freeze=False, event_state="IDLE")

    # BUY_YES: fill should be above effective ask
    sig_yes = Signal(direction="BUY_YES", P_kalshi=0.50)
    fill_yes = await paper.execute_order(sig_yes, 50, ob, model)
    assert fill_yes.price > 0.50  # slippage makes it worse

    # BUY_NO: fill should be below effective bid
    sig_no = Signal(direction="BUY_NO", P_kalshi=0.48)
    fill_no = await paper.execute_order(sig_no, 50, ob, model)
    assert fill_no.price < 0.48  # slippage makes it worse

async def test_db_position_lifecycle():
    """Position goes through PENDING → OPEN → AWAITING_SETTLEMENT → SETTLED."""
    pos_id = await db.create_position(status="PENDING", ...)
    assert (await db.get_position(pos_id)).status == "PENDING"

    await db.update_position_status(pos_id, "OPEN")
    assert (await db.get_position(pos_id)).status == "OPEN"

    await db.update_position_status(pos_id, "AWAITING_SETTLEMENT")
    await db.settle_position(pos_id, settlement_price=1.0, pnl=0.55)
    assert (await db.get_position(pos_id)).status == "SETTLED"

async def test_reserve_confirm_release():
    """Exposure reservation prevents overallocation."""
    # Reserve 15% for match A
    res_a = await reserve_exposure("match_a", "ticker", 0.15, 10000, False)
    assert res_a is not None

    # Total cap is 20%, so match B can only get 5%
    res_b = await reserve_exposure("match_b", "ticker", 0.15, 10000, False)
    # res_b should be capped at 5% (0.05 * 10000 = 500)
    reservation = await db.get_reservation(res_b)
    assert reservation.amount <= 500

    # Release match A reservation
    await release_reservation(res_a)

    # Now match C can use the freed 15%
    res_c = await reserve_exposure("match_c", "ticker", 0.15, 10000, False)
    assert res_c is not None
```

### Tier 3: Property-Based Tests (hypothesis library)

Invariants that must hold for ANY valid input.

```python
from hypothesis import given, strategies as st

@given(
    P_true=st.floats(min_value=0.01, max_value=0.99),
    sigma=st.floats(min_value=0.0, max_value=0.1),
    z=st.floats(min_value=1.0, max_value=2.5),
)
def test_p_cons_direction_invariant(P_true, sigma, z):
    """Buy Yes P_cons is always ≤ P_true, Buy No P_cons is always ≥ P_true."""
    p_yes = compute_conservative_P(P_true, sigma, "BUY_YES", z)
    p_no = compute_conservative_P(P_true, sigma, "BUY_NO", z)
    assert p_yes <= P_true + 1e-9
    assert p_no >= P_true - 1e-9

@given(
    S_home=st.integers(min_value=0, max_value=5),
    S_away=st.integers(min_value=0, max_value=5),
    t=st.floats(min_value=0.1, max_value=1.4),
)
def test_p_true_valid_probability(S_home, S_away, t):
    """P_true must be a valid probability distribution."""
    model = build_mock_model(S=(S_home, S_away), t=t)
    P = compute_P_true(model)
    for market, prob in P.items():
        assert 0 <= prob <= 1, f"P({market}) = {prob} out of [0,1]"

@given(
    entry=st.floats(min_value=0.05, max_value=0.95),
    settlement=st.sampled_from([0.0, 1.0]),
    qty=st.integers(min_value=1, max_value=1000),
)
def test_settlement_sign_consistency(entry, settlement, qty):
    """
    Buy Yes wins when settlement=1.0 → PnL > 0 (before fees)
    Buy No wins when settlement=0.0 → PnL > 0 (before fees)
    """
    pos_yes = Position(direction="BUY_YES", entry_price=entry, quantity=qty)
    pnl_yes = compute_realized_pnl(pos_yes, settlement, fee_rate=0.0)
    if settlement == 1.0:
        assert pnl_yes > 0  # Yes won, Buy Yes profits
    else:
        assert pnl_yes < 0  # Yes lost, Buy Yes loses

    pos_no = Position(direction="BUY_NO", entry_price=entry, quantity=qty)
    pnl_no = compute_realized_pnl(pos_no, settlement, fee_rate=0.0)
    if settlement == 0.0:
        assert pnl_no > 0   # Yes lost, Buy No profits
    else:
        assert pnl_no < 0   # Yes won, Buy No loses

@given(
    mu_H_before=st.floats(min_value=0.1, max_value=3.0),
    mu_A_before=st.floats(min_value=0.1, max_value=3.0),
)
def test_goal_increases_scoring_team_win_prob(mu_H_before, mu_A_before):
    """After home goal, P(home win) must increase."""
    model_before = build_mock_model(S=(0,0), mu_H=mu_H_before, mu_A=mu_A_before)
    P_before = compute_P_true(model_before)

    model_after = build_mock_model(S=(1,0), mu_H=mu_H_before, mu_A=mu_A_before)
    P_after = compute_P_true(model_after)

    assert P_after["home_win"] > P_before["home_win"]
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install pytest hypothesis
      - run: pytest tests/unit/ -v --tb=short
    # Must pass: blocks merge

  integration:
    runs-on: ubuntu-latest
    services:
      postgres: { image: "postgres:16", env: { POSTGRES_DB: test } }
      redis: { image: "redis:7-alpine" }
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/integration/ -v --tb=short
    # Must pass: blocks merge

  property:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/property/ -v --hypothesis-seed=0
    # Must pass: blocks merge

  backtest:
    runs-on: ubuntu-latest
    steps:
      - run: python -m src.calibration.step_3_6_backtest --fast
    # Advisory: does not block merge (too slow for CI, run nightly)
```

### Pre-deployment Checklist

Before any live trading session:

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| All unit tests | `pytest tests/unit/` | 100% pass |
| All integration tests | `pytest tests/integration/` | 100% pass |
| All property tests | `pytest tests/property/` | 100% pass, 1000+ examples each |
| Step 3.6 backtest | `python -m step_3_6_backtest` | Go/No-Go = GO |
| Paper trading graduation | Check Phase 0 criteria | All 8 criteria met |

---

## Deployment Configuration

### Cloud Provider Resources

| Resource | Spec | Purpose |
|----------|------|---------|
| Orchestrator VM | 2 vCPU, 4 GB RAM | Scheduler + Orchestrator + Phase 2 |
| Match engine VMs | 1 vCPU, 1 GB RAM (per container) | Phase 3+4 per match |
| Phase 1 GPU VM | 4 vCPU, 16 GB RAM, T4 GPU | Weekly recalibration |
| PostgreSQL | Managed DB, 2 vCPU, 8 GB RAM | Shared state |
| Redis | Managed, 1 GB | Pubsub + locks + heartbeats |

### Scaling

| Scenario | Concurrent Matches | Resources |
|----------|-------------------|-----------|
| Single league evening | 2-4 | Orchestrator VM handles all |
| Multi-league Saturday (Europe + Americas) | 6-10 | Auto-scale match engine VMs |
| Champions League + MLS + Brasileirão midweek | 10-14 | Pre-provision extra capacity |

```python
# Auto-scaling rule (simplified)
async def scale_check():
    scheduled_next_2h = await db.count_matches_next_hours(2)
    active = await docker.count_running("service=match-engine")

    needed = min(scheduled_next_2h, MAX_CONCURRENT_MATCHES)
    if needed > active + 2:  # buffer of 2
        await cloud.scale_up(needed - active)
```

### Network Security

| Component | Inbound | Outbound |
|-----------|---------|----------|
| Orchestrator | SSH (admin only) | Goalserve, Odds-API, Kalshi, Docker socket |
| Match container | None (no inbound) | Goalserve, Odds-API, Kalshi, PostgreSQL, Redis |
| PostgreSQL | Match containers + Orchestrator only | None |
| Redis | Match containers + Orchestrator only | None |

> **Goalserve IP whitelist:** register the Orchestrator VM's static IP with Goalserve.
> Match containers route through the same VM's network, sharing the whitelisted IP.

### Secret Management

API keys and database credentials must not be stored in plaintext in code,
docker-compose files, or container environment variables in production.

| Environment | Method | Details |
|-------------|--------|---------|
| **Local dev** | `.env` file (gitignored) | `docker-compose --env-file .env` |
| **Production** | Cloud secret store | AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault |
| **CI/CD** | Pipeline secrets | GitHub Actions secrets, injected at deploy time |

```yaml
# Production: secrets injected by cloud provider, not in docker-compose
# Example: AWS ECS task definition pulls from Secrets Manager
# The docker-compose.yml env vars (ODDS_API_KEY, etc.) are dev-only placeholders.
```

**Requirements:**
- All API keys (`ODDS_API_KEY`, `GOALSERVE_API_KEY`, `KALSHI_API_KEY`) from secret store
- Database credentials (`DB_PASSWORD`) from secret store, never hardcoded
- Secrets rotated quarterly; rotation triggers container restart via orchestrator
- Audit log of secret access (provided by cloud secret store natively)

---

## Failure Modes & Recovery

### Orchestrator Single Point of Failure — Design Trade-off

The orchestrator is a single process that manages scheduling, container lifecycle,
and monitoring. This is a deliberate trade-off for the current system scale.

**Why not HA (High Availability) orchestrator:**
At the current scale (≤8 concurrent matches, ~80 matches/week), running dual
orchestrators with leader election (e.g., via Redis or Raft) adds significant
complexity for minimal benefit. The risk window during orchestrator downtime
is mitigated by the following design:

**Mitigation 1 — Container self-sufficiency:**
Running match containers operate independently. If the orchestrator dies for 30 minutes,
all running matches continue pricing and trading normally. Containers manage their own
WebSocket connections, Phase 3 pricing, Phase 4 execution, and settlement polling.
The orchestrator is only needed for starting new matches and monitoring.

**Mitigation 2 — Cloud auto-restart:**
The orchestrator runs as a systemd service with `Restart=always` (Linux) or
as a cloud-managed service (AWS ECS service, GCP Cloud Run) with auto-recovery.
Typical restart time: 5-30 seconds.

**Mitigation 3 — Missed trigger recovery:**
On restart, `recover_orchestrator_state()` catches:
- Matches in `SCHEDULED` whose `phase2_trigger` has passed → run Phase 2 immediately
- Matches in `PHASE2_DONE` whose `phase3_trigger` has passed → launch container immediately
- Matches in `PHASE3_RUNNING` → check if container is still alive, resume monitoring

**What can go wrong during orchestrator downtime:**
- A match scheduled to start during downtime is delayed (but recovered on restart)
- Container crash during downtime goes undetected (positions frozen, alert delayed)
- No heartbeat monitoring → stale heartbeats accumulate (auto-clears on restart)

**When to upgrade to HA:**
Consider dual orchestrators when concurrent matches routinely exceed 15,
or when the system trades with real capital exceeding $50,000.

| Failure | Detection | Automated Response | Manual Action |
|---------|-----------|-------------------|---------------|
| Match container crash | Exit code ≠ 0 | Freeze positions, alert | Review logs, manual close if needed |
| Orchestrator crash | Systemd watchdog | Auto-restart, resume from DB state | Check for missed triggers |
| PostgreSQL down | Connection failure | All containers freeze (no new orders) | Restore DB, containers auto-reconnect |
| Redis down | Connection failure | Heartbeats stop, orchestrator alerts | Restart Redis, containers reconnect |
| Odds-API WS disconnect | No message for 5s | Fallback to Live Score only | Monitor for reconnection |
| Goalserve REST failure | 5 consecutive HTTP errors | Freeze match orders | Check Goalserve status, IP whitelist |
| Kalshi WS disconnect | No message for 10s | Cancel working orders, reconnect | Verify API key validity |
| Bankroll depleted | balance < min_threshold | Stop all new entries globally | Deposit or review strategy |
| Phase 1 validation failure | Step 1.5 GO/NO-GO | Keep old params, alert | Investigate drift, manual retrain |
| PostgreSQL transient failure | Connection error on query | Retry via pool; block order submission if 3 retries fail | Check managed DB health |
| PENDING position stale | position.status = PENDING for > 5 min | Alert for reconciliation | Match PENDING against Kalshi fills manually |
| Kalshi settlement timeout | AWAITING_SETTLEMENT > 6 hours | Alert, keep positions open | Check Kalshi market status manually |

### State Recovery on Restart

```python
async def recover_orchestrator_state():
    """
    Called on orchestrator startup.
    Resume from last known DB state.
    """
    # 1. Find matches that were running
    running = await db.fetch_all("""
        SELECT * FROM match_schedule
        WHERE status IN ('PHASE2_RUNNING', 'PHASE3_RUNNING')
    """)

    for match in running:
        if match.status == "PHASE2_RUNNING":
            # Phase 2 is fast — just re-run
            await orchestrator.start_match_lifecycle(match)

        elif match.status == "PHASE3_RUNNING":
            # Check if container is still alive
            container_alive = await docker.is_running(match.container_id)
            if container_alive:
                # Resume monitoring
                await orchestrator.monitor_container(match.match_id, match.container_id)
            else:
                # Container died during outage — freeze and alert
                await db.update_status(match.match_id, "FAILED")
                await orchestrator.emergency_freeze(match.match_id)

    # 2. Re-schedule pending matches
    pending = await db.fetch_all("""
        SELECT * FROM match_schedule
        WHERE status = 'SCHEDULED'
          AND kickoff_utc > NOW()
    """)
    for match in pending:
        scheduler.enqueue(match)

    log.info(f"Recovery complete: {len(running)} running, {len(pending)} pending")
```

---

## Data Flow Summary

```
[Every 6 hours]
  Match Discovery (Goalserve fixtures + Kalshi markets)
      │
      ▼
  match_schedule table (SCHEDULED)
      │
      ▼
[Kickoff - 65 min]
  Orchestrator: Phase 2 (in-process, ~10s)
      │
      ├── SKIP → done
      ▼
  match_schedule (PHASE2_DONE)
      │
[Kickoff - 2 min]
      ▼
  Orchestrator: Launch match container
      │
      ▼
  ┌─────────────────────────────┐
  │  Match Container             │
  │                             │
  │  Phase 2.5: Init            │──── PostgreSQL: load params
  │       │                     │──── Redis: heartbeat start
  │       ▼                     │
  │  Phase 3: Pricing Loop      │──── Odds-API WS: bet365 odds
  │  (tick + odds + score)      │──── Goalserve REST: events
  │       │                     │
  │       ▼                     │
  │  Phase 4: Execution         │──── Kalshi WS: order book
  │  (signals + orders)         │──── Kalshi REST: orders
  │       │                     │──── PostgreSQL: positions, exposure
  │       ▼                     │──── Redis: risk locks
  │  Match Finished             │
  │       │                     │
  │  Settlement + Analytics     │──── PostgreSQL: P&L, metrics
  │       │                     │
  │  Exit (code 0)              │
  └─────────────┬───────────────┘
                │
                ▼
  Orchestrator: Archive logs, update status
                │
                ▼
  [Weekly / On-trigger]
  Phase 1 Worker (GPU container)
      │
      ▼
  production_params table (new version)
      │
      ▼
  Running containers notified via Redis pubsub
```