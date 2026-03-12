# Deployment Guide — Post-Sprint 8

Steps from data collection through live trading. Execute in order.
Each step has a verification check — do not proceed until it passes.

---

## Step 1: Data Collection Results

**Prerequisite:** `scripts/collect_all_historical.py` has finished running locally.

```
Review the data saved in data/commentaries/. Print a summary table:

| League | ID | Seasons Found | Total Matches | Total Goals | Total Red Cards | Status |

For any league with 0 seasons (e.g., MLS 1440 via soccerhistory), try direct commentaries date probing:
- Pick 10 dates spread across the missing season (e.g., every 5th Saturday)
- Call /commentaries/{league_id}?date={DD.MM.YYYY}&json=1
- If matches are found, run full date iteration for that season

Also check: python scripts/odds_backfill.py --leagues all --dry-run
Print how many odds events are cached per league.

Print final summary. Git commit -m "data: collection summary + gap fill" and push.
```

**Verify:** Every tradable league has at least 2 seasons of data with red cards present.

---

## Step 2: Phase 1 Full Retrain

**Prerequisite:** Step 1 verified.

```
Retrain Phase 1 on the full dataset (all 8 leagues, all collected seasons).
Load all match data from data/commentaries/.
Run step_1_4_nll_optimize with σ_a grid search {0.1, 0.3, 0.5, 1.0}.
Run step_1_5_validation with 5-fold chronological CV.

Print:
- Per-league: match count, Brier Score, Q matrix condition number
- Overall: best σ_a, ΔBS vs Betfair Exchange, Go/No-Go verdict
- Parameter summary: b[6], gamma_H[4], gamma_A[4], tau_H, tau_A

Save winning params to production_params table (is_active=True).
Git commit -m "phase1: full multi-league retrain" and push.
Then: git tag phase1-full-retrain && git push --tags
```

**Verify:**
- [ ] ΔBS < 0 (model beats Exchange baseline) on at least tier_1 leagues
- [ ] Q matrix condition number < 1000 for all leagues
- [ ] tau_H, tau_A ∈ [0.1, 5.0]
- [ ] No NaN/Inf in any parameter
- [ ] production_params table has active row

---

## Step 3: Dashboard Dockerfiles

**Prerequisite:** Step 2 verified.

```
Build docker/dashboard-api/Dockerfile (Python 3.11 + FastAPI + uvicorn)
and docker/dashboard-ui/Dockerfile (Node 20 + Next.js).

dashboard-api:
- Copy dashboard/api/ and src/common/ (for db.py, redis_client.py)
- pip install fastapi uvicorn asyncpg redis pydantic
- ENTRYPOINT: uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000

dashboard-ui:
- Copy dashboard/ui/
- npm install && npm run build
- ENTRYPOINT: npx next start -p 3000

Test: docker-compose --profile dashboard up --build
Verify: curl http://localhost:8000/api/matches returns JSON
Verify: curl http://localhost:3001 returns HTML

Git commit -m "docker: dashboard Dockerfiles" and push.
```

**Verify:**
- [ ] `http://localhost:8000/api/matches` returns `[]` (no matches yet, but valid JSON)
- [ ] `http://localhost:3001` loads Command Center page
- [ ] `http://localhost:3000` loads Grafana login (admin/admin)

---

## Step 4: Full Stack Local Start

**Prerequisite:** Step 3 verified.

```
Start the full stack locally and verify all services are healthy:

1. docker-compose up -d
2. docker-compose --profile dashboard up -d
3. Verify each service:
   - PostgreSQL: docker exec mmpp-postgres pg_isready → "accepting connections"
   - Redis: docker exec mmpp-redis redis-cli ping → "PONG"
   - Orchestrator: docker logs mmpp-orchestrator --tail 20 → no crash, shows "scheduler started"
   - Prometheus: curl http://localhost:9090/-/healthy → "ok"
   - Grafana: curl http://localhost:3000/api/health → "ok"
   - Dashboard API: curl http://localhost:8000/api/system/status → JSON response
   - Dashboard UI: curl http://localhost:3001 → HTML response

4. Run make migrate to ensure DB schema is up to date
5. Verify production_params has active row:
   docker exec mmpp-postgres psql -U trader -d soccer_trading \
     -c "SELECT version, is_active FROM production_params WHERE is_active=true"

Print service health table. Fix any failures before proceeding.
Git commit -m "deployment: full stack verified" and push.
```

**Verify:** All 7 services healthy, production_params active.

---

## Step 5: Paper Trading Start

**Prerequisite:** Step 4 verified. Live matches must be happening (check league schedules).

The orchestrator automatically:
1. Discovers upcoming matches every 6h
2. Runs Phase 2 at kickoff - 65min
3. Launches match container at kickoff - 2min
4. Container runs Phase 3+4 in paper mode
5. Settlement polls Kalshi after match ends

**Monitor via:**
- Command Center: `http://localhost:3001` — shows active/upcoming matches
- Grafana System Overview: `http://localhost:3000` → System Overview dashboard
- Orchestrator logs: `docker logs -f mmpp-orchestrator`
- Match container logs: `docker logs -f <container_id>`

**If no matches appear:**
```
Check orchestrator logs for errors:
docker logs mmpp-orchestrator --tail 100 | grep -i "error\|warn\|discovery"

Common issues:
- No matches in next 48h for tradable leagues → wait for match day
- Kalshi has no soccer markets for this match → check Kalshi API
- Goalserve API error → check API key and connectivity
- Phase 2 sanity check = SKIP → model disagrees too much, check logs
```

**If matches appear but no trades:**
```
Check match container logs:
docker logs <container_id> --tail 200

Common issues:
- No edge detected (EV < THETA_ENTRY for all markets) → normal, wait for more matches
- Kalshi order book empty or stale → check Kalshi WS connection
- ob_freeze stuck → check Odds-API WS connection
- All signals HOLD → model and market agree, no edge exists
```

---

## Step 6: Monitor Graduation Criteria (2-4 weeks)

**Prerequisite:** Paper trades are being generated.

Track these 8 criteria. All must pass for Phase A transition.

| # | Criterion | Target | Where to Check |
|---|-----------|--------|----------------|
| 1 | Paper trades | ≥ 50 | Grafana: Paper Validation → Paper Trades Count |
| 2 | Edge realization | 0.6 - 1.5 | Grafana: Paper Validation → Edge Realization |
| 3 | Brier Score | Phase 1.5 ± 0.03 | Grafana: Model Health → Brier Score Rolling 20 |
| 4 | Max drawdown | < 15% | Grafana: Paper Validation → Max Drawdown |
| 5 | Directional correctness | 100% | Check: no BUY_YES P&L has wrong sign pattern |
| 6 | Paper realism score | > 0.85 | Grafana: Paper Validation → Realism Score |
| 7 | No system crashes | 0 crashes | Grafana: System Overview → Alerts Firing |
| 8 | THETA_ENTRY calibrated | Done | Compute from paper data: breakeven edge + margin |

**Weekly check prompt:**
```
Query the paper trading results:
1. SELECT COUNT(*) FROM positions WHERE is_paper=true AND status='SETTLED'
2. Compute edge realization: avg(realized_pnl) / avg(EV_at_entry) for settled paper positions
3. Compute rolling 20-match Brier Score from tick_snapshots
4. SELECT drawdown_pct FROM bankroll_snapshot WHERE mode='paper' ORDER BY created_at DESC LIMIT 1
5. Check for any container exits with code != 0

Print graduation checklist with current values vs targets.
```

---

## Step 7: Phase A Transition (Live Trading)

**Prerequisite:** All 8 graduation criteria passed.

```
Transition to live trading:

1. Update config:
   - config/system.yaml: trading.mode = "live"
   - Set real Kalshi API credentials in .env

2. Fund Kalshi account (start small, e.g., $500-1000)

3. Update bankroll:
   docker exec mmpp-postgres psql -U trader -d soccer_trading \
     -c "UPDATE bankroll SET balance=500.00 WHERE mode='live'"

4. Restart orchestrator:
   docker-compose restart orchestrator

5. Verify first live match:
   - Watch Command Center for [Live Mode 🟢] badge
   - First trade should appear in PositionTable
   - Check Kalshi account for matching order

6. Monitor closely for first 48 hours:
   - Keep Grafana Risk dashboard open
   - Set phone alerts for critical severity
   - Be ready to emergency stop:
     docker exec mmpp-postgres psql -U trader -d soccer_trading \
       -c "UPDATE bankroll SET balance=0 WHERE mode='live'"
```

**Emergency stop:**
```bash
# Option 1: Zero bankroll (no new trades, existing positions settle normally)
docker exec mmpp-postgres psql -U trader -d soccer_trading \
  -c "UPDATE bankroll SET balance=0 WHERE mode='live'"

# Option 2: Kill all containers (immediate, positions stay open on Kalshi)
docker-compose down

# Option 3: Switch back to paper
# Edit config/system.yaml: trading.mode = "paper"
docker-compose restart orchestrator
```

---

## Quick Reference — Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| React Dashboard | http://localhost:3001 | Command Center, Match Deep Dive, P&L |
| Dashboard API | http://localhost:8000 | REST + WebSocket |
| Grafana | http://localhost:3000 | 6 operational dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Raw metrics |

## Quick Reference — Useful Commands

```bash
# Start everything
docker-compose up -d && docker-compose --profile dashboard up -d

# Stop everything
docker-compose --profile dashboard down && docker-compose down

# View orchestrator logs
docker logs -f mmpp-orchestrator

# View match container logs
docker logs -f $(docker ps --filter "label=service=match-engine" -q | head -1)

# Check paper P&L
docker exec mmpp-postgres psql -U trader -d soccer_trading \
  -c "SELECT SUM(realized_pnl) as total_pnl, COUNT(*) as trades FROM positions WHERE is_paper=true AND status='SETTLED'"

# Check active matches
docker exec mmpp-postgres psql -U trader -d soccer_trading \
  -c "SELECT match_id, status, trading_mode FROM match_schedule WHERE status='PHASE3_RUNNING'"

# Force Phase 1 retrain
docker exec mmpp-orchestrator python -m src.calibration.phase1_worker
```
