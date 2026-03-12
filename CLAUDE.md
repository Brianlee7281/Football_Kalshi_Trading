# CLAUDE.md — MMPP Soccer Live Trading System

Real-time soccer betting: MMPP model → true probabilities → edge detection → Kalshi execution.

## Architecture (4 phases per match)

1. **Phase 1** (offline weekly): Train MMPP params → `production_params` table
2. **Phase 2** (kickoff -65min): Backsolve intensities, sanity check → GO/SKIP
3. **Phase 3** (live 90min): Price P_true/sec via MC simulation → `dict` per market
4. **Phase 4** (live 90min): Incremental Kelly → Kalshi orders (paper or live)

Infra: Docker (1 container/match), PostgreSQL, Redis, Prometheus/Grafana, FastAPI+Next.js dashboard.

## Project Layout

```
src/calibration/    Phase 1 (Steps 1.1-1.5)
src/prematch/       Phase 2 (Steps 2.1-2.5)
src/engine/         Phase 3 (tick loop, MC, events) — mode-invariant
src/execution/      Phase 4 (signals, Kelly, orders) — paper/live branching
src/match_engine/   Container entry point
src/orchestrator/   Scheduler + lifecycle
src/common/         DB, Redis, logging, metrics, shared types
src/clients/        Goalserve, Odds-API, Kalshi API clients
dashboard/          api/ (FastAPI) + ui/ (Next.js)
tests/              unit/ + integration/ + property/
```

## Rules & Conventions

Detailed rules in `.claude/rules/` — **read these before writing any code:**

| File | What |
|------|------|
| `.claude/rules/coding.md` | Python style, naming, imports, error handling, testing rules |
| `.claude/rules/patterns.md` | Key system patterns (Phase 3→4 interface, paper/live, risk limits, tick clock) |
| `.claude/rules/workflow.md` | Sprint plan, what to read before coding each module |

## Design Docs (`docs/`)

Read the relevant doc BEFORE coding its module:

| Module | Read First |
|--------|-----------|
| `src/calibration/` | `docs/phase1.md` |
| `src/prematch/` | `docs/phase2.md` |
| `src/engine/` | `docs/phase3.md` |
| `src/execution/` | `docs/phase4.md` |
| `src/orchestrator/`, `docker/` | `docs/orchestration.md` |
| `dashboard/` | `docs/dashboard.md` + `docs/dashboard_decomposition.md` |
| Any config work | `docs/config_reference.md` |
| Any DB work | `docs/schema.sql` |
| Starting a sprint | `docs/implementation_roadmap.md` |

## Commands

```bash
make up          # docker-compose (postgres, redis, prometheus, grafana)
make test        # pytest all tiers
make lint        # mypy + ruff
make migrate     # psql -f sql/schema.sql
```

## Current Progress

```
[x] Design docs complete (9 files, 9300+ lines)
[x] Project blueprint (docs/blueprint.md)
[ ] Sprint 1: API clients + Step 1.1
[ ] Sprint 2: Steps 1.2-1.3
[ ] Sprint 3: Steps 1.4-1.5
[ ] Sprint 4: Phase 2+3 (single match replay)
[ ] Sprint 5: Phase 4 (paper on live Kalshi)
[ ] Sprint 6: Orchestration (Docker e2e)
[ ] Sprint 7: Dashboard
[ ] Sprint 8: Tests + CI
```

> **Now working on:** Sprint 1 — repo init, schema.sql, API clients