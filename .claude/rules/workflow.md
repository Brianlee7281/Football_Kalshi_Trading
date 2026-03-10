# Workflow Rules

## Before Coding Any Module

1. Read `docs/implementation_roadmap.md` for the current sprint's task list
2. Read the design doc listed in CLAUDE.md for that module
3. Read `.claude/rules/coding.md` for style rules
4. Read `.claude/rules/patterns.md` for system patterns
5. Check `docs/config_reference.md` if the module uses any config values
6. Check `docs/schema.sql` if the module touches the database

## Sprint Plan

| Sprint | Build | Design Doc | Key Files |
|--------|-------|------------|-----------|
| S1 | API clients + Step 1.1 | phase1.md | `src/clients/*.py`, `src/calibration/step_1_1_intervals.py`, `src/common/types.py` |
| S2 | Steps 1.2-1.3 | phase1.md | `src/calibration/step_1_2_*.py`, `step_1_3_*.py` |
| S3 | Steps 1.4-1.5 | phase1.md | `src/calibration/step_1_4_*.py`, `step_1_5_*.py` |
| S4 | Phase 2 + Phase 3 | phase2.md, phase3.md | `src/prematch/*.py`, `src/engine/*.py` |
| S5 | Phase 4 (paper) | phase4.md | `src/execution/*.py`, `src/match_engine/main.py` |
| S6 | Orchestration | orchestration.md | `src/orchestrator/*.py`, `docker/*`, `sql/schema.sql` |
| S7 | Dashboard | dashboard.md | `dashboard/api/*`, `dashboard/ui/*`, `monitoring/*` |
| S8 | Tests + CI | orchestration.md (Testing) | `tests/**`, `.github/workflows/*` |

## Sprint Completion Criteria

Each sprint ends with:
1. All new code has type hints and passes `mypy --strict`
2. Unit tests written for all math functions
3. Manual verification: run the module with real (or mock) data, check outputs are in expected range
4. Update `CLAUDE.md` → "Current Progress" section

## Module Creation Checklist

When creating a new `.py` file:
- [ ] Add `__init__.py` to package if missing
- [ ] Import shared types from `src/common/types.py`
- [ ] Add structlog logger: `log = structlog.get_logger()`
- [ ] Add type hints to all functions
- [ ] Create corresponding test file in `tests/unit/`
- [ ] If it uses DB: go through `src/common/db.py`, not direct asyncpg

## When Stuck

- Math question → check the design doc's pseudocode and validation examples
- "Is this how P_true flows?" → `.claude/rules/patterns.md` #1
- "Paper or live?" → `.claude/rules/patterns.md` #2
- Config value → `docs/config_reference.md`
- DB schema → `docs/schema.sql`
- Full architecture → `docs/blueprint.md`
