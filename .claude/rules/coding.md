# Coding Rules

## Python

- Python 3.11+. async/await for all I/O.
- Type hints required on all function signatures. Run `mypy --strict`.
- Dataclasses for data containers (not dicts). Shared types → `src/common/types.py`.
- asyncpg for PostgreSQL, always via pool in `src/common/db.py`. No raw SQL outside `db.py`.
- structlog for logging. Bind `match_id` and `component` at module level.
- Prometheus client for metrics. Define in `src/common/metrics.py`, import elsewhere.
- No global mutable state. Pass config/model/db as arguments or attributes.

## Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE`
- DB tables: `snake_case`
- Config YAML keys: `snake_case`

## Imports (this order)

```python
# Standard library
import asyncio
import time

# Third party
import numpy as np
import structlog

# Internal — always absolute
from src.common.types import Signal, Position
from src.common.db import db_pool
from src.common.metrics import tick_latency
```

## Error Handling

- API clients: retry with exponential backoff (`src/engine/resilient_ws.py` pattern).
- DB writes for positions: 2-phase pattern (PENDING → OPEN). See `docs/orchestration.md` DB Resilience.
- Never swallow exceptions silently. Log at minimum.

## Testing

- Every new function gets a unit test in `tests/unit/`.
- Write the test FIRST if the function involves math (Kelly, EV, P_cons, settlement).
- Use `hypothesis` for invariants (P ∈ [0,1], directional correctness).
- Integration tests mock external APIs, never call real endpoints.
- Property tests: `tests/property/` with `hypothesis` library.
