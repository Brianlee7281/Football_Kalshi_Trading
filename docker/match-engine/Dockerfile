# docker/match-engine/Dockerfile
#
# Phase 3+4 runtime — one container per live match.
# Injected env vars: MATCH_ID, LEAGUE_ID, TRADING_MODE, PARAM_VERSION,
#   A_H, A_A, C_TIME, KALSHI_TICKERS, ODDS_API_EVENT_ID,
#   DB_URL, REDIS_URL, ODDS_API_KEY, GOALSERVE_API_KEY, KALSHI_API_KEY
#
# Build: docker build -t soccer-live-engine:latest -f docker/match-engine/Dockerfile .
# Run:   launched automatically by ContainerManager (never by hand)

FROM python:3.11-slim

# ── System deps needed by Numba / SciPy ────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ─────────────────────────────────────────────────────────────
COPY docker/match-engine/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application source ──────────────────────────────────────────────────────
COPY src/ /app/src/
COPY config/ /app/config/

# ── Numba AOT warm-up ───────────────────────────────────────────────────────
# Trigger JIT compilation at image-build time so the first-tick latency on a
# live match is not spent on compilation. Numba caches the compiled artifact
# inside the image — no disk writes needed at runtime.
RUN python - <<'PYEOF'
import numpy as np
from src.engine.mc_core import mc_simulate_remaining

_b      = np.zeros(6, dtype=np.float64)
_gH     = np.ones(4,  dtype=np.float64)
_gA     = np.ones(4,  dtype=np.float64)
_dH     = np.zeros(5, dtype=np.float64)
_dA     = np.zeros(5, dtype=np.float64)
_Qd     = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
_Qoff   = np.eye(4, dtype=np.float64)
_bounds = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], dtype=np.float64)

mc_simulate_remaining(
    t_now=0.0, T_end=90.0,
    S_H=0, S_A=0, state=0, score_diff=0,
    a_H=1.2, a_A=0.9,
    b=_b, gamma_H=_gH, gamma_A=_gA,
    delta_H=_dH, delta_A=_dA,
    Q_diag=_Qd, Q_off=_Qoff,
    basis_bounds=_bounds,
    N=10, seed=42,
)
print("Numba AOT warm-up complete")
PYEOF

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# restart_policy=no is enforced by ContainerManager at container launch.
ENTRYPOINT ["python", "-m", "src.match_engine.main"]
