# docker/orchestrator/Dockerfile
#
# Orchestrator — scheduler, match discovery, lifecycle management.
# Manages match containers via the Docker socket (mounted at runtime).
#
# Build: docker build -t soccer-orchestrator:latest -f docker/orchestrator/Dockerfile .
# Run:   see docker/docker-compose.yml (orchestrator service)
#
# The Docker socket (/var/run/docker.sock) must be bind-mounted at runtime
# so aiodocker can launch/inspect/stop match-engine containers:
#   volumes:
#     - /var/run/docker.sock:/var/run/docker.sock

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # curl used by health-check probe in docker-compose
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ─────────────────────────────────────────────────────────────
COPY docker/orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Application source ──────────────────────────────────────────────────────
COPY src/ /app/src/
COPY config/ /app/config/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Restart policy managed by docker-compose / systemd (Restart=always).
# The orchestrator is stateless between restarts — recovery.py re-reads DB.
ENTRYPOINT ["python", "-m", "src.orchestrator.main"]
