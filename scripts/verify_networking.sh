#!/usr/bin/env bash
# scripts/verify_networking.sh
#
# Verify Docker-in-Docker networking for the MMPP trading stack.
#
# What it tests:
#   1. From inside the orchestrator: pg_isready -h postgres (compose service)
#   2. From inside the orchestrator: redis-cli -h redis ping (compose service)
#   3. Spawn a temporary match-engine container on mmpp-net and check:
#      a. pg_isready -h postgres   (same hostname, different container)
#      b. redis-cli -h redis ping  (same hostname, different container)
#   4. Print DNS resolution for postgres + redis from both contexts
#
# Usage (from repo root):
#   docker compose -f docker/docker-compose.yml up -d postgres redis orchestrator
#   bash scripts/verify_networking.sh
#
# Expected output: all checks print OK.
# If any check fails, a container spawned by ContainerManager will also fail
# to resolve postgres/redis — fix the network before deploying.

set -euo pipefail

COMPOSE_FILE="docker/docker-compose.yml"
NETWORK="mmpp-net"
ORCHESTRATOR="mmpp-orchestrator"
# Use a lightweight image that has pg_isready and redis-cli
TEST_IMAGE="postgres:16-alpine"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "=== MMPP Network Verification ==="
echo "Network: ${NETWORK}"
echo ""

# ── 1. Orchestrator → Postgres ────────────────────────────────────────────
echo "--- 1. orchestrator → postgres (pg_isready) ---"
docker exec "${ORCHESTRATOR}" sh -c \
  "pg_isready -h postgres -U postgres -d soccer_trading -t 5" \
  && ok "orchestrator can reach postgres" \
  || fail "orchestrator CANNOT reach postgres"

# ── 2. Orchestrator → Redis ───────────────────────────────────────────────
echo ""
echo "--- 2. orchestrator → redis (redis-cli ping) ---"
docker exec "${ORCHESTRATOR}" sh -c \
  "redis-cli -h redis ping" \
  | grep -q "PONG" \
  && ok "orchestrator can reach redis" \
  || fail "orchestrator CANNOT reach redis"

# ── 3. DNS from orchestrator ──────────────────────────────────────────────
echo ""
echo "--- 3. DNS resolution from orchestrator ---"
PG_IP=$(docker exec "${ORCHESTRATOR}" sh -c "getent hosts postgres | awk '{print \$1}'" 2>/dev/null || echo "")
REDIS_IP=$(docker exec "${ORCHESTRATOR}" sh -c "getent hosts redis | awk '{print \$1}'" 2>/dev/null || echo "")
echo "  postgres → ${PG_IP:-<unresolved>}"
echo "  redis    → ${REDIS_IP:-<unresolved>}"
[ -n "${PG_IP}" ]    && ok "postgres DNS resolved" || fail "postgres DNS unresolved in orchestrator"
[ -n "${REDIS_IP}" ] && ok "redis DNS resolved"    || fail "redis DNS unresolved in orchestrator"

# ── 4. Spawn a test container on mmpp-net (simulates match-engine launch) ─
echo ""
echo "--- 4. Spawned container on mmpp-net → postgres + redis ---"
TEST_CTR="mmpp-net-verify-$$"

cleanup() { docker rm -f "${TEST_CTR}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d \
  --name "${TEST_CTR}" \
  --network "${NETWORK}" \
  "${TEST_IMAGE}" \
  sleep 30 \
  >/dev/null

# pg_isready from spawned container
docker exec "${TEST_CTR}" \
  pg_isready -h postgres -U postgres -d soccer_trading -t 5 \
  && ok "spawned container can reach postgres" \
  || fail "spawned container CANNOT reach postgres — Docker-in-Docker networking broken"

# redis-cli from spawned container
docker exec "${TEST_CTR}" sh -c \
  "apk add --no-cache redis-cli -q 2>/dev/null; redis-cli -h redis ping" \
  2>/dev/null | grep -q "PONG" \
  && ok "spawned container can reach redis" \
  || fail "spawned container CANNOT reach redis — Docker-in-Docker networking broken"

# DNS from spawned container
echo ""
echo "--- 5. DNS resolution from spawned container ---"
PG_IP2=$(docker exec "${TEST_CTR}" sh -c "getent hosts postgres | awk '{print \$1}'" 2>/dev/null || echo "")
REDIS_IP2=$(docker exec "${TEST_CTR}" sh -c "getent hosts redis | awk '{print \$1}'" 2>/dev/null || echo "")
echo "  postgres → ${PG_IP2:-<unresolved>}"
echo "  redis    → ${REDIS_IP2:-<unresolved>}"
[ -n "${PG_IP2}" ]    && ok "postgres DNS resolved from spawned container" \
                       || fail "postgres DNS unresolved from spawned container"
[ -n "${REDIS_IP2}" ] && ok "redis DNS resolved from spawned container" \
                        || fail "redis DNS unresolved from spawned container"

echo ""
echo "=== All checks passed — Docker-in-Docker networking OK ==="
