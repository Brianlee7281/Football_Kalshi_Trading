# dashboard/api/routes/system.py
#
# REST endpoints:
#   GET /api/system/status   → SystemStatus

from __future__ import annotations

import json as _json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from dashboard.api.deps import Pool
from dashboard.api.models import (
    ConnectionHealth,
    ContainerStatus,
    SystemStatus,
)

router = APIRouter(prefix="/api/system", tags=["system"])

# Statuses that represent an active or upcoming match
_ACTIVE_STATUSES = (
    "SCHEDULED",
    "PHASE2_RUNNING",
    "PHASE2_DONE",
    "PHASE3_RUNNING",
    "SETTLING",
)


def _j(v: Any) -> dict[str, Any] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return _json.loads(v)  # type: ignore[no-any-return]
    return dict(v)


@router.get("/status", response_model=SystemStatus)
async def system_status(pool: Pool) -> SystemStatus:
    """Container status, connection health, and active param version."""
    now_utc = datetime.now(UTC)

    async with pool.acquire() as conn:
        # ── active/upcoming containers ────────────────────────────────────────
        match_rows = await conn.fetch(
            """
            SELECT match_id, status, container_id, updated_at
            FROM match_schedule
            WHERE status = ANY($1::text[])
            ORDER BY kickoff_utc
            LIMIT 50
            """,
            list(_ACTIVE_STATUSES),
        )

        # ── active param version ──────────────────────────────────────────────
        param_row = await conn.fetchrow(
            """
            SELECT version, created_at
            FROM production_params
            WHERE is_active = TRUE
            LIMIT 1
            """,
        )

        # ── matches since last retrain ────────────────────────────────────────
        matches_since: int | None = None
        if param_row is not None:
            cnt_row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt
                FROM match_schedule
                WHERE status IN ('PHASE3_RUNNING', 'SETTLING', 'FINISHED', 'ARCHIVED')
                  AND created_at >= $1
                """,
                param_row["created_at"],
            )
            matches_since = int(cnt_row["cnt"]) if cnt_row else 0

        # ── DB connectivity (if we reached here, it's connected) ──────────────
        db_msg_age = (
            (now_utc - now_utc).total_seconds()  # 0 — just executed a query
        )

    # Build container list
    containers: list[ContainerStatus] = []
    for row in match_rows:
        updated_at = row["updated_at"]
        if updated_at is not None:
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            age_s = (now_utc - updated_at).total_seconds()
        else:
            age_s = None

        containers.append(
            ContainerStatus(
                match_id=row["match_id"],
                status=row["status"],
                uptime_min=None,        # container start time not in match_schedule
                heartbeat_age_s=age_s,  # proxy: seconds since status last updated
                container_id=row["container_id"],
            )
        )

    # Connection health — DB verified above; Redis unknown from REST route
    connections: list[ConnectionHealth] = [
        ConnectionHealth(
            service="PostgreSQL",
            status="connected",
            last_message_age_s=db_msg_age,
            detail=None,
        ),
        ConnectionHealth(
            service="Redis",
            status="unknown",  # Redis health checked via WebSocket route
            last_message_age_s=None,
            detail="Redis health reported via /ws/live",
        ),
    ]

    return SystemStatus(
        containers=containers,
        connections=connections,
        param_version=(
            int(param_row["version"]) if param_row is not None else None
        ),
        param_trained_at=(
            param_row["created_at"] if param_row is not None else None
        ),
        matches_since_retrain=matches_since,
    )
