"""Orchestrator state recovery — resume from DB on restart.

Called once at startup before the main event loop begins.  Scans
``match_schedule`` for matches in intermediate states and re-enqueues or
re-monitors them so no match is silently dropped across a restart.

Recovery rules
--------------
PHASE2_RUNNING
    Phase 2 is fast (<10s). Re-run immediately via
    ``MatchLifecycleManager.start_match_lifecycle``.

PHASE2_DONE (phase3_trigger already past)
    The container was never launched (orchestrator crashed between Phase 2
    completion and Phase 3 trigger).  Launch immediately via
    ``MatchLifecycleManager.start_live_engine``.

PHASE3_RUNNING
    A container should be running.  Check Docker:
      • Container alive → resume monitoring in a background task.
      • Container dead  → mark FAILED + emergency_freeze.

SCHEDULED (phase2_trigger already past)
    Trigger was missed during downtime.  Start Phase 2 immediately.

All other statuses (PHASE2_DONE within window, SKIPPED, FAILED,
SETTLING, FINISHED, ARCHIVED) require no action.

Reference: docs/orchestration.md — State Recovery on Restart
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import asyncpg

from src.common.logging import get_logger

logger = get_logger("recovery")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def recover_orchestrator_state(
    pool: asyncpg.Pool,
    lifecycle: Any,
    container_manager: Any,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Resume orchestrator from last known DB state after a restart.

    Scans ``match_schedule`` for matches in intermediate states and either
    re-runs Phase 2, resumes container monitoring, or launches containers
    that missed their trigger window.

    Args:
        pool: asyncpg connection pool.
        lifecycle: ``MatchLifecycleManager`` instance.
        container_manager: ``ContainerManager`` instance.

    Returns:
        Dict with counts: ``{
            "phase2_rerun": int,
            "phase2_done_launched": int,
            "phase3_resumed": int,
            "phase3_failed": int,
            "scheduled_triggered": int,
        }``
    """
    now = now if now is not None else datetime.now(UTC)
    counts: dict[str, int] = {
        "phase2_rerun": 0,
        "phase2_done_launched": 0,
        "phase3_resumed": 0,
        "phase3_failed": 0,
        "scheduled_triggered": 0,
    }

    # ── Fetch all matches requiring action ───────────────────────────────
    async with pool.acquire() as conn:
        rows: list[Any] = await conn.fetch(
            """
            SELECT match_id, status, container_id,
                   phase2_trigger, phase3_trigger, kickoff_utc
            FROM match_schedule
            WHERE status IN ('PHASE2_RUNNING', 'PHASE2_DONE', 'PHASE3_RUNNING', 'SCHEDULED')
            ORDER BY kickoff_utc ASC
            """
        )

    logger.info(
        "recovery_scan",
        total_matches=len(rows),
        now=now.isoformat(),
    )

    for row in rows:
        match_id: str = row["match_id"]
        status: str = row["status"]

        try:
            if status == "PHASE2_RUNNING":
                await _recover_phase2_running(match_id, row, lifecycle, counts)

            elif status == "PHASE2_DONE":
                await _recover_phase2_done(match_id, row, lifecycle, now, counts)

            elif status == "PHASE3_RUNNING":
                await _recover_phase3_running(
                    match_id, row, lifecycle, container_manager, counts
                )

            elif status == "SCHEDULED":
                await _recover_scheduled(match_id, row, lifecycle, now, counts)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "recovery_match_error",
                match_id=match_id,
                status=status,
                error=str(exc),
            )

    logger.info("recovery_complete", **counts)
    return counts


# ---------------------------------------------------------------------------
# Per-status recovery handlers
# ---------------------------------------------------------------------------


async def _recover_phase2_running(
    match_id: str,
    row: Any,
    lifecycle: Any,
    counts: dict[str, int],
) -> None:
    """Phase 2 was in-flight — re-run it immediately.

    Phase 2 is idempotent and fast (<10s), so re-running is always safe.
    The lifecycle manager will update status from PHASE2_RUNNING again.

    Args:
        match_id: Goalserve match ID.
        row: asyncpg Record from match_schedule.
        lifecycle: MatchLifecycleManager instance.
        counts: Mutable counter dict to update.
    """
    logger.info("recovery_phase2_rerun", match_id=match_id)
    asyncio.create_task(
        lifecycle.start_match_lifecycle(row),
        name=f"recovery-phase2-{match_id}",
    )
    counts["phase2_rerun"] += 1


async def _recover_phase2_done(
    match_id: str,
    row: Any,
    lifecycle: Any,
    now: datetime,
    counts: dict[str, int],
) -> None:
    """Phase 2 is done but Phase 3 container was not yet launched.

    If phase3_trigger has already passed, launch immediately.
    Otherwise the normal TriggerExecutor.tick() will fire at the right time.

    Args:
        match_id: Goalserve match ID.
        row: asyncpg Record from match_schedule.
        lifecycle: MatchLifecycleManager instance.
        now: Current UTC datetime.
        counts: Mutable counter dict to update.
    """
    phase3_trigger = _ensure_utc(row["phase3_trigger"])
    if phase3_trigger <= now:
        logger.info(
            "recovery_phase2_done_trigger_missed",
            match_id=match_id,
            phase3_trigger=phase3_trigger.isoformat(),
        )
        asyncio.create_task(
            lifecycle.start_live_engine(row),
            name=f"recovery-phase3-{match_id}",
        )
        counts["phase2_done_launched"] += 1
    else:
        logger.info(
            "recovery_phase2_done_within_window",
            match_id=match_id,
            phase3_trigger=phase3_trigger.isoformat(),
        )


async def _recover_phase3_running(
    match_id: str,
    row: Any,
    lifecycle: Any,
    container_manager: Any,
    counts: dict[str, int],
) -> None:
    """A match container should be running — verify and resume monitoring.

    Args:
        match_id: Goalserve match ID.
        row: asyncpg Record from match_schedule.
        lifecycle: MatchLifecycleManager instance.
        container_manager: ContainerManager instance.
        counts: Mutable counter dict to update.
    """
    container_id: str | None = row["container_id"]

    if not container_id:
        logger.warning(
            "recovery_phase3_no_container_id",
            match_id=match_id,
        )
        await lifecycle.emergency_freeze(match_id)
        counts["phase3_failed"] += 1
        return

    alive = await _is_container_alive(container_manager, container_id)

    if alive:
        logger.info(
            "recovery_phase3_container_alive",
            match_id=match_id,
            container_id=container_id[:12],
        )
        # Resume monitoring in the background — pass a thin proxy object
        proxy = _ContainerProxy(container_id)
        asyncio.create_task(
            lifecycle._monitor_container(match_id, proxy),
            name=f"recovery-monitor-{match_id}",
        )
        counts["phase3_resumed"] += 1
    else:
        logger.error(
            "recovery_phase3_container_dead",
            match_id=match_id,
            container_id=container_id[:12],
        )
        await _mark_failed(lifecycle, match_id)
        counts["phase3_failed"] += 1


async def _recover_scheduled(
    match_id: str,
    row: Any,
    lifecycle: Any,
    now: datetime,
    counts: dict[str, int],
) -> None:
    """A SCHEDULED match whose phase2_trigger has passed during downtime.

    Kick off Phase 2 immediately so the match can still be traded.
    If the kickoff has already passed, the sanity check in Phase 2 will
    produce a SKIP verdict and no container will be launched.

    Args:
        match_id: Goalserve match ID.
        row: asyncpg Record from match_schedule.
        lifecycle: MatchLifecycleManager instance.
        now: Current UTC datetime.
        counts: Mutable counter dict to update.
    """
    phase2_trigger = _ensure_utc(row["phase2_trigger"])
    if phase2_trigger <= now:
        logger.info(
            "recovery_scheduled_trigger_missed",
            match_id=match_id,
            phase2_trigger=phase2_trigger.isoformat(),
        )
        asyncio.create_task(
            lifecycle.start_match_lifecycle(row),
            name=f"recovery-scheduled-{match_id}",
        )
        counts["scheduled_triggered"] += 1
    else:
        logger.debug(
            "recovery_scheduled_future",
            match_id=match_id,
            phase2_trigger=phase2_trigger.isoformat(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _is_container_alive(container_manager: Any, container_id: str) -> bool:
    """Return True if the Docker container is in a running state.

    Args:
        container_manager: ContainerManager instance.
        container_id: Docker container ID string.

    Returns:
        True if container status is "running", False otherwise or on error.
    """
    try:
        proxy = _ContainerProxy(container_id)
        info = await container_manager.inspect(proxy)
        state = info.get("State", {})
        return str(state.get("Status", "")) == "running"
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "container_alive_check_failed",
            container_id=container_id[:12],
            error=str(exc),
        )
        return False


async def _mark_failed(lifecycle: Any, match_id: str) -> None:
    """Update status to FAILED and emit emergency freeze.

    Args:
        lifecycle: MatchLifecycleManager instance.
        match_id: Goalserve match ID.
    """
    from src.orchestrator.lifecycle import _update_status  # avoid circular at module level

    try:
        await _update_status(lifecycle._pool, match_id, "FAILED")
    except Exception as exc:  # noqa: BLE001
        logger.warning("recovery_mark_failed_db_error", match_id=match_id, error=str(exc))
    await lifecycle.emergency_freeze(match_id)


def _ensure_utc(dt: Any) -> datetime:
    """Return a timezone-aware UTC datetime from an asyncpg timestamptz value.

    asyncpg returns ``datetime`` objects with UTC timezone already set for
    ``TIMESTAMPTZ`` columns.  This function handles the rare case where the
    value is naive (no tzinfo) by attaching UTC.

    Args:
        dt: datetime (possibly naive) from asyncpg Record.

    Returns:
        Timezone-aware UTC datetime.
    """
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    # Fallback: attempt ISO parse
    return datetime.fromisoformat(str(dt)).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Thin proxy for re-attaching to an existing container by ID
# ---------------------------------------------------------------------------


class _ContainerProxy:
    """Minimal object that satisfies ContainerManager's container.id contract.

    When resuming monitoring after a restart, we only have the container_id
    string from the DB — no live aiodocker Container object.  This proxy
    lets the existing inspect/stop/remove/monitor_container code work without
    modification.

    Args:
        container_id: Full Docker container ID string.
    """

    def __init__(self, container_id: str) -> None:
        self.id: str = container_id
