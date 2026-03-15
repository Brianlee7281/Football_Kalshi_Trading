"""Match lifecycle manager — SCHEDULED → PHASE2_RUNNING → PHASE2_DONE →
PHASE3_RUNNING → SETTLING → FINISHED → ARCHIVED.

The MatchLifecycleManager runs inside the Orchestrator process and drives one
match through its full lifecycle:

  Phase 2 (in-process):  run_phase2() calls src/prematch/pipeline.py
  Phase 3+4 (container): launch_match_container() delegates to ContainerManager
  Monitoring:            monitor_container() polls every 10s until exit

Concurrency: each match runs in its own asyncio task.  The orchestrator
creates one MatchLifecycleManager per Orchestrator instance (not per match);
start_match_lifecycle is the per-match entry point.

Reference: docs/orchestration.md Component 2: Orchestrator
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from src.common.logging import get_logger
from src.common.types import Phase2Result

logger = get_logger("lifecycle")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MATCH_DURATION: timedelta = timedelta(hours=9)
HEARTBEAT_STALE_S: float = 60.0
MONITOR_POLL_S: float = 10.0


# ---------------------------------------------------------------------------
# MatchLifecycleManager
# ---------------------------------------------------------------------------


class MatchLifecycleManager:
    """Drives a match through its lifecycle inside the Orchestrator process.

    Args:
        db_pool: asyncpg connection pool (shared with the Orchestrator).
        redis: Connected Redis client (used for heartbeat + freeze events).
        container_manager: ContainerManager instance (Task 6.6, typed Any).
        goalserve_client: GoalserveClient for Phase 2 data collection.
        odds_client: OddsApiClient for Phase 2 odds data (optional).
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis: Any,
        container_manager: Any,
        goalserve_client: Any,
        odds_client: Any = None,
    ) -> None:
        self._pool = db_pool
        self._redis = redis
        self._container_manager = container_manager
        self._goalserve = goalserve_client
        self._odds = odds_client

    # ------------------------------------------------------------------
    # Public entry points (called by TriggerExecutor)
    # ------------------------------------------------------------------

    async def start_match_lifecycle(self, match: Any) -> None:
        """Run the full lifecycle from SCHEDULED through ARCHIVED.

        Called by TriggerExecutor when phase2_trigger time is reached.
        Accepts a raw asyncpg Record or a MatchSchedule-like dict/object.

        Args:
            match: Row from match_schedule (asyncpg Record or MatchSchedule).
        """
        match_id: str = match["match_id"] if hasattr(match, "__getitem__") else str(match.match_id)

        logger.info("lifecycle_start", match_id=match_id)
        await _update_status(self._pool, match_id, "PHASE2_RUNNING")

        try:
            # ── Phase 2: in-process ──────────────────────────────────────
            phase2_result = await self._run_phase2(match_id, match)

            if phase2_result.verdict == "SKIP":
                await _update_status(self._pool, match_id, "SKIPPED")
                logger.info(
                    "lifecycle_skipped",
                    match_id=match_id,
                    warning=phase2_result.warning,
                )
                return

            await _update_status(self._pool, match_id, "PHASE2_DONE")
            await _store_phase2_params(self._pool, match_id, phase2_result)
            logger.info(
                "phase2_complete",
                match_id=match_id,
                verdict=phase2_result.verdict,
                a_H=round(phase2_result.a_H, 4),
                a_A=round(phase2_result.a_A, 4),
            )

            # Phase 3+4 container is launched by start_live_engine() when
            # phase3_trigger is reached (called separately by TriggerExecutor).

        except Exception as exc:  # noqa: BLE001
            await _update_status(self._pool, match_id, "FAILED", error=str(exc))
            logger.error(
                "lifecycle_failed",
                match_id=match_id,
                error=str(exc),
            )
            await self.emergency_freeze(match_id)

    async def start_live_engine(self, match: Any) -> None:
        """Launch Phase 3+4 container when phase3_trigger is reached.

        Called by TriggerExecutor when a PHASE2_DONE match reaches
        phase3_trigger time.

        Args:
            match: Row from match_schedule (asyncpg Record or MatchSchedule).
        """
        match_id: str = match["match_id"] if hasattr(match, "__getitem__") else str(match.match_id)

        logger.info("live_engine_start", match_id=match_id)

        try:
            phase2_result = await _load_phase2_params(self._pool, match_id)

            container = await self._launch_match_container(match_id, match, phase2_result)
            container_id: str = getattr(container, "id", str(container))

            await _update_status(
                self._pool, match_id, "PHASE3_RUNNING", container_id=container_id
            )
            logger.info(
                "container_launched",
                match_id=match_id,
                container_id=container_id,
            )

            # Monitor in the background so TriggerExecutor is not blocked.
            asyncio.create_task(
                self._monitor_container(match_id, container),
                name=f"monitor-{match_id}",
            )

        except Exception as exc:  # noqa: BLE001
            await _update_status(self._pool, match_id, "FAILED", error=str(exc))
            logger.error(
                "live_engine_start_failed",
                match_id=match_id,
                error=str(exc),
            )
            await self.emergency_freeze(match_id)

    # ------------------------------------------------------------------
    # Phase 2 — in-process execution
    # ------------------------------------------------------------------

    async def _run_phase2(self, match_id: str, match: Any) -> Phase2Result:
        """Run the Phase 2 pipeline inside the Orchestrator process.

        Loads production params from DB, then calls
        ``src.prematch.pipeline.run_phase2``.

        Args:
            match_id: Goalserve match ID.
            match: Row from match_schedule (asyncpg Record or MatchSchedule).

        Returns:
            Phase2Result with verdict and back-solved intensities.
        """
        from src.prematch.pipeline import run_phase2  # local import avoids circular deps

        params = await _load_production_params(self._pool)

        league_id: int = int(
            match["league_id"] if hasattr(match, "__getitem__") else match.league_id
        )
        odds_event_id: str | None = (
            match.get("odds_api_event_id")
            if hasattr(match, "get")
            else getattr(match, "odds_api_event_id", None)
        )

        phase2_result, _ = await run_phase2(
            gs_client=self._goalserve,
            match_id=match_id,
            league_id=league_id,
            params=params.get("params", {}),
            odds_client=self._odds,
            odds_event_id=odds_event_id,
            xgb_model_path=params.get("xgb_model_path"),
            feature_mask=params.get("feature_mask"),
            median_values=params.get("median_values"),
        )
        return phase2_result

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    async def _launch_match_container(
        self,
        match_id: str,
        match: Any,
        phase2_result: Phase2Result,
    ) -> Any:
        """Delegate container launch to ContainerManager (Task 6.6).

        Args:
            match_id: Goalserve match ID.
            match: Row from match_schedule.
            phase2_result: Phase 2 back-solve results.

        Returns:
            Container object (type defined by ContainerManager).
        """
        return await self._container_manager.launch(
            match_id=match_id,
            match=match,
            phase2_result=phase2_result,
        )

    async def _monitor_container(self, match_id: str, container: Any) -> None:
        """Poll container status every 10s until it exits or times out.

        On clean exit (code 0):  status → SETTLING → FINISHED → ARCHIVED.
        On crash (non-zero):     status → FAILED, emergency_freeze called.
        On heartbeat stale:      emergency_freeze called (container kept running).
        On 9h timeout:           container stopped, status → FAILED.
        On inspect 404 (3x):    container gone, status → FAILED.

        Args:
            match_id: Goalserve match ID.
            container: Container object returned by launch.
        """
        start_time = datetime.now(UTC)
        inspect_failures: int = 0
        max_inspect_failures: int = 3

        while True:
            await asyncio.sleep(MONITOR_POLL_S)

            # ── Check container state ────────────────────────────────────
            try:
                status_info: dict[str, Any] = await self._container_manager.inspect(container)
                state = status_info.get("State", {})
                inspect_failures = 0  # reset on success
            except Exception as exc:  # noqa: BLE001
                inspect_failures += 1
                logger.warning(
                    "container_inspect_failed",
                    match_id=match_id,
                    error=str(exc),
                    attempt=inspect_failures,
                    max_attempts=max_inspect_failures,
                )
                if inspect_failures >= max_inspect_failures:
                    # Capture crash output before giving up — container may
                    # still exist as a stopped container even if inspect fails.
                    await _log_container_output(
                        self._container_manager, match_id, container,
                    )
                    logger.error(
                        "container_inspect_gave_up",
                        match_id=match_id,
                        attempts=inspect_failures,
                    )
                    await _update_status(self._pool, match_id, "FAILED")
                    await self.emergency_freeze(match_id)
                    break
                continue

            if state.get("Status") == "exited":
                exit_code: int = int(state.get("ExitCode", 1))
                if exit_code == 0:
                    logger.info("container_exited_clean", match_id=match_id)
                    await _update_status(self._pool, match_id, "SETTLING")
                    await _settle_match(self._pool, match_id)
                    await _update_status(self._pool, match_id, "FINISHED")
                else:
                    # Capture crash output BEFORE container is removed
                    await _log_container_output(
                        self._container_manager, match_id, container,
                    )
                    logger.error(
                        "container_exited_error",
                        match_id=match_id,
                        exit_code=exit_code,
                    )
                    await _update_status(self._pool, match_id, "FAILED")
                    await self.emergency_freeze(match_id)
                break

            # ── Safety timeout ───────────────────────────────────────────
            elapsed = datetime.now(UTC) - start_time
            if elapsed > MAX_MATCH_DURATION:
                logger.error(
                    "container_max_duration_exceeded",
                    match_id=match_id,
                    elapsed_h=elapsed.total_seconds() / 3600,
                )
                try:
                    await self._container_manager.stop(container)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "container_stop_failed",
                        match_id=match_id,
                        error=str(exc),
                    )
                await _update_status(self._pool, match_id, "FAILED")
                await self.emergency_freeze(match_id)
                break

            # ── Heartbeat staleness ──────────────────────────────────────
            await self._check_heartbeat(match_id)

        # ── Cleanup (always runs) ────────────────────────────────────────
        try:
            await self._container_manager.remove(container)
        except Exception as exc:  # noqa: BLE001
            logger.warning("container_remove_failed", match_id=match_id, error=str(exc))

        await _archive_logs(self._pool, match_id, getattr(container, "id", str(container)))
        await _update_status(self._pool, match_id, "ARCHIVED")
        logger.info("lifecycle_archived", match_id=match_id)

    async def _check_heartbeat(self, match_id: str) -> None:
        """Check Redis heartbeat; call emergency_freeze if stale.

        Args:
            match_id: Goalserve match ID.
        """
        try:
            raw = await self._redis.get(f"heartbeat:{match_id}")
            if raw is not None:
                age = time.time() - float(raw)
                if age > HEARTBEAT_STALE_S:
                    logger.error(
                        "heartbeat_stale",
                        match_id=match_id,
                        age_s=round(age, 1),
                    )
                    await self.emergency_freeze(match_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "heartbeat_check_failed",
                match_id=match_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Emergency procedures
    # ------------------------------------------------------------------

    async def emergency_freeze(self, match_id: str) -> None:
        """Freeze all activity for a match without closing positions.

        Best-effort: publishes a EMERGENCY_FREEZE event to Redis so other
        components can react.  Positions remain open for manual review.

        Args:
            match_id: Goalserve match ID.
        """
        logger.critical("emergency_freeze", match_id=match_id)

        # Publish freeze event to Redis (best-effort)
        try:
            await self._redis.publish(
                "match_events",
                json.dumps(
                    {
                        "type": "EMERGENCY_FREEZE",
                        "match_id": match_id,
                        "timestamp": time.time(),
                    }
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "emergency_freeze_publish_failed",
                match_id=match_id,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# DB helpers (module-level, no ORM)
# ---------------------------------------------------------------------------


async def _update_status(
    pool: asyncpg.Pool,
    match_id: str,
    status: str,
    *,
    container_id: str | None = None,
    error: str | None = None,
) -> None:
    """UPDATE match_schedule.status (and optionally container_id / error).

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match ID.
        status: New status string.
        container_id: Docker container ID (set when launching Phase 3).
        error: Error message (logged; not stored in DB schema currently).
    """
    if error:
        logger.error("match_status_error", match_id=match_id, status=status, error=error)

    if container_id is not None:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE match_schedule
                SET status = $2, container_id = $3, updated_at = NOW()
                WHERE match_id = $1
                """,
                match_id,
                status,
                container_id,
            )
    else:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE match_schedule
                SET status = $2, updated_at = NOW()
                WHERE match_id = $1
                """,
                match_id,
                status,
            )

    logger.info("match_status_updated", match_id=match_id, status=status)


async def _store_phase2_params(
    pool: asyncpg.Pool,
    match_id: str,
    result: Phase2Result,
) -> None:
    """Persist Phase 2 results to match_schedule.phase2_params JSONB column.

    The column stores a_H, a_A, C_time and the verdict so the container
    can be launched with these values as environment variables.

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match ID.
        result: Phase2Result from the pipeline.
    """
    payload = json.dumps(
        {
            "a_H": result.a_H,
            "a_A": result.a_A,
            "C_time": result.C_time,
            "verdict": result.verdict,
            "warning": result.warning,
        }
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE match_schedule
            SET phase2_params = $2::jsonb, updated_at = NOW()
            WHERE match_id = $1
            """,
            match_id,
            payload,
        )


async def _load_phase2_params(
    pool: asyncpg.Pool,
    match_id: str,
) -> Phase2Result:
    """Load Phase 2 results from match_schedule.phase2_params.

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match ID.

    Returns:
        Phase2Result reconstructed from stored JSON.

    Raises:
        RuntimeError: If no phase2_params row is found for the match.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT phase2_params FROM match_schedule WHERE match_id = $1",
            match_id,
        )
    if row is None or row["phase2_params"] is None:
        raise RuntimeError(f"No phase2_params found for match_id={match_id!r}")

    data: dict[str, Any] = (
        json.loads(row["phase2_params"])
        if isinstance(row["phase2_params"], str)
        else row["phase2_params"]
    )
    return Phase2Result(
        a_H=float(data["a_H"]),
        a_A=float(data["a_A"]),
        C_time=float(data["C_time"]),
        verdict=str(data["verdict"]),
        warning=data.get("warning"),
    )


async def _load_production_params(pool: asyncpg.Pool) -> dict[str, Any]:
    """Load the active Phase 1 production params from the DB.

    Args:
        pool: asyncpg connection pool.

    Returns:
        Dict with keys: version, params, xgb_model_path, feature_mask,
        median_values, is_active, created_at.

    Raises:
        RuntimeError: If no active params row is found.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM production_params WHERE is_active = TRUE LIMIT 1"
        )
    if row is None:
        raise RuntimeError("No active production_params found in DB")

    result: dict[str, Any] = dict(row)
    # Decode JSONB fields that asyncpg may return as strings
    for key in ("params", "feature_mask", "median_values"):
        val = result.get(key)
        if isinstance(val, str):
            result[key] = json.loads(val)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _settle_match(pool: asyncpg.Pool, match_id: str) -> None:
    """Post-match analytics placeholder — settlement is done inside the container.

    The container polls Kalshi for resolution and writes SETTLED positions
    before exiting with code 0.  This function is a hook for any additional
    orchestrator-level post-match work (e.g. bankroll snapshot).

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match ID.
    """
    logger.info("settle_match_analytics", match_id=match_id)


async def _log_container_output(
    container_manager: Any,
    match_id: str,
    container: Any,
    *,
    tail: int = 50,
) -> None:
    """Fetch and log the last N lines of container stdout/stderr.

    Called before container removal on crash so the error output is
    captured in orchestrator logs for debugging.

    Args:
        container_manager: ContainerManager instance.
        match_id: Goalserve match ID.
        container: Container object (has .id attribute).
        tail: Number of lines to fetch (default 50).
    """
    try:
        import aiodocker

        container_id = getattr(container, "id", str(container))
        async with aiodocker.Docker() as docker:
            c = await docker.containers.get(container_id)
            log_lines: list[str] | bytes = await c.log(
                stdout=True, stderr=True, tail=tail,
            )

        if isinstance(log_lines, list):
            output = "".join(log_lines)
        elif isinstance(log_lines, bytes):
            output = log_lines.decode("utf-8", errors="replace")
        else:
            output = str(log_lines)

        logger.error(
            "container_crash_output",
            match_id=match_id,
            container_id=container_id[:12],
            output=output[-3000:],  # cap at 3000 chars
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "container_crash_output_fetch_failed",
            match_id=match_id,
            error=str(exc),
        )


async def _archive_logs(pool: asyncpg.Pool, match_id: str, container_id: str) -> None:
    """Archive container logs (placeholder for T6.6 ContainerManager integration).

    Args:
        pool: asyncpg connection pool.
        match_id: Goalserve match ID.
        container_id: Docker container ID.
    """
    logger.info("archive_logs", match_id=match_id, container_id=container_id)
