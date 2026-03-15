"""Unit tests for src/orchestrator/recovery.py.

Tests cover:
  - recover_orchestrator_state: counts returned for each status bucket
  - PHASE2_RUNNING: task spawned via start_match_lifecycle
  - PHASE2_DONE (trigger past): task spawned via start_live_engine
  - PHASE2_DONE (trigger future): no task spawned
  - PHASE3_RUNNING (alive): monitor task resumed
  - PHASE3_RUNNING (dead): FAILED status + freeze
  - PHASE3_RUNNING (no container_id): freeze called
  - SCHEDULED (trigger past): task spawned via start_match_lifecycle
  - SCHEDULED (trigger future): no task spawned
  - _is_container_alive: running→True, exited→False, error→False
  - _ensure_utc: naive datetime gains UTC, aware datetime unchanged
  - row-level error doesn't abort entire recovery loop
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.recovery import (
    _ContainerProxy,
    _ensure_utc,
    _is_container_alive,
    recover_orchestrator_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 26, 14, 0, tzinfo=UTC)
_PAST = _NOW - timedelta(hours=1)
_FUTURE = _NOW + timedelta(hours=1)


def _make_pool(rows: list[dict]):
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[_DictRow(r) for r in rows])
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


class _DictRow:
    """Behaves like both a dict and an asyncpg Record for test purposes."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):
        return self._data[key]

    def get(self, key: str, default=None):
        return self._data.get(key, default)


def _make_lifecycle():
    lc = MagicMock()
    lc.start_match_lifecycle = AsyncMock()
    lc.start_live_engine = AsyncMock()
    lc.emergency_freeze = AsyncMock()
    lc._monitor_container = AsyncMock()
    lc._pool = MagicMock()
    return lc


def _make_cm(alive: bool = True):
    cm = MagicMock()
    state = "running" if alive else "exited"
    cm.inspect = AsyncMock(
        return_value={"State": {"Status": state, "ExitCode": 0 if alive else 1}}
    )
    return cm


def _row(
    match_id: str = "gs-001",
    status: str = "SCHEDULED",
    container_id: str | None = None,
    phase2_trigger: datetime = _PAST,
    phase3_trigger: datetime = _PAST,
    kickoff_utc: datetime = _NOW,
) -> dict:
    return {
        "match_id": match_id,
        "status": status,
        "container_id": container_id,
        "phase2_trigger": phase2_trigger,
        "phase3_trigger": phase3_trigger,
        "kickoff_utc": kickoff_utc,
    }


# ---------------------------------------------------------------------------
# recover_orchestrator_state — counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_empty_returns_zero_counts() -> None:
    pool = _make_pool([])
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result == {
        "phase2_rerun": 0,
        "phase2_done_launched": 0,
        "phase3_resumed": 0,
        "phase3_failed": 0,
        "scheduled_triggered": 0,
        "auto_finished": 0,
    }


@pytest.mark.asyncio
async def test_recovery_phase2_running_count() -> None:
    rows = [_row(status="PHASE2_RUNNING")]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase2_rerun"] == 1


@pytest.mark.asyncio
async def test_recovery_phase2_done_trigger_past_count() -> None:
    rows = [_row(status="PHASE2_DONE", phase3_trigger=_PAST)]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase2_done_launched"] == 1


@pytest.mark.asyncio
async def test_recovery_phase2_done_trigger_future_no_launch() -> None:
    rows = [_row(status="PHASE2_DONE", phase3_trigger=_FUTURE)]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task") as mock_task:
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase2_done_launched"] == 0
    mock_task.assert_not_called()


@pytest.mark.asyncio
async def test_recovery_phase3_running_alive_resumed() -> None:
    rows = [_row(status="PHASE3_RUNNING", container_id="abc123")]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm(alive=True)
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase3_resumed"] == 1
    assert result["phase3_failed"] == 0


@pytest.mark.asyncio
async def test_recovery_phase3_running_dead_fails() -> None:
    rows = [_row(status="PHASE3_RUNNING", container_id="abc123")]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm(alive=False)
    with patch("src.orchestrator.lifecycle._update_status", new=AsyncMock()):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase3_failed"] == 1
    assert result["phase3_resumed"] == 0
    lc.emergency_freeze.assert_called_once_with("gs-001")


@pytest.mark.asyncio
async def test_recovery_phase3_no_container_id_fails() -> None:
    rows = [_row(status="PHASE3_RUNNING", container_id=None)]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase3_failed"] == 1
    lc.emergency_freeze.assert_called_once_with("gs-001")


@pytest.mark.asyncio
async def test_recovery_scheduled_trigger_past() -> None:
    rows = [_row(status="SCHEDULED", phase2_trigger=_PAST)]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["scheduled_triggered"] == 1


@pytest.mark.asyncio
async def test_recovery_scheduled_trigger_future_no_launch() -> None:
    rows = [_row(status="SCHEDULED", phase2_trigger=_FUTURE)]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()
    with patch("asyncio.create_task") as mock_task:
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["scheduled_triggered"] == 0
    mock_task.assert_not_called()


# ---------------------------------------------------------------------------
# recover_orchestrator_state — multiple matches mixed statuses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_multiple_matches() -> None:
    rows = [
        _row(match_id="gs-001", status="PHASE2_RUNNING"),
        _row(match_id="gs-002", status="PHASE3_RUNNING", container_id="ctr-abc"),
        _row(match_id="gs-003", status="SCHEDULED", phase2_trigger=_PAST),
    ]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm(alive=True)
    with patch("asyncio.create_task"):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)
    assert result["phase2_rerun"] == 1
    assert result["phase3_resumed"] == 1
    assert result["scheduled_triggered"] == 1


# ---------------------------------------------------------------------------
# recover_orchestrator_state — error in one match doesn't abort loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_error_in_one_match_continues() -> None:
    rows = [
        _row(match_id="gs-bad", status="PHASE2_RUNNING"),
        _row(match_id="gs-ok", status="SCHEDULED", phase2_trigger=_PAST),
    ]
    pool = _make_pool(rows)
    lc = _make_lifecycle()
    cm = _make_cm()

    call_count = {"n": 0}

    def create_task_side_effect(coro, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Close the coroutine to avoid RuntimeWarning
            coro.close()
            raise RuntimeError("boom")
        # Let second one through
        task = MagicMock()
        coro.close()
        return task

    with patch("asyncio.create_task", side_effect=create_task_side_effect):
        result = await recover_orchestrator_state(pool, lc, cm, now=_NOW)

    # Second match still processed
    assert result["scheduled_triggered"] == 1


# ---------------------------------------------------------------------------
# _is_container_alive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_container_alive_running() -> None:
    cm = _make_cm(alive=True)
    assert await _is_container_alive(cm, "abc123") is True


@pytest.mark.asyncio
async def test_is_container_alive_exited() -> None:
    cm = _make_cm(alive=False)
    assert await _is_container_alive(cm, "abc123") is False


@pytest.mark.asyncio
async def test_is_container_alive_error_returns_false() -> None:
    cm = MagicMock()
    cm.inspect = AsyncMock(side_effect=RuntimeError("Docker error"))
    assert await _is_container_alive(cm, "abc123") is False


# ---------------------------------------------------------------------------
# _ensure_utc
# ---------------------------------------------------------------------------


def test_ensure_utc_aware_unchanged() -> None:
    dt = datetime(2026, 3, 26, 15, 0, tzinfo=UTC)
    result = _ensure_utc(dt)
    assert result.tzinfo is not None
    assert result == dt


def test_ensure_utc_naive_gets_utc() -> None:
    naive = datetime(2026, 3, 26, 15, 0)
    result = _ensure_utc(naive)
    assert result.tzinfo is UTC


def test_ensure_utc_string_fallback() -> None:
    result = _ensure_utc("2026-03-26T15:00:00")
    assert isinstance(result, datetime)
    assert result.tzinfo is UTC


# ---------------------------------------------------------------------------
# _ContainerProxy
# ---------------------------------------------------------------------------


def test_container_proxy_id() -> None:
    proxy = _ContainerProxy("full-container-id-here")
    assert proxy.id == "full-container-id-here"
