"""Unit tests for src/orchestrator/lifecycle.py.

Tests cover:
  - _update_status: plain status, with container_id
  - _store_phase2_params: JSON written to DB
  - _load_phase2_params: happy path, missing row raises RuntimeError
  - _load_production_params: happy path, missing row raises RuntimeError
  - MatchLifecycleManager.start_match_lifecycle: happy path, SKIP verdict, exception
  - MatchLifecycleManager.start_live_engine: happy path, exception
  - MatchLifecycleManager.emergency_freeze: publishes EMERGENCY_FREEZE event
  - MatchLifecycleManager._check_heartbeat: fresh, stale, missing
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.types import Phase2Result
from src.orchestrator.lifecycle import (
    HEARTBEAT_STALE_S,
    MatchLifecycleManager,
    _load_phase2_params,
    _load_production_params,
    _store_phase2_params,
    _update_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(*, fetchrow_return=None, execute_return=None):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock(return_value=execute_return or "UPDATE 1")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


def _make_redis():
    redis = MagicMock()
    redis.publish = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    return redis


def _make_container_manager():
    cm = MagicMock()
    cm.launch = AsyncMock(return_value=MagicMock(id="container-abc"))
    cm.inspect = AsyncMock(return_value={"State": {"Status": "exited", "ExitCode": 0}})
    cm.stop = AsyncMock()
    cm.remove = AsyncMock()
    return cm


def _make_match_row(
    match_id="gs-001",
    league_id=1204,
    odds_api_event_id="EVT-001",
):
    row = {
        "match_id": match_id,
        "league_id": league_id,
        "odds_api_event_id": odds_api_event_id,
        "kalshi_tickers": ["SOCCER-EPL-ARS-CHE-20260326-YES"],
    }
    return row


def _make_phase2_result(verdict="GO"):
    return Phase2Result(
        a_H=1.2, a_A=0.9, C_time=1.05, verdict=verdict, warning=None
    )


def _make_manager(pool, redis=None, container_manager=None):
    return MatchLifecycleManager(
        db_pool=pool,
        redis=redis or _make_redis(),
        container_manager=container_manager or _make_container_manager(),
        goalserve_client=MagicMock(),
        odds_client=None,
    )


# ---------------------------------------------------------------------------
# _update_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_status_plain() -> None:
    pool, conn = _make_pool()
    await _update_status(pool, "gs-001", "PHASE2_RUNNING")
    conn.execute.assert_called_once()
    sql = conn.execute.call_args.args[0]
    assert "status" in sql.lower()
    assert "PHASE2_RUNNING" in conn.execute.call_args.args


@pytest.mark.asyncio
async def test_update_status_with_container_id() -> None:
    pool, conn = _make_pool()
    await _update_status(pool, "gs-001", "PHASE3_RUNNING", container_id="ctr-123")
    conn.execute.assert_called_once()
    assert "ctr-123" in conn.execute.call_args.args


# ---------------------------------------------------------------------------
# _store_phase2_params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_phase2_params_writes_json() -> None:
    pool, conn = _make_pool()
    result = _make_phase2_result()
    await _store_phase2_params(pool, "gs-001", result)
    conn.execute.assert_called_once()
    # The payload JSON is passed as a positional arg
    args = conn.execute.call_args.args
    payload_arg = args[2]  # sql, match_id, payload
    data = json.loads(payload_arg)
    assert data["a_H"] == pytest.approx(1.2)
    assert data["verdict"] == "GO"


# ---------------------------------------------------------------------------
# _load_phase2_params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_phase2_params_happy_path() -> None:
    payload = json.dumps({"a_H": 1.2, "a_A": 0.9, "C_time": 1.05, "verdict": "GO"})
    pool, _ = _make_pool(fetchrow_return={"phase2_params": payload})
    result = await _load_phase2_params(pool, "gs-001")
    assert result.a_H == pytest.approx(1.2)
    assert result.verdict == "GO"


@pytest.mark.asyncio
async def test_load_phase2_params_missing_raises() -> None:
    pool, _ = _make_pool(fetchrow_return=None)
    with pytest.raises(RuntimeError, match="No phase2_params"):
        await _load_phase2_params(pool, "gs-001")


@pytest.mark.asyncio
async def test_load_phase2_params_null_column_raises() -> None:
    pool, _ = _make_pool(fetchrow_return={"phase2_params": None})
    with pytest.raises(RuntimeError, match="No phase2_params"):
        await _load_phase2_params(pool, "gs-001")


# ---------------------------------------------------------------------------
# _load_production_params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_production_params_happy_path() -> None:
    row = {
        "version": 3,
        "params": json.dumps({"b": [1.0]}),
        "xgb_model_path": "s3://bucket/model.xgb",
        "feature_mask": json.dumps(["feat_a", "feat_b"]),
        "median_values": json.dumps({"feat_a": 0.5}),
        "is_active": True,
        "created_at": None,
    }
    pool, _ = _make_pool(fetchrow_return=row)
    data = await _load_production_params(pool)
    assert data["version"] == 3
    assert data["params"] == {"b": [1.0]}
    assert data["feature_mask"] == ["feat_a", "feat_b"]


@pytest.mark.asyncio
async def test_load_production_params_missing_raises() -> None:
    pool, _ = _make_pool(fetchrow_return=None)
    with pytest.raises(RuntimeError, match="No active production_params"):
        await _load_production_params(pool)


# ---------------------------------------------------------------------------
# MatchLifecycleManager.start_match_lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_match_lifecycle_happy_path() -> None:
    """Phase 2 returns GO → status updated to PHASE2_DONE."""
    pool, conn = _make_pool()
    redis = _make_redis()
    manager = _make_manager(pool, redis)
    match = _make_match_row()

    phase2_result = _make_phase2_result(verdict="GO")
    prod_params_row = {
        "version": 1,
        "params": json.dumps({}),
        "xgb_model_path": "model.xgb",
        "feature_mask": json.dumps([]),
        "median_values": json.dumps({}),
        "is_active": True,
        "created_at": None,
    }

    with (
        patch(
            "src.orchestrator.lifecycle._load_production_params",
            new=AsyncMock(return_value=dict(prod_params_row)),
        ),
        patch(
            "src.prematch.pipeline.run_phase2",
            new=AsyncMock(return_value=(phase2_result, MagicMock())),
        ),
    ):
        await manager.start_match_lifecycle(match)

    # At least two execute calls: PHASE2_RUNNING + PHASE2_DONE + store params
    assert conn.execute.call_count >= 2
    statuses = [call.args[2] for call in conn.execute.call_args_list if len(call.args) >= 3]
    assert "PHASE2_RUNNING" in statuses
    assert "PHASE2_DONE" in statuses


@pytest.mark.asyncio
async def test_start_match_lifecycle_skip_verdict() -> None:
    """Phase 2 returns SKIP → status set to SKIPPED, not PHASE2_DONE."""
    pool, conn = _make_pool()
    manager = _make_manager(pool)
    match = _make_match_row()

    phase2_result = _make_phase2_result(verdict="SKIP")
    prod_params_row = {
        "version": 1,
        "params": "{}",
        "xgb_model_path": "model.xgb",
        "feature_mask": "[]",
        "median_values": "{}",
        "is_active": True,
        "created_at": None,
    }

    with (
        patch(
            "src.orchestrator.lifecycle._load_production_params",
            new=AsyncMock(return_value=prod_params_row),
        ),
        patch(
            "src.prematch.pipeline.run_phase2",
            new=AsyncMock(return_value=(phase2_result, MagicMock())),
        ),
    ):
        await manager.start_match_lifecycle(match)

    statuses = [call.args[2] for call in conn.execute.call_args_list if len(call.args) >= 3]
    assert "SKIPPED" in statuses
    assert "PHASE2_DONE" not in statuses


@pytest.mark.asyncio
async def test_start_match_lifecycle_exception_sets_failed() -> None:
    """Unhandled exception → FAILED status, emergency_freeze called."""
    pool, conn = _make_pool()
    redis = _make_redis()
    manager = _make_manager(pool, redis)
    match = _make_match_row()

    with patch(
        "src.orchestrator.lifecycle._load_production_params",
        new=AsyncMock(side_effect=RuntimeError("DB down")),
    ):
        await manager.start_match_lifecycle(match)

    statuses = [call.args[2] for call in conn.execute.call_args_list if len(call.args) >= 3]
    assert "FAILED" in statuses
    redis.publish.assert_called_once()


# ---------------------------------------------------------------------------
# MatchLifecycleManager.start_live_engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_live_engine_happy_path() -> None:
    """Launches container, updates to PHASE3_RUNNING, spawns monitor task."""
    pool, conn = _make_pool(
        fetchrow_return={
            "phase2_params": json.dumps(
                {"a_H": 1.2, "a_A": 0.9, "C_time": 1.05, "verdict": "GO"}
            )
        }
    )
    redis = _make_redis()
    cm = _make_container_manager()
    manager = _make_manager(pool, redis, cm)
    match = _make_match_row()

    # Patch asyncio.create_task to avoid actual background task
    with patch("asyncio.create_task") as mock_create_task:
        await manager.start_live_engine(match)

    statuses = [call.args[2] for call in conn.execute.call_args_list if len(call.args) >= 3]
    assert "PHASE3_RUNNING" in statuses
    cm.launch.assert_called_once()
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_start_live_engine_launch_failure() -> None:
    """Container launch failure → FAILED status, freeze published."""
    pool, conn = _make_pool(
        fetchrow_return={
            "phase2_params": json.dumps(
                {"a_H": 1.2, "a_A": 0.9, "C_time": 1.05, "verdict": "GO"}
            )
        }
    )
    redis = _make_redis()
    cm = _make_container_manager()
    cm.launch = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
    manager = _make_manager(pool, redis, cm)
    match = _make_match_row()

    await manager.start_live_engine(match)

    statuses = [call.args[2] for call in conn.execute.call_args_list if len(call.args) >= 3]
    assert "FAILED" in statuses
    redis.publish.assert_called_once()


# ---------------------------------------------------------------------------
# MatchLifecycleManager.emergency_freeze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_freeze_publishes_event() -> None:
    pool, _ = _make_pool()
    redis = _make_redis()
    manager = _make_manager(pool, redis)

    await manager.emergency_freeze("gs-001")

    redis.publish.assert_called_once()
    channel, raw = redis.publish.call_args.args
    assert channel == "match_events"
    payload = json.loads(raw)
    assert payload["type"] == "EMERGENCY_FREEZE"
    assert payload["match_id"] == "gs-001"


@pytest.mark.asyncio
async def test_emergency_freeze_suppresses_redis_error() -> None:
    pool, _ = _make_pool()
    redis = _make_redis()
    redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))
    manager = _make_manager(pool, redis)

    # Should not raise
    await manager.emergency_freeze("gs-001")


# ---------------------------------------------------------------------------
# MatchLifecycleManager._check_heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_heartbeat_fresh_no_freeze() -> None:
    pool, _ = _make_pool()
    redis = _make_redis()
    redis.get = AsyncMock(return_value=str(time.time()))  # just now
    manager = _make_manager(pool, redis)

    await manager._check_heartbeat("gs-001")

    redis.publish.assert_not_called()


@pytest.mark.asyncio
async def test_check_heartbeat_stale_triggers_freeze() -> None:
    pool, _ = _make_pool()
    redis = _make_redis()
    stale_ts = str(time.time() - HEARTBEAT_STALE_S - 10)
    redis.get = AsyncMock(return_value=stale_ts)
    manager = _make_manager(pool, redis)

    await manager._check_heartbeat("gs-001")

    redis.publish.assert_called_once()  # emergency_freeze → publish


@pytest.mark.asyncio
async def test_check_heartbeat_missing_no_freeze() -> None:
    pool, _ = _make_pool()
    redis = _make_redis()
    redis.get = AsyncMock(return_value=None)
    manager = _make_manager(pool, redis)

    await manager._check_heartbeat("gs-001")

    redis.publish.assert_not_called()
