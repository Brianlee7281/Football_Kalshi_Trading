"""Unit tests for match_engine components.

Tests cover:
  - MatchEngineConfig.from_env: happy path, missing required vars, invalid values
  - heartbeat_emitter: publishes to Redis, no-redis guard, final heartbeat
"""

from __future__ import annotations

import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.match_engine.config import MatchEngineConfig
from src.match_engine.heartbeat import heartbeat_emitter


# ---------------------------------------------------------------------------
# MatchEngineConfig.from_env
# ---------------------------------------------------------------------------


def _base_env() -> dict[str, str]:
    return {
        "MATCH_ID": "match-abc-123",
        "TRADING_MODE": "paper",
        "PARAM_VERSION": "7",
        "KALSHI_TICKERS": json.dumps(["SOCC-M1-YES", "SOCC-M2-YES"]),
        "DB_URL": "postgresql://trader:pass@postgres:5432/trading",
        "REDIS_URL": "redis://redis:6379",
        "KALSHI_API_KEY": "kalshi-key-abc",
        "ODDS_API_KEY": "odds-key-xyz",
        "GOALSERVE_API_KEY": "gs-key-123",
    }


def test_config_from_env_happy_path() -> None:
    """All required env vars present → config populated correctly."""
    with patch.dict(os.environ, _base_env(), clear=True):
        cfg = MatchEngineConfig.from_env()

    assert cfg.match_id == "match-abc-123"
    assert cfg.trading_mode == "paper"
    assert cfg.param_version == 7
    assert cfg.kalshi_tickers == ["SOCC-M1-YES", "SOCC-M2-YES"]
    assert cfg.db_url == "postgresql://trader:pass@postgres:5432/trading"
    assert cfg.redis_url == "redis://redis:6379"
    assert cfg.kalshi_api_key == "kalshi-key-abc"
    assert cfg.odds_api_key == "odds-key-xyz"
    assert cfg.goalserve_api_key == "gs-key-123"
    # defaults
    assert cfg.fee_rate == pytest.approx(0.07)
    assert cfg.z == pytest.approx(1.645)
    assert cfg.K_frac == pytest.approx(0.25)


def test_config_live_mode() -> None:
    """TRADING_MODE=live is accepted."""
    env = {**_base_env(), "TRADING_MODE": "live"}
    with patch.dict(os.environ, env, clear=True):
        cfg = MatchEngineConfig.from_env()
    assert cfg.trading_mode == "live"


def test_config_invalid_trading_mode() -> None:
    """Unknown TRADING_MODE raises ValueError."""
    env = {**_base_env(), "TRADING_MODE": "simulation"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="TRADING_MODE"):
            MatchEngineConfig.from_env()


def test_config_missing_match_id() -> None:
    """Missing MATCH_ID raises KeyError."""
    env = {k: v for k, v in _base_env().items() if k != "MATCH_ID"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(KeyError):
            MatchEngineConfig.from_env()


def test_config_invalid_param_version() -> None:
    """Non-integer PARAM_VERSION raises ValueError."""
    env = {**_base_env(), "PARAM_VERSION": "not-a-number"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError):
            MatchEngineConfig.from_env()


def test_config_empty_kalshi_tickers() -> None:
    """Empty KALSHI_TICKERS parses to empty list."""
    env = {**_base_env(), "KALSHI_TICKERS": "[]"}
    with patch.dict(os.environ, env, clear=True):
        cfg = MatchEngineConfig.from_env()
    assert cfg.kalshi_tickers == []


def test_config_custom_fee_rate_and_k_frac() -> None:
    """FEE_RATE and K_FRAC env overrides are respected."""
    env = {**_base_env(), "FEE_RATE": "0.05", "K_FRAC": "0.50", "Z_SCORE": "2.0"}
    with patch.dict(os.environ, env, clear=True):
        cfg = MatchEngineConfig.from_env()
    assert cfg.fee_rate == pytest.approx(0.05)
    assert cfg.K_frac == pytest.approx(0.50)
    assert cfg.z == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# heartbeat_emitter
# ---------------------------------------------------------------------------


def _model_with_redis(*, engine_phase: str = "FIRST_HALF") -> MagicMock:
    model = MagicMock()
    model.match_id = "match-001"
    model.engine_phase = engine_phase
    model.redis = MagicMock()
    model.redis.set = AsyncMock()
    return model


@pytest.mark.asyncio
async def test_heartbeat_emitter_no_redis_returns_immediately() -> None:
    """model.redis is None → returns without error, no Redis calls."""
    model = _model_with_redis()
    model.redis = None

    await heartbeat_emitter(model)  # should return immediately


@pytest.mark.asyncio
async def test_heartbeat_emitter_publishes_to_redis() -> None:
    """Emits heartbeat to Redis key heartbeat:{match_id}."""
    model = _model_with_redis(engine_phase="FIRST_HALF")

    # After one sleep tick, set FINISHED so the loop exits
    call_count = {"n": 0}

    async def _fake_sleep(_: float) -> None:
        call_count["n"] += 1
        model.engine_phase = "FINISHED"

    with patch("src.match_engine.heartbeat.asyncio.sleep", side_effect=_fake_sleep):
        await heartbeat_emitter(model)

    # Redis.set should have been called at least once
    model.redis.set.assert_called()
    call_args = model.redis.set.call_args_list[0]
    key = call_args.args[0]
    assert key == "heartbeat:match-001"


@pytest.mark.asyncio
async def test_heartbeat_emitter_already_finished() -> None:
    """engine_phase=FINISHED → no loop iterations, just final heartbeat."""
    model = _model_with_redis(engine_phase="FINISHED")

    await heartbeat_emitter(model)

    # Only the final heartbeat call (from _emit_final_heartbeat)
    assert model.redis.set.call_count <= 1


@pytest.mark.asyncio
async def test_heartbeat_emitter_suppresses_redis_error() -> None:
    """Redis error during heartbeat is suppressed (liveness, not critical path)."""
    model = _model_with_redis(engine_phase="FIRST_HALF")
    model.redis.set = AsyncMock(side_effect=ConnectionError("redis down"))

    async def _fake_sleep(_: float) -> None:
        model.engine_phase = "FINISHED"

    with patch("src.match_engine.heartbeat.asyncio.sleep", side_effect=_fake_sleep):
        await heartbeat_emitter(model)  # no exception raised


@pytest.mark.asyncio
async def test_heartbeat_key_format() -> None:
    """Heartbeat key includes the match_id."""
    model = _model_with_redis(engine_phase="FIRST_HALF")
    model.match_id = "soccer-epl-ars-che-20260301"

    async def _fake_sleep(_: float) -> None:
        model.engine_phase = "FINISHED"

    with patch("src.match_engine.heartbeat.asyncio.sleep", side_effect=_fake_sleep):
        await heartbeat_emitter(model)

    keys_set = [call.args[0] for call in model.redis.set.call_args_list]
    assert any("soccer-epl-ars-che-20260301" in k for k in keys_set)
