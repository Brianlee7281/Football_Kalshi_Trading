"""Unit tests for src/common/redis_client.py.

Tests cover:
  - exposure_lock: acquires and releases the lock
  - publish_tick_to_dashboard: happy path payload, exception suppressed
  - publish_signal_to_dashboard: happy path payload, exception suppressed
  - subscribe_to_channels: calls subscribe on pubsub
  - unsubscribe_from_channels: calls unsubscribe + aclose; error suppressed
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.redis_client import (
    exposure_lock,
    publish_signal_to_dashboard,
    publish_tick_to_dashboard,
    subscribe_to_channels,
    unsubscribe_from_channels,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_redis() -> MagicMock:
    """Mock redis.asyncio.Redis with publish and lock."""
    redis = MagicMock()
    redis.publish = AsyncMock()

    # lock() returns a context manager
    lock_cm = MagicMock()
    lock_cm.__aenter__ = AsyncMock(return_value=None)
    lock_cm.__aexit__ = AsyncMock(return_value=False)
    redis.lock = MagicMock(return_value=lock_cm)

    # pubsub()
    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.aclose = AsyncMock()
    redis.pubsub = MagicMock(return_value=pubsub)

    return redis


def _make_model(
    *,
    match_id: str = "match-001",
    t: float = 10.5,
    engine_phase: str = "FIRST_HALF",
) -> MagicMock:
    model = MagicMock()
    model.match_id = match_id
    model.t = t
    model.engine_phase = engine_phase
    model.bet365_implied = {"home_win": 0.50}
    model.cooldown = False
    model.ob_freeze = False
    model.event_state = "IDLE"
    model.μ_H = 1.1
    model.μ_A = 0.9
    model.S = [1, 0]
    return model


def _make_signal(
    *,
    direction: str = "BUY_YES",
    EV: float = 0.04,
    P_cons: float = 0.58,
    P_kalshi: float = 0.54,
    alignment_status: str = "ALIGNED",
    kelly_multiplier: float = 0.8,
) -> MagicMock:
    signal = MagicMock()
    signal.direction = direction
    signal.EV = EV
    signal.P_cons = P_cons
    signal.P_kalshi = P_kalshi
    signal.alignment_status = alignment_status
    signal.kelly_multiplier = kelly_multiplier
    return signal


# ---------------------------------------------------------------------------
# exposure_lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exposure_lock_acquires_and_releases() -> None:
    """exposure_lock acquires redis.lock('exposure_lock') and releases on exit."""
    redis = _make_redis()

    acquired = False
    async with exposure_lock(redis):
        acquired = True

    assert acquired
    redis.lock.assert_called_once_with("exposure_lock", timeout=2.0)


@pytest.mark.asyncio
async def test_exposure_lock_custom_timeout() -> None:
    """Custom timeout is forwarded to redis.lock."""
    redis = _make_redis()
    async with exposure_lock(redis, timeout=5.0):
        pass
    redis.lock.assert_called_once_with("exposure_lock", timeout=5.0)


# ---------------------------------------------------------------------------
# publish_tick_to_dashboard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_tick_publishes_to_correct_channel() -> None:
    """Publishes to tick:{match_id} channel."""
    redis = _make_redis()
    model = _make_model(match_id="match-abc")

    await publish_tick_to_dashboard(
        redis, model, {"home_win": 0.55}, {"home_win": 0.002}, True
    )

    redis.publish.assert_called_once()
    channel = redis.publish.call_args.args[0]
    assert channel == "tick:match-abc"


@pytest.mark.asyncio
async def test_publish_tick_payload_structure() -> None:
    """Tick payload has required keys with correct values."""
    redis = _make_redis()
    model = _make_model(match_id="match-abc", t=15.0, engine_phase="FIRST_HALF")

    await publish_tick_to_dashboard(
        redis, model, {"home_win": 0.55}, {"home_win": 0.002}, False
    )

    raw = redis.publish.call_args.args[1]
    payload = json.loads(raw)

    assert payload["type"] == "tick"
    assert payload["match_id"] == "match-abc"
    assert payload["t"] == pytest.approx(15.0)
    assert payload["engine_phase"] == "FIRST_HALF"
    assert payload["P_true"] == {"home_win": 0.55}
    assert payload["sigma_MC"] == {"home_win": 0.002}
    assert payload["order_allowed"] is False
    assert "score" in payload
    assert "mu_H" in payload


@pytest.mark.asyncio
async def test_publish_tick_suppresses_redis_error() -> None:
    """Redis publish failure is suppressed (fire-and-forget)."""
    redis = _make_redis()
    redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))
    model = _make_model()

    # Should not raise
    await publish_tick_to_dashboard(redis, model, {}, {}, True)


@pytest.mark.asyncio
async def test_publish_tick_sigma_mc_key_name() -> None:
    """JSON key is 'sigma_MC' (not Greek σ) to match TypeScript types."""
    redis = _make_redis()
    model = _make_model()

    await publish_tick_to_dashboard(redis, model, {}, {"home_win": 0.001}, True)
    raw = redis.publish.call_args.args[1]
    payload = json.loads(raw)
    assert "sigma_MC" in payload
    assert "σ_MC" not in payload


# ---------------------------------------------------------------------------
# publish_signal_to_dashboard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_signal_publishes_to_correct_channel() -> None:
    """Publishes to signal:{match_id} channel."""
    redis = _make_redis()
    model = _make_model(match_id="match-xyz")
    signal = _make_signal()

    await publish_signal_to_dashboard(redis, model, "SOCC-M1-YES", signal)

    channel = redis.publish.call_args.args[0]
    assert channel == "signal:match-xyz"


@pytest.mark.asyncio
async def test_publish_signal_payload_structure() -> None:
    """Signal payload contains all required keys."""
    redis = _make_redis()
    model = _make_model()
    signal = _make_signal(direction="BUY_NO", EV=0.06, P_kalshi=0.42)

    await publish_signal_to_dashboard(redis, model, "SOCC-M2-YES", signal)

    raw = redis.publish.call_args.args[1]
    payload = json.loads(raw)

    assert payload["type"] == "signal"
    assert payload["match_id"] == model.match_id
    assert payload["ticker"] == "SOCC-M2-YES"
    assert payload["direction"] == "BUY_NO"
    assert payload["EV"] == pytest.approx(0.06)
    assert payload["P_kalshi"] == pytest.approx(0.42)
    assert "timestamp" in payload
    assert "alignment" in payload
    assert "kelly_multiplier" in payload


@pytest.mark.asyncio
async def test_publish_signal_suppresses_redis_error() -> None:
    """Redis publish failure is suppressed (fire-and-forget)."""
    redis = _make_redis()
    redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))
    model = _make_model()
    signal = _make_signal()

    # Should not raise
    await publish_signal_to_dashboard(redis, model, "SOCC-M1-YES", signal)


# ---------------------------------------------------------------------------
# subscribe_to_channels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_to_channels_calls_subscribe() -> None:
    """subscribe_to_channels creates a pubsub and subscribes."""
    redis = _make_redis()
    channels = ["tick:match-001", "signal:match-001"]

    pubsub = await subscribe_to_channels(redis, channels)

    redis.pubsub.assert_called_once()
    pubsub.subscribe.assert_called_once_with(*channels)


@pytest.mark.asyncio
async def test_subscribe_returns_pubsub_object() -> None:
    """Returns the PubSub object for the caller to iterate."""
    redis = _make_redis()
    pubsub = await subscribe_to_channels(redis, ["tick:match-001"])
    assert pubsub is redis.pubsub.return_value


# ---------------------------------------------------------------------------
# unsubscribe_from_channels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsubscribe_calls_unsubscribe_and_aclose() -> None:
    """unsubscribe_from_channels calls unsubscribe and aclose."""
    redis = _make_redis()
    pubsub = redis.pubsub()
    channels = ["tick:match-001"]

    await unsubscribe_from_channels(pubsub, channels)

    pubsub.unsubscribe.assert_called_once_with(*channels)
    pubsub.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_unsubscribe_suppresses_error() -> None:
    """Errors during unsubscribe are suppressed."""
    redis = _make_redis()
    pubsub = redis.pubsub()
    pubsub.unsubscribe = AsyncMock(side_effect=ConnectionError("redis down"))

    # Should not raise
    await unsubscribe_from_channels(pubsub, ["tick:match-001"])
