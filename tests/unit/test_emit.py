"""Unit tests for emit_to_phase4 — queue push + stale replacement.

Tests:
  - emit_to_phase4 puts TickData on the queue with correct P_true/sigma_MC
  - queue full → old tick replaced (always-fresh guarantee)
  - order_allowed flag is forwarded correctly

Reference: docs/phase3.md §emit_to_phase4, patterns.md #1
"""

from __future__ import annotations

import pytest

from src.engine.emit import emit_to_phase4
from src.engine.model import FIRST_HALF, LiveFootballQuantModel


def _make_model() -> LiveFootballQuantModel:
    model = LiveFootballQuantModel()
    model.engine_phase = FIRST_HALF
    model.t = 30.0
    return model


_P_TRUE = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20, "over_25": 0.65}
_SIGMA_MC = {"home_win": 0.005, "draw": 0.004, "away_win": 0.003, "over_25": 0.006}


# ---------------------------------------------------------------------------
# emit_to_phase4 puts TickData on queue
# ---------------------------------------------------------------------------


def test_emit_puts_tick_on_queue() -> None:
    """emit_to_phase4 puts a TickData onto model.phase4_queue."""
    model = _make_model()
    assert model.phase4_queue.empty()

    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=True, model=model)

    assert not model.phase4_queue.empty()


def test_emit_tick_data_p_true() -> None:
    """TickData.P_true matches the emitted dict."""
    model = _make_model()
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=True, model=model)

    tick = model.phase4_queue.get_nowait()
    assert tick.P_true == _P_TRUE


def test_emit_tick_data_sigma_mc() -> None:
    """TickData.sigma_MC matches the emitted dict."""
    model = _make_model()
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=True, model=model)

    tick = model.phase4_queue.get_nowait()
    assert tick.sigma_MC == _SIGMA_MC


def test_emit_tick_data_order_allowed_true() -> None:
    """TickData.order_allowed=True when no freeze or cooldown."""
    model = _make_model()
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=True, model=model)

    tick = model.phase4_queue.get_nowait()
    assert tick.order_allowed is True


def test_emit_tick_data_order_allowed_false() -> None:
    """TickData.order_allowed=False when explicitly passed False."""
    model = _make_model()
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=False, model=model)

    tick = model.phase4_queue.get_nowait()
    assert tick.order_allowed is False


# ---------------------------------------------------------------------------
# Queue full → old tick replaced (always-fresh guarantee)
# ---------------------------------------------------------------------------


def test_queue_full_old_tick_replaced() -> None:
    """When queue is full, old tick is discarded and new tick takes its place."""
    model = _make_model()

    # Emit first tick
    p_true_old = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30, "over_25": 0.45}
    emit_to_phase4(p_true_old, _SIGMA_MC, order_allowed=True, model=model)
    assert model.phase4_queue.full()

    # Emit second tick (queue is full — old must be replaced)
    p_true_new = {"home_win": 0.60, "draw": 0.25, "away_win": 0.15, "over_25": 0.70}
    emit_to_phase4(p_true_new, _SIGMA_MC, order_allowed=True, model=model)

    # Queue should still be size 1 (maxsize=1 — not overflowed)
    assert model.phase4_queue.qsize() == 1

    # The tick on the queue should be the NEW one
    tick = model.phase4_queue.get_nowait()
    assert tick.P_true == p_true_new


def test_queue_full_does_not_raise() -> None:
    """emit_to_phase4 never raises even when queue is already full."""
    model = _make_model()

    # Fill the queue
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=True, model=model)

    # Should not raise on second emit
    emit_to_phase4(_P_TRUE, _SIGMA_MC, order_allowed=False, model=model)
