"""Unit tests for LiveFootballQuantModel — Task 4.2."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.common.types import Phase2Result
from src.engine.model import (
    EVENT_IDLE,
    FIRST_HALF,
    WAITING_FOR_KICKOFF,
    LiveFootballQuantModel,
)


def _make_phase2() -> Phase2Result:
    return Phase2Result(a_H=-4.27, a_A=-4.35, C_time=14.5, verdict="GO")


def _make_params() -> dict:  # type: ignore[type-arg]
    """Minimal production_params dict for testing."""
    Q = np.diag([-0.012, -0.008, -0.008, -0.005])
    return {
        "b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gamma_H": [0.0, -0.15, 0.10, -0.25],
        "gamma_A": [0.0, 0.10, -0.15, -0.20],
        "delta_H": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "delta_A": [0.10, 0.05, 0.0, -0.05, -0.10],
        "beta_H": 0.0,
        "kappa_H": 0.0,
        "tau_H": 1.0,
        "beta_A": 0.0,
        "kappa_A": 0.0,
        "tau_A": 1.0,
        "Q_global": Q.tolist(),
        "Q_by_delta_S": {},
        "n_matches": 380,
        "n_goals": 1247,
        "league_id": 1204,
    }


# ---------------------------------------------------------------------------
# Init from Phase2Result
# ---------------------------------------------------------------------------


def test_model_init_from_phase2_t_zero() -> None:
    """model.t starts at 0.0."""
    model = LiveFootballQuantModel.from_phase2(
        _make_phase2(), _make_params(), match_id="test"
    )
    assert model.t == 0.0


def test_model_init_from_phase2_score_zero() -> None:
    """model.score starts at (0, 0)."""
    model = LiveFootballQuantModel.from_phase2(
        _make_phase2(), _make_params(), match_id="test"
    )
    assert model.score == (0, 0)
    assert model.score_home == 0
    assert model.score_away == 0


def test_model_init_from_phase2_X_zero() -> None:
    """model.current_state_X starts at 0 (no red cards)."""
    model = LiveFootballQuantModel.from_phase2(
        _make_phase2(), _make_params(), match_id="test"
    )
    assert model.current_state_X == 0


def test_model_init_engine_phase_waiting() -> None:
    """Engine phase starts at WAITING_FOR_KICKOFF."""
    model = LiveFootballQuantModel.from_phase2(
        _make_phase2(), _make_params(), match_id="test"
    )
    assert model.engine_phase == WAITING_FOR_KICKOFF


def test_model_init_event_state_idle() -> None:
    """Event state starts at IDLE."""
    model = LiveFootballQuantModel.from_phase2(
        _make_phase2(), _make_params(), match_id="test"
    )
    assert model.event_state == EVENT_IDLE


# ---------------------------------------------------------------------------
# phase4_queue
# ---------------------------------------------------------------------------


def test_phase4_queue_maxsize_one() -> None:
    """phase4_queue is created with maxsize=1."""
    model = LiveFootballQuantModel()
    assert model.phase4_queue.maxsize == 1


def test_phase4_queue_stale_replacement() -> None:
    """emit_tick replaces stale tick when queue is full."""
    from src.common.types import TickData

    model = LiveFootballQuantModel()

    tick_old = TickData(
        P_true={"home_win": 0.3},
        sigma_MC={"home_win": 0.002},
        order_allowed=True,
    )
    tick_new = TickData(
        P_true={"home_win": 0.6},
        sigma_MC={"home_win": 0.002},
        order_allowed=True,
    )

    model.emit_tick(tick_old)
    assert model.phase4_queue.full()

    model.emit_tick(tick_new)
    assert model.phase4_queue.full()

    retrieved = model.phase4_queue.get_nowait()
    assert retrieved.P_true["home_win"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# State mutations
# ---------------------------------------------------------------------------


def test_update_score_updates_delta_S() -> None:
    model = LiveFootballQuantModel()
    model.update_score(2, 1)
    assert model.score == (2, 1)
    assert model.delta_S == 1


def test_transition_state() -> None:
    model = LiveFootballQuantModel()
    model.transition_state(2)
    assert model.current_state_X == 2


def test_order_allowed_false_during_cooldown() -> None:
    model = LiveFootballQuantModel()
    model.cooldown = True
    assert not model.order_allowed


def test_order_allowed_false_during_ob_freeze() -> None:
    model = LiveFootballQuantModel()
    model.ob_freeze = True
    assert not model.order_allowed


def test_is_active_false_in_waiting() -> None:
    model = LiveFootballQuantModel()
    assert not model.is_active


def test_is_active_true_in_first_half() -> None:
    model = LiveFootballQuantModel()
    model.engine_phase = FIRST_HALF
    assert model.is_active


def test_Q_diag_property() -> None:
    """Q_diag returns the diagonal of Q matrix."""
    model = LiveFootballQuantModel.from_phase2(_make_phase2(), _make_params())
    q_diag = model.Q_diag
    assert q_diag.shape == (4,)
    assert all(q_diag[i] < 0 for i in range(4))  # all departure rates negative


# ---------------------------------------------------------------------------
# event_queue field
# ---------------------------------------------------------------------------


def test_event_queue_initialized() -> None:
    """event_queue is an EventQueue instance."""
    from src.engine.event_queue import EventQueue

    model = LiveFootballQuantModel()
    assert isinstance(model.event_queue, EventQueue)
    assert len(model.event_queue) == 0
