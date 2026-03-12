"""Integration: Phase 2 → Phase 3 pipeline end-to-end.

Tests the Phase2Result → LiveFootballQuantModel initialization path
and Phase 3→4 tick emission via the queue, using mock API clients.

Reference: docs/phase2.md Step 2.5, docs/phase3.md
"""

from __future__ import annotations

import pytest

from src.common.types import Phase2Result, TickData
from src.engine.mc_pricing import analytical_pricing, compute_mc_stderr
from src.engine.model import (
    EVENT_IDLE,
    FIRST_HALF,
    HALFTIME,
    SECOND_HALF,
    WAITING_FOR_KICKOFF,
    LiveFootballQuantModel,
)

from .conftest import make_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phase1_params() -> dict[str, object]:
    """Minimal Phase 1 production_params for model initialization."""
    return {
        "b": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "gamma_H": [0.0, 0.0, 0.0, 0.0],
        "gamma_A": [0.0, 0.0, 0.0, 0.0],
        "delta_H": [0.0, 0.0, 0.0, 0.0, 0.0],
        "delta_A": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Q_global": [
            [-0.01, 0.005, 0.005, 0.0],
            [0.0, -0.01, 0.0, 0.01],
            [0.01, 0.0, -0.01, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "Q_by_delta_S": {},
    }


def _phase2_result() -> Phase2Result:
    return Phase2Result(
        a_H=1.5,
        a_A=1.2,
        C_time=0.95,
        verdict="GO",
    )


# ---------------------------------------------------------------------------
# Phase 2 → Model initialization
# ---------------------------------------------------------------------------


def test_model_from_phase2_basic() -> None:
    """LiveFootballQuantModel.from_phase2 sets all key attributes."""
    p2 = _phase2_result()
    params = _phase1_params()

    model = LiveFootballQuantModel.from_phase2(
        p2, params,
        match_id="m123",
        league_id=42,
        trading_mode="paper",
        bankroll=10_000.0,
    )

    assert model.match_id == "m123"
    assert model.a_H == 1.5
    assert model.a_A == 1.2
    assert model.bankroll == 10_000.0
    assert model.is_paper is True
    assert model.engine_phase == WAITING_FOR_KICKOFF
    assert model.event_state == EVENT_IDLE
    assert model.order_allowed is True


def test_model_from_phase2_parameters_loaded() -> None:
    """Phase 1 parameters (b, gamma, delta, Q) are loaded into model."""
    p2 = _phase2_result()
    params = _phase1_params()

    model = LiveFootballQuantModel.from_phase2(p2, params)

    assert model.b.shape == (6,)
    assert model.gamma_H.shape == (4,)
    assert model.gamma_A.shape == (4,)
    assert model.delta_H.shape == (5,)
    assert model.delta_A.shape == (5,)
    assert model.Q.shape == (4, 4)


def test_model_from_phase2_basis_bounds() -> None:
    """Basis bounds computed from alpha values."""
    p2 = _phase2_result()
    params = _phase1_params()

    model = LiveFootballQuantModel.from_phase2(
        p2, params, alpha_1_mean=2.0, alpha_2_mean=4.0,
    )

    # [0, 15, 30, 47, 62, 77, 96]
    assert model.basis_bounds[0] == 0.0
    assert model.basis_bounds[3] == pytest.approx(47.0)
    assert model.basis_bounds[6] == pytest.approx(96.0)
    assert model.T_exp == pytest.approx(96.0)


def test_model_from_phase2_pricing_mode() -> None:
    """delta_significant=True → MC pricing mode."""
    p2 = _phase2_result()
    params = _phase1_params()

    model_analytical = LiveFootballQuantModel.from_phase2(
        p2, params, delta_significant=False,
    )
    model_mc = LiveFootballQuantModel.from_phase2(
        p2, params, delta_significant=True,
    )

    assert model_analytical.pricing_mode == "analytical"
    assert model_mc.pricing_mode == "mc"


def test_model_from_phase2_sanity_verdict_propagated() -> None:
    """Phase 2 sanity verdict propagated to model."""
    p2 = Phase2Result(a_H=1.5, a_A=1.2, C_time=0.95, verdict="GO_WITH_CAUTION")
    params = _phase1_params()

    model = LiveFootballQuantModel.from_phase2(p2, params)
    assert model.sanity_verdict == "GO_WITH_CAUTION"


# ---------------------------------------------------------------------------
# Engine phase transitions
# ---------------------------------------------------------------------------


def test_engine_phase_transitions() -> None:
    """Model supports the full phase lifecycle."""
    p2 = _phase2_result()
    params = _phase1_params()
    model = LiveFootballQuantModel.from_phase2(p2, params)

    assert model.engine_phase == WAITING_FOR_KICKOFF

    model.engine_phase = FIRST_HALF
    assert model.is_active is True

    model.enter_halftime()
    assert model.engine_phase == HALFTIME
    assert model.is_active is False

    model.exit_halftime()
    assert model.engine_phase == SECOND_HALF
    assert model.is_active is True

    model.finish()
    assert model.engine_phase == "FINISHED"
    assert model.is_active is False


# ---------------------------------------------------------------------------
# Score state mutation
# ---------------------------------------------------------------------------


def test_score_update_and_delta_s() -> None:
    """update_score correctly updates score and ΔS."""
    model = make_model()
    model.update_score(1, 0)
    assert model.score == (1, 0)
    assert model.delta_S == 1

    model.update_score(1, 2)
    assert model.score == (1, 2)
    assert model.delta_S == -1


def test_markov_state_transition() -> None:
    """transition_state updates current_state_X."""
    model = make_model()
    assert model.current_state_X == 0

    model.transition_state(2)
    assert model.current_state_X == 2


# ---------------------------------------------------------------------------
# Phase 3 → 4 queue emission
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_emit_tick_to_phase4_queue() -> None:
    """emit_tick puts TickData on phase4_queue."""
    model = make_model()
    tick = TickData(
        P_true={"home_win": 0.55},
        sigma_MC={"home_win": 0.005},
        order_allowed=True,
    )

    model.emit_tick(tick)
    result = model.phase4_queue.get_nowait()

    assert result.P_true["home_win"] == 0.55
    assert result.sigma_MC["home_win"] == 0.005
    assert result.order_allowed is True


@pytest.mark.anyio
async def test_emit_tick_replaces_stale() -> None:
    """Second emit replaces first when queue is full (maxsize=1)."""
    model = make_model()

    tick1 = TickData(P_true={"home_win": 0.50}, sigma_MC={"home_win": 0.005}, order_allowed=True)
    tick2 = TickData(P_true={"home_win": 0.65}, sigma_MC={"home_win": 0.003}, order_allowed=True)

    model.emit_tick(tick1)
    model.emit_tick(tick2)

    result = model.phase4_queue.get_nowait()
    assert result.P_true["home_win"] == 0.65


# ---------------------------------------------------------------------------
# Analytical pricing → Phase 4 flow
# ---------------------------------------------------------------------------


def test_analytical_pricing_to_phase4_format() -> None:
    """analytical_pricing output matches Phase 4 expected dict format."""
    P_true = analytical_pricing(mu_H=1.5, mu_A=1.2, score=(0, 0))
    sigma_MC = compute_mc_stderr(P_true, N=50_000, analytical=True)

    # P_true has all 7 markets
    expected_markets = {"home_win", "draw", "away_win", "over_15", "over_25", "over_35", "btts_yes"}
    assert set(P_true.keys()) == expected_markets
    assert set(sigma_MC.keys()) == expected_markets

    # All values are valid floats
    for market in expected_markets:
        assert 0.0 <= P_true[market] <= 1.0 + 1e-12
        assert sigma_MC[market] >= 0.0

    # Analytical σ_MC has floor of 0.005
    for _market, sigma in sigma_MC.items():
        assert sigma >= 0.005


# ---------------------------------------------------------------------------
# order_allowed gating
# ---------------------------------------------------------------------------


def test_order_allowed_respects_cooldown() -> None:
    """Cooldown active → order_allowed = False."""
    model = make_model()
    assert model.order_allowed is True

    model.cooldown = True
    assert model.order_allowed is False


def test_order_allowed_respects_ob_freeze() -> None:
    """ob_freeze → order_allowed = False."""
    model = make_model()
    model.ob_freeze = True
    assert model.order_allowed is False


def test_order_allowed_respects_event_state() -> None:
    """Non-IDLE event_state → order_allowed = False."""
    model = make_model()
    model.event_state = "PRELIMINARY_DETECTED"
    assert model.order_allowed is False
