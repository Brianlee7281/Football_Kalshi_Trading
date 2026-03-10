"""Tests for src/calibration/step_1_4_nll_optimize.py."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.calibration.step_1_4_nll_optimize import (
    GoalEvent,
    IntervalData,
    MatchData,
    MMPPModel,
    OptimizationResult,
    _ds_to_bin,
    _time_to_basis,
    compute_nll,
    delta_lookup_from_params,
    optimize_nll,
    parametric_delta,
    prepare_match_data,
)
from src.common.types import IntervalRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval(
    *,
    match_id: str = "m1",
    t_start: float = 0.0,
    t_end: float = 45.0,
    state_X: int = 0,
    delta_S: int = 0,
    home_goal_times: list[float] | None = None,
    away_goal_times: list[float] | None = None,
    goal_delta_before: list[int] | None = None,
    alpha_1: float = 0.0,
    alpha_2: float = 0.0,
    T_m: float = 90.0,
    is_halftime: bool = False,
) -> IntervalRecord:
    return IntervalRecord(
        match_id=match_id,
        t_start=t_start,
        t_end=t_end,
        state_X=state_X,
        delta_S=delta_S,
        home_goal_times=home_goal_times or [],
        away_goal_times=away_goal_times or [],
        goal_delta_before=goal_delta_before or [],
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        T_m=T_m,
        is_halftime=is_halftime,
    )


def _synthetic_matches(n: int = 10) -> tuple[list[MatchData], np.ndarray, np.ndarray]:
    """Create synthetic match data with ~1.3 goals per team per match.

    Returns (match_data, a_H_init, a_A_init).
    """
    rng = np.random.default_rng(42)
    match_data: list[MatchData] = []

    for m in range(n):
        md = MatchData(match_idx=m)

        # 6 intervals of 15 min each
        for basis_idx in range(6):
            md.intervals.append(IntervalData(
                basis_idx=basis_idx,
                state_X=0,
                ds_bin=2,  # ΔS=0
                duration=15.0,
            ))

        # Poisson goals (~1.3 per team)
        n_home = rng.poisson(1.3)
        n_away = rng.poisson(1.3)
        for _ in range(n_home):
            md.home_goal_log_lambdas.append(GoalEvent(
                basis_idx=rng.integers(0, 6),
                state_X=0,
                ds_bin=2,
            ))
        for _ in range(n_away):
            md.away_goal_log_lambdas.append(GoalEvent(
                basis_idx=rng.integers(0, 6),
                state_X=0,
                ds_bin=2,
            ))
        match_data.append(md)

    # Initial log-intensities: ln(1.3 / 90)
    a_init = np.full(n, math.log(1.3 / 90.0), dtype=np.float64)
    return match_data, a_init.copy(), a_init.copy()


# ---------------------------------------------------------------------------
# NLL decreases monotonically
# ---------------------------------------------------------------------------


class TestNLLMonotonic:
    def test_nll_decreases_over_100_steps(self) -> None:
        """NLL should decrease monotonically over optimization steps."""
        match_data, a_H, a_A = _synthetic_matches(10)
        result = optimize_nll(
            match_data, a_H, a_A,
            sigma_a=0.5, lr=1e-3, num_epochs=100,
        )
        losses = result.loss_history
        assert len(losses) == 100
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-6, (
                f"NLL increased at step {i}: {losses[i - 1]:.6f} → {losses[i]:.6f}"
            )


# ---------------------------------------------------------------------------
# log_tau parameterization
# ---------------------------------------------------------------------------


class TestLogTauParameterization:
    def test_log_tau_zero_gives_tau_one(self) -> None:
        """log_tau=0.0 → τ = exp(0) = 1.0."""
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        assert model.get_tau_H().item() == pytest.approx(1.0, abs=1e-6)
        assert model.get_tau_A().item() == pytest.approx(1.0, abs=1e-6)

    def test_log_tau_negative_gives_small_tau(self) -> None:
        """log_tau=-2.3 → τ ≈ 0.1 (clamped lower bound)."""
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        with torch.no_grad():
            model.log_tau_H.fill_(-2.3)
        tau = model.get_tau_H().item()
        assert tau == pytest.approx(0.1, abs=0.01)

    def test_log_tau_large_gives_clamped_tau(self) -> None:
        """log_tau=1.6 → τ ≈ 5.0 (clamped upper bound)."""
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        with torch.no_grad():
            model.log_tau_H.fill_(1.7)  # exp(1.7) ≈ 5.47 → clamped to 5.0
        tau = model.get_tau_H().item()
        assert tau == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# σ_a regularization
# ---------------------------------------------------------------------------


class TestSigmaARegularization:
    def test_large_sigma_a_keeps_a_near_init(self) -> None:
        """σ_a → ∞ (e.g., 1e6) → a_H, a_A ≈ XGBoost prediction (regularization dominates)."""
        match_data, a_H, a_A = _synthetic_matches(5)
        # Very small sigma_a means STRONG regularization → a stays near init
        result = optimize_nll(
            match_data, a_H, a_A,
            sigma_a=0.001, lr=1e-3, num_epochs=200,
        )
        np.testing.assert_allclose(result.a_H, a_H, atol=0.05)
        np.testing.assert_allclose(result.a_A, a_A, atol=0.05)

    def test_sigma_a_zero_no_nan(self) -> None:
        """σ_a = 0 → regularization term skipped, no NaN."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(
            match_data, a_H, a_A,
            sigma_a=0.0, lr=1e-3, num_epochs=50,
        )
        assert all(np.isfinite(result.loss_history))
        assert np.all(np.isfinite(result.a_H))
        assert np.all(np.isfinite(result.a_A))


# ---------------------------------------------------------------------------
# Output b array
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_b_has_6_elements(self) -> None:
        """Output b array has exactly 6 elements (6 basis periods × 15 min)."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        assert result.b.shape == (6,)

    def test_gamma_H_structure(self) -> None:
        """gamma_H has 4 elements (states X=0,1,2,3), gamma_H[0] = 0.0 (reference)."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        assert result.gamma_H.shape == (4,)
        assert result.gamma_H[0] == pytest.approx(0.0, abs=1e-10)

    def test_gamma_A_structure(self) -> None:
        """gamma_A has 4 elements, gamma_A[0] = 0.0 (reference)."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        assert result.gamma_A.shape == (4,)
        assert result.gamma_A[0] == pytest.approx(0.0, abs=1e-10)

    def test_delta_lookups_shape(self) -> None:
        """delta_H and delta_A each have 5 elements."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        assert result.delta_H.shape == (5,)
        assert result.delta_A.shape == (5,)

    def test_delta_zero_at_center(self) -> None:
        """δ(0) = 0 by construction for both teams."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        # ΔS=0 is index 2
        assert result.delta_H[2] == pytest.approx(0.0, abs=1e-10)
        assert result.delta_A[2] == pytest.approx(0.0, abs=1e-10)

    def test_gamma_H3_additive(self) -> None:
        """gamma_H[3] = gamma_H[1] + gamma_H[2] (additivity)."""
        match_data, a_H, a_A = _synthetic_matches(5)
        result = optimize_nll(match_data, a_H, a_A, num_epochs=10)
        assert result.gamma_H[3] == pytest.approx(
            result.gamma_H[1] + result.gamma_H[2], abs=1e-6,
        )


# ---------------------------------------------------------------------------
# Parametric delta function
# ---------------------------------------------------------------------------


class TestParametricDelta:
    def test_delta_zero_at_s_zero(self) -> None:
        """δ(0) = 0 for any parameter values."""
        s = torch.tensor([0.0])
        d = parametric_delta(s, torch.tensor(0.3), torch.tensor(0.5), torch.tensor(1.0))
        assert d.item() == pytest.approx(0.0, abs=1e-10)

    def test_linear_at_small_s(self) -> None:
        """For small |s| and large τ, δ ≈ (β + κ/τ)·s."""
        tau = torch.tensor(100.0)
        beta = torch.tensor(0.1)
        kappa = torch.tensor(0.2)
        s = torch.tensor([0.01])
        d = parametric_delta(s, beta, kappa, tau)
        expected = (0.1 + 0.2 / 100.0) * 0.01
        assert d.item() == pytest.approx(expected, rel=0.01)

    def test_delta_lookup_shape(self) -> None:
        lookup = delta_lookup_from_params(0.1, 0.2, 1.0)
        assert lookup.shape == (5,)
        assert lookup[2] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Time-to-basis mapping
# ---------------------------------------------------------------------------


class TestTimeToBasis:
    def test_first_half_periods(self) -> None:
        assert _time_to_basis(0.0) == 0
        assert _time_to_basis(14.9) == 0
        assert _time_to_basis(15.0) == 1
        assert _time_to_basis(29.9) == 1
        assert _time_to_basis(30.0) == 2
        assert _time_to_basis(44.9) == 2

    def test_second_half_periods(self) -> None:
        assert _time_to_basis(45.0) == 3
        assert _time_to_basis(59.9) == 3
        assert _time_to_basis(60.0) == 4
        assert _time_to_basis(74.9) == 4
        assert _time_to_basis(75.0) == 5
        assert _time_to_basis(89.9) == 5

    def test_with_stoppage(self) -> None:
        """With α₁=3, halftime is at 48, second half shifts."""
        assert _time_to_basis(44.0, alpha_1=3.0) == 2
        assert _time_to_basis(48.0, alpha_1=3.0) == 3  # HT boundary
        assert _time_to_basis(62.9, alpha_1=3.0) == 3
        assert _time_to_basis(63.0, alpha_1=3.0) == 4


# ---------------------------------------------------------------------------
# ds_to_bin
# ---------------------------------------------------------------------------


class TestDsToBin:
    def test_bins(self) -> None:
        assert _ds_to_bin(-5) == 0
        assert _ds_to_bin(-2) == 0
        assert _ds_to_bin(-1) == 1
        assert _ds_to_bin(0) == 2
        assert _ds_to_bin(1) == 3
        assert _ds_to_bin(2) == 4
        assert _ds_to_bin(5) == 4


# ---------------------------------------------------------------------------
# prepare_match_data
# ---------------------------------------------------------------------------


class TestPrepareMatchData:
    def test_basic_match(self) -> None:
        intervals = [
            _make_interval(match_id="m1", t_start=0, t_end=45, state_X=0, delta_S=0),
            _make_interval(match_id="m1", t_start=45, t_end=90, state_X=0, delta_S=0),
        ]
        result = prepare_match_data({"m1": intervals}, ["m1"])
        assert len(result) == 1
        assert result[0].match_idx == 0
        assert len(result[0].intervals) == 2

    def test_goals_extracted(self) -> None:
        intervals = [
            _make_interval(
                match_id="m1", t_start=0, t_end=45,
                home_goal_times=[23.0], away_goal_times=[30.0],
                goal_delta_before=[0, 1],
            ),
        ]
        result = prepare_match_data({"m1": intervals}, ["m1"])
        assert len(result[0].home_goal_log_lambdas) == 1
        assert len(result[0].away_goal_log_lambdas) == 1
        assert result[0].home_goal_log_lambdas[0].ds_bin == 2  # pre-goal ΔS=0 → bin 2
        assert result[0].away_goal_log_lambdas[0].ds_bin == 3  # pre-goal ΔS=1 → bin 3

    def test_halftime_skipped(self) -> None:
        intervals = [
            _make_interval(match_id="m1", t_start=0, t_end=45),
            _make_interval(match_id="m1", t_start=45, t_end=45, is_halftime=True),
            _make_interval(match_id="m1", t_start=45, t_end=90),
        ]
        # The halftime interval has duration 0, so it's skipped
        result = prepare_match_data({"m1": intervals}, ["m1"])
        assert len(result[0].intervals) == 2


# ---------------------------------------------------------------------------
# NLL computation sanity
# ---------------------------------------------------------------------------


class TestComputeNLL:
    def test_nll_finite(self) -> None:
        """NLL should be finite for default initialization."""
        match_data, a_H, a_A = _synthetic_matches(3)
        model = MMPPModel(3, torch.tensor(a_H, dtype=torch.float32), torch.tensor(a_A, dtype=torch.float32))
        loss = compute_nll(model, match_data)
        assert torch.isfinite(loss)

    def test_nll_decreases_with_gradient_step(self) -> None:
        """A single gradient step should decrease NLL."""
        match_data, a_H, a_A = _synthetic_matches(5)
        model = MMPPModel(5, torch.tensor(a_H, dtype=torch.float32), torch.tensor(a_A, dtype=torch.float32))
        loss_before = compute_nll(model, match_data).item()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = compute_nll(model, match_data)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        loss_after = compute_nll(model, match_data).item()
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# Clamping tests
# ---------------------------------------------------------------------------


class TestParameterClamping:
    def test_b_clamped(self) -> None:
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        with torch.no_grad():
            model.b.fill_(10.0)
        b = model.get_b_clamped()
        assert torch.all(b <= 0.5)

    def test_gamma_H_clamped(self) -> None:
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        with torch.no_grad():
            model.gamma_H_raw[0] = 5.0  # should clamp to 0
            model.gamma_H_raw[1] = -5.0  # should clamp to 0
        gamma_H = model.get_gamma_H()
        assert gamma_H[1].item() <= 0.0  # γ^H_1 ∈ [-1.5, 0]
        assert gamma_H[2].item() >= 0.0  # γ^H_2 ∈ [0, 1.5]

    def test_tau_clamped_range(self) -> None:
        model = MMPPModel(1, torch.zeros(1), torch.zeros(1))
        with torch.no_grad():
            model.log_tau_H.fill_(-10.0)
        assert model.get_tau_H().item() >= 0.1
        with torch.no_grad():
            model.log_tau_H.fill_(10.0)
        assert model.get_tau_H().item() <= 5.0


# ---------------------------------------------------------------------------
# _make_interval helper for prepare_match_data (with is_halftime)
# ---------------------------------------------------------------------------


def _make_interval_ht(
    *,
    match_id: str = "m1",
    t_start: float = 45.0,
    t_end: float = 45.0,
    is_halftime: bool = True,
) -> IntervalRecord:
    return IntervalRecord(
        match_id=match_id,
        t_start=t_start,
        t_end=t_end,
        state_X=0,
        delta_S=0,
        is_halftime=is_halftime,
    )
