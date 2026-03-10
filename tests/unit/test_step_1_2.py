"""Tests for src/calibration/step_1_2_Q_estimation.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.step_1_2_Q_estimation import (
    _ds_to_bin,
    apply_state3_additivity,
    estimate_Q_by_delta_S,
    estimate_Q_global,
    normalize_Q_off_diagonal,
)
from src.common.types import IntervalRecord, RedCardTransition


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
    rc_transitions: list[RedCardTransition] | None = None,
    is_halftime: bool = False,
) -> IntervalRecord:
    return IntervalRecord(
        match_id=match_id,
        t_start=t_start,
        t_end=t_end,
        state_X=state_X,
        delta_S=delta_S,
        red_card_transitions=rc_transitions or [],
        is_halftime=is_halftime,
    )


def _intervals_with_red_cards() -> list[IntervalRecord]:
    """Synthetic dataset: 100 matches, 5 home reds at state 0→1, 3 away reds at 0→2.

    Each match has ~95 min in state 0, then the rest in the new state.
    """
    intervals: list[IntervalRecord] = []

    # 92 clean matches (no red cards): full 95 min in state 0
    for i in range(92):
        intervals.append(_make_interval(match_id=f"clean_{i}", t_start=0, t_end=47, state_X=0))
        intervals.append(_make_interval(match_id=f"clean_{i}", t_start=47, t_end=98, state_X=0))

    # 5 home red cards at minute 60 (state 0→1)
    for i in range(5):
        intervals.append(_make_interval(
            match_id=f"home_rc_{i}", t_start=0, t_end=60, state_X=0,
            rc_transitions=[RedCardTransition(minute=60, team="localteam", from_state=0, to_state=1)],
        ))
        intervals.append(_make_interval(
            match_id=f"home_rc_{i}", t_start=60, t_end=98, state_X=1,
        ))

    # 3 away red cards at minute 70 (state 0→2)
    for i in range(3):
        intervals.append(_make_interval(
            match_id=f"away_rc_{i}", t_start=0, t_end=70, state_X=0,
            rc_transitions=[RedCardTransition(minute=70, team="visitorteam", from_state=0, to_state=2)],
        ))
        intervals.append(_make_interval(
            match_id=f"away_rc_{i}", t_start=70, t_end=98, state_X=2,
        ))

    return intervals


# ---------------------------------------------------------------------------
# estimate_Q_global
# ---------------------------------------------------------------------------


class TestEstimateQGlobal:
    def test_rows_sum_to_zero(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_diagonal_entries_negative(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        for i in range(4):
            assert Q[i, i] <= 0.0

    def test_off_diagonal_entries_nonneg(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert Q[i, j] >= 0.0

    def test_shape(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        assert Q.shape == (4, 4)

    def test_transition_rates_reasonable(self) -> None:
        """5 home reds / total state-0 time ≈ small positive rate."""
        Q = estimate_Q_global(_intervals_with_red_cards())
        assert Q[0, 1] > 0.0  # home red from state 0
        assert Q[0, 2] > 0.0  # away red from state 0
        assert Q[0, 1] > Q[0, 2]  # 5 > 3 transitions

    def test_no_transitions_from_state_3(self) -> None:
        """No data for state 3 → off-diagonal should be 0."""
        Q = estimate_Q_global(_intervals_with_red_cards())
        for j in range(4):
            if j != 3:
                assert Q[3, j] == 0.0
        assert Q[3, 3] == 0.0  # no dwell time → diagonal also 0


class TestEstimateQGlobalNoRedCards:
    def test_zero_transitions_all_off_diagonal_zero(self) -> None:
        """With zero red card transitions → Q off-diagonal = 0."""
        intervals = [
            _make_interval(match_id="a", t_start=0, t_end=47, state_X=0),
            _make_interval(match_id="a", t_start=47, t_end=98, state_X=0),
            _make_interval(match_id="b", t_start=0, t_end=47, state_X=0),
            _make_interval(match_id="b", t_start=47, t_end=98, state_X=0),
        ]
        Q = estimate_Q_global(intervals)
        # All off-diagonal should be zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert Q[i, j] == 0.0
        # Diagonal should also be zero (no outgoing transitions)
        for i in range(4):
            assert Q[i, i] == 0.0

    def test_rows_still_sum_to_zero(self) -> None:
        intervals = [_make_interval(t_start=0, t_end=95, state_X=0)]
        Q = estimate_Q_global(intervals)
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


class TestEstimateQGlobalHalftimeExcluded:
    def test_halftime_intervals_skipped(self) -> None:
        """Halftime intervals should not contribute dwell time or transitions."""
        intervals = [
            _make_interval(t_start=0, t_end=45, state_X=0),
            _make_interval(t_start=45, t_end=45, state_X=0, is_halftime=True),
            _make_interval(t_start=45, t_end=95, state_X=0),
        ]
        Q = estimate_Q_global(intervals)
        # Should still work (90 min dwell, no transitions)
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# normalize_Q_off_diagonal
# ---------------------------------------------------------------------------


class TestNormalizeQOffDiagonal:
    def test_rows_sum_to_one(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_off = normalize_Q_off_diagonal(Q)
        # Row 0 has outgoing transitions → should sum to 1.0
        assert Q_off[0, :].sum() == pytest.approx(1.0, abs=1e-10)

    def test_diagonal_is_zero(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_off = normalize_Q_off_diagonal(Q)
        for i in range(4):
            assert Q_off[i, i] == 0.0

    def test_entries_nonneg(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_off = normalize_Q_off_diagonal(Q)
        assert np.all(Q_off >= 0.0)

    def test_zero_outgoing_row_stays_zero(self) -> None:
        """State 3 has no transitions → normalized row all zeros."""
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_off = normalize_Q_off_diagonal(Q)
        np.testing.assert_allclose(Q_off[3, :], 0.0, atol=1e-10)

    def test_proportions_preserved(self) -> None:
        """Ratio of q(0→1)/q(0→2) should be preserved in normalized form."""
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_off = normalize_Q_off_diagonal(Q)
        ratio_Q = Q[0, 1] / Q[0, 2]
        ratio_off = Q_off[0, 1] / Q_off[0, 2]
        assert ratio_Q == pytest.approx(ratio_off, rel=1e-10)


# ---------------------------------------------------------------------------
# estimate_Q_by_delta_S
# ---------------------------------------------------------------------------


class TestEstimateQByDeltaS:
    def test_returns_five_bins(self) -> None:
        result = estimate_Q_by_delta_S(_intervals_with_red_cards())
        assert len(result) == 5
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_each_bin_rows_sum_to_zero(self) -> None:
        result = estimate_Q_by_delta_S(_intervals_with_red_cards())
        for b, Q in result.items():
            row_sums = Q.sum(axis=1)
            np.testing.assert_allclose(row_sums, 0.0, atol=1e-10, err_msg=f"bin {b}")

    def test_shrinkage_toward_global_when_low_data(self) -> None:
        """Bins with no data should equal the global Q exactly."""
        intervals = _intervals_with_red_cards()
        # All intervals have delta_S=0 → bin 2. Other bins have 0 dwell → w=0.
        Q_global = estimate_Q_global(intervals)
        result = estimate_Q_by_delta_S(intervals)

        # Bin 0 (ΔS ≤ -2): no data → should be pure global Q
        np.testing.assert_allclose(result[0], Q_global, atol=1e-10)

    def test_high_threshold_gives_global(self) -> None:
        """With extremely high threshold, all bins shrink to global."""
        intervals = _intervals_with_red_cards()
        Q_global = estimate_Q_global(intervals)
        result = estimate_Q_by_delta_S(intervals, T_threshold=1e12)
        for b, Q in result.items():
            np.testing.assert_allclose(Q, Q_global, atol=1e-10, err_msg=f"bin {b}")

    def test_low_threshold_gives_empirical(self) -> None:
        """With very low threshold, bin 2 should equal its empirical Q."""
        intervals = _intervals_with_red_cards()
        result = estimate_Q_by_delta_S(intervals, T_threshold=1.0)
        Q_global = estimate_Q_global(intervals)
        # Bin 2 has all the data → w ≈ 1.0 → should equal global Q
        # (because all data IS in bin 2, so empirical ≈ global)
        np.testing.assert_allclose(result[2], Q_global, atol=1e-6)


# ---------------------------------------------------------------------------
# apply_state3_additivity
# ---------------------------------------------------------------------------


class TestApplyState3Additivity:
    def test_state3_transitions_filled(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_patched = apply_state3_additivity(Q)
        # q(1→3) should equal q(0→2)
        assert Q_patched[1, 3] == pytest.approx(Q_patched[0, 2], abs=1e-10)
        # q(2→3) should equal q(0→1)
        assert Q_patched[2, 3] == pytest.approx(Q_patched[0, 1], abs=1e-10)

    def test_rows_still_sum_to_zero(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_patched = apply_state3_additivity(Q)
        row_sums = Q_patched.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_off_diagonal_nonneg(self) -> None:
        Q = estimate_Q_global(_intervals_with_red_cards())
        Q_patched = apply_state3_additivity(Q)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert Q_patched[i, j] >= 0.0


# ---------------------------------------------------------------------------
# _ds_to_bin
# ---------------------------------------------------------------------------


class TestDsToBin:
    def test_extreme_negative(self) -> None:
        assert _ds_to_bin(-5) == 0
        assert _ds_to_bin(-2) == 0

    def test_negative_one(self) -> None:
        assert _ds_to_bin(-1) == 1

    def test_zero(self) -> None:
        assert _ds_to_bin(0) == 2

    def test_positive_one(self) -> None:
        assert _ds_to_bin(1) == 3

    def test_extreme_positive(self) -> None:
        assert _ds_to_bin(2) == 4
        assert _ds_to_bin(5) == 4


# ---------------------------------------------------------------------------
# Empty input edge case
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_intervals_returns_zero_Q(self) -> None:
        Q = estimate_Q_global([])
        np.testing.assert_allclose(Q, 0.0, atol=1e-10)
        assert Q.shape == (4, 4)

    def test_empty_intervals_Q_off_normalized(self) -> None:
        Q = estimate_Q_global([])
        Q_off = normalize_Q_off_diagonal(Q)
        np.testing.assert_allclose(Q_off, 0.0, atol=1e-10)
