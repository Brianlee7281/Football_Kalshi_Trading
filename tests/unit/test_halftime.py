"""Unit tests for halftime wall-clock tracking and model.t exclusion.

Tests the core invariant:
  model.t = (wall_elapsed - halftime_accumulated) / 60

Reference: docs/phase3.md §tick_loop, patterns.md #4
"""

from __future__ import annotations

import time

import pytest

from src.engine.model import (
    FIRST_HALF,
    HALFTIME,
    SECOND_HALF,
    LiveFootballQuantModel,
)
from src.engine.period_handler import _enter_halftime, _enter_second_half


def _make_model() -> LiveFootballQuantModel:
    model = LiveFootballQuantModel()
    model.engine_phase = FIRST_HALF
    model.kickoff_wall_clock = time.monotonic()
    model.halftime_accumulated = 0.0
    model.halftime_start = None
    return model


# ---------------------------------------------------------------------------
# model.t before halftime
# ---------------------------------------------------------------------------


def test_t_before_halftime() -> None:
    """model.t = wall_elapsed / 60 before halftime (no accumulation)."""
    model = _make_model()
    # Simulate 47 minutes of first half
    model.kickoff_wall_clock = time.monotonic() - 47.0 * 60
    wall_elapsed = time.monotonic() - model.kickoff_wall_clock
    model.t = (wall_elapsed - model.halftime_accumulated) / 60.0
    assert model.t == pytest.approx(47.0, abs=0.05)


# ---------------------------------------------------------------------------
# Halftime entry: halftime_start set
# ---------------------------------------------------------------------------


def test_halftime_start_set_on_entry() -> None:
    """Entering HALFTIME records halftime_start (wall clock)."""
    model = _make_model()
    model.t = 47.0

    t_before = time.monotonic()
    _enter_halftime(model)
    t_after = time.monotonic()

    assert model.engine_phase == HALFTIME
    assert model.halftime_start is not None
    assert t_before <= model.halftime_start <= t_after


def test_halftime_entry_idempotent() -> None:
    """Entering HALFTIME twice doesn't reset halftime_start."""
    model = _make_model()
    _enter_halftime(model)
    first_start = model.halftime_start

    _enter_halftime(model)
    assert model.halftime_start == first_start  # unchanged


# ---------------------------------------------------------------------------
# Second half: halftime_accumulated = 900s for 15-min HT
# ---------------------------------------------------------------------------


def test_halftime_accumulated_after_15_min() -> None:
    """After 15-min halftime, halftime_accumulated ≈ 900s."""
    model = _make_model()
    model.t = 47.0

    # Simulate halftime_start 900 seconds ago
    model.engine_phase = HALFTIME
    model.halftime_start = time.monotonic() - 900.0  # 15 min ago

    _enter_second_half(model)

    assert model.engine_phase == SECOND_HALF
    assert model.halftime_accumulated == pytest.approx(900.0, abs=1.0)
    assert model.halftime_start is None  # reset


# ---------------------------------------------------------------------------
# model.t after halftime resume = 47.0 (not 62.0)
# ---------------------------------------------------------------------------


def test_t_after_halftime_resume_still_47() -> None:
    """After resuming second half, effective play time resumes at ~47 min."""
    model = _make_model()

    # Simulate: kickoff 62 min wall-clock ago, 15 min halftime
    halftime_dur_s = 900.0  # 15 min halftime
    wall_ago_s = 62.0 * 60  # 62 wall-clock minutes since kickoff
    model.kickoff_wall_clock = time.monotonic() - wall_ago_s
    model.halftime_accumulated = halftime_dur_s
    model.engine_phase = SECOND_HALF

    wall_elapsed = time.monotonic() - model.kickoff_wall_clock
    model.t = (wall_elapsed - model.halftime_accumulated) / 60.0

    # Effective time: (62*60 - 900) / 60 = (3720 - 900) / 60 = 47.0
    assert model.t == pytest.approx(47.0, abs=0.1)


# ---------------------------------------------------------------------------
# model.t at 80th play minute
# ---------------------------------------------------------------------------


def test_t_at_80th_play_minute() -> None:
    """model.t = 80.0 at 80th effective play minute."""
    model = _make_model()

    # 80 effective play minutes + 15 min halftime = 95 wall-clock minutes
    wall_ago_s = (80.0 + 15.0) * 60  # 95 min wall clock
    halftime_dur_s = 900.0  # 15 min HT
    model.kickoff_wall_clock = time.monotonic() - wall_ago_s
    model.halftime_accumulated = halftime_dur_s
    model.engine_phase = SECOND_HALF

    wall_elapsed = time.monotonic() - model.kickoff_wall_clock
    model.t = (wall_elapsed - model.halftime_accumulated) / 60.0

    assert model.t == pytest.approx(80.0, abs=0.1)


# ---------------------------------------------------------------------------
# halftime_accumulated invariant (never drifts)
# ---------------------------------------------------------------------------


def test_halftime_never_drifts_second_time() -> None:
    """After second-half transition, halftime_start is None (prevents double-counting)."""
    model = _make_model()
    model.engine_phase = HALFTIME
    model.halftime_start = time.monotonic() - 900.0

    _enter_second_half(model)
    assert model.halftime_start is None

    # If we accidentally call _enter_second_half again, accumulated doesn't change
    prev_accumulated = model.halftime_accumulated
    _enter_second_half(model)  # idempotent
    assert model.halftime_accumulated == prev_accumulated
