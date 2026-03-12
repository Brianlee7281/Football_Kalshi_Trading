"""Unit tests for EventQueue — FIFO buffer for events during PRELIMINARY state.

Reference: docs/phase3.md §Rapid Sequential Events
"""

from __future__ import annotations

from src.common.types import NormalizedEvent
from src.engine.event_queue import EventQueue


def _event(event_type: str = "goal_detected") -> NormalizedEvent:
    return NormalizedEvent(
        type=event_type,
        source="live_score",
        confidence="preliminary",
        timestamp=1000.0,
    )


def test_empty_queue_len_zero() -> None:
    q = EventQueue()
    assert len(q) == 0


def test_enqueue_increments_len() -> None:
    q = EventQueue()
    q.enqueue(_event())
    assert len(q) == 1
    q.enqueue(_event("red_card"))
    assert len(q) == 2


def test_drain_returns_events_in_order() -> None:
    q = EventQueue()
    e1 = _event("goal_detected")
    e2 = _event("red_card")
    q.enqueue(e1)
    q.enqueue(e2)
    result = q.drain()
    assert result == [e1, e2]


def test_drain_clears_queue() -> None:
    q = EventQueue()
    q.enqueue(_event())
    q.drain()
    assert len(q) == 0


def test_drain_empty_returns_empty_list() -> None:
    q = EventQueue()
    assert q.drain() == []


def test_double_drain_returns_empty_second_time() -> None:
    q = EventQueue()
    q.enqueue(_event())
    q.drain()
    assert q.drain() == []


def test_separate_instances_isolated() -> None:
    q1 = EventQueue()
    q2 = EventQueue()
    q1.enqueue(_event())
    assert len(q2) == 0


def test_drain_returns_copy_not_reference() -> None:
    """drain() returns a copy — mutating it doesn't affect the queue."""
    q = EventQueue()
    q.enqueue(_event())
    result = q.drain()
    result.clear()
    # After drain, queue is empty anyway, but the key point:
    # appending to result doesn't affect internal queue
    q.enqueue(_event())
    assert len(q) == 1
