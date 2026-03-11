"""EventQueue — buffer for events that arrive during PRELIMINARY state.

When the engine is in PRELIMINARY_DETECTED (waiting for VAR confirmation
of event A), any new event B is queued here. Once event A resolves
(confirmed or rolled back), the queue is drained and all pending events
are dispatched.

Rule 3 from patterns.md: Events during PRELIMINARY are queued (EventQueue),
drained on confirmation.

Reference: docs/phase3.md §Rapid Sequential Events
"""

from __future__ import annotations

from src.common.logging import get_logger
from src.common.types import NormalizedEvent

logger = get_logger("event_queue")


class EventQueue:
    """FIFO buffer for events arriving during PRELIMINARY state."""

    def __init__(self) -> None:
        self._queue: list[NormalizedEvent] = []

    def enqueue(self, event: NormalizedEvent) -> None:
        """Add an event to the queue."""
        self._queue.append(event)
        logger.info(
            "event_queued",
            event_type=event.type,
            pending=len(self._queue),
        )

    def drain(self) -> list[NormalizedEvent]:
        """Return and clear all queued events."""
        events = self._queue.copy()
        self._queue.clear()
        return events

    def __len__(self) -> int:
        return len(self._queue)
