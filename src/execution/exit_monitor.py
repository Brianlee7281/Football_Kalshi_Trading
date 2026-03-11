"""Exit monitor — per-position exit evaluation loop (Phase 4).

Runs concurrently with signal_generator.  Each iteration evaluates all
open positions against the six exit triggers defined in exit_logic.py.
Positions with a non-None ExitSignal are closed via ExecutionRouter.

Reference: docs/phase4.md Step 4.4, docs/blueprint.md exit_monitor.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("exit_monitor")

FINISHED = "FINISHED"
_POLL_INTERVAL_S: float = 1.0  # evaluate exits once per second


async def exit_monitor(model: LiveFootballQuantModel) -> None:
    """Monitor open positions and close those that trigger an exit condition.

    Polls DB for open positions every second.  For each position, evaluates
    all six exit triggers from exit_logic.evaluate_exit.  If an exit is
    signalled, submits an opposing order via model.execution and updates the
    position status.

    This is a stub implementation for Sprint 5.  Full implementation
    (position DB queries, opposing-order submission) is wired in Sprint 6
    when the DB layer is available.

    Args:
        model: Live match model (provides engine_phase, db_pool, execution).
    """
    logger.info("exit_monitor_started", match_id=model.match_id)

    while model.engine_phase != FINISHED:
        await asyncio.sleep(_POLL_INTERVAL_S)
        # Sprint 6: query open positions, call evaluate_exit per position,
        # submit opposing orders for any ExitSignal returned.

    logger.info("exit_monitor_finished", match_id=model.match_id)
