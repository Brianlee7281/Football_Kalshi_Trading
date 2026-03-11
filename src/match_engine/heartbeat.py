"""Redis heartbeat emitter for match container liveness monitoring.

Emits a heartbeat to Redis every 10 seconds.  The orchestrator monitors
``heartbeat:{match_id}`` and alerts if the age exceeds 60 seconds.

The key is set with a 120-second TTL so stale heartbeats auto-expire
if the container crashes without cleaning up.

Reference: docs/orchestration.md Container Entry Point (heartbeat_emitter)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.engine.model import LiveFootballQuantModel

logger = get_logger("heartbeat")

HEARTBEAT_INTERVAL_S: float = 10.0
HEARTBEAT_TTL_S: int = 120  # Redis TTL in seconds

FINISHED = "FINISHED"


async def heartbeat_emitter(model: LiveFootballQuantModel) -> None:
    """Emit a Redis heartbeat every 10 seconds until match FINISHED.

    Stores ``str(time.time())`` at key ``heartbeat:{match_id}`` with a
    120-second expiry.  The orchestrator reads this key to verify liveness.

    Args:
        model: Live match model.  Uses ``model.redis``, ``model.match_id``,
               and ``model.engine_phase``.
    """
    if model.redis is None:
        logger.warning("heartbeat_skipped_no_redis", match_id=model.match_id)
        return

    key = f"heartbeat:{model.match_id}"
    logger.info("heartbeat_started", match_id=model.match_id, interval_s=HEARTBEAT_INTERVAL_S)

    while model.engine_phase != FINISHED:
        try:
            await model.redis.set(key, str(time.time()), ex=HEARTBEAT_TTL_S)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "heartbeat_failed",
                match_id=model.match_id,
                error=str(exc),
            )
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # Final heartbeat on clean exit
    await _emit_final_heartbeat(model)


async def _emit_final_heartbeat(model: LiveFootballQuantModel) -> None:
    """Emit one last heartbeat tagged as FINISHED for the orchestrator."""
    if model.redis is None:
        return
    try:
        key = f"heartbeat:{model.match_id}"
        await model.redis.set(key, f"FINISHED:{time.time()}", ex=HEARTBEAT_TTL_S)
        logger.info("final_heartbeat_emitted", match_id=model.match_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "final_heartbeat_failed",
            match_id=model.match_id,
            error=str(exc),
        )
