# dashboard/api/routes/websocket.py
#
# WebSocket endpoint:
#   WS /ws/live — push live tick, signal, position, and alert data
#
# Uses the FIXED redis_listener from docs/dashboard_decomposition.md Part 2.1:
# subscribe/unsubscribe delta management with no duplicate subscribe bug.

from __future__ import annotations

import json

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.common.logging import get_logger

logger = get_logger("websocket_route")

router = APIRouter(tags=["websocket"])

# Global channels every WS client receives (no per-match subscription needed).
_GLOBAL_CHANNELS = ("position_update", "system_alert")


def _get_redis(ws: WebSocket) -> aioredis.Redis:
    """Retrieve the Redis client stored on ``app.state.redis``."""
    client: aioredis.Redis = ws.app.state.redis
    return client


@router.websocket("/ws/live")
async def live_updates(ws: WebSocket) -> None:
    """Stream Redis pubsub messages to the browser over a single WebSocket.

    Protocol — client sends JSON to manage match subscriptions::

        {"subscribe": ["match_abc", "match_xyz"]}

    The server computes a *delta* against the current subscription set:
    * New match IDs → subscribe to ``tick:{id}``, ``event:{id}``, ``signal:{id}``
    * Removed match IDs → unsubscribe from those channels
    * Global channels (``position_update``, ``system_alert``) are subscribed once.

    Bug fix (Part 2.1): subscriptions are managed via set-delta, not re-issued
    every iteration, avoiding duplicate message delivery.
    """
    import asyncio

    await ws.accept()

    redis = _get_redis(ws)
    subscriptions: set[str] = set()
    pubsub = redis.pubsub()

    # Always subscribe to global channels once.
    await pubsub.subscribe(*_GLOBAL_CHANNELS)

    async def redis_listener() -> None:
        """Forward Redis pubsub messages to the WebSocket client."""
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    await ws.send_json(json.loads(message["data"]))
                except WebSocketDisconnect:
                    break

    listener_task = asyncio.create_task(redis_listener())

    try:
        async for data in ws.iter_json():
            if "subscribe" in data:
                new_subs = set(data["subscribe"])

                # Unsubscribe removed matches.
                for match_id in subscriptions - new_subs:
                    await pubsub.unsubscribe(
                        f"tick:{match_id}",
                        f"event:{match_id}",
                        f"signal:{match_id}",
                    )

                # Subscribe new matches.
                for match_id in new_subs - subscriptions:
                    await pubsub.subscribe(
                        f"tick:{match_id}",
                        f"event:{match_id}",
                        f"signal:{match_id}",
                    )

                subscriptions = new_subs
    except WebSocketDisconnect:
        pass
    finally:
        listener_task.cancel()
        await pubsub.unsubscribe()
        await pubsub.aclose()  # type: ignore[no-untyped-call]
