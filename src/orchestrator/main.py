"""Orchestrator entry point — match discovery, trigger execution, recovery.

Started by Docker via ``python -m src.orchestrator.main``.

Main loop:
  1. Create DB pool + Redis client.
  2. Instantiate API clients, ContainerManager, MatchLifecycleManager.
  3. Run recovery (resume interrupted matches from DB state).
  4. Run initial MatchDiscovery.
  5. Start TriggerExecutor loop (every 30s).
  6. Re-run MatchDiscovery every 6 hours.

Reference: docs/orchestration.md — Data Flow Summary
"""

from __future__ import annotations

import asyncio
import os
import signal

from src.clients.goalserve import GoalserveClient
from src.clients.kalshi import KalshiClient
from src.common.db import create_pool
from src.common.logging import get_logger
from src.common.redis_client import create_client as create_redis
from src.orchestrator.container_manager import ContainerManager
from src.orchestrator.lifecycle import MatchLifecycleManager
from src.orchestrator.recovery import recover_orchestrator_state
from src.orchestrator.scheduler import MatchDiscovery, TriggerExecutor

logger = get_logger("orchestrator.main")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCOVERY_INTERVAL_S: float = 6 * 3600  # 6 hours


# ---------------------------------------------------------------------------
# Periodic discovery task
# ---------------------------------------------------------------------------


async def _discovery_loop(discovery: MatchDiscovery) -> None:
    """Run match discovery immediately, then every DISCOVERY_INTERVAL_S."""
    while True:
        try:
            schedules = await discovery.discover()
            logger.info("discovery_cycle_complete", new_matches=len(schedules))
        except Exception as exc:  # noqa: BLE001
            logger.error("discovery_error", error=str(exc))
        await asyncio.sleep(DISCOVERY_INTERVAL_S)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Orchestrator main — runs until SIGINT / SIGTERM."""

    # ── Environment ───────────────────────────────────────────────────────
    db_url = os.environ.get(
        "DB_URL",
        "postgresql://postgres:postgres@localhost:5432/soccer_trading",
    )
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    trading_mode = os.environ.get("TRADING_MODE", "paper")
    goalserve_key = os.environ.get("GOALSERVE_API_KEY", "")
    kalshi_key = os.environ.get("KALSHI_API_KEY", "")
    kalshi_private_key = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    logger.info(
        "orchestrator_starting",
        trading_mode=trading_mode,
        db_host=db_url.split("@")[1].split("/")[0] if "@" in db_url else "local",
    )

    # ── Connections ───────────────────────────────────────────────────────
    pool = await create_pool(db_url)
    redis = await create_redis(redis_url)
    logger.info("connections_established")

    # ── Fetch active param version ────────────────────────────────────────
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT version FROM production_params WHERE is_active = TRUE"
        )
    param_version = int(row["version"]) if row else 1
    logger.info("param_version_loaded", version=param_version)

    # ── API clients ───────────────────────────────────────────────────────
    goalserve = GoalserveClient(goalserve_key)
    kalshi = KalshiClient(kalshi_key, kalshi_private_key) if kalshi_private_key else None

    # ── Container manager ─────────────────────────────────────────────────
    container_manager = ContainerManager(
        db_url=db_url,
        redis_url=redis_url,
        odds_api_key=odds_api_key,
        goalserve_api_key=goalserve_key,
        kalshi_api_key=kalshi_key,
    )

    # ── Lifecycle manager ─────────────────────────────────────────────────
    lifecycle = MatchLifecycleManager(
        db_pool=pool,
        redis=redis,
        container_manager=container_manager,
        goalserve_client=goalserve,
    )

    # ── Recovery ──────────────────────────────────────────────────────────
    counts = await recover_orchestrator_state(
        pool, lifecycle, container_manager,
    )
    logger.info("recovery_complete", **counts)

    # ── Match discovery ───────────────────────────────────────────────────
    if kalshi is None:
        logger.warning("kalshi_client_not_configured_skipping_discovery")
    else:
        discovery = MatchDiscovery(
            goalserve, kalshi, pool,
            trading_mode=trading_mode,
            param_version=param_version,
        )

    # ── Trigger executor ──────────────────────────────────────────────────
    trigger_executor = TriggerExecutor(pool, lifecycle)

    # ── Graceful shutdown ─────────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    # ── Run tasks ─────────────────────────────────────────────────────────
    tasks: list[asyncio.Task[None]] = []
    tasks.append(asyncio.create_task(trigger_executor.run(), name="trigger_executor"))
    if kalshi is not None:
        tasks.append(asyncio.create_task(_discovery_loop(discovery), name="discovery"))

    logger.info("orchestrator_running", tasks=[t.get_name() for t in tasks])

    # Wait for shutdown signal
    await shutdown_event.wait()
    logger.info("orchestrator_shutting_down")

    # Cancel all tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup
    await pool.close()
    await redis.aclose()
    logger.info("orchestrator_stopped")


if __name__ == "__main__":
    asyncio.run(main())
