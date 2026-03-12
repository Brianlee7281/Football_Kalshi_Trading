"""Docker container management for Phase 3+4 match engine containers.

Each live match runs in an isolated Docker container with pinned environment:
  TRADING_MODE, PARAM_VERSION, A_H, A_A, C_TIME, KALSHI_TICKERS, API keys.

Resource limits: 512MB RAM, 0.5 CPU (cpu_quota=50_000).
Restart policy: "no" — trading containers must NOT auto-restart.

Log archival: on container exit, full stdout/stderr is fetched from Docker
and written to logs/containers/<match_id>/<container_id>.log.

Reference: docs/orchestration.md Component 2: Orchestrator — Lifecycle Manager
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiodocker
import aiodocker.containers

from src.common.logging import get_logger
from src.common.types import Phase2Result

logger = get_logger("container_manager")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_IMAGE_NAME: str = "soccer-live-engine:latest"
_MEM_LIMIT: str = "512m"
_CPU_QUOTA: int = 50_000  # 0.5 CPU (100_000 = 1 full CPU in Docker)
_LOG_ROOT: Path = Path("logs") / "containers"
# Default compose network name — overridden by COMPOSE_NETWORK env var.
# Must match the `name:` field of the mmpp-net network in docker-compose.yml
# so spawned match-engine containers can resolve "postgres" and "redis".
_DEFAULT_NETWORK: str = "mmpp-net"


# ---------------------------------------------------------------------------
# ContainerManager
# ---------------------------------------------------------------------------


class ContainerManager:
    """Manage Docker containers for Phase 3+4 match engine instances.

    Uses ``aiodocker`` for all async Docker operations.  One instance is
    shared by the Orchestrator and can manage multiple concurrent containers.

    Args:
        image: Docker image name and tag.
        db_url: PostgreSQL connection URL (injected into container env).
        redis_url: Redis connection URL (injected into container env).
        odds_api_key: Odds-API key (injected into container env).
        goalserve_api_key: Goalserve API key (injected into container env).
        kalshi_api_key: Kalshi API key (injected into container env).
        log_root: Root directory for archived container logs.
    """

    def __init__(
        self,
        *,
        image: str = _IMAGE_NAME,
        db_url: str = "",
        redis_url: str = "",
        odds_api_key: str = "",
        goalserve_api_key: str = "",
        kalshi_api_key: str = "",
        log_root: Path = _LOG_ROOT,
        compose_network: str = _DEFAULT_NETWORK,
    ) -> None:
        self._image = image
        self._db_url = db_url
        self._redis_url = redis_url
        self._odds_api_key = odds_api_key
        self._goalserve_api_key = goalserve_api_key
        self._kalshi_api_key = kalshi_api_key
        self._log_root = log_root
        self._compose_network = compose_network

    # ------------------------------------------------------------------
    # Public interface (matches lifecycle.py expectations)
    # ------------------------------------------------------------------

    async def launch(
        self,
        *,
        match_id: str,
        match: Any,
        phase2_result: Phase2Result,
    ) -> aiodocker.containers.DockerContainer:
        """Run a match-engine container and return the container object.

        Args:
            match_id: Goalserve match ID.
            match: Row from match_schedule (asyncpg Record or MatchSchedule).
            phase2_result: Back-solved Phase 2 intensities.

        Returns:
            Running ``aiodocker.containers.DockerContainer`` instance.

        Raises:
            aiodocker.exceptions.DockerError: If Docker daemon rejects the
                create/start request.
        """
        env = self._build_env(match_id, match, phase2_result)
        league_id: str = str(
            match["league_id"] if hasattr(match, "__getitem__") else match.league_id
        )
        kalshi_tickers: list[str] = (
            list(match["kalshi_tickers"]) if hasattr(match, "__getitem__")
            else list(getattr(match, "kalshi_tickers", []))
        )

        container_name = f"match-{match_id}"
        config: dict[str, Any] = {
            "Image": self._image,
            "Env": [f"{k}={v}" for k, v in env.items()],
            "HostConfig": {
                "Memory": _parse_mem_limit(_MEM_LIMIT),
                "CpuQuota": _CPU_QUOTA,
                "RestartPolicy": {"Name": "no"},
                # Join the docker-compose network so the container can
                # resolve "postgres" and "redis" by hostname.
                "NetworkMode": self._compose_network,
            },
            "Labels": {
                "service": "match-engine",
                "match_id": match_id,
                "league": league_id,
            },
        }

        async with aiodocker.Docker() as docker:
            container = await docker.containers.create(
                config=config,
                name=container_name,
            )
            await container.start()

        logger.info(
            "container_launched",
            match_id=match_id,
            container_id=container.id[:12],
            image=self._image,
            tickers=kalshi_tickers,
        )
        return container

    async def inspect(
        self, container: aiodocker.containers.DockerContainer
    ) -> dict[str, Any]:
        """Return the raw Docker inspect dict for a container.

        Args:
            container: Container object returned by ``launch``.

        Returns:
            Dict with at minimum ``{"State": {"Status": ..., "ExitCode": ...}}``.
        """
        async with aiodocker.Docker() as docker:
            c = await docker.containers.get(container.id)
            info: dict[str, Any] = await c.show()
        return info

    async def stop(
        self,
        container: aiodocker.containers.DockerContainer,
        *,
        timeout: int = 10,
    ) -> None:
        """Send SIGTERM to the container, then SIGKILL after *timeout* seconds.

        Args:
            container: Container object returned by ``launch``.
            timeout: Seconds to wait for graceful shutdown before SIGKILL.
        """
        async with aiodocker.Docker() as docker:
            c = await docker.containers.get(container.id)
            await c.stop(timeout=timeout)
        logger.info("container_stopped", container_id=container.id[:12])

    async def remove(
        self,
        container: aiodocker.containers.DockerContainer,
        *,
        force: bool = False,
    ) -> None:
        """Remove the container (must be stopped first unless *force* is True).

        Args:
            container: Container object returned by ``launch``.
            force: If True, force-remove even if running.
        """
        async with aiodocker.Docker() as docker:
            c = await docker.containers.get(container.id)
            await c.delete(force=force)
        logger.info("container_removed", container_id=container.id[:12])

    async def archive_logs(
        self,
        match_id: str,
        container: aiodocker.containers.DockerContainer,
    ) -> Path:
        """Fetch container stdout/stderr and write to disk.

        Logs are written to ``<log_root>/<match_id>/<container_id_short>.log``.

        Args:
            match_id: Goalserve match ID (used as subdirectory).
            container: Container object returned by ``launch``.

        Returns:
            Path to the written log file, or the intended path if write failed.
        """
        short_id = container.id[:12]
        log_dir = self._log_root / match_id
        log_path = log_dir / f"{short_id}.log"

        try:
            async with aiodocker.Docker() as docker:
                c = await docker.containers.get(container.id)
                logs: bytes | list[str] = await c.log(stdout=True, stderr=True)

            log_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(logs, list):
                log_path.write_text("".join(logs), encoding="utf-8")
            else:
                log_path.write_bytes(logs if isinstance(logs, bytes) else logs.encode())

            logger.info(
                "container_logs_archived",
                match_id=match_id,
                container_id=short_id,
                path=str(log_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "container_log_archive_failed",
                match_id=match_id,
                container_id=short_id,
                error=str(exc),
            )

        return log_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(
        self,
        match_id: str,
        match: Any,
        phase2_result: Phase2Result,
    ) -> dict[str, str]:
        """Build the environment dict to inject into the container.

        Args:
            match_id: Goalserve match ID.
            match: Row from match_schedule.
            phase2_result: Phase 2 back-solve results.

        Returns:
            Dict mapping env-var name → string value.
        """
        league_id: str = str(
            match["league_id"] if hasattr(match, "__getitem__") else match.league_id
        )
        trading_mode: str = str(
            match.get("trading_mode", "paper")
            if hasattr(match, "get")
            else getattr(match, "trading_mode", "paper")
        )
        param_version: str = str(
            match.get("param_version") or ""
            if hasattr(match, "get")
            else str(getattr(match, "param_version", "") or "")
        )
        kalshi_tickers: str = json.dumps(
            list(match["kalshi_tickers"]) if hasattr(match, "__getitem__")
            else list(getattr(match, "kalshi_tickers", []))
        )
        odds_api_event_id: str = str(
            match.get("odds_api_event_id") or ""
            if hasattr(match, "get")
            else str(getattr(match, "odds_api_event_id", "") or "")
        )

        return {
            # Match identity
            "MATCH_ID": match_id,
            "LEAGUE_ID": league_id,
            # Trading mode + params (pinned at launch — never reloaded mid-match)
            "TRADING_MODE": trading_mode,
            "PARAM_VERSION": param_version,
            # Phase 2 back-solved intensities
            "A_H": str(phase2_result.a_H),
            "A_A": str(phase2_result.a_A),
            "C_TIME": str(phase2_result.C_time),
            # Market identifiers
            "KALSHI_TICKERS": kalshi_tickers,
            "ODDS_API_EVENT_ID": odds_api_event_id,
            # Infrastructure URLs
            "DB_URL": self._db_url,
            "REDIS_URL": self._redis_url,
            # API keys
            "ODDS_API_KEY": self._odds_api_key,
            "GOALSERVE_API_KEY": self._goalserve_api_key,
            "KALSHI_API_KEY": self._kalshi_api_key,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_container_manager(config: dict[str, Any]) -> ContainerManager:
    """Build a ContainerManager from the system config dict.

    Reads API keys from environment variables (never from config files).

    Args:
        config: Loaded system config (from ``src.common.config_loader.load_config``).

    Returns:
        Ready-to-use ContainerManager instance.
    """
    return ContainerManager(
        image=config.get("docker", {}).get("image", _IMAGE_NAME),
        db_url=os.environ.get("DB_URL", ""),
        redis_url=os.environ.get("REDIS_URL", ""),
        odds_api_key=os.environ.get("ODDS_API_KEY", ""),
        goalserve_api_key=os.environ.get("GOALSERVE_API_KEY", ""),
        kalshi_api_key=os.environ.get("KALSHI_API_KEY", ""),
        # COMPOSE_NETWORK must match the `name:` field of the mmpp-net
        # network defined in docker-compose.yml so spawned containers
        # can resolve postgres/redis by hostname.
        compose_network=os.environ.get("COMPOSE_NETWORK", _DEFAULT_NETWORK),
    )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _parse_mem_limit(limit: str) -> int:
    """Convert a Docker memory string (e.g. '512m') to bytes.

    Args:
        limit: Memory string like '512m', '1g', '256k'.

    Returns:
        Integer byte count.

    Raises:
        ValueError: If the suffix is not recognised.
    """
    limit = limit.strip().lower()
    suffixes: dict[str, int] = {"k": 1024, "m": 1024**2, "g": 1024**3}
    for suffix, multiplier in suffixes.items():
        if limit.endswith(suffix):
            return int(limit[: -len(suffix)]) * multiplier
    return int(limit)
