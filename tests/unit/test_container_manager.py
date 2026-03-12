"""Unit tests for src/orchestrator/container_manager.py.

Tests cover:
  - _parse_mem_limit: 512m, 1g, 256k, raw int
  - ContainerManager._build_env: all required keys present, values correct
  - ContainerManager.launch: correct config passed to Docker, container returned
  - ContainerManager.inspect: returns state dict
  - ContainerManager.stop: calls stop on inner container
  - ContainerManager.remove: calls delete on inner container
  - ContainerManager.archive_logs: writes log file, suppresses error
  - create_container_manager: reads env vars, uses config image
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.types import Phase2Result
from src.orchestrator.container_manager import (
    ContainerManager,
    _parse_mem_limit,
    create_container_manager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase2() -> Phase2Result:
    return Phase2Result(a_H=1.2, a_A=0.8, C_time=1.05, verdict="GO")


def _make_match(
    match_id: str = "gs-001",
    league_id: int = 1204,
    trading_mode: str = "paper",
    param_version: int | None = 3,
    kalshi_tickers: list[str] | None = None,
    odds_api_event_id: str | None = "EVT-001",
) -> dict:
    return {
        "match_id": match_id,
        "league_id": league_id,
        "trading_mode": trading_mode,
        "param_version": param_version,
        "kalshi_tickers": kalshi_tickers or ["SOCCER-EPL-ARS-CHE-YES"],
        "odds_api_event_id": odds_api_event_id,
    }


def _make_manager(tmp_path: Path) -> ContainerManager:
    return ContainerManager(
        image="soccer-live-engine:test",
        db_url="postgresql://localhost/test",
        redis_url="redis://localhost",
        odds_api_key="odds-key",
        goalserve_api_key="gs-key",
        kalshi_api_key="kalshi-key",
        log_root=tmp_path / "logs",
    )


def _make_aiodocker_mock(container_id: str = "abc123def456"):
    """Return a mock aiodocker.Docker context manager."""
    inner_container = AsyncMock()
    inner_container.id = container_id
    inner_container.start = AsyncMock()
    inner_container.stop = AsyncMock()
    inner_container.delete = AsyncMock()
    inner_container.show = AsyncMock(
        return_value={"State": {"Status": "running", "ExitCode": 0}}
    )
    inner_container.log = AsyncMock(return_value=[b"log line 1\n", b"log line 2\n"])

    containers_mock = MagicMock()
    containers_mock.create = AsyncMock(return_value=inner_container)
    containers_mock.get = AsyncMock(return_value=inner_container)

    docker_instance = MagicMock()
    docker_instance.containers = containers_mock
    docker_instance.__aenter__ = AsyncMock(return_value=docker_instance)
    docker_instance.__aexit__ = AsyncMock(return_value=None)

    return docker_instance, inner_container


# ---------------------------------------------------------------------------
# _parse_mem_limit
# ---------------------------------------------------------------------------


def test_parse_mem_limit_megabytes() -> None:
    assert _parse_mem_limit("512m") == 512 * 1024**2


def test_parse_mem_limit_gigabytes() -> None:
    assert _parse_mem_limit("1g") == 1024**3


def test_parse_mem_limit_kilobytes() -> None:
    assert _parse_mem_limit("256k") == 256 * 1024


def test_parse_mem_limit_raw_int() -> None:
    assert _parse_mem_limit("1073741824") == 1073741824


def test_parse_mem_limit_uppercase() -> None:
    assert _parse_mem_limit("512M") == 512 * 1024**2


# ---------------------------------------------------------------------------
# ContainerManager._build_env
# ---------------------------------------------------------------------------


def test_build_env_all_keys_present(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match()
    env = manager._build_env("gs-001", match, _make_phase2())

    required_keys = {
        "MATCH_ID", "LEAGUE_ID", "TRADING_MODE", "PARAM_VERSION",
        "A_H", "A_A", "C_TIME", "KALSHI_TICKERS", "ODDS_API_EVENT_ID",
        "DB_URL", "REDIS_URL", "ODDS_API_KEY", "GOALSERVE_API_KEY", "KALSHI_API_KEY",
    }
    assert required_keys.issubset(env.keys())


def test_build_env_phase2_values(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match()
    env = manager._build_env("gs-001", match, _make_phase2())
    assert env["A_H"] == "1.2"
    assert env["A_A"] == "0.8"
    assert env["C_TIME"] == "1.05"


def test_build_env_kalshi_tickers_json(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match(kalshi_tickers=["SOCCER-EPL-ARS-CHE-YES", "SOCCER-EPL-ARS-CHE-NO"])
    env = manager._build_env("gs-001", match, _make_phase2())
    tickers = json.loads(env["KALSHI_TICKERS"])
    assert tickers == ["SOCCER-EPL-ARS-CHE-YES", "SOCCER-EPL-ARS-CHE-NO"]


def test_build_env_trading_mode(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match(trading_mode="live")
    env = manager._build_env("gs-001", match, _make_phase2())
    assert env["TRADING_MODE"] == "live"


def test_build_env_api_keys_from_constructor(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match()
    env = manager._build_env("gs-001", match, _make_phase2())
    assert env["ODDS_API_KEY"] == "odds-key"
    assert env["GOALSERVE_API_KEY"] == "gs-key"
    assert env["KALSHI_API_KEY"] == "kalshi-key"


def test_build_env_null_param_version_becomes_empty(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match(param_version=None)
    env = manager._build_env("gs-001", match, _make_phase2())
    assert env["PARAM_VERSION"] == ""


# ---------------------------------------------------------------------------
# ContainerManager.launch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_creates_and_starts_container(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    match = _make_match()
    docker_instance, inner = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        container = await manager.launch(
            match_id="gs-001", match=match, phase2_result=_make_phase2()
        )

    docker_instance.containers.create.assert_called_once()
    inner.start.assert_called_once()
    assert container is inner


@pytest.mark.asyncio
async def test_launch_config_has_env_list(tmp_path: Path) -> None:
    """Config passed to docker.containers.create must have Env as list of KEY=VALUE."""
    manager = _make_manager(tmp_path)
    match = _make_match()
    docker_instance, _ = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.launch(match_id="gs-001", match=match, phase2_result=_make_phase2())

    config_arg = docker_instance.containers.create.call_args.kwargs["config"]
    env_list: list[str] = config_arg["Env"]
    env_dict = dict(item.split("=", 1) for item in env_list)
    assert env_dict["MATCH_ID"] == "gs-001"
    assert env_dict["TRADING_MODE"] == "paper"


@pytest.mark.asyncio
async def test_launch_resource_limits(tmp_path: Path) -> None:
    """Container must be launched with memory limit and cpu quota."""
    manager = _make_manager(tmp_path)
    match = _make_match()
    docker_instance, _ = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.launch(match_id="gs-001", match=match, phase2_result=_make_phase2())

    config_arg = docker_instance.containers.create.call_args.kwargs["config"]
    host_cfg = config_arg["HostConfig"]
    assert host_cfg["Memory"] == 512 * 1024**2
    assert host_cfg["CpuQuota"] == 50_000
    assert host_cfg["RestartPolicy"]["Name"] == "no"


@pytest.mark.asyncio
async def test_launch_container_name(tmp_path: Path) -> None:
    """Container name must be match-<match_id>."""
    manager = _make_manager(tmp_path)
    match = _make_match(match_id="gs-007")
    docker_instance, _ = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.launch(match_id="gs-007", match=match, phase2_result=_make_phase2())

    name_arg = docker_instance.containers.create.call_args.kwargs["name"]
    assert name_arg == "match-gs-007"


@pytest.mark.asyncio
async def test_launch_network_mode(tmp_path: Path) -> None:
    """Container must join the compose network via HostConfig.NetworkMode."""
    manager = ContainerManager(
        image="soccer-live-engine:test",
        compose_network="mmpp-net",
        log_root=tmp_path / "logs",
    )
    match = _make_match()
    docker_instance, _ = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.launch(match_id="gs-001", match=match, phase2_result=_make_phase2())

    config_arg = docker_instance.containers.create.call_args.kwargs["config"]
    assert config_arg["HostConfig"]["NetworkMode"] == "mmpp-net"


# ---------------------------------------------------------------------------
# ContainerManager.inspect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_returns_state(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    docker_instance, inner = _make_aiodocker_mock()
    inner.show = AsyncMock(return_value={"State": {"Status": "exited", "ExitCode": 0}})

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        result = await manager.inspect(inner)

    assert result["State"]["Status"] == "exited"
    assert result["State"]["ExitCode"] == 0


# ---------------------------------------------------------------------------
# ContainerManager.stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_calls_container_stop(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    docker_instance, inner = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.stop(inner)

    inner.stop.assert_called_once_with(timeout=10)


# ---------------------------------------------------------------------------
# ContainerManager.remove
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_calls_container_delete(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    docker_instance, inner = _make_aiodocker_mock()

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        await manager.remove(inner)

    inner.delete.assert_called_once_with(force=False)


# ---------------------------------------------------------------------------
# ContainerManager.archive_logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_logs_writes_file(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    docker_instance, inner = _make_aiodocker_mock()
    inner.log = AsyncMock(return_value=["log line 1\n", "log line 2\n"])
    inner.id = "abc123def456"

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        log_path = await manager.archive_logs("gs-001", inner)

    assert log_path.exists()
    content = log_path.read_text()
    assert "log line 1" in content


@pytest.mark.asyncio
async def test_archive_logs_returns_path_on_error(tmp_path: Path) -> None:
    """Errors during log fetch are suppressed; intended path still returned."""
    manager = _make_manager(tmp_path)
    docker_instance, inner = _make_aiodocker_mock()
    inner.log = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
    inner.id = "abc123def456"

    with patch("src.orchestrator.container_manager.aiodocker.Docker", return_value=docker_instance):
        log_path = await manager.archive_logs("gs-001", inner)

    # Path returned even though write failed
    assert log_path.name == "abc123def456.log"


# ---------------------------------------------------------------------------
# create_container_manager
# ---------------------------------------------------------------------------


def test_create_container_manager_reads_env(tmp_path: Path) -> None:
    config: dict = {"docker": {"image": "custom-engine:v2"}}
    env_overrides = {
        "DB_URL": "postgresql://prod",
        "REDIS_URL": "redis://prod",
        "ODDS_API_KEY": "odds-prod",
        "GOALSERVE_API_KEY": "gs-prod",
        "KALSHI_API_KEY": "kalshi-prod",
        "COMPOSE_NETWORK": "custom-net",
    }
    with patch.dict(os.environ, env_overrides):
        manager = create_container_manager(config)

    assert manager._image == "custom-engine:v2"
    assert manager._db_url == "postgresql://prod"
    assert manager._kalshi_api_key == "kalshi-prod"
    assert manager._compose_network == "custom-net"


def test_create_container_manager_default_network(tmp_path: Path) -> None:
    """COMPOSE_NETWORK defaults to mmpp-net when env var not set."""
    env = {k: v for k, v in os.environ.items() if k != "COMPOSE_NETWORK"}
    with patch.dict(os.environ, env, clear=True):
        manager = create_container_manager({})
    assert manager._compose_network == "mmpp-net"


def test_create_container_manager_defaults_image(tmp_path: Path) -> None:
    with patch.dict(os.environ, {}, clear=False):
        manager = create_container_manager({})
    assert manager._image == "soccer-live-engine:latest"
