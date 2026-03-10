"""Load config/system.yaml with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "system.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides.

    Convention: MMPP_<SECTION>__<KEY> overrides config[section][key].
    Example: MMPP_TRADING__MODE=live -> config["trading"]["mode"] = "live"
    """
    prefix = "MMPP_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        if len(parts) != 2:
            continue
        section, key = parts
        if section in config and isinstance(config[section], dict):
            existing = config[section].get(key)
            if existing is not None:
                config[section][key] = _cast_value(env_value, type(existing))
            else:
                config[section][key] = env_value
    return config


def _cast_value(value: str, target_type: type[Any]) -> Any:
    """Cast a string env value to the target type."""
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def load_config(
    config_path: Path | str | None = None,
    overlay_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load system configuration from YAML with optional overlay and env overrides.

    Args:
        config_path: Path to base system.yaml. Defaults to config/system.yaml.
        overlay_path: Optional overlay file (e.g., system.paper.yaml).

    Returns:
        Merged configuration dict.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    if overlay_path:
        overlay = Path(overlay_path)
        if overlay.exists():
            with open(overlay) as f:
                overlay_data: dict[str, Any] = yaml.safe_load(f) or {}
            config = _deep_merge(config, overlay_data)

    config = _apply_env_overrides(config)

    return config
