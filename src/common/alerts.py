"""Alert routing — Slack webhook, SMS (Twilio), and Redis pubsub.

Severity routing:
  critical → SMS + Slack + Redis (system_alert channel)
  warning  → Slack + Redis
  info     → Redis only

Alert definitions from docs/dashboard.md Part 4.

Environment variables:
  SLACK_WEBHOOK        — Slack incoming-webhook URL
  TWILIO_ACCOUNT_SID   — Twilio account SID
  TWILIO_AUTH_TOKEN    — Twilio auth token
  TWILIO_FROM_NUMBER   — Twilio sender phone number
  ALERT_PHONE          — SMS recipient for critical alerts
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import redis.asyncio as aioredis

from src.common.logging import get_logger

logger = get_logger("alerts")

# ---------------------------------------------------------------------------
# Config (read from env)
# ---------------------------------------------------------------------------

SLACK_WEBHOOK: str | None = os.environ.get("SLACK_WEBHOOK")
TWILIO_ACCOUNT_SID: str | None = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN: str | None = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER: str | None = os.environ.get("TWILIO_FROM_NUMBER")
ALERT_PHONE: str | None = os.environ.get("ALERT_PHONE")

# Severity type
Severity = str  # "critical" | "warning" | "info"

_SEVERITY_EMOJI: dict[str, str] = {
    "critical": "\U0001f534",  # 🔴
    "warning": "\U0001f7e1",  # 🟡
    "info": "\u2139\ufe0f",  # ℹ️
}

# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


async def _send_slack(
    severity: Severity,
    title: str,
    details: dict[str, Any],
) -> None:
    """Post a rich message to Slack via incoming webhook."""
    if not SLACK_WEBHOOK:
        logger.warning("slack_webhook_not_configured", title=title)
        return

    emoji = _SEVERITY_EMOJI.get(severity, "")
    detail_lines = "\n".join(f"*{k}:* {v}" for k, v in details.items())

    payload: dict[str, Any] = {
        "text": f"{emoji} {title}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": detail_lines},
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(SLACK_WEBHOOK, json=payload)
            resp.raise_for_status()
        logger.info("slack_alert_sent", title=title, severity=severity)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "slack_alert_failed",
            title=title,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# SMS (Twilio)
# ---------------------------------------------------------------------------


async def _send_sms(title: str) -> None:
    """Send an SMS via Twilio for critical alerts."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_PHONE]):
        logger.warning("twilio_not_configured", title=title)
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "From": TWILIO_FROM_NUMBER,
        "To": ALERT_PHONE,
        "Body": f"MMPP CRITICAL: {title}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                data=data,
                auth=(TWILIO_ACCOUNT_SID or "", TWILIO_AUTH_TOKEN or ""),
            )
            resp.raise_for_status()
        logger.info("sms_alert_sent", title=title, phone=ALERT_PHONE)
    except Exception as exc:  # noqa: BLE001
        logger.error("sms_alert_failed", title=title, error=str(exc))


# ---------------------------------------------------------------------------
# Redis publish (dashboard AlertBanner)
# ---------------------------------------------------------------------------


async def _publish_redis(
    redis_client: aioredis.Redis,
    severity: Severity,
    title: str,
    details: dict[str, Any],
) -> None:
    """Publish alert to Redis ``system_alert`` channel for the dashboard."""
    payload = json.dumps(
        {
            "type": "alert",
            "severity": severity,
            "title": title,
            "details": {k: str(v) for k, v in details.items()},
            "timestamp": time.time(),
        }
    )
    try:
        await redis_client.publish("system_alert", payload)
        logger.debug("redis_alert_published", title=title, severity=severity)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "redis_alert_publish_failed",
            title=title,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def send_alert(
    severity: Severity,
    title: str,
    details: dict[str, Any],
    *,
    redis_client: aioredis.Redis,
) -> None:
    """Route an alert to the appropriate channels based on severity.

    Routing:
      critical → SMS + Slack + Redis
      warning  → Slack + Redis
      info     → Redis only

    Args:
        severity: ``"critical"``, ``"warning"``, or ``"info"``.
        title: Short alert title (e.g. ``"Drawdown > 15%"``).
        details: Key-value pairs with context (e.g. ``{"value": "16.2%"}``).
        redis_client: Connected async Redis client for dashboard publishing.
    """
    logger.info(
        "alert_triggered",
        severity=severity,
        title=title,
        details=details,
    )

    # Always publish to Redis (dashboard banner)
    await _publish_redis(redis_client, severity, title, details)

    # Slack for warning + critical
    if severity in ("critical", "warning"):
        await _send_slack(severity, title, details)

    # SMS for critical only
    if severity == "critical":
        await _send_sms(title)


# ---------------------------------------------------------------------------
# Convenience helpers for common alert definitions (dashboard.md Part 4)
# ---------------------------------------------------------------------------


async def alert_container_crash(
    redis_client: aioredis.Redis,
    match_id: str,
    exit_code: int,
) -> None:
    """Critical: Container exited with non-zero code."""
    await send_alert(
        "critical",
        "Container crash",
        {"match_id": match_id, "exit_code": str(exit_code)},
        redis_client=redis_client,
    )


async def alert_drawdown_exceeded(
    redis_client: aioredis.Redis,
    drawdown_pct: float,
) -> None:
    """Critical: Max drawdown exceeded 15%."""
    await send_alert(
        "critical",
        "Drawdown > 15%",
        {"value": f"{drawdown_pct:.1f}%", "action": "Stop all new entries"},
        redis_client=redis_client,
    )


async def alert_db_unreachable(redis_client: aioredis.Redis) -> None:
    """Critical: Database unreachable after 3 consecutive failures."""
    await send_alert(
        "critical",
        "DB unreachable",
        {"action": "All containers freeze"},
        redis_client=redis_client,
    )


async def alert_stale_pending(
    redis_client: aioredis.Redis,
    match_id: str,
    ticker: str,
    age_min: float,
) -> None:
    """Critical: Position stuck in PENDING > 5 min."""
    await send_alert(
        "critical",
        "Stale PENDING position",
        {"match_id": match_id, "ticker": ticker, "age": f"{age_min:.1f} min"},
        redis_client=redis_client,
    )


async def alert_bankroll_low(
    redis_client: aioredis.Redis,
    balance: float,
) -> None:
    """Critical: Bankroll below minimum $500."""
    await send_alert(
        "critical",
        "Bankroll below minimum",
        {"balance": f"${balance:.2f}", "action": "Stop all new entries"},
        redis_client=redis_client,
    )


async def alert_heartbeat_dead(
    redis_client: aioredis.Redis,
    match_id: str,
    age_s: float,
) -> None:
    """Critical: Heartbeat dead > 60s."""
    await send_alert(
        "critical",
        "Heartbeat dead",
        {"match_id": match_id, "age": f"{age_s:.0f}s"},
        redis_client=redis_client,
    )


async def alert_exposure_high(
    redis_client: aioredis.Redis,
    exposure_pct: float,
) -> None:
    """Warning: Exposure > 15%."""
    await send_alert(
        "warning",
        "Exposure > 15%",
        {"value": f"{exposure_pct:.1f}%", "action": "Review positions"},
        redis_client=redis_client,
    )


async def alert_tick_overrun(
    redis_client: aioredis.Redis,
    p99_seconds: float,
) -> None:
    """Warning: Tick overrun p99 > 3s."""
    await send_alert(
        "warning",
        "Tick overrun",
        {"p99": f"{p99_seconds:.2f}s", "action": "Check MC performance"},
        redis_client=redis_client,
    )


async def alert_odds_api_disconnected(redis_client: aioredis.Redis) -> None:
    """Warning: Odds-API WS disconnected > 30s."""
    await send_alert(
        "warning",
        "Odds-API WS disconnected",
        {"action": "Fallback mode active"},
        redis_client=redis_client,
    )


async def alert_brier_drifting(
    redis_client: aioredis.Redis,
    brier_score: float,
) -> None:
    """Warning: Brier score drifting outside baseline +/- 0.03."""
    await send_alert(
        "warning",
        "Brier Score drifting",
        {"value": f"{brier_score:.4f}", "action": "Consider retrain"},
        redis_client=redis_client,
    )


async def alert_edge_realization_low(
    redis_client: aioredis.Redis,
    edge_real: float,
) -> None:
    """Warning: Edge realization < 0.5 rolling 20 matches."""
    await send_alert(
        "warning",
        "Edge realization low",
        {"value": f"{edge_real:.2f}", "action": "Review model assumptions"},
        redis_client=redis_client,
    )


async def alert_new_param_version(
    redis_client: aioredis.Redis,
    version: int,
) -> None:
    """Info: Phase 1 retrain complete."""
    await send_alert(
        "info",
        "New param version",
        {"version": str(version), "note": "Check validation scores"},
        redis_client=redis_client,
    )


async def alert_match_skipped(
    redis_client: aioredis.Redis,
    match_id: str,
    reason: str,
) -> None:
    """Info: Phase 2 sanity = SKIP."""
    await send_alert(
        "info",
        "Match skipped",
        {"match_id": match_id, "reason": reason},
        redis_client=redis_client,
    )


async def alert_paper_graduation_ready(redis_client: aioredis.Redis) -> None:
    """Info: All 8 graduation criteria met."""
    await send_alert(
        "info",
        "Paper graduation ready",
        {"note": "Consider Phase A transition"},
        redis_client=redis_client,
    )


async def alert_adaptive_param_change(
    redis_client: aioredis.Redis,
    param_name: str,
    old_value: str,
    new_value: str,
) -> None:
    """Info: Step 4.6 adaptive tuning changed a parameter."""
    await send_alert(
        "info",
        "Adaptive param change",
        {"param": param_name, "old": old_value, "new": new_value},
        redis_client=redis_client,
    )
