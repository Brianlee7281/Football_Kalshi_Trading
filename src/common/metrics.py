"""Prometheus metrics definitions for the MMPP Soccer Live Trading System.

All metrics are defined here and imported by modules that need them.
Never define metrics inside individual modules — centralise here to avoid
duplicate registration errors in multi-process environments.

Reference: CLAUDE.md coding rules — "Define in src/common/metrics.py, import elsewhere."
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Phase 3: Tick loop
# ---------------------------------------------------------------------------

tick_latency: Histogram = Histogram(
    "phase3_tick_latency_seconds",
    "Wall-clock duration of a single Phase 3 tick (pricing + emit)",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

tick_overrun_total: Counter = Counter(
    "phase3_tick_overrun_total",
    "Number of ticks that exceeded 1s wall-clock budget",
    ["severity"],  # "warn" (>1s) or "critical" (>3s)
)

phase3_pricing_mode: Gauge = Gauge(
    "phase3_pricing_mode",
    "Current pricing mode: 1=MC, 0=analytical",
    ["match_id"],
)

# ---------------------------------------------------------------------------
# Phase 3: MC pricing
# ---------------------------------------------------------------------------

mc_stale_total: Counter = Counter(
    "phase3_mc_stale_total",
    "Number of MC pricing results discarded as stale",
    ["match_id"],
)

# ---------------------------------------------------------------------------
# Phase 3 → Phase 4: emit
# ---------------------------------------------------------------------------

emit_queue_full_total: Counter = Counter(
    "phase3_emit_queue_full_total",
    "Number of times the Phase 4 queue was full (stale tick replaced)",
    ["match_id"],
)

redis_publish_error_total: Counter = Counter(
    "phase3_redis_publish_error_total",
    "Number of Redis publish errors in emit_to_phase4",
    ["match_id"],
)

# ---------------------------------------------------------------------------
# Phase 4: execution
# ---------------------------------------------------------------------------

orders_placed_total: Counter = Counter(
    "phase4_orders_placed_total",
    "Total orders placed",
    ["match_id", "market", "direction", "mode"],  # mode: paper|live
)

kelly_fraction: Gauge = Gauge(
    "phase4_kelly_fraction",
    "Most recent incremental Kelly fraction for a market",
    ["match_id", "market"],
)

phase4_queue_depth: Gauge = Gauge(
    "phase4_queue_depth",
    "Phase 3→4 asyncio.Queue depth (should be 0 or 1)",
)

# ---------------------------------------------------------------------------
# Match lifecycle
# ---------------------------------------------------------------------------

matches_started_total: Counter = Counter(
    "matches_started_total",
    "Total matches started",
    ["league"],
)

matches_completed_total: Counter = Counter(
    "matches_completed_total",
    "Total matches completed",
    ["status"],  # FINISHED | FAILED | SKIPPED
)

active_match_containers: Gauge = Gauge(
    "active_match_containers",
    "Number of currently running match containers",
)

# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

orders_submitted_total: Counter = Counter(
    "orders_submitted_total",
    "Total orders submitted to Kalshi",
    ["direction", "match_id"],
)

orders_filled_total: Counter = Counter(
    "orders_filled_total",
    "Total orders that received fills",
    ["direction"],
)

position_pnl: Histogram = Histogram(
    "position_pnl",
    "Realised P&L per settled position (dollars)",
    buckets=[-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5],
)

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

total_exposure_ratio: Gauge = Gauge(
    "total_exposure_ratio",
    "Total portfolio exposure as fraction of bankroll",
)

bankroll_balance: Gauge = Gauge(
    "bankroll_balance",
    "Current bankroll balance in dollars",
    ["mode"],  # live | paper
)

max_drawdown_pct: Gauge = Gauge(
    "max_drawdown_pct",
    "Current maximum drawdown percentage",
)

# ---------------------------------------------------------------------------
# Latency: external data sources
# ---------------------------------------------------------------------------

odds_api_ws_latency: Histogram = Histogram(
    "odds_api_ws_latency_seconds",
    "Odds-API WebSocket message processing latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

live_score_poll_latency: Histogram = Histogram(
    "live_score_poll_latency_seconds",
    "Goalserve REST live-score poll latency",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0],
)

mc_compute_latency: Histogram = Histogram(
    "mc_compute_latency_seconds",
    "Monte Carlo pricing computation latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)

order_fill_latency: Histogram = Histogram(
    "order_fill_latency_seconds",
    "Kalshi order fill latency (submit to fill confirmation)",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

heartbeat_age_seconds: Gauge = Gauge(
    "heartbeat_age_seconds",
    "Seconds since last heartbeat from a match container",
    ["match_id"],
)
