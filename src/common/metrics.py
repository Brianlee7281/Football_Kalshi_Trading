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
