-- =============================================================================
-- MMPP Soccer Live Trading System — Database Schema
-- =============================================================================
-- 
-- PostgreSQL 16+
-- Run once on fresh database: psql -d soccer_trading -f schema.sql
--
-- Tables: 9
-- Functions: 1
-- Indexes: 8
--
-- Referenced by: orchestration.md, phase4.md, dashboard.md
-- =============================================================================

BEGIN;

-- =============================================================================
-- 1. match_schedule — Match lifecycle management (Orchestrator)
--    Source: orchestration.md Component 1 (Scheduler)
-- =============================================================================

CREATE TABLE match_schedule (
    match_id            TEXT PRIMARY KEY,           -- Goalserve match ID (unified across all phases)
    league_id           INTEGER NOT NULL,
    kickoff_utc         TIMESTAMPTZ NOT NULL,
    phase2_trigger      TIMESTAMPTZ NOT NULL,       -- kickoff - 65 min
    phase3_trigger      TIMESTAMPTZ NOT NULL,       -- kickoff - 2 min
    kalshi_tickers      JSONB NOT NULL,             -- ["SOCCER-EPL-ARS-v-CHE-WINNER", ...]
    odds_api_event_id   TEXT,
    status              TEXT DEFAULT 'SCHEDULED',
        -- State machine:
        -- SCHEDULED → PHASE2_RUNNING → PHASE2_DONE → PHASE3_RUNNING
        --   → SETTLING → FINISHED
        -- SCHEDULED → SKIPPED (sanity check failed)
        -- PHASE3_RUNNING → FAILED (container crash)
        -- FINISHED → ARCHIVED (logs archived, container removed)
    container_id        TEXT,                        -- Docker container ID (set at Phase 3 launch)
    trading_mode        TEXT DEFAULT 'paper',        -- 'paper' or 'live'
    param_version       INTEGER,                     -- pinned at container launch (Step 1 version)
    home_team           TEXT,                        -- e.g. 'Arsenal'
    away_team           TEXT,                        -- e.g. 'Chelsea'
    phase2_params       JSONB,                       -- Phase 2 result (a_H, a_A, C_time, verdict)
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_match_schedule_trigger ON match_schedule(phase2_trigger)
    WHERE status = 'SCHEDULED';

CREATE INDEX idx_match_schedule_status ON match_schedule(status);

-- =============================================================================
-- 2. bankroll — Account balance tracking (paper/live isolated)
--    Source: orchestration.md Component 2 (Orchestrator)
-- =============================================================================

CREATE TABLE bankroll (
    id              SERIAL PRIMARY KEY,
    mode            TEXT NOT NULL DEFAULT 'live',    -- 'paper' or 'live'
    balance         NUMERIC(12, 4) NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(mode)                                     -- one row per mode
);

-- Initialize both bankrolls
INSERT INTO bankroll (mode, balance) VALUES ('live', 0);        -- real deposit
INSERT INTO bankroll (mode, balance) VALUES ('paper', 10000);   -- virtual $10,000

-- =============================================================================
-- 3. positions — Trade position tracking
--    Source: orchestration.md Component 3 + phase4.md Step 4.5/4.6
-- =============================================================================

CREATE TABLE positions (
    id                  SERIAL PRIMARY KEY,
    match_id            TEXT NOT NULL REFERENCES match_schedule(match_id),
    market_ticker       TEXT NOT NULL,               -- Kalshi ticker
    direction           TEXT NOT NULL,               -- BUY_YES | BUY_NO
    entry_price         NUMERIC(6, 4) NOT NULL,      -- Yes-space price (0.01 - 0.99)
    quantity            INTEGER NOT NULL,
    status              TEXT DEFAULT 'OPEN',
        -- PENDING → OPEN → AWAITING_SETTLEMENT → SETTLED
        -- PENDING → (deleted if order not filled)
        -- OPEN → CLOSED (early exit via Step 4.4)
    is_paper            BOOLEAN NOT NULL DEFAULT FALSE,
    entry_time          TIMESTAMPTZ NOT NULL,
    exit_time           TIMESTAMPTZ,
    exit_price          NUMERIC(6, 4),
    settlement_price    NUMERIC(6, 4),               -- 1.00 (Yes won) or 0.00 (No won)
    realized_pnl        NUMERIC(12, 4),
    fill_delay          NUMERIC(6, 3),               -- paper only: simulated fill delay (seconds)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_positions_match_status ON positions(match_id, status);
CREATE INDEX idx_positions_paper ON positions(is_paper, status);
CREATE INDEX idx_positions_settled ON positions(status, created_at)
    WHERE status = 'SETTLED';

-- =============================================================================
-- 4. exposure_cache — Per-match exposure summary (hot cache for risk limits)
--    Source: orchestration.md Component 3 (Risk Limits)
-- =============================================================================

CREATE TABLE exposure_cache (
    match_id            TEXT PRIMARY KEY REFERENCES match_schedule(match_id),
    is_paper            BOOLEAN NOT NULL DEFAULT FALSE,
    total_exposure      NUMERIC(12, 4) NOT NULL,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 5. exposure_reservation — Atomic reserve-confirm-release for cross-container safety
--    Source: orchestration.md Component 3 (Risk Limits)
-- =============================================================================

CREATE TABLE exposure_reservation (
    id              SERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES match_schedule(match_id),
    market_ticker   TEXT NOT NULL,
    amount          NUMERIC(12, 4) NOT NULL,
    is_paper        BOOLEAN NOT NULL DEFAULT FALSE,
    status          TEXT DEFAULT 'RESERVED',         -- RESERVED → CONFIRMED | RELEASED
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reservation_stale ON exposure_reservation(created_at)
    WHERE status = 'RESERVED';

-- =============================================================================
-- 6. get_total_exposure() — Cross-container exposure query (includes reservations)
--    Source: orchestration.md Component 3 (Risk Limits)
-- =============================================================================

CREATE OR REPLACE FUNCTION get_total_exposure(p_is_paper BOOLEAN)
RETURNS NUMERIC AS $$
    SELECT
        COALESCE(SUM(e.total_exposure), 0)
        + COALESCE((
            SELECT SUM(amount) FROM exposure_reservation
            WHERE is_paper = p_is_paper AND status = 'RESERVED'
        ), 0)
    FROM exposure_cache e
    WHERE e.is_paper = p_is_paper
      AND e.match_id IN (
        SELECT match_id FROM match_schedule WHERE status = 'PHASE3_RUNNING'
    );
$$ LANGUAGE SQL;

-- =============================================================================
-- 7. production_params — Phase 1 calibration results (versioned)
--    Source: orchestration.md Component 4 (Phase 1 Worker)
-- =============================================================================

CREATE TABLE production_params (
    version             SERIAL PRIMARY KEY,
    params              JSONB NOT NULL,              -- b[], gamma_H/A, delta_H/A, Q, etc.
    xgb_model_path      TEXT NOT NULL,               -- S3/local path to .xgb file
    feature_mask        JSONB NOT NULL,              -- selected features from Step 1.3
    validation          JSONB NOT NULL,              -- Step 1.5 Go/No-Go results
    sanity_thresholds   JSONB NOT NULL,              -- {go_threshold, hold_threshold, ou_threshold}
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT FALSE
);

-- Only one version is active at a time
CREATE UNIQUE INDEX idx_active_params ON production_params(is_active)
    WHERE is_active = TRUE;

-- =============================================================================
-- 8. event_log — Every in-match event (goals, cards, ob_freeze, cooldown, etc.)
--    Source: orchestration.md Component 7 (Logging)
-- =============================================================================

CREATE TABLE event_log (
    id              BIGSERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES match_schedule(match_id),
    event_type      TEXT NOT NULL,                   -- goal_confirmed, red_card, ob_freeze,
                                                     -- cooldown_start, cooldown_end, period_change,
                                                     -- var_cancel, substitution, source_failure
    source          TEXT NOT NULL,                   -- odds_api, live_score, system
    payload         JSONB,                           -- event-specific data
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_match_time ON event_log(match_id, created_at);

-- =============================================================================
-- 9. tick_snapshots — Per-second pricing snapshots (sampled in normal play)
--    Source: orchestration.md Component 7 (Logging)
--    Note: σ_MC is JSONB (per-market dict) since v3
-- =============================================================================

CREATE TABLE tick_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL,                   -- no FK for insert performance
    t               NUMERIC(6, 2) NOT NULL,          -- effective play time (minutes)
    mu_H            NUMERIC(8, 4),
    mu_A            NUMERIC(8, 4),
    P_true          JSONB,                           -- {"home_win": 0.55, "draw": 0.25, ...}
    P_kalshi        JSONB,                           -- {"home_win_ask": 0.52, "home_win_bid": 0.50, ...}
    P_bet365        JSONB,                           -- {"home_win": 0.54, ...}
    sigma_MC        JSONB,                           -- {"home_win": 0.0022, ...} (per-market)
    engine_phase    TEXT,                             -- FIRST_HALF, HALFTIME, SECOND_HALF, FINISHED
    event_state     TEXT,                             -- IDLE, PRELIMINARY_DETECTED, CONFIRMED
    cooldown        BOOLEAN,
    ob_freeze       BOOLEAN,
    order_allowed   BOOLEAN,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tick_match_time ON tick_snapshots(match_id, t);

-- Partition by month for retention management (optional, recommended for production)
-- CREATE TABLE tick_snapshots_2025_03 PARTITION OF tick_snapshots
--     FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- =============================================================================
-- 10. parameter_change_log — Adaptive tuning audit trail (Step 4.6)
--     Source: phase4.md Step 4.6 (Feedback Loop)
-- =============================================================================

CREATE TABLE parameter_change_log (
    id              SERIAL PRIMARY KEY,
    parameter_name  TEXT NOT NULL,                   -- K_frac, z, THETA_ENTRY, DIVERGENT_MULTIPLIER, etc.
    old_value       NUMERIC(10, 6),
    new_value       NUMERIC(10, 6),
    trigger_metric  TEXT,                            -- edge_realization, brier_trend, etc.
    trigger_value   NUMERIC(10, 6),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 11. bankroll_snapshot — Hourly bankroll recording for drawdown tracking
--     Source: orchestration.md CRON schedule
-- =============================================================================

CREATE TABLE bankroll_snapshot (
    id              SERIAL PRIMARY KEY,
    mode            TEXT NOT NULL,                   -- 'paper' or 'live'
    balance         NUMERIC(12, 4) NOT NULL,
    high_water_mark NUMERIC(12, 4) NOT NULL,
    drawdown_pct    NUMERIC(6, 4) NOT NULL,          -- (hwm - balance) / hwm
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_bankroll_snapshot_mode_time ON bankroll_snapshot(mode, created_at);

-- =============================================================================
-- 12. ticker_mapping — Kalshi ticker → model market key mapping
--     Source: phase4.md signal_generator (Market Key Mapping)
-- =============================================================================

CREATE TABLE ticker_mapping (
    id              SERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES match_schedule(match_id),
    kalshi_ticker   TEXT NOT NULL,                   -- "SOCCER-EPL-ARS-v-CHE-25MAR15-WINNER"
    model_key       TEXT NOT NULL,                   -- "home_win", "over_25", "btts_yes"
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(match_id, kalshi_ticker)
);

-- =============================================================================
-- Summary
-- =============================================================================
-- Tables:  12
-- Functions: 1 (get_total_exposure)
-- Indexes: 10
--
-- Estimated storage (per match, 90 min):
--   tick_snapshots: ~5,400 rows × 500B = ~2.7 MB
--   event_log: ~20-50 rows × 200B = ~10 KB
--   positions: ~1-5 rows × 200B = ~1 KB
--   Total per match: ~3 MB
--   Weekly (80 matches, 8 leagues): ~240 MB
--   Monthly: ~1 GB
--   Yearly: ~12 GB (before retention policy)
-- =============================================================================

COMMIT;
