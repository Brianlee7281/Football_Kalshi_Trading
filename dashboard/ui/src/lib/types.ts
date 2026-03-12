// dashboard/ui/src/lib/types.ts
//
// Shared TypeScript types for the MMPP Trading Dashboard.
// Field names match Pydantic models exactly (sigma_MC, not σ_MC).
//
// References:
//   docs/dashboard_decomposition.md Part 1.2 (WebSocket message types)
//   dashboard/api/models.py (Pydantic models)

// ── Shared value types ──────────────────────────────────────────────────────

export interface MarketProbs {
  home_win: number;
  draw: number;
  away_win: number;
  over_25?: number;
  under_25?: number;
  btts_yes?: number;
  btts_no?: number;
}

export interface Score {
  home: number;
  away: number;
}

// ── Inbound WebSocket messages (server → client) ────────────────────────────

export interface TickMessage {
  type: "tick";
  match_id: string;
  t: number;
  engine_phase: string;
  P_true: MarketProbs;
  sigma_MC: MarketProbs;
  P_bet365: MarketProbs | null;
  order_allowed: boolean;
  cooldown: boolean;
  ob_freeze: boolean;
  event_state: string;
  mu_H: number;
  mu_A: number;
  score: [number, number];
}

export interface EventMessage {
  type: "event";
  match_id: string;
  event_type: string;
  t: number;
  payload: {
    score: [number, number];
    team: string | null;
    minute: number;
    var_cancelled: boolean;
  };
}

export interface SignalMessage {
  type: "signal";
  match_id: string;
  ticker: string;
  direction: string;
  EV: number;
  P_cons: number;
  P_kalshi: number;
  alignment: string;
  kelly_fraction: number;
  fill_qty: number;
  fill_price: number;
  timestamp: number;
}

export interface PositionUpdateMessage {
  type: "new_fill" | "exit" | "settled";
  match_id: string;
  ticker: string;
  direction: string;
  quantity: number;
  price: number;
}

export interface SystemAlertMessage {
  type: "alert";
  severity: "critical" | "warning" | "info";
  title: string;
  details: Record<string, string>;
  timestamp: number;
}

export type WSMessage =
  | TickMessage
  | EventMessage
  | SignalMessage
  | PositionUpdateMessage
  | SystemAlertMessage;

// ── Outbound WebSocket messages (client → server) ───────────────────────────

export interface SubscribeMessage {
  subscribe: string[];
}

// ── REST API response types ─────────────────────────────────────────────────

export interface MatchSummary {
  match_id: string;
  league_id: number;
  kickoff_utc: string;
  status: string;
  trading_mode: string;
  home_team: string | null;
  away_team: string | null;
  score: Score | null;
  param_version: number | null;
}

export interface TickSnapshot {
  match_id: string;
  t: number;
  engine_phase: string | null;
  P_true: MarketProbs | null;
  P_kalshi: MarketProbs | null;
  P_bet365: MarketProbs | null;
  sigma_MC: MarketProbs | null;
  order_allowed: boolean | null;
  cooldown: boolean | null;
  ob_freeze: boolean | null;
  event_state: string | null;
  mu_H: number | null;
  mu_A: number | null;
  score: Score | null;
}

export interface PositionItem {
  id: number;
  match_id: string;
  market_ticker: string;
  direction: string;
  entry_price: number;
  quantity: number;
  status: string;
  is_paper: boolean;
  entry_time: string;
  exit_time: string | null;
  exit_price: number | null;
  settlement_price: number | null;
  realized_pnl: number | null;
}

export interface SignalItem {
  match_id: string;
  ticker: string;
  direction: string;
  EV: number;
  P_cons: number;
  P_kalshi: number;
  alignment: string;
  kelly_multiplier: number;
  timestamp: number;
}

export interface EventItem {
  id: number;
  match_id: string;
  event_type: string;
  source: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

export interface MatchDetail {
  match_id: string;
  league_id: number;
  kickoff_utc: string;
  status: string;
  trading_mode: string;
  home_team: string | null;
  away_team: string | null;
  score: Score | null;
  param_version: number | null;
  latest_tick: TickSnapshot | null;
  positions: PositionItem[];
  recent_events: EventItem[];
}

export interface PnLBreakdown {
  by_league: Record<string, number>;
  by_market: Record<string, number>;
  by_direction: Record<string, number>;
  by_alignment: Record<string, number>;
}

export interface PnLReport {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  edge_realization: number;
  max_drawdown_pct: number;
  sharpe: number | null;
  avg_ev: number | null;
  avg_slippage_cents: number | null;
  brier_vs_exchange: number | null;
  breakdown: PnLBreakdown;
}

export interface ModelHealthReport {
  param_version: number;
  param_trained_at: string;
  brier_score: number;
  brier_vs_exchange: number | null;
  edge_realization: number;
  matches_since_retrain: number;
  brier_by_league: Record<string, number>;
  edge_realization_rolling: number[];
}

export interface GraduationChecklist {
  trade_count: number;
  trades_ok: boolean;
  edge_realization_ok: boolean;
  brier_ok: boolean;
  max_drawdown_ok: boolean;
  realism_score_ok: boolean;
  directional_ok: boolean;
  no_crashes_ok: boolean;
  theta_calibrated: boolean;
  all_pass: boolean;
}

export interface ContainerStatus {
  match_id: string;
  status: string;
  uptime_min: number | null;
  heartbeat_age_s: number | null;
  container_id: string | null;
}

export interface ConnectionHealth {
  service: string;
  status: string;
  last_message_age_s: number | null;
  detail: string | null;
}

export interface SystemStatus {
  containers: ContainerStatus[];
  connections: ConnectionHealth[];
  param_version: number | null;
  param_trained_at: string | null;
  matches_since_retrain: number | null;
  bankroll: number | null;
  exposure_pct: number | null;
  drawdown_pct: number | null;
  trading_mode: string | null;
}
