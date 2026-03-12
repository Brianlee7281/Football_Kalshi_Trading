// dashboard/ui/src/components/StatusBar.tsx
//
// Top-level status bar showing bankroll, exposure, drawdown, mode, and WS status.
// Data: REST /api/system/status polled every 10s (via useSystemStatus).
//
// Edge cases:
//   - WS disconnected → 🔴 + "Connection lost — retrying..."
//   - bankroll null   → "Loading..."

"use client";

import React from "react";

import { exposureColor, formatBankroll, formatPct } from "@/lib/format";
import type { WsStatus } from "@/hooks/useWebSocket";

// ── Props ───────────────────────────────────────────────────────────────────

export interface StatusBarProps {
  bankroll: number | null;
  exposure_pct: number | null;
  drawdown_pct: number | null;
  trading_mode: string | null; // "paper" | "live"
  ws_status: WsStatus;
}

// ── WS indicator ────────────────────────────────────────────────────────────

const WS_INDICATOR: Record<WsStatus, { icon: string; label: string }> = {
  connecting: { icon: "🟡", label: "Connecting..." },
  connected: { icon: "🟢", label: "Connected" },
  reconnecting: { icon: "🟡", label: "Reconnecting..." },
  disconnected: { icon: "🔴", label: "Connection lost — retrying..." },
};

// ── Component ───────────────────────────────────────────────────────────────

export function StatusBar({
  bankroll,
  exposure_pct,
  drawdown_pct,
  trading_mode,
  ws_status,
}: StatusBarProps) {
  const ws = WS_INDICATOR[ws_status];

  // Mode badge
  const modeIcon = trading_mode === "live" ? "🟢" : "🟡";
  const modeLabel = trading_mode
    ? `${trading_mode.charAt(0).toUpperCase()}${trading_mode.slice(1)} Mode`
    : "Unknown";

  return (
    <div className="flex items-center justify-between bg-gray-900 px-4 py-2 text-sm text-gray-100">
      {/* Bankroll */}
      <span>
        Bankroll:{" "}
        {bankroll != null ? (
          <span className="font-semibold">{formatBankroll(bankroll)}</span>
        ) : (
          <span className="text-gray-400">Loading...</span>
        )}
      </span>

      {/* Exposure */}
      <span>
        Exposure:{" "}
        {exposure_pct != null ? (
          <span className={`font-semibold ${exposureColor(exposure_pct)}`}>
            {formatPct(exposure_pct)}
          </span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
        <span className="text-gray-500"> / 20%</span>
      </span>

      {/* Drawdown */}
      <span>
        Drawdown:{" "}
        {drawdown_pct != null ? (
          <span className="font-semibold">{formatPct(drawdown_pct)}</span>
        ) : (
          <span className="text-gray-400">—</span>
        )}
      </span>

      {/* Trading mode badge */}
      <span className="rounded bg-gray-800 px-2 py-0.5">
        {modeIcon} {modeLabel}
      </span>

      {/* WebSocket status */}
      <span>
        {ws.icon}{" "}
        <span
          className={
            ws_status === "connected"
              ? "text-green-500"
              : ws_status === "disconnected"
                ? "text-red-500"
                : "text-yellow-500"
          }
        >
          {ws.label}
        </span>
      </span>
    </div>
  );
}
