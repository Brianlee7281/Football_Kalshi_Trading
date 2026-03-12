// dashboard/ui/src/components/PositionTable.tsx
//
// Columns: Market | Dir | Entry | Qty | Current | Unreal P&L | Status
// Current price: from latest tick P_kalshi.
// Unrealized P&L: computed client-side, directional.
// Color: positive green, negative red.
// Status badges: OPEN 🟢, AWAITING_SETTLEMENT 🟡, SETTLED ⚪
//
// Edge case: 0 positions → "No open positions for this match"

"use client";

import React from "react";

import type { MarketProbs, PositionItem } from "@/lib/types";
import { directionBg, formatCents, formatPnL, pnlColor } from "@/lib/format";

// ── Helpers ─────────────────────────────────────────────────────────────────

const STATUS_BADGES: Record<string, { icon: string; color: string }> = {
  OPEN: { icon: "🟢", color: "text-green-700" },
  AWAITING_SETTLEMENT: { icon: "🟡", color: "text-yellow-700" },
  SETTLED: { icon: "⚪", color: "text-gray-500" },
  PENDING: { icon: "⏳", color: "text-gray-400" },
  CLOSED: { icon: "⚪", color: "text-gray-500" },
};

/**
 * Compute unrealized P&L.
 * BUY_YES: qty * (current - entry)
 * BUY_NO:  qty * (entry - current)   (profit when price drops)
 */
function unrealizedPnl(
  position: PositionItem,
  currentPrice: number | null,
): number | null {
  if (currentPrice == null) return null;
  if (position.direction === "BUY_YES") {
    return position.quantity * (currentPrice - position.entry_price);
  }
  // BUY_NO
  return position.quantity * (position.entry_price - currentPrice);
}

/** Map a ticker to its market key in MarketProbs (best-effort heuristic). */
function tickerToMarketKey(ticker: string): string | null {
  const lower = ticker.toLowerCase();
  if (lower.includes("winner") || lower.includes("home")) return "home_win";
  if (lower.includes("draw")) return "draw";
  if (lower.includes("away")) return "away_win";
  if (lower.includes("over")) return "over_25";
  if (lower.includes("under")) return "under_25";
  if (lower.includes("btts")) return "btts_yes";
  return null;
}

// ── Props ───────────────────────────────────────────────────────────────────

export interface PositionTableProps {
  positions: PositionItem[];
  P_kalshi: MarketProbs | null;
}

// ── Component ───────────────────────────────────────────────────────────────

export function PositionTable({ positions, P_kalshi }: PositionTableProps) {
  if (positions.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        No open positions for this match
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="w-full text-xs">
        <thead className="bg-gray-100 text-left text-gray-600">
          <tr>
            <th className="px-3 py-2">Market</th>
            <th className="px-3 py-2">Dir</th>
            <th className="px-3 py-2">Entry</th>
            <th className="px-3 py-2">Qty</th>
            <th className="px-3 py-2">Current</th>
            <th className="px-3 py-2">Unreal P&L</th>
            <th className="px-3 py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos) => {
            const marketKey = tickerToMarketKey(pos.market_ticker);
            const currentPrice =
              P_kalshi && marketKey
                ? (P_kalshi[marketKey as keyof MarketProbs] ?? null)
                : null;
            const pnl = unrealizedPnl(pos, currentPrice);
            const badge = STATUS_BADGES[pos.status] ?? STATUS_BADGES.PENDING;

            return (
              <tr
                key={pos.id}
                className={`border-t border-gray-100 ${directionBg(pos.direction)}`}
              >
                <td className="px-3 py-1.5 font-mono">
                  {pos.market_ticker}
                </td>
                <td className="px-3 py-1.5">
                  <span
                    className={
                      pos.direction === "BUY_YES"
                        ? "font-semibold text-green-700"
                        : "font-semibold text-red-700"
                    }
                  >
                    {pos.direction}
                  </span>
                </td>
                <td className="px-3 py-1.5">
                  {formatCents(pos.entry_price)}
                </td>
                <td className="px-3 py-1.5">{pos.quantity}</td>
                <td className="px-3 py-1.5">
                  {currentPrice != null ? formatCents(currentPrice) : "—"}
                </td>
                <td className={`px-3 py-1.5 font-semibold ${pnl != null ? pnlColor(pnl) : "text-gray-400"}`}>
                  {pnl != null ? formatPnL(pnl) : "—"}
                </td>
                <td className={`px-3 py-1.5 ${badge.color}`}>
                  {badge.icon} {pos.status}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
