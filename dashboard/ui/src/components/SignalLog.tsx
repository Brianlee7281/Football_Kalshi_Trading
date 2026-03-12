// dashboard/ui/src/components/SignalLog.tsx
//
// Scrollable table of trading signals, newest at top, max 50 visible.
// Columns: Time | Ticker | Direction | EV | Kelly% | Alignment | Outcome
// Color: BUY_YES green tint, BUY_NO red tint, HOLD gray.
//
// Edge case: 0 signals → "No signals generated yet"

"use client";

import React from "react";

import type { SignalItem } from "@/lib/types";
import { directionBg, formatEdge, formatPct, formatTimestamp } from "@/lib/format";

// ── Props ───────────────────────────────────────────────────────────────────

export interface SignalLogProps {
  signals: SignalItem[];
}

// ── Component ───────────────────────────────────────────────────────────────

export function SignalLog({ signals }: SignalLogProps) {
  if (signals.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        No signals generated yet
      </div>
    );
  }

  // Newest first, max 50
  const sorted = [...signals]
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 50);

  return (
    <div className="max-h-96 overflow-y-auto rounded-lg border border-gray-200">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-100 text-left text-gray-600">
          <tr>
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Ticker</th>
            <th className="px-3 py-2">Direction</th>
            <th className="px-3 py-2">EV</th>
            <th className="px-3 py-2">Kelly%</th>
            <th className="px-3 py-2">Alignment</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((signal, i) => (
            <tr
              key={`${signal.ticker}-${signal.timestamp}-${i}`}
              className={`border-t border-gray-100 ${directionBg(signal.direction)}`}
            >
              <td className="px-3 py-1.5 text-gray-500">
                {formatTimestamp(signal.timestamp)}
              </td>
              <td className="px-3 py-1.5 font-mono">{signal.ticker}</td>
              <td className="px-3 py-1.5">
                <span
                  className={
                    signal.direction === "BUY_YES"
                      ? "font-semibold text-green-700"
                      : signal.direction === "BUY_NO"
                        ? "font-semibold text-red-700"
                        : "text-gray-500"
                  }
                >
                  {signal.direction}
                </span>
              </td>
              <td className="px-3 py-1.5">{formatEdge(signal.EV)}</td>
              <td className="px-3 py-1.5">
                {formatPct(signal.kelly_multiplier)}
              </td>
              <td className="px-3 py-1.5">
                <span
                  className={`rounded px-1.5 py-0.5 text-xs ${
                    signal.alignment === "ALIGNED"
                      ? "bg-green-100 text-green-700"
                      : signal.alignment === "DIVERGENT"
                        ? "bg-red-100 text-red-700"
                        : "bg-gray-100 text-gray-500"
                  }`}
                >
                  {signal.alignment}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
