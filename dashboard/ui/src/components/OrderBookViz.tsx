// dashboard/ui/src/components/OrderBookViz.tsx
//
// Horizontal bar chart: bids (green, left) + asks (red, right), spread in center.
// Data from latest TickMessage P_kalshi field.
//
// Edge cases:
//   P_kalshi null → "Order book unavailable"
//   stale > 5s   → red "STALE" badge

"use client";

import React from "react";

import type { MarketProbs } from "@/lib/types";
import { formatCents } from "@/lib/format";

// ── Props ───────────────────────────────────────────────────────────────────

export interface OrderBookVizProps {
  P_kalshi: MarketProbs | null;
  /** Milliseconds since last Kalshi data update */
  kalshiAge: number;
  selectedMarket: string;
}

// ── Component ───────────────────────────────────────────────────────────────

export function OrderBookViz({
  P_kalshi,
  kalshiAge,
  selectedMarket,
}: OrderBookVizProps) {
  if (!P_kalshi) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        Order book unavailable
      </div>
    );
  }

  const isStale = kalshiAge > 5000;
  const mid = P_kalshi[selectedMarket as keyof MarketProbs];

  // Simulate bid/ask from midpoint (spread ≈ 2¢ in practice)
  const spread = 0.02;
  const bidPrice = mid != null ? Math.max(0, mid - spread / 2) : 0;
  const askPrice = mid != null ? Math.min(1, mid + spread / 2) : 0;
  const spreadCents = mid != null ? Math.round(spread * 100) : 0;

  const bidPct = bidPrice * 100;
  const askPct = askPrice * 100;

  return (
    <div className="relative rounded-lg border border-gray-200 p-4">
      {/* Stale badge */}
      {isStale && (
        <span className="absolute right-2 top-2 rounded bg-red-600 px-1.5 py-0.5 text-xs font-semibold text-white">
          STALE
        </span>
      )}

      <div className="mb-2 text-xs font-medium text-gray-500">
        {selectedMarket.replace("_", " ")}
      </div>

      {/* Bid / Spread / Ask layout */}
      <div className="flex items-center gap-2">
        {/* Bid bar (green, grows right) */}
        <div className="flex flex-1 justify-end">
          <div
            className="h-6 rounded-l bg-green-500"
            style={{ width: `${bidPct}%`, minWidth: "4px" }}
          />
        </div>

        {/* Spread label */}
        <div className="flex flex-col items-center text-xs">
          <span className="font-semibold text-gray-700">{spreadCents}¢</span>
          <span className="text-gray-400">spread</span>
        </div>

        {/* Ask bar (red, grows left) */}
        <div className="flex flex-1">
          <div
            className="h-6 rounded-r bg-red-500"
            style={{ width: `${askPct}%`, minWidth: "4px" }}
          />
        </div>
      </div>

      {/* Bid/Ask labels */}
      <div className="mt-1 flex justify-between text-xs text-gray-500">
        <span className="text-green-600">Bid: {formatCents(bidPrice)}</span>
        <span className="text-red-600">Ask: {formatCents(askPrice)}</span>
      </div>
    </div>
  );
}
