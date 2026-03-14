// dashboard/ui/src/components/PriceChart.tsx
//
// Recharts LineChart showing P_true, P_kalshi, P_bet365 with σ_MC confidence band.
// Event annotations as vertical dashed lines at goal/red card t values.
// Market selector tabs: home_win | draw | away_win | over_25.
//
// Edge cases:
//   0 ticks   → "Waiting for match to start..."
//   < 10 ticks → chart + "Data collecting..." label

"use client";

import React, { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { EventItem, TickSnapshot } from "@/lib/types";
import { formatProb, formatTime } from "@/lib/format";

// ── Constants ───────────────────────────────────────────────────────────────

const MARKET_TABS = ["home_win", "draw", "away_win", "over_25"] as const;
type MarketKey = (typeof MARKET_TABS)[number];

const TAB_LABELS: Record<MarketKey, string> = {
  home_win: "Home Win",
  draw: "Draw",
  away_win: "Away Win",
  over_25: "Over 2.5",
};

// Event types that get annotation lines
const ANNOTATION_EVENTS = new Set(["goal_confirmed", "red_card"]);

// ── Props ───────────────────────────────────────────────────────────────────

export interface PriceChartProps {
  ticks: TickSnapshot[];
  events: EventItem[];
}

// ── Component ───────────────────────────────────────────────────────────────

export function PriceChart({ ticks, events }: PriceChartProps) {
  const [market, setMarket] = useState<MarketKey>("home_win");

  // Build chart data
  const chartData = useMemo(
    () =>
      ticks.map((tick) => {
        const pTrue = tick.P_true?.[market] ?? null;
        const sigma = tick.sigma_MC?.[market] ?? 0;
        const upper = pTrue != null ? Math.min(1, pTrue + 1.96 * sigma) : null;
        const lower = pTrue != null ? Math.max(0, pTrue - 1.96 * sigma) : null;
        return {
          t: tick.t,
          P_true: pTrue,
          P_kalshi: tick.P_kalshi?.[market] ?? null,
          P_bet365: tick.P_bet365?.[market] ?? null,
          // Recharts range area: [lower, upper] tuple for the confidence band
          band: upper != null && lower != null ? [lower, upper] : null,
        };
      }),
    [ticks, market],
  );

  // Event annotation lines (goal, red card) with score labels
  const annotations = useMemo(
    () =>
      events
        .filter((e) => ANNOTATION_EVENTS.has(e.event_type))
        .map((e) => {
          const payload = e.payload as Record<string, unknown> | null;
          const minute = (payload?.minute as number) ?? 0;
          const score = payload?.score as [number, number] | undefined;
          const icon = e.event_type === "goal_confirmed" ? "⚽" : "🟥";
          const scoreStr = score ? `${score[0]}-${score[1]}` : "";
          return {
            t: minute,
            type: e.event_type,
            label: `${scoreStr} ${icon} ${minute}'`,
          };
        }),
    [events],
  );

  // ── Empty states ────────────────────────────────────────────────────────

  if (ticks.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        Waiting for match to start...
      </div>
    );
  }

  return (
    <div>
      {/* Market selector tabs */}
      <div className="mb-3 flex gap-1">
        {MARKET_TABS.map((key) => (
          <button
            key={key}
            onClick={() => setMarket(key)}
            className={`rounded px-3 py-1 text-xs font-medium ${
              market === key
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {TAB_LABELS[key]}
          </button>
        ))}
      </div>

      {/* Data collecting label */}
      {ticks.length < 10 && (
        <div className="mb-2 text-xs text-yellow-600">Data collecting...</div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="t"
            type="number"
            domain={[0, "auto"]}
            tickFormatter={(v: number) => formatTime(v)}
            label={{ value: "Time", position: "insideBottomRight", offset: -5 }}
            stroke="#9ca3af"
            fontSize={11}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => formatProb(v)}
            stroke="#9ca3af"
            fontSize={11}
          />
          <Tooltip
            formatter={(value: number) => formatProb(value)}
            labelFormatter={(label: number) => formatTime(label)}
          />

          {/* σ_MC 95% confidence band (P_true ± 1.96σ) */}
          <Area
            dataKey="band"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.12}
            isAnimationActive={false}
            name="95% CI"
          />

          {/* Lines */}
          <Line
            dataKey="P_true"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="P_true"
            isAnimationActive={false}
            connectNulls
          />
          <Line
            dataKey="P_kalshi"
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            name="P_kalshi"
            isAnimationActive={false}
            connectNulls
          />
          <Line
            dataKey="P_bet365"
            stroke="#9ca3af"
            strokeWidth={1}
            strokeDasharray="5 3"
            dot={false}
            name="P_bet365"
            isAnimationActive={false}
            connectNulls
          />

          {/* Event annotations — vertical dashed lines with score labels */}
          {annotations.map((a, i) => (
            <ReferenceLine
              key={`${a.type}-${i}`}
              x={a.t}
              stroke={a.type === "goal_confirmed" ? "#f59e0b" : "#ef4444"}
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: a.label, position: "top", fontSize: 10 }}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
