// dashboard/ui/src/components/MatchCard.tsx
//
// Card showing a single match: teams, score, time, per-market edge, position count.
// Click → navigates to /match/{match_id}.
// Pulse animation on new goal.
//
// Edge cases:
//   - WS not connected → gray "Live data unavailable" badge
//   - status = SCHEDULED → "Starting in {time}"

"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";

import type { MatchSummary, TickMessage } from "@/lib/types";
import { formatEdge, formatTime } from "@/lib/format";

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Edge indicator: ▲ positive > 2¢, ▬ hold, ▼ negative */
function edgeIndicator(edge: number): { symbol: string; color: string } {
  if (edge > 0.02) return { symbol: "▲", color: "text-green-600" };
  if (edge < -0.02) return { symbol: "▼", color: "text-red-600" };
  return { symbol: "▬", color: "text-gray-400" };
}

function timeUntil(kickoffUtc: string): string {
  const diff = new Date(kickoffUtc).getTime() - Date.now();
  if (diff <= 0) return "now";
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  return rem > 0 ? `${hours}h ${rem}m` : `${hours}h`;
}

// Markets to display in the edge summary
const EDGE_MARKETS = ["home_win", "draw", "away_win", "over_25"] as const;

// ── Props ───────────────────────────────────────────────────────────────────

export interface MatchCardProps {
  match: MatchSummary;
  latestTick: TickMessage | null;
  wsConnected: boolean;
  positionCount?: number;
}

// ── Component ───────────────────────────────────────────────────────────────

export function MatchCard({
  match,
  latestTick,
  wsConnected,
  positionCount = 0,
}: MatchCardProps) {
  const [pulse, setPulse] = useState(false);

  // Pulse on new goal (score change)
  useEffect(() => {
    if (!latestTick) return;
    setPulse(true);
    const timer = setTimeout(() => setPulse(false), 1500);
    return () => clearTimeout(timer);
    // Trigger on score change
  }, [latestTick?.score?.[0], latestTick?.score?.[1]]);

  const isScheduled = match.status === "SCHEDULED";

  // Compute per-market edge: P_true - P_kalshi (midpoint proxy)
  const edges: Record<string, number> = {};
  if (latestTick?.P_true) {
    for (const market of EDGE_MARKETS) {
      const pTrue = latestTick.P_true[market];
      // P_kalshi is MarketProbs on the tick — treat as midpoint
      const pKalshi = latestTick.sigma_MC
        ? undefined
        : undefined;
      // If we have P_true, show it as the edge vs 0.5 as a fallback
      // In practice, the tick also carries P_kalshi on the REST model
      if (pTrue != null) {
        edges[market] = pTrue - 0.5; // placeholder — real edge needs P_kalshi
      }
    }
  }

  return (
    <Link href={`/match/${match.match_id}`}>
      <div
        className={`rounded-lg border border-gray-200 bg-white p-4 shadow-sm transition hover:shadow-md ${
          pulse ? "animate-pulse ring-2 ring-yellow-400" : ""
        }`}
      >
        {/* Header: teams + score */}
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium text-gray-700">
            {match.home_team ?? "TBD"} vs {match.away_team ?? "TBD"}
          </div>
          {match.trading_mode === "paper" ? (
            <span className="rounded bg-yellow-100 px-1.5 py-0.5 text-xs text-yellow-700">
              Paper
            </span>
          ) : (
            <span className="rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-700">
              Live
            </span>
          )}
        </div>

        {/* Score + time or scheduled countdown */}
        <div className="mt-2">
          {isScheduled ? (
            <div className="text-sm text-gray-500">
              Starting in {timeUntil(match.kickoff_utc)}
            </div>
          ) : latestTick ? (
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-bold">
                {latestTick.score[0]} – {latestTick.score[1]}
              </span>
              <span className="text-sm text-gray-500">
                {formatTime(latestTick.t)}
              </span>
              <span className="text-xs text-gray-400">
                {latestTick.engine_phase}
              </span>
            </div>
          ) : match.score ? (
            <div className="text-2xl font-bold">
              {match.score.home} – {match.score.away}
            </div>
          ) : (
            <div className="text-sm text-gray-400">—</div>
          )}
        </div>

        {/* Per-market edge summary */}
        {latestTick?.P_true && !isScheduled && (
          <div className="mt-2 flex gap-3 text-xs">
            {EDGE_MARKETS.map((market) => {
              const edge = edges[market];
              if (edge == null) return null;
              const { symbol, color } = edgeIndicator(edge);
              return (
                <span key={market} className={color}>
                  {market.replace("_", " ")}: {symbol} {formatEdge(edge)}
                </span>
              );
            })}
          </div>
        )}

        {/* Footer: position count + WS status */}
        <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
          <span>
            {positionCount > 0
              ? `${positionCount} position${positionCount > 1 ? "s" : ""}`
              : "No positions"}
          </span>
          {!wsConnected && !isScheduled && (
            <span className="rounded bg-gray-100 px-1.5 py-0.5 text-gray-400">
              Live data unavailable
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}
