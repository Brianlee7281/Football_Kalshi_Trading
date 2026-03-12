// dashboard/ui/src/app/page.tsx
//
// Command Center — lists active matches (MatchCards) and upcoming matches.
//
// Data:
//   REST /api/matches polled every 5s (useMatches)
//   WS tick per active match (useWebSocket context)
//
// Edge cases:
//   0 running matches → "No active matches. Next: {home} vs {away} in {time}"
//   0 upcoming        → "No matches scheduled in the next 48 hours."

"use client";

import React, { useContext, useEffect, useMemo, useState } from "react";

import type { TickMessage, WSMessage } from "@/lib/types";
import { useMatches } from "@/hooks/useApi";
import { WebSocketContext } from "@/hooks/useLiveTick";
import { MatchCard } from "@/components/MatchCard";

// ── Helpers ─────────────────────────────────────────────────────────────────

const ACTIVE_STATUSES = new Set([
  "PHASE2_RUNNING",
  "PHASE2_DONE",
  "PHASE3_RUNNING",
  "SETTLING",
]);

function timeUntil(kickoffUtc: string): string {
  const diff = new Date(kickoffUtc).getTime() - Date.now();
  if (diff <= 0) return "now";
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const rem = minutes % 60;
  return rem > 0 ? `${hours}h ${rem}m` : `${hours}h`;
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function CommandCenter() {
  const { data: matches } = useMatches();
  const wsCtx = useContext(WebSocketContext);
  const wsConnected = wsCtx?.status === "connected";

  // Track latest tick per match from WS
  const [ticks, setTicks] = useState<Record<string, TickMessage>>({});

  // Subscribe to all active matches
  const activeMatches = useMemo(
    () => (matches ?? []).filter((m) => ACTIVE_STATUSES.has(m.status)),
    [matches],
  );
  const upcomingMatches = useMemo(
    () => (matches ?? []).filter((m) => m.status === "SCHEDULED"),
    [matches],
  );

  // Subscribe WS to all active match IDs
  useEffect(() => {
    if (!wsCtx) return;
    const ids = activeMatches.map((m) => m.match_id);
    wsCtx.subscribe(ids);
  }, [activeMatches, wsCtx]);

  // Collect ticks from lastMessage
  useEffect(() => {
    if (!wsCtx?.lastMessage) return;
    const msg = wsCtx.lastMessage;
    if (msg.type === "tick") {
      setTicks((prev) => ({ ...prev, [msg.match_id]: msg }));
    }
  }, [wsCtx?.lastMessage]);

  return (
    <div className="mx-auto max-w-6xl p-6">
      <h1 className="mb-6 text-2xl font-bold text-gray-900">
        Command Center
      </h1>

      {/* Active matches */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold text-gray-700">
          Active Matches
        </h2>
        {activeMatches.length > 0 ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {activeMatches.map((match) => (
              <MatchCard
                key={match.match_id}
                match={match}
                latestTick={ticks[match.match_id] ?? null}
                wsConnected={wsConnected ?? false}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center text-sm text-gray-500">
            {upcomingMatches.length > 0 ? (
              <>
                No active matches. Next:{" "}
                {upcomingMatches[0].home_team ?? "TBD"} vs{" "}
                {upcomingMatches[0].away_team ?? "TBD"} in{" "}
                {timeUntil(upcomingMatches[0].kickoff_utc)}
              </>
            ) : (
              "No active matches."
            )}
          </div>
        )}
      </section>

      {/* Upcoming matches */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-gray-700">Upcoming</h2>
        {upcomingMatches.length > 0 ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {upcomingMatches.map((match) => (
              <MatchCard
                key={match.match_id}
                match={match}
                latestTick={null}
                wsConnected={wsConnected ?? false}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center text-sm text-gray-500">
            No matches scheduled in the next 48 hours.
          </div>
        )}
      </section>
    </div>
  );
}
