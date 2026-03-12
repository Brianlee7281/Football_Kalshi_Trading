// dashboard/ui/src/app/match/[id]/page.tsx
//
// Match Deep Dive page — full match detail with live WS updates.
//
// Layout:
//   MatchHeader (teams, score, phase, time, status flags)
//   PriceChart  (P_true/P_kalshi/P_bet365 + σ_MC band + event annotations)
//   OrderBookViz (bid/ask bars, spread, stale indicator)
//   SignalLog    (newest first, direction-colored rows)
//   PositionTable (unrealized P&L computed client-side)
//   EventTimeline (chronological, with icons)
//
// Data:
//   REST: /api/match/{id} (detail) + /api/match/{id}/ticks + /api/match/{id}/events
//   WS: tick:{id}, signal:{id}, event:{id}

"use client";

import React, { useContext, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import type {
  EventItem,
  SignalMessage,
  TickMessage,
  TickSnapshot,
} from "@/lib/types";
import { formatTime } from "@/lib/format";
import { useMatchDetail, useMatchEvents, useMatchTicks } from "@/hooks/useApi";
import { WebSocketContext } from "@/hooks/useLiveTick";
import { PriceChart } from "@/components/PriceChart";
import { OrderBookViz } from "@/components/OrderBookViz";
import { SignalLog } from "@/components/SignalLog";
import { PositionTable } from "@/components/PositionTable";
import { EventTimeline } from "@/components/EventTimeline";

// ── Page ────────────────────────────────────────────────────────────────────

export default function MatchDeepDive() {
  const params = useParams();
  const matchId = params.id as string;

  // REST data
  const { data: detail } = useMatchDetail(matchId);
  const { data: restTicks } = useMatchTicks(matchId, 1);
  const { data: restEvents } = useMatchEvents(matchId);

  // WS context for live updates
  const wsCtx = useContext(WebSocketContext);

  // Live state accumulated from WS
  const [liveTicks, setLiveTicks] = useState<TickSnapshot[]>([]);
  const [liveEvents, setLiveEvents] = useState<EventItem[]>([]);
  const [liveSignals, setLiveSignals] = useState<SignalMessage[]>([]);
  const [latestTick, setLatestTick] = useState<TickMessage | null>(null);
  const [lastTickTime, setLastTickTime] = useState(Date.now());

  // Subscribe to this match
  useEffect(() => {
    if (wsCtx) {
      wsCtx.subscribe([matchId]);
    }
  }, [matchId, wsCtx]);

  // Process WS messages
  useEffect(() => {
    if (!wsCtx?.lastMessage) return;
    const msg = wsCtx.lastMessage;

    if (msg.type === "tick" && msg.match_id === matchId) {
      setLatestTick(msg);
      setLastTickTime(Date.now());
      // Append to live ticks for chart
      const snapshot: TickSnapshot = {
        match_id: msg.match_id,
        t: msg.t,
        engine_phase: msg.engine_phase,
        P_true: msg.P_true,
        P_kalshi: null, // WS tick doesn't carry full P_kalshi MarketProbs
        P_bet365: msg.P_bet365,
        sigma_MC: msg.sigma_MC,
        order_allowed: msg.order_allowed,
        cooldown: msg.cooldown,
        ob_freeze: msg.ob_freeze,
        event_state: msg.event_state,
        mu_H: msg.mu_H,
        mu_A: msg.mu_A,
        score: null,
      };
      setLiveTicks((prev) => [...prev, snapshot]);
    }

    if (msg.type === "signal" && msg.match_id === matchId) {
      setLiveSignals((prev) => [...prev, msg]);
    }

    if (msg.type === "event" && msg.match_id === matchId) {
      const item: EventItem = {
        id: Date.now(),
        match_id: msg.match_id,
        event_type: msg.event_type,
        source: "live",
        payload: msg.payload as Record<string, unknown>,
        created_at: new Date().toISOString(),
      };
      setLiveEvents((prev) => [...prev, item]);
    }
  }, [wsCtx?.lastMessage, matchId]);

  // Merge REST + WS ticks
  const allTicks = useMemo(() => {
    const base = restTicks ?? [];
    // Avoid duplicates — WS ticks have t > last REST tick
    const lastRestT = base.length > 0 ? base[base.length - 1].t : -1;
    const newLive = liveTicks.filter((t) => t.t > lastRestT);
    return [...base, ...newLive];
  }, [restTicks, liveTicks]);

  // Merge REST + WS events
  const allEvents = useMemo(() => {
    const base = restEvents ?? [];
    const baseIds = new Set(base.map((e) => e.id));
    const newLive = liveEvents.filter((e) => !baseIds.has(e.id));
    return [...base, ...newLive];
  }, [restEvents, liveEvents]);

  // Signals: from detail + live WS
  const allSignals = useMemo(() => {
    // SignalItem from REST and SignalMessage from WS have slightly different shapes
    // Normalize to SignalItem for the SignalLog
    const fromWs = liveSignals.map((s) => ({
      match_id: s.match_id,
      ticker: s.ticker,
      direction: s.direction,
      EV: s.EV,
      P_cons: s.P_cons,
      P_kalshi: s.P_kalshi,
      alignment: s.alignment,
      kelly_multiplier: s.kelly_fraction,
      timestamp: s.timestamp,
    }));
    return fromWs;
  }, [liveSignals]);

  // Kalshi age for stale detection
  const kalshiAge = Date.now() - lastTickTime;

  // Loading state
  if (!detail) {
    return (
      <div className="flex h-64 items-center justify-center text-gray-500">
        Loading match...
      </div>
    );
  }

  const score = latestTick
    ? `${latestTick.score[0]} – ${latestTick.score[1]}`
    : detail.score
      ? `${detail.score.home} – ${detail.score.away}`
      : "— – —";

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      {/* Match Header */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">
              {detail.home_team ?? "TBD"} vs {detail.away_team ?? "TBD"}
            </h1>
            <div className="mt-1 flex items-baseline gap-3 text-sm text-gray-500">
              <span>{detail.status}</span>
              {latestTick && (
                <>
                  <span>{latestTick.engine_phase}</span>
                  <span>{formatTime(latestTick.t)}</span>
                </>
              )}
              {latestTick?.cooldown && (
                <span className="rounded bg-yellow-100 px-1.5 py-0.5 text-xs text-yellow-700">
                  Cooldown
                </span>
              )}
              {latestTick?.ob_freeze && (
                <span className="rounded bg-blue-100 px-1.5 py-0.5 text-xs text-blue-700">
                  OB Freeze
                </span>
              )}
            </div>
          </div>
          <div className="text-3xl font-bold text-gray-900">{score}</div>
        </div>
      </div>

      {/* Price Chart + Order Book side by side */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <h2 className="mb-2 text-sm font-semibold text-gray-700">
            Price Chart
          </h2>
          <PriceChart ticks={allTicks} events={allEvents} />
        </div>
        <div>
          <h2 className="mb-2 text-sm font-semibold text-gray-700">
            Order Book
          </h2>
          <OrderBookViz
            P_kalshi={latestTick?.P_true ?? detail.latest_tick?.P_kalshi ?? null}
            kalshiAge={kalshiAge}
            selectedMarket="home_win"
          />
        </div>
      </div>

      {/* Signal Log + Position Table side by side */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h2 className="mb-2 text-sm font-semibold text-gray-700">
            Signals
          </h2>
          <SignalLog signals={allSignals} />
        </div>
        <div>
          <h2 className="mb-2 text-sm font-semibold text-gray-700">
            Positions
          </h2>
          <PositionTable
            positions={detail.positions}
            P_kalshi={detail.latest_tick?.P_kalshi ?? null}
          />
        </div>
      </div>

      {/* Event Timeline */}
      <div>
        <h2 className="mb-2 text-sm font-semibold text-gray-700">
          Event Timeline
        </h2>
        <EventTimeline events={allEvents} />
      </div>
    </div>
  );
}
