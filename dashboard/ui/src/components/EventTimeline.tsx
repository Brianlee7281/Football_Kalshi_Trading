// dashboard/ui/src/components/EventTimeline.tsx
//
// Vertical timeline, chronological (newest at bottom).
// Icons: ⚽ goal, 🟥 red card, 🔄 substitution, ⏸ period change, ❄️ ob_freeze, 🧊 cooldown
// Each event: icon + time + description + score after.
//
// Edge case: 0 events → "No events yet — match in progress"

"use client";

import React from "react";

import type { EventItem } from "@/lib/types";

// ── Icon mapping ────────────────────────────────────────────────────────────

const EVENT_ICONS: Record<string, string> = {
  goal_confirmed: "⚽",
  goal_preliminary: "⚽",
  red_card: "🟥",
  substitution: "🔄",
  period_change: "⏸",
  ob_freeze: "❄️",
  cooldown: "🧊",
};

function eventIcon(eventType: string): string {
  return EVENT_ICONS[eventType] ?? "📋";
}

function eventDescription(event: EventItem): string {
  const payload = event.payload as Record<string, unknown> | null;
  const team = (payload?.team as string) ?? "";
  const minute = payload?.minute as number | undefined;
  const timeStr = minute != null ? `${minute}'` : "";

  switch (event.event_type) {
    case "goal_confirmed":
      return `Goal ${team ? `(${team})` : ""} ${timeStr}`.trim();
    case "goal_preliminary":
      return `Goal (preliminary) ${team ? `(${team})` : ""} ${timeStr}`.trim();
    case "red_card":
      return `Red card ${team ? `(${team})` : ""} ${timeStr}`.trim();
    case "substitution":
      return `Substitution ${team ? `(${team})` : ""} ${timeStr}`.trim();
    case "period_change":
      return `Period change ${timeStr}`.trim();
    case "ob_freeze":
      return `Order book freeze ${timeStr}`.trim();
    case "cooldown":
      return `Cooldown active ${timeStr}`.trim();
    default:
      return `${event.event_type} ${timeStr}`.trim();
  }
}

function scoreAfter(event: EventItem): string | null {
  const payload = event.payload as Record<string, unknown> | null;
  const score = payload?.score as [number, number] | undefined;
  if (score) return `${score[0]} – ${score[1]}`;
  return null;
}

// ── Props ───────────────────────────────────────────────────────────────────

export interface EventTimelineProps {
  events: EventItem[];
}

// ── Component ───────────────────────────────────────────────────────────────

export function EventTimeline({ events }: EventTimelineProps) {
  if (events.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        No events yet — match in progress
      </div>
    );
  }

  // Chronological order (newest at bottom)
  const sorted = [...events].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );

  return (
    <div className="max-h-96 overflow-y-auto">
      <div className="relative border-l-2 border-gray-200 pl-6">
        {sorted.map((event) => {
          const score = scoreAfter(event);
          return (
            <div key={event.id} className="relative mb-4">
              {/* Dot on the timeline */}
              <div className="absolute -left-[29px] top-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-white text-sm">
                {eventIcon(event.event_type)}
              </div>

              {/* Content */}
              <div className="text-sm">
                <span className="text-gray-800">
                  {eventDescription(event)}
                </span>
                {score && (
                  <span className="ml-2 rounded bg-gray-100 px-1.5 py-0.5 text-xs font-semibold text-gray-600">
                    {score}
                  </span>
                )}
                <div className="mt-0.5 text-xs text-gray-400">
                  {new Date(event.created_at).toLocaleTimeString("en-GB", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                    hour12: false,
                  })}
                  <span className="ml-2 text-gray-300">{event.source}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
