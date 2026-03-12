// dashboard/ui/src/components/AlertBanner.tsx
//
// Alert banner that displays system alerts from the WebSocket.
//
// Auto-dismiss rules:
//   critical → sticky (requires manual dismiss)
//   warning  → auto-dismiss after 30s
//   info     → auto-dismiss after 10s

"use client";

import React, { useCallback, useEffect, useState } from "react";

import type { SystemAlertMessage } from "@/lib/types";

// ── Constants ───────────────────────────────────────────────────────────────

const DISMISS_MS: Record<SystemAlertMessage["severity"], number | null> = {
  critical: null, // sticky
  warning: 30_000,
  info: 10_000,
};

const SEVERITY_STYLES: Record<SystemAlertMessage["severity"], string> = {
  critical: "bg-red-600 text-white",
  warning: "bg-yellow-500 text-gray-900",
  info: "bg-blue-500 text-white",
};

// ── Types ───────────────────────────────────────────────────────────────────

interface TrackedAlert {
  alert: SystemAlertMessage;
  id: number;
}

// ── Props ───────────────────────────────────────────────────────────────────

export interface AlertBannerProps {
  alerts: SystemAlertMessage[];
}

// ── Component ───────────────────────────────────────────────────────────────

let nextId = 0;

export function AlertBanner({ alerts }: AlertBannerProps) {
  const [visible, setVisible] = useState<TrackedAlert[]>([]);

  // Track new alerts as they arrive
  useEffect(() => {
    if (alerts.length === 0) return;

    const latest = alerts[alerts.length - 1];
    const id = nextId++;
    const tracked: TrackedAlert = { alert: latest, id };
    setVisible((prev) => [...prev, tracked]);

    // Auto-dismiss for non-critical
    const timeout = DISMISS_MS[latest.severity];
    if (timeout != null) {
      const timer = setTimeout(() => {
        setVisible((prev) => prev.filter((a) => a.id !== id));
      }, timeout);
      return () => clearTimeout(timer);
    }
  }, [alerts]);

  const dismiss = useCallback((id: number) => {
    setVisible((prev) => prev.filter((a) => a.id !== id));
  }, []);

  if (visible.length === 0) return null;

  return (
    <div className="space-y-1">
      {visible.map(({ alert, id }) => (
        <div
          key={id}
          className={`flex items-center justify-between px-4 py-2 text-sm ${SEVERITY_STYLES[alert.severity]}`}
          role="alert"
        >
          <div>
            <span className="font-semibold">{alert.title}</span>
            {alert.details &&
              Object.entries(alert.details).map(([k, v]) => (
                <span key={k} className="ml-3 opacity-80">
                  {k}: {v}
                </span>
              ))}
          </div>
          <button
            onClick={() => dismiss(id)}
            className="ml-4 opacity-70 hover:opacity-100"
            aria-label="Dismiss alert"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
