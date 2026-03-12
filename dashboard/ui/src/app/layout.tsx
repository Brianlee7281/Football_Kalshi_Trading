// dashboard/ui/src/app/layout.tsx
//
// Root layout — wraps all pages with WebSocketProvider, StatusBar, AlertBanner.
// Required by Next.js App Router for <html> and <body> tags.

"use client";

import React, { useCallback, useState } from "react";

import type { SystemAlertMessage } from "@/lib/types";
import type { WSMessage } from "@/lib/types";
import { useWebSocket } from "@/hooks/useWebSocket";
import { WebSocketContext } from "@/hooks/useLiveTick";
import { useSystemStatus } from "@/hooks/useApi";
import { StatusBar } from "@/components/StatusBar";
import { AlertBanner } from "@/components/AlertBanner";

// ── WebSocket wrapper that provides context + collects alerts ────────────────

function Providers({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<SystemAlertMessage[]>([]);

  const handleMessage = useCallback((msg: WSMessage) => {
    if (msg.type === "alert") {
      setAlerts((prev) => [...prev, msg]);
    }
  }, []);

  const ws = useWebSocket(handleMessage);
  const { data: systemStatus } = useSystemStatus();

  return (
    <WebSocketContext.Provider value={ws}>
      <StatusBar
        bankroll={systemStatus?.bankroll ?? null}
        exposure_pct={systemStatus?.exposure_pct ?? null}
        drawdown_pct={systemStatus?.drawdown_pct ?? null}
        trading_mode={systemStatus?.trading_mode ?? null}
        ws_status={ws.status}
      />
      <AlertBanner alerts={alerts} />
      {children}
    </WebSocketContext.Provider>
  );
}

// ── Root layout ──────────────────────────────────────────────────────────────

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
