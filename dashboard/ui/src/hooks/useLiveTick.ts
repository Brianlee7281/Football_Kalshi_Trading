// dashboard/ui/src/hooks/useLiveTick.ts
//
// Subscribe to a match_id over WebSocket and return the latest TickMessage.
//
// Requires a parent <WebSocketProvider> that owns the single shared connection.
// This hook registers/unregisters a per-match listener via the provider context.

"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

import type { TickMessage, WSMessage } from "@/lib/types";

import type { UseWebSocketReturn, WsStatus } from "./useWebSocket";

// ── Context (set up by WebSocketProvider in layout.tsx) ─────────────────────

export const WebSocketContext = createContext<UseWebSocketReturn | null>(null);

function useWsContext(): UseWebSocketReturn {
  const ctx = useContext(WebSocketContext);
  if (!ctx) {
    throw new Error("useLiveTick must be used within a <WebSocketProvider>");
  }
  return ctx;
}

// ── Hook ────────────────────────────────────────────────────────────────────

export interface UseLiveTickReturn {
  tick: TickMessage | null;
  wsStatus: WsStatus;
  lastMessageAge: number;
}

export function useLiveTick(matchId: string | null): UseLiveTickReturn {
  const { status, lastMessageAge, lastMessage, subscribe } = useWsContext();
  const [tick, setTick] = useState<TickMessage | null>(null);
  const prevMatchId = useRef<string | null>(null);

  // Subscribe when matchId changes
  useEffect(() => {
    if (matchId !== prevMatchId.current) {
      prevMatchId.current = matchId;
      setTick(null);
    }
    if (matchId) {
      subscribe([matchId]);
    } else {
      subscribe([]);
    }
  }, [matchId, subscribe]);

  // Update tick from lastMessage
  useEffect(() => {
    if (
      lastMessage &&
      lastMessage.type === "tick" &&
      matchId != null &&
      lastMessage.match_id === matchId
    ) {
      setTick(lastMessage);
    }
  }, [lastMessage, matchId]);

  return { tick, wsStatus: status, lastMessageAge };
}
