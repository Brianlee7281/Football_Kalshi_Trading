// dashboard/ui/src/hooks/useWebSocket.ts
//
// Central WebSocket hook with exponential backoff reconnection.
// Reference: docs/dashboard_decomposition.md Part 2.2
//
// Exposes:
//   status: "connecting" | "connected" | "reconnecting" | "disconnected"
//   lastMessageAge: number (ms since last message)
//   subscribe(matchIds): send subscribe message
//   lastMessage: WSMessage | null

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { SubscribeMessage, WSMessage } from "@/lib/types";

// ── Reconnection constants ──────────────────────────────────────────────────

const BACKOFF_BASE = 1000; // 1s
const BACKOFF_MAX = 30000; // 30s
const MAX_RETRIES = 10;

// ── Staleness ticker interval ───────────────────────────────────────────────

const AGE_TICK_MS = 500;

// ── Types ───────────────────────────────────────────────────────────────────

export type WsStatus =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

export interface UseWebSocketReturn {
  status: WsStatus;
  lastMessageAge: number;
  lastMessage: WSMessage | null;
  subscribe: (matchIds: string[]) => void;
}

// ── Hook ────────────────────────────────────────────────────────────────────

const WS_URL =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_WS_URL ?? `ws://${window.location.host}/ws/live`
    : "ws://localhost:8000/ws/live";

export function useWebSocket(
  onMessage?: (msg: WSMessage) => void,
): UseWebSocketReturn {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [lastMessageAge, setLastMessageAge] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const lastMsgTimeRef = useRef<number>(Date.now());
  const currentSubsRef = useRef<string[]>([]);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  // Staleness ticker — updates lastMessageAge every AGE_TICK_MS
  useEffect(() => {
    const id = setInterval(() => {
      setLastMessageAge(Date.now() - lastMsgTimeRef.current);
    }, AGE_TICK_MS);
    return () => clearInterval(id);
  }, []);

  // ── Connect / reconnect logic ───────────────────────────────────────────

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      retriesRef.current = 0;

      // Re-send current subscriptions on reconnect
      if (currentSubsRef.current.length > 0) {
        const msg: SubscribeMessage = { subscribe: currentSubsRef.current };
        ws.send(JSON.stringify(msg));
      }
    };

    ws.onmessage = (event) => {
      lastMsgTimeRef.current = Date.now();
      setLastMessageAge(0);

      try {
        const parsed: WSMessage = JSON.parse(event.data as string);
        setLastMessage(parsed);
        onMessageRef.current?.(parsed);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;

      if (retriesRef.current >= MAX_RETRIES) {
        setStatus("disconnected");
        return;
      }

      setStatus("reconnecting");
      const delay = Math.min(
        BACKOFF_BASE * 2 ** retriesRef.current,
        BACKOFF_MAX,
      );
      retriesRef.current += 1;
      setTimeout(connect, delay);
    };

    ws.onerror = () => {
      // onclose will fire after onerror — reconnection handled there
    };
  }, []);

  // Initial connect
  useEffect(() => {
    connect();
    return () => {
      // Prevent reconnection on unmount
      retriesRef.current = MAX_RETRIES;
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Subscribe helper ────────────────────────────────────────────────────

  const subscribe = useCallback((matchIds: string[]) => {
    currentSubsRef.current = matchIds;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const msg: SubscribeMessage = { subscribe: matchIds };
      ws.send(JSON.stringify(msg));
    }
  }, []);

  return { status, lastMessageAge, lastMessage, subscribe };
}
