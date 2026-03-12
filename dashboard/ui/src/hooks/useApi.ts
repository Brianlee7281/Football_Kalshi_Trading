// dashboard/ui/src/hooks/useApi.ts
//
// REST fetch hooks for the MMPP Trading Dashboard.
// Provides useFetch (one-shot) and usePolling (interval-based) helpers.
//
// All endpoints go through the API base URL configured via NEXT_PUBLIC_API_URL
// (defaults to "" for same-origin requests via Next.js proxy).

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  GraduationChecklist,
  MatchDetail,
  MatchSummary,
  ModelHealthReport,
  PnLReport,
  PositionItem,
  SystemStatus,
  TickSnapshot,
  EventItem,
} from "@/lib/types";

// ── Base URL ────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// ── Generic fetch wrapper ───────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── useFetch: one-shot fetch on mount or when deps change ───────────────────

interface UseFetchReturn<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refetch: () => void;
}

export function useFetch<T>(
  path: string | null,
): UseFetchReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!path);

  const fetchData = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    setError(null);
    try {
      const result = await apiFetch<T>(path);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, error, loading, refetch: fetchData };
}

// ── usePolling: fetch on an interval ────────────────────────────────────────

export function usePolling<T>(
  path: string | null,
  intervalMs: number,
): UseFetchReturn<T> {
  const result = useFetch<T>(path);
  const refetchRef = useRef(result.refetch);
  refetchRef.current = result.refetch;

  useEffect(() => {
    if (!path) return;
    const id = setInterval(() => refetchRef.current(), intervalMs);
    return () => clearInterval(id);
  }, [path, intervalMs]);

  return result;
}

// ── Typed endpoint hooks ────────────────────────────────────────────────────

/** GET /api/matches — poll every 5s */
export function useMatches(
  status?: string,
  date?: string,
) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (date) params.set("date", date);
  const qs = params.toString();
  const path = `/api/matches${qs ? `?${qs}` : ""}`;
  return usePolling<MatchSummary[]>(path, 5000);
}

/** GET /api/match/{id} — one-shot */
export function useMatchDetail(matchId: string | null) {
  return useFetch<MatchDetail>(matchId ? `/api/match/${matchId}` : null);
}

/** GET /api/match/{id}/ticks — one-shot (historical ticks for chart) */
export function useMatchTicks(matchId: string | null, downsample = 1) {
  const path = matchId
    ? `/api/match/${matchId}/ticks?downsample=${downsample}`
    : null;
  return useFetch<TickSnapshot[]>(path);
}

/** GET /api/match/{id}/events — one-shot */
export function useMatchEvents(matchId: string | null) {
  return useFetch<EventItem[]>(matchId ? `/api/match/${matchId}/events` : null);
}

/** GET /api/positions — poll every 5s */
export function usePositions(
  matchId?: string,
  status?: string,
) {
  const params = new URLSearchParams();
  if (matchId) params.set("match_id", matchId);
  if (status) params.set("status", status);
  const qs = params.toString();
  const path = `/api/positions${qs ? `?${qs}` : ""}`;
  return usePolling<PositionItem[]>(path, 5000);
}

/** GET /api/analytics/pnl — one-shot */
export function usePnLReport(params?: Record<string, string>) {
  const qs = params ? new URLSearchParams(params).toString() : "";
  return useFetch<PnLReport>(`/api/analytics/pnl${qs ? `?${qs}` : ""}`);
}

/** GET /api/analytics/model-health — one-shot */
export function useModelHealth() {
  return useFetch<ModelHealthReport>("/api/analytics/model-health");
}

/** GET /api/analytics/paper-graduation — one-shot */
export function useGraduation() {
  return useFetch<GraduationChecklist>("/api/analytics/paper-graduation");
}

/** GET /api/system/status — poll every 10s */
export function useSystemStatus() {
  return usePolling<SystemStatus>("/api/system/status", 10000);
}
