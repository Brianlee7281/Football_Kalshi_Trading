// dashboard/ui/src/app/system/page.tsx
//
// System Operations page — container status, connection health, alerts, param info.
// Data: REST /api/system/status polled every 10s.
//
// Sections:
//   ContainerTable  — match_id, status, uptime, heartbeat, healthy badge
//   ConnectionPanel — service health for Odds-API, Goalserve, Kalshi, PostgreSQL, Redis
//   AlertHistory    — recent system alerts
//   ParamVersionInfo — active param version, trained_at, matches since retrain
//
// Edge case: heartbeat > 60s → red "UNRESPONSIVE" badge

"use client";

import React from "react";

import type { ConnectionHealth, ContainerStatus } from "@/lib/types";
import { formatDuration, heartbeatDot } from "@/lib/format";
import { useSystemStatus } from "@/hooks/useApi";

// ── ContainerTable ──────────────────────────────────────────────────────────

function ContainerTable({
  containers,
}: {
  containers: ContainerStatus[];
}) {
  if (containers.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-sm text-gray-500">
        No active containers
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="w-full text-sm">
        <thead className="bg-gray-100 text-left text-xs text-gray-600">
          <tr>
            <th className="px-4 py-2">Match ID</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2">Uptime</th>
            <th className="px-4 py-2">Heartbeat</th>
            <th className="px-4 py-2">Health</th>
          </tr>
        </thead>
        <tbody>
          {containers.map((c) => {
            const healthy =
              c.heartbeat_age_s != null && c.heartbeat_age_s < 60;
            const unresponsive =
              c.heartbeat_age_s != null && c.heartbeat_age_s >= 60;

            return (
              <tr
                key={c.match_id}
                className={`border-t border-gray-100 ${
                  unresponsive ? "bg-red-50" : ""
                }`}
              >
                <td className="px-4 py-2 font-mono text-xs">{c.match_id}</td>
                <td className="px-4 py-2">{c.status}</td>
                <td className="px-4 py-2 text-gray-500">
                  {c.uptime_min != null
                    ? formatDuration(c.uptime_min * 60)
                    : "—"}
                </td>
                <td className="px-4 py-2 text-gray-500">
                  {c.heartbeat_age_s != null
                    ? `${c.heartbeat_age_s.toFixed(0)}s ago`
                    : "—"}
                </td>
                <td className="px-4 py-2">
                  {unresponsive ? (
                    <span className="rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white">
                      UNRESPONSIVE
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className={`inline-block h-2.5 w-2.5 ${heartbeatDot(healthy)}`}
                      />
                      <span className="text-xs text-gray-500">
                        {healthy ? "Healthy" : "Unknown"}
                      </span>
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── ConnectionPanel ─────────────────────────────────────────────────────────

function ConnectionPanel({
  connections,
}: {
  connections: ConnectionHealth[];
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">Connections</h3>
      <div className="space-y-2">
        {connections.map((conn) => (
          <div
            key={conn.service}
            className="flex items-center justify-between rounded px-3 py-2 hover:bg-gray-50"
          >
            <span className="text-sm text-gray-700">{conn.service}</span>
            <div className="flex items-center gap-2">
              {conn.last_message_age_s != null && (
                <span className="text-xs text-gray-400">
                  {conn.last_message_age_s.toFixed(0)}s ago
                </span>
              )}
              <span
                className={`rounded px-2 py-0.5 text-xs font-semibold ${
                  conn.status === "connected"
                    ? "bg-green-100 text-green-700"
                    : conn.status === "reconnecting" ||
                        conn.status === "polling"
                      ? "bg-yellow-100 text-yellow-700"
                      : conn.status === "unknown"
                        ? "bg-gray-100 text-gray-500"
                        : "bg-red-100 text-red-700"
                }`}
              >
                {conn.status}
              </span>
            </div>
          </div>
        ))}
        {connections.length === 0 && (
          <div className="text-sm text-gray-400">No connection data</div>
        )}
      </div>
    </div>
  );
}

// ── AlertHistory ────────────────────────────────────────────────────────────

function AlertHistory() {
  // Alert history would come from a dedicated endpoint or SystemStatus.
  // For now, show a placeholder that will be wired up when the alert
  // REST endpoint is available.
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">
        Recent Alerts
      </h3>
      <div className="text-sm text-gray-400">No recent alerts</div>
    </div>
  );
}

// ── ParamVersionInfo ────────────────────────────────────────────────────────

function ParamVersionInfo({
  version,
  trainedAt,
  matchesSince,
}: {
  version: number | null;
  trainedAt: string | null;
  matchesSince: number | null;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">
        Model Parameters
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">Active Version</span>
          <span className="font-semibold text-gray-800">
            {version != null ? `v${version}` : "—"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Trained At</span>
          <span className="text-gray-700">
            {trainedAt
              ? new Date(trainedAt).toLocaleString("en-GB", {
                  dateStyle: "medium",
                  timeStyle: "short",
                })
              : "—"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Matches Since Retrain</span>
          <span className="font-semibold text-gray-800">
            {matchesSince ?? "—"}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function SystemOperations() {
  const { data: status, loading } = useSystemStatus();

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-2xl font-bold text-gray-900">System Operations</h1>

      {loading && !status && (
        <div className="text-center text-sm text-gray-500">Loading...</div>
      )}

      {/* Container table */}
      <section>
        <h2 className="mb-2 text-lg font-semibold text-gray-700">
          Match Containers
        </h2>
        <ContainerTable containers={status?.containers ?? []} />
      </section>

      {/* Connection panel + Param info side by side */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <ConnectionPanel connections={status?.connections ?? []} />
        <ParamVersionInfo
          version={status?.param_version ?? null}
          trainedAt={status?.param_trained_at ?? null}
          matchesSince={status?.matches_since_retrain ?? null}
        />
      </div>

      {/* Alert history */}
      <AlertHistory />
    </div>
  );
}
