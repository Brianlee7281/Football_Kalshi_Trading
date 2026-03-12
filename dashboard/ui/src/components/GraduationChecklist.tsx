// dashboard/ui/src/components/GraduationChecklist.tsx
//
// 8 criteria for paper→live graduation, each with pass/fail badge.
// "Ready for Live" banner when all_pass = true.
// < 50 trades → progress bar: "{count}/50 trades"

"use client";

import React from "react";

import type { GraduationChecklist as GraduationData } from "@/lib/types";

// ── Criteria definitions ────────────────────────────────────────────────────

interface Criterion {
  label: string;
  target: string;
  key: keyof GraduationData;
}

const CRITERIA: Criterion[] = [
  { label: "Trade Count", target: ">= 50", key: "trades_ok" },
  { label: "Edge Realization", target: "0.6 – 1.5", key: "edge_realization_ok" },
  { label: "Brier Score", target: "Phase 1.5 baseline +/- 0.03", key: "brier_ok" },
  { label: "Max Drawdown", target: "< 15%", key: "max_drawdown_ok" },
  { label: "Paper Realism", target: "> 0.85", key: "realism_score_ok" },
  { label: "Directional Correctness", target: "100%", key: "directional_ok" },
  { label: "No System Crashes", target: "0 crashes", key: "no_crashes_ok" },
  { label: "THETA_ENTRY Calibrated", target: "Calibrated", key: "theta_calibrated" },
];

// ── Props ───────────────────────────────────────────────────────────────────

export interface GraduationChecklistProps {
  data: GraduationData;
}

// ── Component ───────────────────────────────────────────────────────────────

export function GraduationChecklist({ data }: GraduationChecklistProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-4 text-sm font-semibold text-gray-700">
        Paper Graduation Checklist
      </h3>

      {/* Trade count progress bar when < 50 */}
      {data.trade_count < 50 && (
        <div className="mb-4">
          <div className="mb-1 flex justify-between text-xs text-gray-500">
            <span>Trade progress</span>
            <span>{data.trade_count}/50 trades</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div
              className="h-2 rounded-full bg-blue-500 transition-all"
              style={{ width: `${Math.min(100, (data.trade_count / 50) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Criteria list */}
      <div className="space-y-2">
        {CRITERIA.map((criterion) => {
          const passed = data[criterion.key] as boolean;
          return (
            <div
              key={criterion.key}
              className="flex items-center justify-between rounded px-3 py-2 text-sm"
            >
              <div>
                <span className="font-medium text-gray-800">
                  {criterion.label}
                </span>
                <span className="ml-2 text-xs text-gray-400">
                  Target: {criterion.target}
                </span>
              </div>
              <span
                className={`rounded px-2 py-0.5 text-xs font-semibold ${
                  passed
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {passed ? "PASS" : "FAIL"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Ready for Live banner */}
      {data.all_pass ? (
        <div className="mt-4 rounded-lg bg-green-600 px-4 py-3 text-center text-sm font-semibold text-white">
          Ready for Live Trading
        </div>
      ) : (
        <div className="mt-4 rounded-lg bg-gray-100 px-4 py-3 text-center text-sm text-gray-500">
          {CRITERIA.filter((c) => data[c.key] as boolean).length}/{CRITERIA.length} criteria passed
        </div>
      )}
    </div>
  );
}
