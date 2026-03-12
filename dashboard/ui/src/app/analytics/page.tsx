// dashboard/ui/src/app/analytics/page.tsx
//
// P&L Analytics page with filters and breakdowns.
//
// Filters: date range, league, market, direction, paper/live
// Breakdowns: by_league, by_market, by_direction, by_alignment
// GraduationChecklist: 8 criteria with pass/fail badges
//
// Edge case: 0 trades → "No completed trades yet. Paper trading in progress."

"use client";

import React, { useMemo, useState } from "react";

import { formatPct, formatPnL, pnlColor } from "@/lib/format";
import { useGraduation, usePnLReport } from "@/hooks/useApi";
import { GraduationChecklist } from "@/components/GraduationChecklist";

// ── Filter state ────────────────────────────────────────────────────────────

interface Filters {
  date_from: string;
  date_to: string;
  league: string;
  market: string;
  direction: string;
  mode: string;
}

const INITIAL_FILTERS: Filters = {
  date_from: "",
  date_to: "",
  league: "",
  market: "",
  direction: "",
  mode: "",
};

// ── Breakdown table ─────────────────────────────────────────────────────────

function BreakdownTable({
  title,
  data,
}: {
  title: string;
  data: Record<string, number>;
}) {
  const entries = Object.entries(data).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">{title}</h3>
      <div className="space-y-1">
        {entries.map(([key, value]) => (
          <div
            key={key}
            className="flex items-center justify-between rounded px-2 py-1.5 text-sm hover:bg-gray-50"
          >
            <span className="text-gray-600">{key.replace("_", " ")}</span>
            <span className={`font-semibold ${pnlColor(value)}`}>
              {formatPnL(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function PnLAnalytics() {
  const [filters, setFilters] = useState<Filters>(INITIAL_FILTERS);

  // Build query params from non-empty filters
  const queryParams = useMemo(() => {
    const params: Record<string, string> = {};
    if (filters.date_from) params.date_from = filters.date_from;
    if (filters.date_to) params.date_to = filters.date_to;
    if (filters.league) params.league = filters.league;
    if (filters.market) params.market = filters.market;
    if (filters.direction) params.direction = filters.direction;
    if (filters.mode) params.mode = filters.mode;
    return Object.keys(params).length > 0 ? params : undefined;
  }, [filters]);

  const { data: report, loading: reportLoading } = usePnLReport(queryParams);
  const { data: graduation } = useGraduation();

  const updateFilter = (key: keyof Filters, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-2xl font-bold text-gray-900">P&L Analytics</h1>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 rounded-lg border border-gray-200 bg-white p-4">
        <div>
          <label className="block text-xs text-gray-500">From</label>
          <input
            type="date"
            value={filters.date_from}
            onChange={(e) => updateFilter("date_from", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500">To</label>
          <input
            type="date"
            value={filters.date_to}
            onChange={(e) => updateFilter("date_to", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500">League</label>
          <select
            value={filters.league}
            onChange={(e) => updateFilter("league", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">All</option>
            <option value="EPL">EPL</option>
            <option value="La Liga">La Liga</option>
            <option value="Serie A">Serie A</option>
            <option value="Bundesliga">Bundesliga</option>
            <option value="Ligue 1">Ligue 1</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500">Market</label>
          <select
            value={filters.market}
            onChange={(e) => updateFilter("market", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">All</option>
            <option value="home_win">Home Win</option>
            <option value="draw">Draw</option>
            <option value="away_win">Away Win</option>
            <option value="over_25">Over 2.5</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500">Direction</label>
          <select
            value={filters.direction}
            onChange={(e) => updateFilter("direction", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">All</option>
            <option value="BUY_YES">BUY_YES</option>
            <option value="BUY_NO">BUY_NO</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500">Mode</label>
          <select
            value={filters.mode}
            onChange={(e) => updateFilter("mode", e.target.value)}
            className="rounded border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">All</option>
            <option value="paper">Paper</option>
            <option value="live">Live</option>
          </select>
        </div>
      </div>

      {/* Loading */}
      {reportLoading && (
        <div className="text-center text-sm text-gray-500">
          Loading analytics...
        </div>
      )}

      {/* Empty state */}
      {report && report.total_trades === 0 && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center text-sm text-gray-500">
          No completed trades yet. Paper trading in progress.
        </div>
      )}

      {/* Summary stats */}
      {report && report.total_trades > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-6">
            <StatCard label="Total Trades" value={String(report.total_trades)} />
            <StatCard label="Win Rate" value={formatPct(report.win_rate)} />
            <StatCard
              label="Total P&L"
              value={formatPnL(report.total_pnl)}
              color={pnlColor(report.total_pnl)}
            />
            <StatCard
              label="Edge Realization"
              value={`${report.edge_realization.toFixed(2)}x`}
            />
            <StatCard
              label="Max Drawdown"
              value={formatPct(report.max_drawdown_pct)}
            />
            <StatCard
              label="Sharpe"
              value={report.sharpe != null ? report.sharpe.toFixed(2) : "—"}
            />
          </div>

          {/* Breakdowns */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <BreakdownTable title="By League" data={report.breakdown.by_league} />
            <BreakdownTable title="By Market" data={report.breakdown.by_market} />
            <BreakdownTable
              title="By Direction"
              data={report.breakdown.by_direction}
            />
            <BreakdownTable
              title="By Alignment"
              data={report.breakdown.by_alignment}
            />
          </div>
        </>
      )}

      {/* Graduation Checklist */}
      {graduation && (
        <div className="mt-6">
          <h2 className="mb-3 text-lg font-semibold text-gray-700">
            Paper Graduation
          </h2>
          <GraduationChecklist data={graduation} />
        </div>
      )}
    </div>
  );
}

// ── Stat card helper ────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${color ?? "text-gray-900"}`}>
        {value}
      </div>
    </div>
  );
}
