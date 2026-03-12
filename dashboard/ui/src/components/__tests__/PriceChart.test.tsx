import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Mock recharts to avoid canvas/SVG issues in jsdom
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ComposedChart: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Line: () => null,
  Area: () => null,
  ReferenceLine: () => null,
}));

import { PriceChart } from "../PriceChart";

describe("PriceChart", () => {
  it("0 ticks shows 'Waiting for match to start...'", () => {
    render(<PriceChart ticks={[]} events={[]} />);
    expect(
      screen.getByText("Waiting for match to start..."),
    ).toBeInTheDocument();
  });

  it("< 10 ticks shows 'Data collecting...'", () => {
    const ticks = Array.from({ length: 5 }, (_, i) => ({
      match_id: "m1",
      t: i,
      engine_phase: "RUNNING",
      P_true: { home_win: 0.5, draw: 0.3, away_win: 0.2 },
      P_kalshi: null,
      P_bet365: null,
      sigma_MC: { home_win: 0.002, draw: 0.002, away_win: 0.002 },
      order_allowed: true,
      cooldown: false,
      ob_freeze: false,
      event_state: "NORMAL",
      mu_H: 1.0,
      mu_A: 1.0,
      score: null,
    }));

    render(<PriceChart ticks={ticks} events={[]} />);
    expect(screen.getByText("Data collecting...")).toBeInTheDocument();
  });
});
