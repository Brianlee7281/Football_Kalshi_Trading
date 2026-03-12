import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MatchCard } from "../MatchCard";
import type { MatchSummary } from "@/lib/types";

// Mock next/link to render a plain anchor
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

afterEach(() => {
  cleanup();
});

const baseMatch: MatchSummary = {
  match_id: "m1",
  league_id: 39,
  kickoff_utc: new Date(Date.now() + 3600_000).toISOString(), // 1h from now
  status: "SCHEDULED",
  trading_mode: "paper",
  home_team: "ARS",
  away_team: "CHE",
  score: null,
  param_version: 1,
};

describe("MatchCard", () => {
  it("SCHEDULED shows 'Starting in'", () => {
    render(
      <MatchCard
        match={baseMatch}
        latestTick={null}
        wsConnected={true}
      />,
    );
    expect(screen.getByText(/Starting in/)).toBeInTheDocument();
  });

  it("no WS shows 'Live data unavailable' for LIVE match", () => {
    const liveMatch: MatchSummary = {
      ...baseMatch,
      status: "PHASE3_RUNNING",
    };
    render(
      <MatchCard
        match={liveMatch}
        latestTick={null}
        wsConnected={false}
      />,
    );
    expect(screen.getByText("Live data unavailable")).toBeInTheDocument();
  });
});
