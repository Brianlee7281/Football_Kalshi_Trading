import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PositionTable } from "../PositionTable";
import type { MarketProbs, PositionItem } from "@/lib/types";

const buyYesPosition: PositionItem = {
  id: 1,
  match_id: "m1",
  market_ticker: "SOCCER-EPL-ARS-v-CHE-WINNER-HOME",
  direction: "BUY_YES",
  entry_price: 0.55,
  quantity: 10,
  status: "OPEN",
  is_paper: true,
  entry_time: "2026-03-10T20:05:00Z",
  exit_time: null,
  exit_price: null,
  settlement_price: null,
  realized_pnl: null,
};

const buyNoPosition: PositionItem = {
  id: 2,
  match_id: "m1",
  market_ticker: "SOCCER-EPL-ARS-v-CHE-DRAW",
  direction: "BUY_NO",
  entry_price: 0.70,
  quantity: 5,
  status: "OPEN",
  is_paper: true,
  entry_time: "2026-03-10T20:10:00Z",
  exit_time: null,
  exit_price: null,
  settlement_price: null,
  realized_pnl: null,
};

const P_kalshi: MarketProbs = {
  home_win: 0.60,
  draw: 0.25,
  away_win: 0.15,
};

describe("PositionTable", () => {
  it("computes unrealized P&L correctly for BUY_YES", () => {
    // BUY_YES: qty * (current - entry) = 10 * (0.60 - 0.55) = $0.50
    render(
      <PositionTable positions={[buyYesPosition]} P_kalshi={P_kalshi} />,
    );
    expect(screen.getByText("+$0.50")).toBeInTheDocument();
  });

  it("computes unrealized P&L correctly for BUY_NO", () => {
    // BUY_NO: qty * (entry - current) = 5 * (0.70 - 0.25) = $2.25
    render(
      <PositionTable positions={[buyNoPosition]} P_kalshi={P_kalshi} />,
    );
    expect(screen.getByText("+$2.25")).toBeInTheDocument();
  });

  it("shows 'No open positions' when empty", () => {
    render(<PositionTable positions={[]} P_kalshi={null} />);
    expect(
      screen.getByText("No open positions for this match"),
    ).toBeInTheDocument();
  });

  it("shows dash when P_kalshi is null", () => {
    render(
      <PositionTable positions={[buyYesPosition]} P_kalshi={null} />,
    );
    // Current and P&L columns should show "—"
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });
});
