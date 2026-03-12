import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBar } from "../StatusBar";

describe("StatusBar", () => {
  it("exposure < 15% is green", () => {
    render(
      <StatusBar
        bankroll={10000}
        exposure_pct={0.10}
        drawdown_pct={0.05}
        trading_mode="paper"
        ws_status="connected"
      />,
    );
    const exposure = screen.getByText("10.0%");
    expect(exposure.className).toContain("text-green-600");
  });

  it("exposure 15-20% is yellow", () => {
    render(
      <StatusBar
        bankroll={10000}
        exposure_pct={0.17}
        drawdown_pct={0.05}
        trading_mode="paper"
        ws_status="connected"
      />,
    );
    const exposure = screen.getByText("17.0%");
    expect(exposure.className).toContain("text-yellow-600");
  });

  it("exposure > 20% is red", () => {
    render(
      <StatusBar
        bankroll={10000}
        exposure_pct={0.22}
        drawdown_pct={0.05}
        trading_mode="paper"
        ws_status="connected"
      />,
    );
    const exposure = screen.getByText("22.0%");
    expect(exposure.className).toContain("text-red-600");
  });

  it("null bankroll shows 'Loading...'", () => {
    render(
      <StatusBar
        bankroll={null}
        exposure_pct={null}
        drawdown_pct={null}
        trading_mode="paper"
        ws_status="connecting"
      />,
    );
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });
});
