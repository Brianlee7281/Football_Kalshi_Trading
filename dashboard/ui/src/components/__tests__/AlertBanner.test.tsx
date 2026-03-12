import React from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AlertBanner } from "../AlertBanner";
import type { SystemAlertMessage } from "@/lib/types";

afterEach(() => {
  cleanup();
});

const criticalAlert: SystemAlertMessage = {
  type: "alert",
  severity: "critical",
  title: "Container crash",
  details: { match_id: "m1", exit_code: "1" },
  timestamp: Date.now() / 1000,
};

const infoAlert: SystemAlertMessage = {
  type: "alert",
  severity: "info",
  title: "New param version",
  details: { version: "3" },
  timestamp: Date.now() / 1000,
};

describe("AlertBanner", () => {
  it("critical alert stays visible (no auto-dismiss)", () => {
    vi.useFakeTimers();

    render(<AlertBanner alerts={[criticalAlert]} />);
    expect(screen.getByText("Container crash")).toBeInTheDocument();

    // Advance well past any auto-dismiss window
    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    // Critical should still be visible
    expect(screen.getByText("Container crash")).toBeInTheDocument();

    vi.useRealTimers();
  });

  it("info alert auto-dismisses after 10s", () => {
    vi.useFakeTimers();

    render(<AlertBanner alerts={[infoAlert]} />);
    expect(screen.getByText("New param version")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(11_000);
    });

    expect(screen.queryByText("New param version")).not.toBeInTheDocument();

    vi.useRealTimers();
  });
});
