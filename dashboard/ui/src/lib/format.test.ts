// dashboard/ui/src/lib/format.test.ts
//
// Test cases from docs/dashboard_decomposition.md Part 6.

import { describe, expect, it } from "vitest";

import {
  directionBg,
  exposureColor,
  formatBankroll,
  formatCents,
  formatCount,
  formatDuration,
  formatEdge,
  formatLatency,
  formatPct,
  formatPnL,
  formatProb,
  formatProbPct,
  formatSigma,
  formatTime,
  heartbeatDot,
  pnlColor,
  wsStatusColor,
} from "./format";

// ── formatProb ──────────────────────────────────────────────────────────────

describe("formatProb", () => {
  it("rounds to 2 decimal places", () => {
    expect(formatProb(0.5523)).toBe("0.55");
  });
  it("formats 0.0", () => {
    expect(formatProb(0.0)).toBe("0.00");
  });
  it("formats 1.0", () => {
    expect(formatProb(1.0)).toBe("1.00");
  });
});

// ── formatProbPct ───────────────────────────────────────────────────────────

describe("formatProbPct", () => {
  it("converts to percent with 1 decimal", () => {
    expect(formatProbPct(0.5523)).toBe("55.2%");
  });
});

// ── formatEdge ──────────────────────────────────────────────────────────────

describe("formatEdge", () => {
  it("positive edge", () => {
    expect(formatEdge(0.042)).toBe("+4.2¢");
  });
  it("negative edge", () => {
    expect(formatEdge(-0.013)).toBe("-1.3¢");
  });
  it("zero edge", () => {
    expect(formatEdge(0.0)).toBe("+0.0¢");
  });
});

// ── formatPnL ───────────────────────────────────────────────────────────────

describe("formatPnL", () => {
  it("positive P&L", () => {
    expect(formatPnL(12.5)).toBe("+$12.50");
  });
  it("negative P&L", () => {
    expect(formatPnL(-3.2)).toBe("-$3.20");
  });
  it("zero P&L", () => {
    expect(formatPnL(0.0)).toBe("$0.00");
  });
});

// ── formatBankroll ──────────────────────────────────────────────────────────

describe("formatBankroll", () => {
  it("formats with commas and 2 decimals", () => {
    expect(formatBankroll(9847.32)).toBe("$9,847.32");
  });
  it("formats round number", () => {
    expect(formatBankroll(10000)).toBe("$10,000.00");
  });
});

// ── formatPct ───────────────────────────────────────────────────────────────

describe("formatPct", () => {
  it("formats exposure percent", () => {
    expect(formatPct(0.124)).toBe("12.4%");
  });
  it("formats kelly fraction", () => {
    expect(formatPct(0.0265)).toBe("2.7%");
  });
});

// ── formatTime ──────────────────────────────────────────────────────────────

describe("formatTime", () => {
  it("regular time", () => {
    expect(formatTime(67.3)).toBe("67'");
  });
  it("second half stoppage time", () => {
    expect(formatTime(93, { period: 2, regular: 90 })).toBe("90+3'");
  });
  it("first half stoppage time", () => {
    expect(formatTime(47, { period: 1, regular: 45 })).toBe("45+2'");
  });
});

// ── formatSigma ─────────────────────────────────────────────────────────────

describe("formatSigma", () => {
  it("formats to 4 decimal places", () => {
    expect(formatSigma(0.00224)).toBe("0.0022");
  });
});

// ── formatCents ─────────────────────────────────────────────────────────────

describe("formatCents", () => {
  it("formats 0.65 → 65¢", () => {
    expect(formatCents(0.65)).toBe("65¢");
  });
  it("formats 0.05 → 5¢", () => {
    expect(formatCents(0.05)).toBe("5¢");
  });
});

// ── formatLatency ───────────────────────────────────────────────────────────

describe("formatLatency", () => {
  it("milliseconds", () => {
    expect(formatLatency(0.0076)).toBe("7.6ms");
  });
  it("seconds", () => {
    expect(formatLatency(1.2)).toBe("1.2s");
  });
});

// ── formatDuration ──────────────────────────────────────────────────────────

describe("formatDuration", () => {
  it("formats seconds to Xm Ys", () => {
    expect(formatDuration(1395)).toBe("23m 15s");
  });
});

// ── formatCount ─────────────────────────────────────────────────────────────

describe("formatCount", () => {
  it("comma-separated count", () => {
    expect(formatCount(2412)).toBe("2,412");
  });
});

// ── Color helpers ───────────────────────────────────────────────────────────

describe("pnlColor", () => {
  it("green for positive", () => {
    expect(pnlColor(10)).toBe("text-green-600");
  });
  it("red for negative", () => {
    expect(pnlColor(-5)).toBe("text-red-600");
  });
  it("gray for zero", () => {
    expect(pnlColor(0)).toBe("text-gray-500");
  });
});

describe("exposureColor", () => {
  it("green for 0-15%", () => {
    expect(exposureColor(0.1)).toBe("text-green-600");
  });
  it("yellow for 15-20%", () => {
    expect(exposureColor(0.16)).toBe("text-yellow-600");
  });
  it("red for >20%", () => {
    expect(exposureColor(0.21)).toBe("text-red-600");
  });
});

describe("directionBg", () => {
  it("green for BUY_YES", () => {
    expect(directionBg("BUY_YES")).toBe("bg-green-50");
  });
  it("red for BUY_NO", () => {
    expect(directionBg("BUY_NO")).toBe("bg-red-50");
  });
  it("gray for HOLD", () => {
    expect(directionBg("HOLD")).toBe("bg-gray-50");
  });
});

describe("wsStatusColor", () => {
  it("green for connected", () => {
    expect(wsStatusColor("connected")).toBe("text-green-500");
  });
  it("yellow for reconnecting", () => {
    expect(wsStatusColor("reconnecting")).toBe("text-yellow-500");
  });
  it("red for disconnected", () => {
    expect(wsStatusColor("disconnected")).toBe("text-red-500");
  });
});

describe("heartbeatDot", () => {
  it("green for healthy", () => {
    expect(heartbeatDot(true)).toBe("bg-green-500 rounded-full");
  });
  it("red for stale", () => {
    expect(heartbeatDot(false)).toBe("bg-red-500 rounded-full");
  });
});
