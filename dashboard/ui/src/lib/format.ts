// dashboard/ui/src/lib/format.ts
//
// All dashboard formatting in one place.
// Reference: docs/dashboard_decomposition.md Part 5

/** Probability → 2 decimal places.  formatProb(0.5523) → "0.55" */
export function formatProb(p: number): string {
  return p.toFixed(2);
}

/** Probability as percent → 1 decimal + %.  formatProbPct(0.5523) → "55.2%" */
export function formatProbPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** Edge (EV in dollars) → signed cents with 1 decimal.  formatEdge(0.042) → "+4.2¢" */
export function formatEdge(ev: number): string {
  const cents = ev * 100;
  const sign = cents >= 0 ? "+" : "";
  return `${sign}${cents.toFixed(1)}¢`;
}

/** P&L → signed dollar amount, 2 decimals.  formatPnL(12.5) → "+$12.50" */
export function formatPnL(pnl: number): string {
  if (pnl > 0) return `+$${pnl.toFixed(2)}`;
  if (pnl < 0) return `-$${Math.abs(pnl).toFixed(2)}`;
  return `$${pnl.toFixed(2)}`;
}

/** Bankroll → comma-separated dollar amount.  formatBankroll(9847.32) → "$9,847.32" */
export function formatBankroll(amount: number): string {
  return `$${amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** Ratio → 1 decimal percent.  formatPct(0.124) → "12.4%" */
export function formatPct(ratio: number): string {
  // Round via integers to avoid floating-point issues (e.g. 0.0265*100 = 2.649...)
  const scaled = Math.round(ratio * 1000) / 10;
  return `${scaled.toFixed(1)}%`;
}

/** Match time → integer minutes with prime symbol.
 *  formatTime(67.3) → "67'"
 *  formatTime(93, {period: 2, regular: 90}) → "90+3'"
 *  formatTime(47, {period: 1, regular: 45}) → "45+2'"
 */
export function formatTime(
  t: number,
  opts?: { period?: number; regular?: number },
): string {
  if (opts?.period != null && opts?.regular != null) {
    const elapsed = Math.floor(t);
    const regular = opts.regular;
    if (elapsed > regular) {
      return `${regular}+${elapsed - regular}'`;
    }
  }
  return `${Math.floor(t)}'`;
}

/** σ_MC → 4 decimal places.  formatSigma(0.00224) → "0.0022" */
export function formatSigma(sigma: number): string {
  return sigma.toFixed(4);
}

/** Price in [0,1] → integer cents.  formatCents(0.65) → "65¢" */
export function formatCents(price: number): string {
  return `${Math.round(price * 100)}¢`;
}

/** Latency in seconds → human-friendly.  formatLatency(0.0076) → "7.6ms" */
export function formatLatency(seconds: number): string {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)}ms`;
  }
  return `${seconds.toFixed(1)}s`;
}

/** Unix timestamp → HH:MM:SS in local timezone. */
export function formatTimestamp(unix: number): string {
  const d = new Date(unix * 1000);
  return d.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Date object or ISO string → YYYY-MM-DD. */
export function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toISOString().slice(0, 10);
}

/** Duration in seconds → "Xm Ys".  formatDuration(1395) → "23m 15s" */
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

/** Integer count → comma-separated.  formatCount(2412) → "2,412" */
export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

// ── Color helpers (return Tailwind class strings) ───────────────────────────

export function pnlColor(pnl: number): string {
  if (pnl > 0) return "text-green-600";
  if (pnl < 0) return "text-red-600";
  return "text-gray-500";
}

export function exposureColor(pct: number): string {
  if (pct > 0.2) return "text-red-600";
  if (pct > 0.15) return "text-yellow-600";
  return "text-green-600";
}

export function directionBg(direction: string): string {
  if (direction === "BUY_YES") return "bg-green-50";
  if (direction === "BUY_NO") return "bg-red-50";
  return "bg-gray-50";
}

export function wsStatusColor(
  status: "connected" | "reconnecting" | "disconnected",
): string {
  if (status === "connected") return "text-green-500";
  if (status === "reconnecting") return "text-yellow-500";
  return "text-red-500";
}

export function heartbeatDot(healthy: boolean): string {
  return healthy ? "bg-green-500 rounded-full" : "bg-red-500 rounded-full";
}
