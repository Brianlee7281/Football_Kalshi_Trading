# Key System Patterns

These patterns are non-negotiable. Every piece of code must follow them.

## 1. Phase 3→4 Interface (dict, not float)

Phase 3 emits `P_true: dict` and `σ_MC: dict` keyed by market:
```python
{"P_true": {"home_win": 0.55, "over_25": 0.65}, "σ_MC": {"home_win": 0.0022, "over_25": 0.0021}, "order_allowed": True}
```
Pushed to `model.phase4_queue` (asyncio.Queue, maxsize=1, stale ticks replaced).

Phase 4 `signal_generator` decomposes dict→float per Kalshi ticker:
```python
for ticker in model.active_tickers:
    market_key = model.ticker_to_model_key[ticker]
    P_true_float = P_true_dict[market_key]
    σ_MC_float = σ_MC_dict[market_key]
    # → generate_signal(), compute_kelly(), execute_with_reservation()
```

## 2. Paper/Live Mode

`ExecutionRouter` in `src/execution/execution_router.py` dispatches based on `TRADING_MODE` env var.
All upstream logic (pricing, signals, Kelly) is IDENTICAL in both modes. Only fill mechanism differs.
Phase 3 is completely mode-invariant.

## 3. Risk Limits — Reserve-Confirm-Release

Never hold Redis lock during order execution. Pattern:
1. **Reserve** (lock <10ms): write to `exposure_reservation` table, lock released
2. **Execute** (no lock, 1-5s): submit order to Kalshi
3. **Confirm/Release** (lock <10ms): update reservation status

`get_total_exposure()` includes RESERVED amounts, preventing overallocation.

## 4. Wall-Clock Tick with Halftime Exclusion

```python
model.t = (wall_elapsed - model.halftime_accumulated) / 60  # effective play minutes
```
NEVER `model.t += 1/60`. Halftime is excluded via `halftime_accumulated` tracking.
`handle_period_change` records halftime start/end.

## 5. Parameter Version Pinning

Container loads `PARAM_VERSION` at startup, keeps it for entire match.
New Phase 1 params apply to NEXT match only. Never reload mid-match.

## 6. Incremental Kelly

```python
f_incremental = max(0, f_optimal - existing_fraction)
```
If already at optimal allocation, f=0 (no new order). If edge grew, add the difference.
Opposite-direction positions must be closed first (exit logic, not Kelly).

## 7. Per-Market σ_MC

```python
σ_MC = {market: math.sqrt(p * (1-p) / N) for market, p in P_true.items()}
```
Not a single float. Each market has its own uncertainty based on its probability.

## 8. Settlement Polling

Don't assume instant settlement. Poll Kalshi for resolution (60s intervals, up to 6h).
Positions stay in `AWAITING_SETTLEMENT` until resolved.

## 9. Rapid Sequential Events

Cooldown blocks ORDERS, not STATE UPDATES. New goal during cooldown:
- State (S, ΔS, μ) updates immediately
- Cooldown timer RESETS (cancel old, start new 15s)
- Events during PRELIMINARY are queued (EventQueue), drained on confirmation
