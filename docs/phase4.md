# Phase 4: Arbitrage & Execution 


## Overview

After three prior phases that refine the mathematical probability model (`P_true`),
this is the execution front line where model output is converted into real money.

No matter how good the model is, poor design in order-book friction handling
or money management can still lead to ruin.

Using the probability from the Phase 3 engine,
we detect edge in Kalshi markets,
submit orders based on Kelly,
and feed post-match outcomes back into the system.
This is decomposed into 6 steps.

### Paradigm Shift: Defensive → Adaptive

The 2-layer event detection system (Odds-API + Goalserve) fundamentally changes Phase 4 strategy:

```
Original: event detection (3–8s) > Kalshi MM (1–2s) → always defensive
Odds-API + Goalserve: event detection (<1s) ≤ Kalshi MM (1–2s) → can be aggressive depending on conditions
```

Aggressiveness is still increased **gradually** along the Phase 0 → A → B → C roadmap.

### Buy Yes vs Buy No — Probability Space Convention

In this document, all probabilities and prices are represented in the **Yes-probability space**:

| Symbol | Meaning | Space |
|------|------|------|
| P_true | Model-estimated true event probability | Yes probability |
| P_kalshi_buy | Kalshi Best Ask (Yes buy price) | Yes probability |
| P_kalshi_sell | Kalshi Best Bid (Yes sell price) | Yes probability |
| P_bet365 | bet365 implied probability | Yes probability |
| entry_price | Fill price at entry | Yes probability |
| settlement | Settlement at expiry (100¢ or 0¢) | Yes probability |

**Buy No position:**
- Equivalent to selling Yes → `entry_price` is the Yes sell price
- No wins (= Yes settles at 0¢) → profit = `entry_price`
- No loses (= Yes settles at 100¢) → loss = `(1 - entry_price)`

> **Core v2 principle:** In all Buy No formulas, do **not** apply `(1 - P)` conversion.
> `P_cons`, `P_kalshi`, `P_bet365`, and `entry_price` are all compared directly in Yes space.
> Conversion is only used in direction-specific payoff calculations (win/loss branches).

---

## Input Data

**Phase 3 outputs (every second):**

| Item | Type | Description | Update Frequency |
|------|------|------|-------------|
| P_true(t) | `dict` | True probability per market in Yes space (e.g., `{"home_win": 0.55, "over_25": 0.65}`) | every 1s |
| σ_MC(t) | `dict` | Monte Carlo standard error per market (e.g., `{"home_win": 0.0022, "over_25": 0.0021}`) | every 1s |
| order_allowed | `bool` | NOT cooldown AND NOT ob_freeze AND event_state == IDLE | every 1s + on events |
| event_state | `str` | IDLE / PRELIMINARY / CONFIRMED | on events |
| pricing_mode | `str` | Analytical / Monte Carlo | switches on events |
| engine_phase | `str` | FIRST_HALF / HALFTIME / SECOND_HALF / FINISHED | on period change |
| μ_H, μ_A | `float` | Expected remaining goals | every 1s (logging) |
| **P_bet365(t)** | **`dict`** | **bet365 in-play implied probability per market (Yes space)** | **every push (<1s)** |

> **Market key mapping:** Phase 3 outputs use model keys (`home_win`, `over_25`, `btts_yes`).
> Phase 4 maps these to Kalshi tickers via `ticker_to_model_key` lookup (see signal_generator).
> Phase 4 functions (`compute_signal_with_vwap`, `compute_kelly`, etc.) receive **float** values
> for a single market — the dict-to-float decomposition happens in `signal_generator`.

**Kalshi API:**

| Endpoint | Use |
|-----------|------|
| WebSocket | Real-time order book (Bid/Ask + Depth) |
| REST `/portfolio/orders` | Submit/cancel orders |
| REST `/portfolio/positions` | Existing position lookup |
| REST `/portfolio/balance` | Account balance |

---

## Step 4.1: Live Order Book Synchronization

### Goal

Receive Kalshi order-book data in real time,
align it on the same timestamp axis as Phase 3 `P_true`,
and add Odds-API Live Odds bet365 prices as a market reference.

### Kalshi Quote Intake

**Bid/Ask separation:**

$$P_{kalshi}^{buy} = \frac{\text{Best Ask (¢)}}{100}, \quad P_{kalshi}^{sell} = \frac{\text{Best Bid (¢)}}{100}$$

**Order-book depth (Depth) — VWAP effective price:**

> **[v2 fix #5] Connect VWAP directly to EV calculation in Step 4.2.**

For order size $Q$, compute the weighted average effective price:

$$P_{effective}(Q) = \frac{\sum_{level} p_{level} \times q_{level}}{\sum_{level} q_{level}} \quad \text{(accumulated until Q contracts)}$$

```python
class OrderBookSync:
    def __init__(self):
        # Kalshi quotes
        self.kalshi_best_bid = None
        self.kalshi_best_ask = None
        self.kalshi_depth_ask = []  # [(price, qty), ...] ascending
        self.kalshi_depth_bid = []  # [(price, qty), ...] descending
        self.kalshi_last_update = 0.0    # timestamp of last Kalshi WS message

        # bet365 reference prices
        self.bet365_implied = {}
        self.bet365_last_update = 0.0    # timestamp of last bet365 odds update

    KALSHI_STALE_THRESHOLD = 5.0    # seconds — skip trading if order book older than this
    BET365_STALE_THRESHOLD = 30.0   # seconds — treat as UNAVAILABLE if older than this

    @property
    def kalshi_is_stale(self) -> bool:
        """True if Kalshi order book data is too old to trade on."""
        if self.kalshi_last_update == 0.0:
            return True
        return (time.time() - self.kalshi_last_update) > self.KALSHI_STALE_THRESHOLD

    @property
    def bet365_is_stale(self) -> bool:
        """True if bet365 data is too old for alignment check."""
        if self.bet365_last_update == 0.0:
            return True
        return (time.time() - self.bet365_last_update) > self.BET365_STALE_THRESHOLD

    def get_bet365_for_alignment(self, market_key: str) -> Optional[float]:
        """Return bet365 implied prob, or None if data is stale.
        Stale data yields UNAVAILABLE alignment (multiplier 0.6) instead of
        false ALIGNED/DIVERGENT on outdated information."""
        if self.bet365_is_stale:
            return None
        return self.bet365_implied.get(market_key)

    def compute_vwap_buy(self, target_qty: int) -> Optional[float]:
        """
        Effective buy price (VWAP) for target_qty contracts.
        Consumes ask levels from low to high.
        Returns None if depth is insufficient.
        """
        if not self.kalshi_depth_ask:
            return None

        filled = 0
        cost = 0.0
        for price, qty in self.kalshi_depth_ask:
            take = min(qty, target_qty - filled)
            cost += price * take
            filled += take
            if filled >= target_qty:
                break

        if filled < target_qty:
            return None  # insufficient depth

        return cost / filled

    def compute_vwap_sell(self, target_qty: int) -> Optional[float]:
        """Effective sell price for target_qty contracts (Bid VWAP)."""
        if not self.kalshi_depth_bid:
            return None

        filled = 0
        revenue = 0.0
        for price, qty in self.kalshi_depth_bid:
            take = min(qty, target_qty - filled)
            revenue += price * take
            filled += take
            if filled >= target_qty:
                break

        if filled < target_qty:
            return None
        return revenue / filled

    def update_bet365(self, odds_api_msg: dict):
        """Odds-API Live Odds WebSocket → convert to bet365 implied probabilities.

        Odds-API WS message format (filtered to Bet365):
        {
            "type": "updated",
            "bookie": "Bet365",
            "markets": [
                {"name": "ML", "odds": [{"home": "1.44", "draw": "3.50", "away": "12.00"}]},
                {"name": "Totals", "odds": [{"hdp": 2.5, "over": "1.90", "under": "1.90"}]}
            ]
        }
        """
        markets = odds_api_msg.get("markets", [])
        for market in markets:
            if market.get("name") == "ML":
                try:
                    odds = market["odds"][0]
                    home_odds = float(odds["home"])
                    draw_odds = float(odds["draw"])
                    away_odds = float(odds["away"])

                    raw_sum = 1/home_odds + 1/draw_odds + 1/away_odds
                    self.bet365_implied["home_win"] = (1/home_odds) / raw_sum
                    self.bet365_implied["draw"] = (1/draw_odds) / raw_sum
                    self.bet365_implied["away_win"] = (1/away_odds) / raw_sum
                except (KeyError, ValueError, IndexError, ZeroDivisionError):
                    pass

            elif market.get("name") == "Totals":
                try:
                    odds = market["odds"][0]
                    over_odds = float(odds["over"])
                    under_odds = float(odds["under"])
                    hdp = float(odds.get("hdp", 2.5))

                    raw_sum = 1/over_odds + 1/under_odds
                    self.bet365_implied[f"over_{hdp}"] = (1/over_odds) / raw_sum
                    self.bet365_implied[f"under_{hdp}"] = (1/under_odds) / raw_sum
                except (KeyError, ValueError, IndexError, ZeroDivisionError):
                    pass
```

### Liquidity Filter

Do not enter if total depth is below minimum threshold:

$$\text{Total Ask Depth} \geq Q_{min} \quad (\text{e.g., } Q_{min} = 20\text{ contracts})$$

### bet365 Reference Price — "Market Alignment Check" (Not Independent Validation)

> **[v2 fix #4] Reclassify bet365 from "independent validator" to "market alignment check."**
>
> Reason: `P_true` and `P_bet365` are **both derived from the same Odds-API feed**,
> especially right after events, where both may simply reflect the same information.
> So this is not truly independent validation, but a directional alignment check.
> Therefore, set `kelly_multiplier` to 0.8 instead of 1.0.
>
> **When bet365 is most useful:**
> - No-event normal ticks: model uses MMPP time decay, bet365 uses market-making dynamics
>   → different information handling → meaningful alignment value
> - Immediately after events: both derived from same Odds-API event signal
>   → low independence → weaker alignment value

**Three uses of bet365 reference price:**

| Use | Method | Step |
|------|------|------|
| **Market alignment check** | Compare model direction vs bet365 direction | Step 4.2 |
| **Sizing adjustment** | 0.8x when aligned, 0.5x when divergent | Step 4.3 |
| **Exit support signal** | Warning when bet365 moves against position | Step 4.4 |

### Outputs

| Item | Description |
|------|------|
| $P_{kalshi}^{buy}(t)$ | Best ask for buying Yes (top of book) |
| $P_{kalshi}^{sell}(t)$ | Best bid for selling Yes (top of book) |
| $P_{effective}^{buy}(Q)$ | VWAP effective buy price for Q contracts |
| $P_{effective}^{sell}(Q)$ | VWAP effective sell price for Q contracts |
| $P_{bet365}(t)$ | bet365 in-play implied probability (by market) |
| liquidity_ok | Liquidity filter pass/fail |
| depth_profile | Size by order-book level |

---

## Step 4.2: Fee-Adjusted Edge Detection (EV Computation)

### Goal

Compare model `P_true` and market `P_kalshi`,
verify positive expected value after fees/slippage,
and classify edge reliability with bet365 market alignment.

### P_true^cons — Directional Conservative Adjustment

```python
def compute_conservative_P(P_true: float, sigma_MC: float,
                            direction: str, z: float = 1.645) -> float:
    """
    Buy Yes: higher P is favorable → use lower bound (conservatively reduce)
    Buy No:  lower P is favorable → use upper bound (conservatively increase)
    """
    if direction == "BUY_YES":
        return P_true - z * sigma_MC
    elif direction == "BUY_NO":
        return P_true + z * sigma_MC
    else:
        return P_true
```

> **Why direction must differ:**
> If one lower bound (`P_true - z·σ`) is used for both directions,
> Buy No gets artificially inflated `(1 - P_cons)`,
> systematically **overestimating** No-side EV.
> The larger MC uncertainty is, the more aggressively the system overbets No.

### Fee-Adjusted EV — 2-Pass VWAP Connection

> **[v2 fix #5] Use VWAP effective price (not best ask/bid) in EV calculation.**
>
> Circular dependency: EV → Kelly → qty → VWAP → EV
> Solution: 2-pass computation.

```python
def compute_signal_with_vwap(
    P_true: float, sigma_MC: float,
    ob_sync: OrderBookSync,
    c: float, z: float, K_frac: float,
    bankroll: float, market_ticker: str
) -> Signal:
    """
    Connect VWAP to EV with 2-pass computation.

    Pass 1: estimate rough quantity with best ask/bid
    Pass 2: compute final EV with VWAP for rough quantity
    """
    # ═══ Pass 1: rough evaluation with best ask/bid ═══
    P_best_ask = ob_sync.kalshi_best_ask
    P_best_bid = ob_sync.kalshi_best_bid

    # Buy Yes side
    P_cons_yes = P_true - z * sigma_MC
    rough_EV_yes = (
        P_cons_yes * (1 - c) * (1 - P_best_ask)
        - (1 - P_cons_yes) * P_best_ask
    )

    # Buy No side
    P_cons_no = P_true + z * sigma_MC
    rough_EV_no = (
        (1 - P_cons_no) * (1 - c) * P_best_bid
        - P_cons_no * (1 - P_best_bid)
    )

    # Direction selection (higher EV)
    if rough_EV_yes > rough_EV_no and rough_EV_yes > THETA_ENTRY:
        direction = "BUY_YES"
        rough_P_kalshi = P_best_ask
        P_cons = P_cons_yes
    elif rough_EV_no > THETA_ENTRY:
        direction = "BUY_NO"
        rough_P_kalshi = P_best_bid
        P_cons = P_cons_no
    else:
        return Signal(direction="HOLD")

    # Rough quantity
    rough_f = rough_kelly(direction, P_cons, rough_P_kalshi, c, K_frac, rough_EV_yes if direction == "BUY_YES" else rough_EV_no)
    rough_qty = int(rough_f * bankroll / rough_P_kalshi)
    if rough_qty < 1:
        return Signal(direction="HOLD")

    # ═══ Pass 2: final EV with VWAP ═══
    if direction == "BUY_YES":
        P_effective = ob_sync.compute_vwap_buy(rough_qty)
    else:
        P_effective = ob_sync.compute_vwap_sell(rough_qty)

    if P_effective is None:
        return Signal(direction="HOLD")  # insufficient depth

    # Final EV (VWAP-based)
    if direction == "BUY_YES":
        final_EV = (
            P_cons * (1 - c) * (1 - P_effective)
            - (1 - P_cons) * P_effective
        )
    else:  # BUY_NO
        final_EV = (
            (1 - P_cons) * (1 - c) * P_effective
            - P_cons * (1 - P_effective)
        )

    if final_EV <= THETA_ENTRY:
        return Signal(direction="HOLD")  # edge disappears after VWAP

    return Signal(
        direction=direction,
        EV=final_EV,
        P_cons=P_cons,
        P_kalshi=P_effective,  # ← VWAP effective price
        rough_qty=rough_qty,
        market_ticker=market_ticker
    )
```

> **THETA_ENTRY calibration:** `THETA_ENTRY` (default 2¢) represents the minimum edge
> after fees and VWAP slippage. During Phase 0 paper trading, measure the empirical
> breakeven edge (fee + average slippage) and adjust `THETA_ENTRY` accordingly.
> If `c = 0.07` and average slippage is 1¢, the breakeven edge is approximately 3-4¢,
> and `THETA_ENTRY` should be set above this level. Include `THETA_ENTRY` in the
> Step 4.6 adaptive tuning loop once sufficient trade data is accumulated.

### Market Alignment Check — bet365 Reference

> **[v2 fix #4] Renamed from "independent validation" to "market alignment check."**
> `kelly_multiplier`: ALIGNED=0.8 (not 1.0), DIVERGENT=0.5.

```python
@dataclass
class MarketAlignment:
    status: str             # "ALIGNED", "DIVERGENT", "UNAVAILABLE"
    kelly_multiplier: float # ALIGNED→0.8, DIVERGENT→0.5, UNAVAILABLE→0.6

def check_market_alignment(
    P_true_cons: float,
    P_kalshi: float,
    P_bet365: Optional[float],
    direction: str
) -> MarketAlignment:
    """
    Check directional alignment between model and bet365.

    This is NOT independent validation:
    - both are derived from the same Odds-API feed
    - captures interpretation gap between model (MMPP) and market (trader+algo)
    - even when aligned, use 0.8 instead of 1.0 (prevent overconfidence)
    """
    if P_bet365 is None:
        return MarketAlignment(
            status="UNAVAILABLE",
            kelly_multiplier=0.6  # conservative when data is missing
        )

    # All comparisons are in Yes probability space
    if direction == "BUY_YES":
        model_says_high = P_true_cons > P_kalshi
        bet365_says_high = P_bet365 > P_kalshi
        aligned = model_says_high and bet365_says_high

    elif direction == "BUY_NO":
        model_says_low = P_true_cons < P_kalshi
        bet365_says_low = P_bet365 < P_kalshi
        aligned = model_says_low and bet365_says_low

    else:
        return MarketAlignment(status="UNAVAILABLE", kelly_multiplier=0.6)

    if aligned:
        return MarketAlignment(
            status="ALIGNED",
            kelly_multiplier=0.8  # [v2] not 1.0 (reflect limited independence)
        )
    else:
        return MarketAlignment(
            status="DIVERGENT",
            kelly_multiplier=0.5
        )
```

### Filtering Conditions

| Condition | Description |
|------|------|
| final_EV > θ_entry | Minimum edge (`θ_entry = 0.02 = 2¢`), **after VWAP** |
| order_allowed = True | NOT cooldown AND NOT ob_freeze |
| event_state == IDLE | Not during preliminary event state |
| liquidity_ok = True | Minimum order-book depth satisfied |
| engine_phase ∈ {FIRST_HALF, SECOND_HALF} | No entry during halftime/finished |
| alignment.status ≠ "DIVERGENT" (initially) | Block entry on divergence (Phase A) |
| no_opposite_position(market, direction) | Block if existing position is in opposite direction (close first) |
| pending_order_count(market) == 0 | Block if a working order already exists for this market |

> **Existing same-direction positions are allowed** — incremental Kelly (Step 4.3)
> computes the additional allocation needed. If already at optimal, `f_incremental = 0`
> and no order is placed. Opposite-direction positions must be closed via exit logic
> (Step 4.4) before entering in the new direction.

> **Filter relaxation by phase evolution:**
> - Phase A: block entry if DIVERGENT (conservative)
> - Phase B: allow DIVERGENT with multiplier 0.5 (data-driven from Step 4.6)
> - Phase C: tune multiplier if Step 4.6 shows positive divergent performance

### Signal Generation

```python
@dataclass
class Signal:
    direction: str              # BUY_YES, BUY_NO, HOLD
    EV: float                   # Final EV after VWAP
    P_cons: float               # Directional conservative P
    P_kalshi: float             # VWAP effective price
    rough_qty: int              # Rough quantity from Pass 1
    alignment_status: str       # ALIGNED, DIVERGENT, UNAVAILABLE
    kelly_multiplier: float     # 0.8, 0.5, 0.6
    market_ticker: str

def generate_signal(P_true, sigma_MC, ob_sync, P_bet365,
                    c, z, K_frac, bankroll, market_ticker) -> Signal:
    """2-pass VWAP + market alignment check"""

    # 2-pass VWAP computation
    base_signal = compute_signal_with_vwap(
        P_true, sigma_MC, ob_sync, c, z, K_frac, bankroll, market_ticker
    )

    if base_signal.direction == "HOLD":
        return base_signal

    # Market alignment check
    alignment = check_market_alignment(
        base_signal.P_cons, base_signal.P_kalshi, P_bet365, base_signal.direction
    )

    return Signal(
        direction=base_signal.direction,
        EV=base_signal.EV,
        P_cons=base_signal.P_cons,
        P_kalshi=base_signal.P_kalshi,
        rough_qty=base_signal.rough_qty,
        alignment_status=alignment.status,
        kelly_multiplier=alignment.kelly_multiplier,
        market_ticker=market_ticker
    )
```

### Output

```
Signal(direction, EV, P_cons, P_kalshi, rough_qty,
       alignment_status, kelly_multiplier, market_ticker)
```

### Market Key Mapping (Phase 3 → Phase 4)

Phase 3 outputs `P_true: dict` and `σ_MC: dict` keyed by model market names.
Phase 4 maps these to Kalshi tickers for per-market signal generation.

```python
# Model key → Kalshi market type
MODEL_TO_KALSHI_TYPE = {
    "home_win":  "match_winner_home",
    "away_win":  "match_winner_away",
    "draw":      "match_winner_draw",
    "over_25":   "over_under_2.5",
    "over_35":   "over_under_3.5",
    "btts_yes":  "btts",
}

# Populated at Phase 2.5 initialization from match's active Kalshi tickers
# Example: {"SOCCER-EPL-ARS-v-CHE-25MAR15-WINNER": "home_win",
#           "SOCCER-EPL-ARS-v-CHE-25MAR15-OU2.5": "over_25"}
ticker_to_model_key: Dict[str, str]
```

### Signal Generator — Multi-Market Orchestration Loop

```python
async def signal_generator(model):
    """
    Core Phase 4 loop. Runs every tick, processes ALL active Kalshi
    markets for this match. Decomposes Phase 3's P_true dict into
    per-market floats and generates signals independently for each market.

    Flow per tick per market:
    1. Extract P_true[market_key] and σ_MC[market_key] from Phase 3 dict
    2. Get order book snapshot for this market's Kalshi ticker
    3. Get bet365 implied probability for this market
    4. Generate signal (EV, direction, alignment) — Step 4.2
    5. Compute incremental Kelly sizing — Step 4.3
    6. Apply risk limits + reserve exposure — Step 4.3/Orchestration
    7. Execute via ExecutionRouter (paper or live) — Step 4.5
    """
    while model.engine_phase != FINISHED:
        # Wait for next tick data from Phase 3
        tick_data = await model.phase4_queue.get()
        P_true_dict = tick_data["P_true"]      # dict: {"home_win": 0.55, ...}
        σ_MC_dict = tick_data["σ_MC"]           # dict: {"home_win": 0.0022, ...}
        order_allowed = tick_data["order_allowed"]

        if not order_allowed:
            continue

        # Process each active Kalshi market
        for ticker in model.active_tickers:
            market_key = model.ticker_to_model_key.get(ticker)
            if market_key is None or market_key not in P_true_dict:
                continue

            # ─── Decompose dict → float for this market ───
            P_true_float = P_true_dict[market_key]
            σ_MC_float = σ_MC_dict[market_key]

            # ─── Per-market order book and bet365 ───
            ob_sync = model.ob_syncs.get(ticker)
            if ob_sync is None or not ob_sync.liquidity_ok:
                continue

            P_bet365 = model.bet365_implied.get(market_key)

            # ─── Step 4.2: Signal generation (per-market float inputs) ───
            signal = generate_signal(
                P_true_float, σ_MC_float, ob_sync, P_bet365,
                c=model.config.fee_rate,
                z=model.config.z,
                K_frac=model.config.K_frac,
                bankroll=model.bankroll,
                market_ticker=ticker,
            )

            if signal.direction == "HOLD":
                continue

            # ─── Step 4.3: Incremental Kelly ───
            existing = await db.get_existing_exposure(
                model.match_id, ticker, signal.direction
            )
            f_incremental = compute_kelly(
                signal, model.config.fee_rate, model.config.K_frac,
                existing_exposure=existing, bankroll=model.bankroll,
            )

            if f_incremental <= 0:
                continue  # already at or above optimal allocation

            amount = f_incremental * model.bankroll

            # ─── Risk limits + execution ───
            # IMPORTANT: await completes the full reserve→execute→confirm cycle
            # before processing the next ticker. This prevents within-container
            # race conditions where ticker B's Kelly reads stale exposure from ticker A.
            fill = await execute_with_reservation(signal, amount, ob_sync, model)

            # ─── Bankroll refresh after fill ───
            # Without this, subsequent Kelly calculations use startup bankroll,
            # overestimating available capital after each fill.
            if fill and fill.quantity > 0:
                fill_cost = fill.price * fill.quantity
                model.bankroll -= fill_cost
                log.info(f"Bankroll updated: -{fill_cost:.2f}, "
                         f"remaining={model.bankroll:.2f}")

                # ─── Redis publish for dashboard ───
                # signal:{match_id} for signal log, position_update for position table
                asyncio.create_task(_publish_signal_to_redis(
                    model, ticker, signal, fill))


async def _publish_signal_to_redis(model, ticker, signal, fill):
    """Publish signal + fill to Redis for live dashboard."""
    try:
        await model.redis.publish(f"signal:{model.match_id}", json.dumps({
            "type": "signal",
            "match_id": model.match_id,
            "ticker": ticker,
            "direction": signal.direction,
            "EV": signal.EV,
            "P_cons": signal.P_cons,
            "P_kalshi": signal.P_kalshi,
            "alignment": signal.alignment_status,
            "kelly_fraction": signal.kelly_multiplier,
            "fill_qty": fill.quantity,
            "fill_price": fill.price,
            "timestamp": time.time(),
        }))
        await model.redis.publish("position_update", json.dumps({
            "type": "new_fill",
            "match_id": model.match_id,
            "ticker": ticker,
            "direction": signal.direction,
            "quantity": fill.quantity,
            "price": fill.price,
        }))
    except Exception:
        pass  # fire-and-forget, Prometheus counter in emit.py
```

> **Queue interface:** `emit_to_phase4()` in Phase 3 pushes `{"P_true": dict, "σ_MC": dict, "order_allowed": bool}`
> to `model.phase4_queue` (asyncio.Queue). The signal_generator consumes from this queue.
> If Phase 4 processing is slower than 1 tick, old ticks are silently dropped
> (queue maxsize=1, put with `put_nowait` replacing the previous item).

---

## Step 4.3: Position Sizing — Fee-Adjusted Kelly Criterion

### Goal

Compute optimal investment fraction $f^*$ that maximizes long-run geometric growth
while keeping ruin probability at zero.

### Directional Fee-Adjusted Kelly

> Since `P_cons` is already adjusted by direction,
> Kelly should also use direction-specific win/loss (`W/L`) payoffs.

> **Derivation:** For a binary bet with win payoff W and loss payoff L,
> standard Kelly gives $f^* = \frac{p \cdot W - q \cdot L}{W \cdot L} = \frac{EV}{W \cdot L}$
> where $p$ = win probability, $q = 1-p$, and EV = $p \cdot W - q \cdot L$.

```python
def compute_kelly(signal: Signal, c: float, K_frac: float,
                  existing_exposure: float = 0.0,
                  bankroll: float = 1.0) -> float:
    """
    Incremental Kelly with directional P_cons + market alignment multiplier.
    P_kalshi is the VWAP effective price from Step 4.2.

    If a position already exists in the same market+direction,
    compute the optimal total allocation and subtract existing exposure
    to get the incremental amount. This prevents:
    - Duplicate orders flooding the same opportunity
    - Oversizing when edge persists across ticks
    - Undersizing when edge has grown since initial entry
    """
    P_cons = signal.P_cons
    P_kalshi = signal.P_kalshi  # VWAP effective price

    if signal.direction == "BUY_YES":
        W = (1 - c) * (1 - P_kalshi)
        L = P_kalshi
    elif signal.direction == "BUY_NO":
        W = (1 - c) * P_kalshi
        L = (1 - P_kalshi)
    else:
        return 0.0

    if W * L <= 0:
        return 0.0

    f_kelly = signal.EV / (W * L)

    # Fractional Kelly
    f_optimal = K_frac * f_kelly

    # Additional adjustment by market alignment
    f_optimal *= signal.kelly_multiplier

    # ─── Incremental sizing ───
    # existing_exposure: fraction of bankroll already allocated to this market+direction
    # f_optimal: total desired allocation based on current edge
    # f_incremental: additional allocation needed (can be 0 if already at optimal)
    existing_fraction = existing_exposure / bankroll if bankroll > 0 else 0.0
    f_incremental = max(0.0, f_optimal - existing_fraction)

    # If edge has shrunk and existing > optimal, do NOT add more (exit logic handles reduction)
    return f_incremental
```

> **Why incremental, not block-if-exists:**
> Edge changes over time. If at tick 1 the edge justified 2% allocation and by tick 100
> the edge has grown (e.g., time decay favoring our position), optimal allocation may be 3%.
> Incremental Kelly adds the 1% difference. Conversely, if edge has shrunk to 1.5% optimal
> but we already have 2%, `f_incremental = 0` and no new order is placed.
> Exit logic (Step 4.4) handles the case where we should reduce.

### Fractional Kelly Policy

| K_frac | Growth (vs Full) | Volatility (vs Full) | Recommended Situation |
|--------|-------------------|-------------------|----------|
| 0.50 | 75% | 50% lower | Strong Brier score, 100+ trades accumulated |
| 0.25 | 44% | 75% lower | **Initial live stage (recommended starting point)** |

Never use Full Kelly (`K_frac = 1.0`).

### Correlated Position Cap Within Same Match

$$\sum_{\text{markets in match}} |f_{invest,i}| \leq f_{match\_cap}$$

If exceeded, scale proportionally:

$$f_{invest,i}^{scaled} = f_{invest,i} \times \frac{f_{match\_cap}}{\sum_i |f_{invest,i}|}$$

### 3-Layer Risk Limits

```python
def apply_risk_limits(f_invest: float, match_id: str,
                      bankroll: float) -> float:
    amount = f_invest * bankroll

    # Layer 1: single order ≤ 3%
    amount = min(amount, bankroll * F_ORDER_CAP)

    # Layer 2: per match ≤ 5%
    current_match_exposure = get_match_exposure(match_id)
    remaining_match = bankroll * F_MATCH_CAP - current_match_exposure
    amount = min(amount, max(0, remaining_match))

    # Layer 3: total portfolio ≤ 20%
    total_exposure = get_total_exposure()
    remaining_total = bankroll * F_TOTAL_CAP - total_exposure
    amount = min(amount, max(0, remaining_total))

    return amount
```

| Layer | Parameter | Default | Meaning |
|-------|---------|--------|------|
| 1 | f_order_cap | 0.03 (3%) | Single order cannot exceed 3% of capital |
| 2 | f_match_cap | 0.05 (5%) | Match exposure cannot exceed 5% |
| 3 | f_total_cap | 0.20 (20%) | Portfolio exposure cannot exceed 20% |

### Liquidity Gate (Extension — Implemented)

After Kelly sizing and risk limits, gate the position against available
order book depth to prevent VWAP slippage on thin markets:

```python
def apply_liquidity_gate(
    target_qty: int,
    ob_sync: OrderBookSync,
    direction: str,
    depth_fraction: float = 0.30,    # consume at most 30% of visible depth
    min_fill_ratio: float = 0.50,    # skip if gated < 50% of Kelly optimal
) -> tuple[int, bool]:
    """Gate position size against available order book depth.

    Prevents slippage traps by capping the order at a fraction of
    visible depth. If the capped size is too small relative to the
    Kelly-optimal size, the trade is skipped entirely.
    """
    if target_qty <= 0:
        return 0, False

    if direction == "BUY_YES":
        available = ob_sync.total_ask_depth()
    elif direction == "BUY_NO":
        available = ob_sync.total_bid_depth()
    else:
        return 0, False

    if available <= 0:
        return 0, False

    max_qty = int(depth_fraction * available)
    gated_qty = min(target_qty, max_qty)

    # Skip if we can't fill enough of the Kelly-optimal size
    if gated_qty < target_qty * min_fill_ratio:
        return 0, False

    return max(gated_qty, 1), True
```

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `depth_fraction` | 0.30 | Max fraction of visible depth to consume |
| `min_fill_ratio` | 0.50 | Skip if gated qty / target qty < this |

**Logic:** if the order book has 100 contracts on the ask side and Kelly wants
50 contracts, the gate caps at 30 (30% of 100). Since 30/50 = 60% > 50%,
the trade proceeds with 30 contracts. If Kelly wanted 80, the gate caps at 30
but 30/80 = 37.5% < 50%, so the trade is skipped — preserving edge for
deeper markets rather than accepting poor fills.

### Cross-Market Covariance Optimization (Extension)

The MC simulator in `mc_core.py` already generates joint (home_goals, away_goals) draws.
The covariance between HW, O2.5, and BTTS is deterministic given $(μ_H, μ_A)$
and can be computed analytically from the bivariate Poisson.

Instead of the blunt per-market Kelly sizing with a per-match cap,
replace with a portfolio optimization that accounts for cross-market correlations:

> **Execution frequency:** this optimization runs **once per new entry signal**
> (not every tick). When a second market in the same match generates a signal,
> re-optimize the joint allocation considering the existing position.

```python
def optimize_portfolio(
    signals: List[Signal],
    cov_matrix: np.ndarray,
    bankroll: float,
    f_match_cap: float = 0.05,
) -> Dict[str, float]:
    """Quadratic program for optimal multi-market allocation.

    Maximizes expected log-growth subject to:
    1. Total match exposure ≤ f_match_cap
    2. Each position non-negative
    3. Correlation-adjusted risk within bounds

    Uses scipy.optimize.minimize with SLSQP.
    """
    from scipy.optimize import minimize

    n = len(signals)
    EVs = np.array([s.EV for s in signals])
    W_L = np.array([s.W * s.L for s in signals])

    # Objective: maximize EV - 0.5 * f^T Σ f (mean-variance)
    def neg_utility(f):
        return -(EVs @ f - 0.5 * f @ cov_matrix @ f)

    constraints = [
        {"type": "ineq", "fun": lambda f: f_match_cap * bankroll - np.sum(np.abs(f))},
    ]
    bounds = [(0, f_match_cap * bankroll / n) for _ in range(n)]
    x0 = np.zeros(n)

    result = minimize(neg_utility, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    return {signals[i].market_ticker: result.x[i] for i in range(n)}
```

**Covariance matrix construction:**

$$\Sigma_{ij} = \text{Cov}(\text{outcome}_i, \text{outcome}_j)$$

For markets derived from the same $(μ_H, μ_A)$:
- HW vs O2.5: positively correlated (high-scoring games favor home wins)
- HW vs BTTS: weakly correlated
- O2.5 vs BTTS: strongly positively correlated

These correlations are computed analytically from the Poisson joint distribution
or empirically from MC simulation output.

**Benefit:** allows larger total allocation per match while controlling
correlation-adjusted risk. The blunt $f_{match\_cap} = 5\%$ is replaced with
a smarter constraint that accounts for natural hedging between positions.

### Final Allocation Amount

$$\text{Amount}_i = \text{apply\_risk\_limits}(f_{invest,i}^{scaled}, \text{match\_id}, \text{Bankroll})$$

$$\text{Contracts}_i = \left\lfloor \frac{\text{Amount}_i}{P_{kalshi,i}} \right\rfloor$$

After risk limits, the liquidity gate further caps contracts against visible depth.

---

## Step 4.4: Position Exit Logic (Exit Signal)

### Goal

Close positions when edge decays or reverses due to changing in-match conditions.

### Six Exit Triggers

> **Trigger interaction:** Triggers are evaluated in order (1→2→3→4).
> Trigger 1 (edge decay) catches gradual EV erosion, while Trigger 2 (edge reversal)
> catches rapid directional flips where EV may still be marginally positive but
> the model has already crossed to the opposite side of the market.
> Both can fire independently — reversal is a safety net for cases where
> decay threshold hasn't been breached yet but the position is directionally wrong.

#### Trigger 1: Edge Decay

```python
def check_edge_decay(position, P_true, sigma_MC, P_kalshi_bid, c, z):
    if position.direction == "BUY_YES":
        P_cons = P_true - z * sigma_MC
    else:
        P_cons = P_true + z * sigma_MC

    current_EV = compute_position_EV(P_cons, P_kalshi_bid, position, c)
    if current_EV < THETA_EXIT:  # 0.005 = 0.5¢
        return ExitSignal(reason="EDGE_DECAY", EV=current_EV)
    return None
```

#### Trigger 2: Edge Reversal

> **[v2 fix #1] Buy No threshold: `(1 - P_kalshi_bid)` → `P_kalshi_bid`**

```python
def check_edge_reversal(position, P_true, sigma_MC, P_kalshi_bid, z):
    """
    Immediate exit if model now evaluates opposite to market.

    All comparisons are in Yes probability space.
    No (1 - P) conversion even for Buy No.
    """
    if position.direction == "BUY_YES":
        P_cons = P_true - z * sigma_MC
        # Reversal if model P(Yes) is θ below market P(Yes)
        if P_cons < P_kalshi_bid - THETA_ENTRY:
            return ExitSignal(reason="EDGE_REVERSAL")

    elif position.direction == "BUY_NO":
        P_cons = P_true + z * sigma_MC
        # [v2 fix] If model P(Yes) is θ above market P(Yes)
        # → model P(No) is lower than market → No position reversed
        if P_cons > P_kalshi_bid + THETA_ENTRY:
            return ExitSignal(reason="EDGE_REVERSAL")
        # ❌ previous: if P_cons > (1 - P_kalshi_bid) + THETA_ENTRY
        # with bid=0.40, required 0.62 → ~20pp too strict

    return None
```

> **Validation:**
> Buy No, `P_kalshi_bid = 0.40`, `θ = 0.02`
> - ❌ v1: `P_cons > (1 - 0.40) + 0.02 = 0.62` → reversal only at 62%
> - ✅ v2: `P_cons > 0.40 + 0.02 = 0.42` → reversal detected at 42%
>
> Buy No bets on "P(Yes) is low."
> So if `P_cons(Yes)` exceeds market + θ, reversal is correct.

#### Trigger 3: Time-Based Expiry Evaluation (Last 3 Minutes)

> **[v2 fix #2] Added direction-specific `E_hold` for Buy No.**

```python
def check_expiry_eval(position, P_true, sigma_MC, P_kalshi_bid, c, z, t, T):
    """
    Near expiry: compare hold-to-settlement vs exit-now.
    E_hold differs by direction.
    """
    if T - t >= 3:
        return None

    if position.direction == "BUY_YES":
        P_cons = P_true - z * sigma_MC
    else:
        P_cons = P_true + z * sigma_MC

    # ─── E_hold: expected value if held to settlement ───
    if position.direction == "BUY_YES":
        # Yes win (prob P_cons): profit = (1 - entry) × (1-c)
        # Yes lose (prob 1-P_cons): loss = entry
        E_hold = (
            P_cons * (1 - c) * (1 - position.entry_price)
            - (1 - P_cons) * position.entry_price
        )

    elif position.direction == "BUY_NO":
        # [v2 fix] No win (prob 1-P_cons): profit = entry × (1-c)
        # No lose (prob P_cons): loss = (1 - entry)
        E_hold = (
            (1 - P_cons) * (1 - c) * position.entry_price
            - P_cons * (1 - position.entry_price)
        )
        # ❌ v1 reused Buy Yes formula → flips No expected value

    # ─── E_exit: expected value if exited now ───
    if position.direction == "BUY_YES":
        # sell Yes at bid
        profit_if_exit = P_kalshi_bid - position.entry_price
    elif position.direction == "BUY_NO":
        # close No = buy Yes at bid to offset
        # No entry sells Yes at entry_price → close buys Yes at P_kalshi_bid
        profit_if_exit = position.entry_price - P_kalshi_bid

    fee_if_exit = c * max(0, profit_if_exit)
    E_exit = profit_if_exit - fee_if_exit

    if E_exit > E_hold:
        return ExitSignal(reason="EXPIRY_EVAL", E_hold=E_hold, E_exit=E_exit)
    return None
```

> **Validation (Buy No):**
> `entry=0.40`, `P_cons=0.35`, `c=0.07`
>
> `E_hold = (1-0.35) × (1-0.07) × 0.40 - 0.35 × (1-0.40)`
> `       = 0.65 × 0.93 × 0.40 - 0.35 × 0.60`
> `       = 0.2418 - 0.21 = +0.0318` (holding is better)
>
> ❌ v1 (Buy Yes formula reused):
> `E_hold = 0.35 × 0.93 × 0.60 - 0.65 × 0.40`
> `       = 0.1953 - 0.26 = -0.0647` (wrongly favors exit)

#### Trigger 4: bet365 Divergence Warning

> **[v2 fix #3] Buy No threshold: `(1 - entry_price)` → `entry_price`**

```python
def check_bet365_divergence(position, P_bet365: float) -> Optional[DivergenceAlert]:
    """
    Warning when bet365 moves against held position direction.
    All comparisons are in Yes probability space.
    """
    if P_bet365 is None:
        return None

    DIVERGENCE_THRESHOLD = 0.05  # 5pp

    if position.direction == "BUY_YES":
        # Yes held: warning if bet365 P(Yes) drops by 5pp below entry
        if P_bet365 < position.entry_price - DIVERGENCE_THRESHOLD:
            return DivergenceAlert(
                severity="WARNING",
                P_bet365=P_bet365,
                P_entry=position.entry_price,
                suggested_action="REDUCE_OR_EXIT"
            )

    elif position.direction == "BUY_NO":
        # [v2 fix] No held (= sold Yes):
        # warning if bet365 P(Yes) rises by 5pp above entry
        # (Yes up is adverse for No position)
        if P_bet365 > position.entry_price + DIVERGENCE_THRESHOLD:
            return DivergenceAlert(
                severity="WARNING",
                P_bet365=P_bet365,
                P_entry=position.entry_price,
                suggested_action="REDUCE_OR_EXIT"
            )
        # ❌ v1: if P_bet365 > (1 - position.entry_price) + 0.05
        # entry=0.40 => v1 needs 0.65 (25pp), v2 needs 0.45 (5pp)

    return None
```

> **Validation (Buy No):**
> `entry=0.40` (sold Yes at 0.40)
> - ❌ v1: `P_bet365 > (1-0.40)+0.05 = 0.65` → needs 25pp move
> - ✅ v2: `P_bet365 > 0.40+0.05 = 0.45` → warns at 5pp (symmetric with Buy Yes)

**Trigger 4 is logging-only initially.**
Enable auto-exit after enough data in Step 4.6.

#### Trigger 5: Position Trimming (Partial Exit)

> When edge weakens but remains above θ_exit, the position stays at its original
> oversized allocation. This dead zone can persist for minutes.
> Trigger 5 checks if the current optimal allocation is materially below the
> existing position, and trims to the new optimal.

```python
def check_position_trim(position, P_true, sigma_MC, P_kalshi_bid,
                        c, z, K_frac, bankroll) -> Optional[ExitSignal]:
    """
    Partial exit: if optimal allocation has shrunk to less than half
    of existing position, trim down to optimal.

    Without this, positions remain oversized when edge weakens
    but stays above θ_exit (0.5¢).
    """
    if position.direction == "BUY_YES":
        P_cons = P_true - z * sigma_MC
    else:
        P_cons = P_true + z * sigma_MC

    # Compute current optimal f
    P_effective = P_kalshi_bid  # use current market for trim evaluation
    if position.direction == "BUY_YES":
        W = (1 - c) * (1 - P_effective)
        L = P_effective
        EV = P_cons * W - (1 - P_cons) * L
    else:
        W = (1 - c) * P_effective
        L = (1 - P_effective)
        EV = (1 - P_cons) * W - P_cons * L

    if EV <= 0 or W * L <= 0:
        return None  # edge_decay (Trigger 1) will handle this

    f_optimal = K_frac * (EV / (W * L)) * position.kelly_multiplier
    existing_fraction = (position.entry_price * position.quantity) / bankroll

    # Trim if optimal is less than half of existing
    if f_optimal < existing_fraction * 0.5:
        trim_qty = position.quantity - int(f_optimal * bankroll / P_effective)
        return ExitSignal(
            reason="POSITION_TRIM",
            trim_quantity=trim_qty,  # partial, not full exit
            f_optimal=f_optimal,
            f_existing=existing_fraction,
        )

    return None
```

#### Trigger 6: Opportunity Cost Exit (Direction Flip)

> If the model now strongly favors the opposite direction but the current
> position's EV is marginally positive, the system is deadlocked:
> can't enter new direction (no_opposite_position filter), won't exit old one.
> This trigger resolves the deadlock by exiting when the opportunity cost is high.

```python
def check_opportunity_cost_exit(position, P_true, sigma_MC,
                                 P_kalshi_ask, P_kalshi_bid,
                                 c, z) -> Optional[ExitSignal]:
    """
    Exit if opposite direction has strong edge AND current position's
    edge has weakened below 2x θ_exit.

    Scenario: was Buy Yes, goal changes dynamic, model now favors Buy No.
    Current Buy Yes EV = +0.8¢ (above θ_exit 0.5¢, so no decay trigger).
    Opposite Buy No EV = +3.5¢ (strong signal being missed).
    """
    # Compute current position's EV
    if position.direction == "BUY_YES":
        P_cons_current = P_true - z * sigma_MC
        current_EV = (P_cons_current * (1-c) * (1-P_kalshi_bid)
                      - (1-P_cons_current) * P_kalshi_bid)
    else:
        P_cons_current = P_true + z * sigma_MC
        current_EV = ((1-P_cons_current) * (1-c) * P_kalshi_bid
                      - P_cons_current * (1-P_kalshi_bid))

    # Compute opposite direction's EV
    if position.direction == "BUY_YES":
        # Opposite = BUY_NO
        P_cons_opp = P_true + z * sigma_MC
        opp_EV = ((1-P_cons_opp) * (1-c) * P_kalshi_bid
                  - P_cons_opp * (1-P_kalshi_bid))
    else:
        # Opposite = BUY_YES
        P_cons_opp = P_true - z * sigma_MC
        opp_EV = (P_cons_opp * (1-c) * (1-P_kalshi_ask)
                  - (1-P_cons_opp) * P_kalshi_ask)

    # Exit if: opposite has strong edge AND current is weak
    if opp_EV > THETA_ENTRY and current_EV < 2 * THETA_EXIT:
        return ExitSignal(
            reason="OPPORTUNITY_COST",
            current_EV=current_EV,
            opposite_EV=opp_EV,
        )

    return None
```

### Full Exit Evaluation Loop

```python
async def evaluate_exit(position, P_true, sigma_MC, P_kalshi_bid,
                        P_kalshi_ask, P_bet365, c, z, t, T,
                        K_frac, bankroll) -> Optional[ExitSignal]:
    """Call this each tick for all open positions.
    6 triggers evaluated in order — first match wins."""

    # Trigger 1: edge decay (EV below 0.5¢)
    exit = check_edge_decay(position, P_true, sigma_MC, P_kalshi_bid, c, z)
    if exit: return exit

    # Trigger 2: edge reversal (model flipped sides)
    exit = check_edge_reversal(position, P_true, sigma_MC, P_kalshi_bid, z)
    if exit: return exit

    # Trigger 3: expiry eval (last 3 minutes)
    exit = check_expiry_eval(position, P_true, sigma_MC, P_kalshi_bid, c, z, t, T)
    if exit: return exit

    # Trigger 4: bet365 divergence warning
    divergence = check_bet365_divergence(position, P_bet365)
    if divergence:
        log.warning(f"bet365 divergence: {divergence}")
        position.had_bet365_divergence = True
        position.divergence_snapshot = {
            "P_bet365": P_bet365,
            "P_kalshi_bid": P_kalshi_bid,
            "P_true": P_true,
            "t": t,
        }
        if BET365_DIVERGENCE_AUTO_EXIT:
            return ExitSignal(reason="BET365_DIVERGENCE")

    # Trigger 5: position trimming (edge weakened but above θ_exit)
    trim = check_position_trim(position, P_true, sigma_MC, P_kalshi_bid,
                               c, z, K_frac, bankroll)
    if trim: return trim

    # Trigger 6: opportunity cost (opposite direction has strong edge)
    opp = check_opportunity_cost_exit(position, P_true, sigma_MC,
                                       P_kalshi_ask, P_kalshi_bid, c, z)
    if opp: return opp

    return None
```

---

## Step 4.5: Order Execution & Risk Management

### Order Types

| Situation | Order Type | Reason |
|------|----------|------|
| Normal entry | Limit Order (Ask + 0~1¢) | Balance fill probability and slippage |
| Urgent exit | Limit Order (Bid - 1¢) | Prioritize quick fill |
| **Rapid Entry** | **Limit Order (Ask + 1¢)** | **Post-event informational edge (conditional)** |
| Low liquidity | Hold order | If slippage > edge, no entry |

> **Kalshi API note:** the order submission code below uses simplified
> field names for clarity. In production, adapt to the Kalshi REST API which
> uses `ticker` to identify markets, `side` ("yes"/"no") for direction,
> and `yes_price` in cents (1-99). The `100 - price_cents` conversion for
> No orders is handled at the API adapter layer, not in the core trading logic.

### Order Submission

```python
async def execute_order(signal: Signal, amount: float,
                        ob_sync: OrderBookSync,
                        urgent: bool = False) -> Optional[FillResult]:
    """Submit order to Kalshi with error handling for rejections."""

    # --- Staleness gate: skip if order book is stale ---
    if ob_sync.kalshi_is_stale:
        log.warning("Kalshi order book stale, skipping order")
        return None

    P_kalshi = signal.P_kalshi  # VWAP effective price
    contracts = int(amount / P_kalshi)

    if contracts < 1:
        return None

    if urgent:
        price_cents = int(ob_sync.kalshi_best_ask * 100) + 1
    else:
        price_cents = int(ob_sync.kalshi_best_ask * 100)

    order = {
        "ticker": signal.market_ticker,
        "action": "buy",
        "side": "yes" if signal.direction == "BUY_YES" else "no",
        "type": "limit",
        "count": contracts,
        "yes_price": price_cents if signal.direction == "BUY_YES"
                     else (100 - price_cents),
    }

    # --- Submit with error handling ---
    try:
        response = await kalshi_api.submit_order(order)
    except KalshiApiError as e:
        if e.code == "market_closed":
            # Market closed (e.g., halftime) — suppress until reopened.
            # Without this, every tick retries for ~15 minutes of halftime.
            log.warning(f"Market closed: {signal.market_ticker}, "
                        f"suppressing orders for this ticker")
            ob_sync.market_closed_tickers.add(signal.market_ticker)
            return None
        elif e.code == "insufficient_balance":
            log.error(f"Insufficient Kalshi balance, halting new orders")
            return None
        elif e.code == "price_out_of_range":
            log.warning(f"Price {price_cents}¢ out of range for "
                        f"{signal.market_ticker}, skipping")
            return None
        else:
            # Transient error — log and skip this tick
            log.error(f"Kalshi order error: {e.code} — {e.message}")
            return None

    order_id = response["order"]["id"]

    filled = await wait_for_fill(order_id, timeout=5)

    if filled.status == "full":
        record_position(signal, filled)
        return filled
    elif filled.status == "partial":
        await kalshi_api.cancel_order(order_id)
        record_position(signal, filled, partial=True)
        return filled
    else:
        await kalshi_api.cancel_order(order_id)
        return None
```

### Execution Router — Paper/Live Mode Switch

```python
class ExecutionRouter:
    """
    Unified execution interface. Phase 4 logic calls this router;
    the router delegates to PaperExecutionLayer or live execute_order
    based on TRADING_MODE config. This ensures all upstream logic
    (signal generation, Kelly sizing, risk limits) is identical
    in both modes — only the fill mechanism differs.
    """

    def __init__(self, trading_mode: str, model):
        self.mode = trading_mode  # "paper" or "live"
        if self.mode == "paper":
            self.paper = PaperExecutionLayer()
        self.model = model

    async def submit_order(self, signal: Signal, amount: float,
                           ob_sync: OrderBookSync,
                           urgent: bool = False) -> Optional[FillResult]:
        if self.mode == "paper":
            return await self.paper.execute_order(
                signal, amount, ob_sync, self.model, urgent
            )
        else:
            return await execute_order(signal, amount, ob_sync, urgent)
```

> **All Phase 4 code calls `model.execution.submit_order(...)` — never the
> live or paper functions directly.** The router is injected at container startup
> based on the `TRADING_MODE` environment variable (see Orchestration doc).

### Paper Fill Simulation

> **[v2 fix #6] VWAP + slippage + partial-fill simulation.**
> **[v3 fix] Directional slippage + fill delay + ob_freeze check during wait.**

```python
class PaperExecutionLayer:
    """
    Simulates realistic order fills for paper trading.
    Shares the same interface as live execution so Phase 4 logic
    is identical in both modes — only the fill mechanism differs.
    """

    def __init__(self, slippage_ticks: int = 1,
                 fill_delay_range: Tuple[float, float] = (1.0, 3.0)):
        self.slippage_ticks = slippage_ticks
        self.fill_delay_range = fill_delay_range  # seconds

    async def execute_order(self, signal: Signal, amount: float,
                            ob_sync: OrderBookSync, model,
                            urgent: bool = False) -> Optional[PaperFill]:
        """
        Paper fill simulation:
        1. VWAP-based fill price (includes book depth)
        2. Directional slippage (adverse direction per side)
        3. Fill delay simulation (1-3s, aborted if ob_freeze)
        4. Re-snapshot order book after delay (price may have moved)
        5. Partial fill based on available depth
        """
        target_qty = int(amount / signal.P_kalshi)
        if target_qty < 1:
            return None

        # ─── Fill delay simulation ───
        # Real orders sit in the book for 1-5 seconds before fill.
        # During this time, events can change the market.
        delay = random.uniform(*self.fill_delay_range)
        await asyncio.sleep(delay)

        # Check if state changed during wait (ob_freeze, new event)
        if model.ob_freeze or model.event_state != "IDLE":
            log.info(f"Paper order cancelled: state changed during {delay:.1f}s wait")
            return None

        # ─── Re-snapshot VWAP after delay ───
        if signal.direction == "BUY_YES":
            P_effective = ob_sync.compute_vwap_buy(target_qty)
        else:
            P_effective = ob_sync.compute_vwap_sell(target_qty)

        if P_effective is None:
            return None  # depth dried up during wait

        # ─── Directional slippage ───
        # BUY_YES: slippage makes price higher (worse for buyer)
        # BUY_NO (= sell Yes): slippage makes price lower (worse for seller)
        slip = self.slippage_ticks * 0.01
        if signal.direction == "BUY_YES":
            fill_price = P_effective + slip
        else:  # BUY_NO
            fill_price = P_effective - slip

        # ─── Partial fill based on post-delay depth ───
        if signal.direction == "BUY_YES":
            available_depth = sum(qty for price, qty in ob_sync.kalshi_depth_ask
                                 if price <= fill_price * 100)
        else:
            available_depth = sum(qty for price, qty in ob_sync.kalshi_depth_bid
                                 if price >= fill_price * 100)

        filled_qty = min(target_qty, available_depth)
        if filled_qty < 1:
            return None

        return PaperFill(
            price=fill_price,
            quantity=filled_qty,
            timestamp=time.time(),
            is_paper=True,
            slippage=abs(fill_price - P_effective),
            partial=(filled_qty < target_qty),
            fill_delay=delay,
        )
```

### Rapid Entry

> **[v2 fix #7] VAR safety wait + conservative P_cons + stricter activation conditions.**

```python
async def post_event_rapid_entry(model, confirmed_event):
    """
    Evaluate immediate post-confirmation entry before cooldown.
    """
    if not RAPID_ENTRY_ENABLED:
        return

    # [v2] VAR safety wait: extra N seconds after CONFIRMED
    # If no score rollback occurs during this period, treat as safe
    await asyncio.sleep(VAR_SAFETY_WAIT)  # default 5s

    # Recheck state after waiting
    if model.event_state != "IDLE":
        return  # new event occurred — abort
    if model.S != confirmed_event.score:
        return  # score changed — possible VAR cancellation

    # Use precomputed P_true
    if not model.preliminary_cache.get("μ_H"):
        return

    P_true = compute_P_from_preliminary(model)
    sigma_MC = model.preliminary_cache.get("sigma_MC", 0.01)

    # [v2] conservative P_cons adjustment (v1 used P_cons=P_true)
    direction = infer_direction(P_true, model.ob_sync.kalshi_best_ask)
    P_cons = compute_conservative_P(P_true, sigma_MC, direction, model.config.z)

    P_bet365 = model.ob_sync.bet365_implied.get(market_key)
    P_kalshi = model.ob_sync.kalshi_best_ask

    if P_bet365 is None or P_kalshi is None:
        return

    # Market alignment check
    alignment = check_market_alignment(P_cons, P_kalshi, P_bet365, direction)

    if alignment.status == "ALIGNED":
        # VWAP-based EV
        rough_qty = estimate_rapid_qty(P_cons, P_kalshi, model)
        P_effective = model.ob_sync.compute_vwap_buy(rough_qty) if direction == "BUY_YES" \
                      else model.ob_sync.compute_vwap_sell(rough_qty)

        if P_effective is None:
            return

        if direction == "BUY_YES":
            EV = P_cons * (1-c) * (1-P_effective) - (1-P_cons) * P_effective
        else:
            EV = (1-P_cons) * (1-c) * P_effective - P_cons * (1-P_effective)

        if EV <= THETA_ENTRY:
            return

        signal = Signal(
            direction=direction, EV=EV, P_cons=P_cons,
            P_kalshi=P_effective, rough_qty=rough_qty,
            alignment_status="ALIGNED", kelly_multiplier=0.8,
            market_ticker=model.active_market
        )
        amount = compute_kelly(signal, c, K_frac)
        amount = apply_risk_limits(amount, model.match_id, model.bankroll)
        await model.execution.execute_order(signal, amount, model.ob_sync, urgent=True)
        log.info(f"RAPID ENTRY: {signal.direction}, EV={signal.EV:.4f}")
```

**Rapid Entry activation conditions (strengthened):**

```python
RAPID_ENTRY_ENABLED = (
    cumulative_trades >= 200
    and edge_realization >= 0.8
    and preliminary_accuracy >= 0.95
    and var_cancellation_rate < 0.03
    and VAR_SAFETY_WAIT >= 5              # [v2] safety wait is configured
    and rapid_entry_hypo_pnl_after_slip > 0  # [v2] remains positive after slippage
)
```

### Trade Log

```python
@dataclass
class TradeLog:
    timestamp: float
    match_id: str
    market_ticker: str
    direction: str              # BUY_YES | BUY_NO | SELL_YES | SELL_NO
    order_type: str             # ENTRY | EXIT_EDGE_DECAY | EXIT_EDGE_REVERSAL
                                # | EXIT_EXPIRY_EVAL | EXIT_BET365_DIVERGENCE
                                # | RAPID_ENTRY
    quantity_ordered: int
    quantity_filled: int
    limit_price: float
    fill_price: float
    P_true_at_order: float
    P_true_cons_at_order: float     # Directional conservative P
    P_kalshi_at_order: float        # VWAP effective price
    P_kalshi_best_at_order: float   # Best ask/bid (for VWAP comparison)
    P_bet365_at_order: float
    EV_adj: float                   # Final EV after VWAP
    sigma_MC: float
    pricing_mode: str
    f_kelly: float
    K_frac: float
    alignment_status: str           # ALIGNED | DIVERGENT | UNAVAILABLE
    kelly_multiplier: float
    cooldown_active: bool
    ob_freeze_active: bool
    event_state: str
    engine_phase: str
    bankroll_before: float
    bankroll_after: float
    is_paper: bool
    paper_slippage: float           # In paper mode: simulated slippage
```

---

## Step 4.6: Post-Match Settlement and Analysis

### Auto-Settlement

> **[v2 fix #8] Added directional settlement branch for Buy No.**

> **Settlement timing:** Kalshi markets may not resolve immediately after match end.
> Resolution can take minutes to hours depending on the market. The system must
> poll for resolution rather than assuming instant settlement.

```python
async def await_settlement(match_id: str, market_tickers: List[str],
                           timeout_hours: float = 6.0) -> Dict[str, float]:
    """
    Poll Kalshi for market resolution after match ends.
    Returns {ticker: settlement_price} once all markets are resolved.

    Positions remain in AWAITING_SETTLEMENT status until resolved.
    If timeout is reached, alert for manual intervention.
    """
    deadline = time.time() + timeout_hours * 3600
    resolved = {}

    while time.time() < deadline:
        for ticker in market_tickers:
            if ticker in resolved:
                continue
            try:
                market = await kalshi_api.get_market(ticker)
                if market["status"] == "resolved":
                    # settlement_price: 1.00 if Yes won, 0.00 if No won
                    resolved[ticker] = float(market["settlement_price"])
                    log.info(f"Market resolved: {ticker} = {resolved[ticker]}")
            except Exception as e:
                log.warning(f"Settlement poll failed for {ticker}: {e}")

        if len(resolved) == len(market_tickers):
            return resolved  # all resolved

        await asyncio.sleep(60)  # poll every 60 seconds

    # Timeout — some markets unresolved
    unresolved = [t for t in market_tickers if t not in resolved]
    log.error(f"Settlement timeout: {unresolved} unresolved after {timeout_hours}h")
    await alert(f"Manual settlement needed: {match_id}, unresolved: {unresolved}")
    return resolved  # return what we have; unresolved positions stay open

async def settle_all_positions(model):
    """
    Called after match finishes. Waits for Kalshi resolution,
    then computes P&L for all positions.
    """
    open_positions = await db.get_open_positions(model.match_id)
    if not open_positions:
        return

    # Update positions to AWAITING_SETTLEMENT
    for pos in open_positions:
        await db.update_position_status(pos.id, "AWAITING_SETTLEMENT")

    # Wait for Kalshi to resolve
    tickers = list(set(pos.market_ticker for pos in open_positions))
    settlements = await await_settlement(model.match_id, tickers)

    # Compute P&L for resolved positions
    for pos in open_positions:
        if pos.market_ticker in settlements:
            pnl = compute_realized_pnl(pos, settlements[pos.market_ticker], FEE_RATE)
            await db.settle_position(pos.id, settlements[pos.market_ticker], pnl)

            # Update bankroll
            mode = "paper" if model.is_paper else "live"
            await db.update_bankroll(mode, pnl)
```

```python
def compute_realized_pnl(position, settlement_price: float,
                          fee_rate: float) -> float:
    """
    Direction-specific realized P&L.

    settlement_price: settlement from Yes perspective (Yes win=1.00, Yes lose=0.00)

    Buy Yes: profit = Settlement - Entry (Yes at 100¢ is profit)
    Buy No:  profit = Entry - Settlement (Yes at 0¢ is profit)

    ❌ v1: Qty × (Settlement - Entry) - Fee → sign flips for Buy No
    ✅ v2: directional branch
    """
    if position.direction == "BUY_YES":
        gross_pnl = (settlement_price - position.entry_price) * position.quantity
    elif position.direction == "BUY_NO":
        gross_pnl = (position.entry_price - settlement_price) * position.quantity
    else:
        gross_pnl = 0

    # Fee applies only to profits
    fee = fee_rate * max(0, gross_pnl)
    return gross_pnl - fee
```

> **Validation:**
>
> | Direction | Entry | Settlement | v1 Result | v2 Result | Actual |
> |------|-------|------------|---------|---------|------|
> | Buy Yes | 0.45 | 1.00 | +0.55 ✅ | +0.55 ✅ | Profit |
> | Buy Yes | 0.45 | 0.00 | -0.45 ✅ | -0.45 ✅ | Loss |
> | Buy No | 0.40 | 0.00 | -0.40 ❌ | +0.40 ✅ | Profit (No wins) |
> | Buy No | 0.40 | 1.00 | +0.60 ❌ | -0.60 ✅ | Loss (No loses) |
>
> In v1, Buy No profit/loss is completely inverted.
> That contaminates all Step 4.6 post-analysis metrics (Brier, edge realization, drawdown, etc.).

### Post-Analysis Metrics — 12 Total

#### Original metrics (1~6)

**1. Match-level P&L:**

$$\text{Match P\&L} = \sum_{i \in \text{positions}} \text{compute\_realized\_pnl}(i)$$

**2. Cumulative Brier Score** (vs Betfair Exchange baseline)

**3. Edge Realization:**

$$\text{Edge Realization} = \frac{\text{Actual average return}}{\text{Expected average } EV_{adj}}$$

**4. Slippage Performance:**

$$\text{Avg Slippage} = \frac{1}{N}\sum_{n} (\text{Fill Price}_n - P_{kalshi,best,n})$$

> Track difference between `P_kalshi_best_at_order` (best ask/bid) and actual fill.
> Since VWAP is in EV, also track slippage between VWAP and actual fill.

**5. Cooldown impact analysis**

**6. ob_freeze impact analysis**

#### New metrics (7~11)

**7. Market alignment value:**

```python
def analyze_alignment_effect(trades):
    aligned = [t for t in trades if t.alignment_status == "ALIGNED"]
    divergent = [t for t in trades if t.alignment_status == "DIVERGENT"]

    return {
        "aligned_avg_return": safe_mean([t.realized_pnl for t in aligned]),
        "divergent_avg_return": safe_mean([t.realized_pnl for t in divergent]),
        "aligned_win_rate": win_rate(aligned),
        "divergent_win_rate": win_rate(divergent),
        "alignment_value": (
            safe_mean([t.realized_pnl for t in aligned])
            - safe_mean([t.realized_pnl for t in divergent])
        ),
    }
```

**8. Directional P_true^cons analysis:**

```python
def analyze_directional_cons(trades):
    yes = [t for t in trades if t.direction == "BUY_YES"]
    no = [t for t in trades if t.direction == "BUY_NO"]

    return {
        "yes_edge_realization": safe_divide(actual_return(yes), expected_EV(yes)),
        "no_edge_realization": safe_divide(actual_return(no), expected_EV(no)),
    }
```

**9. Preliminary accuracy**

**10. Rapid Entry hypothetical P&L:**

> [v2] Include VWAP + slippage in hypothetical P&L for realism.

**11. bet365 divergence warning effectiveness**

**12. Paper Realism Score (Phase 0 only):**

> Measures how realistic paper fills were compared to actual market conditions.

```python
def compute_paper_realism_score(paper_trades: List[TradeLog],
                                 tick_snapshots: List) -> dict:
    """
    For each paper trade, compare:
    - Paper fill price vs actual market mid-price at fill timestamp
    - Paper fill delay vs typical real fill times (from Phase A data, if available)
    - Paper partial fill rate vs actual depth at trade time

    Realism score = 1.0 means paper perfectly matches reality.
    Score < 0.85 suggests paper is too optimistic.
    """
    price_diffs = []
    for trade in paper_trades:
        # Find actual market snapshot closest to paper fill time
        actual_snapshot = find_nearest_snapshot(tick_snapshots, trade.timestamp)
        if actual_snapshot is None:
            continue

        # Compare paper fill price vs actual mid-price
        if trade.direction == "BUY_YES":
            actual_price = actual_snapshot.P_kalshi_ask
        else:
            actual_price = actual_snapshot.P_kalshi_bid

        price_diff = abs(trade.fill_price - actual_price)
        price_diffs.append(price_diff)

    avg_price_diff = safe_mean(price_diffs)
    partial_fill_rate = sum(1 for t in paper_trades if t.partial) / len(paper_trades)

    # Score: penalize large price deviations and low partial fill rates
    price_score = max(0, 1 - avg_price_diff / 0.05)  # 5¢ deviation = 0
    fill_score = 1 - partial_fill_rate * 0.5          # partial fills reduce score

    realism_score = 0.7 * price_score + 0.3 * fill_score

    return {
        "realism_score": realism_score,
        "avg_paper_vs_actual_price_diff": avg_price_diff,
        "partial_fill_rate": partial_fill_rate,
    }
```

### Model Health Dashboard

| Metric | Healthy 🟢 | Warning 🟡 | Risk 🔴 |
|------|---------|---------|---------|
| Brier Score | Phase 1.5 ± 0.02 | ± 0.05 | outside band |
| Edge Realization | 0.7~1.3 | 0.5~0.7 | < 0.5 |
| Max Drawdown | < 10% | 10~20% | > 20% |
| Market alignment value | ALIGNED > DIVERGENT + 1¢ | gap ≈ 0 | ALIGNED < DIVERGENT |
| Preliminary accuracy | > 0.95 | 0.90~0.95 | < 0.90 |
| No-side realization | 0.7~1.3 | > 1.5 (too conservative) | < 0.5 |
| Paper realism score (Phase 0) | > 0.85 | 0.70~0.85 | < 0.70 |

### Feedback Loop — Adaptive Parameter Tuning

> **Change tracking:** all parameter changes are logged to `parameter_change_log` table
> with timestamp, old value, new value, and triggering metric. This enables post-hoc
> analysis of whether parameter adjustments improved or degraded performance.

```python
def adaptive_parameter_update(analytics: dict):
    """Data-driven auto-adjustment of 8 parameters."""

    # 1. K_frac adjustment
    er = analytics["edge_realization"]
    if er >= 0.8:
        K_frac = min(K_frac + 0.05, 0.50)
    elif er < 0.5:
        K_frac = max(K_frac - 0.10, 0.10)

    # 2. Market alignment multiplier adjustment
    av = analytics["alignment_value"]
    if av < 0.005:
        # low alignment value → raise DIVERGENT multiplier
        DIVERGENT_MULTIPLIER = 0.65
    elif av > 0.015:
        DIVERGENT_MULTIPLIER = 0.4

    # 3. Rapid entry activation decision
    if (analytics["preliminary_accuracy"] > 0.95
        and analytics["var_cancellation_rate"] < 0.03
        and analytics["rapid_entry_hypo_pnl_after_slip"] > 0
        and analytics["cumulative_trades"] >= 200):
        RAPID_ENTRY_ENABLED = True

    # 4. z (conservativeness) adjustment — directional
    no_er = analytics["no_edge_realization"]
    if no_er > 1.5:
        z = max(z - 0.2, 1.0)
    elif no_er < 0.5:
        z = min(z + 0.2, 2.0)

    # 5. Phase 1 retraining trigger
    if analytics["brier_score_trend"] == "worsening_3weeks":
        trigger_phase1_recalibration()

    # 6. Cooldown adjustment
    if analytics["cooldown_suppressed_profitable_rate"] > 0.6:
        COOLDOWN_SECONDS = max(COOLDOWN_SECONDS - 2, 8)

    # 7. bet365 divergence auto-exit decision
    if (analytics["bet365_divergence_should_auto_exit"]
        and analytics["bet365_divergence_sample_size"] >= 30):
        BET365_DIVERGENCE_AUTO_EXIT = True

    # 8. THETA_ENTRY adjustment
    avg_slippage = analytics["avg_slippage"]
    breakeven_edge = c + avg_slippage  # fee + observed slippage
    THETA_ENTRY = max(breakeven_edge + 0.005, 0.01)  # at least 0.5¢ above breakeven
```

---

## Phase 4 Pipeline Summary (v3)

```
[Phase 3: P_true(dict), σ_MC(dict), order_allowed, event_state, P_bet365(dict)]
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  signal_generator: Multi-Market Orchestration Loop    [v3]  │
│  • Receive P_true dict from Phase 3 via asyncio.Queue      │
│  • For each active Kalshi ticker:                      │
│    - Map ticker → model_key (e.g., "SOCCER-EPL-ARS-v-CHE-WINNER" → "home_win") │
│    - Extract P_true[key]: float, σ_MC[key]: float          │
│    - Run Steps 4.2 → 4.3 → 4.5 with per-market floats     │
└──────────────────┬──────────────────────────────────────────┘
                   │ (per market, per tick)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4.1: Order-Book Sync (per market ticker)              │
│  • Kalshi WS → Bid/Ask + VWAP buy/sell effective prices │
│  • Odds-API Live Odds WS → bet365 implied probabilities     │
│  • Liquidity filter (Q_min ≥ 20 contracts)                 │
│  Output: P_kalshi^buy, P_kalshi^sell,               │
│          P_effective^buy(Q), P_effective^sell(Q), [v2 VWAP] │
│          P_bet365, liquidity                               │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4.2: Edge Detection (float inputs per market)         │
│  • Directional P_cons: Yes→P-zσ, No→P+zσ                   │
│  • 2-pass VWAP EV calc:                             [v2]    │
│    Pass 1: best ask/bid → rough qty                        │
│    Pass 2: VWAP for rough qty → final EV                   │
│  • Market alignment check:                         [v2]     │
│    → ALIGNED (mult 0.8) / DIVERGENT (0.5)                  │
│       / UNAVAILABLE (0.6)                                  │
│  • Filter: EV>θ (after VWAP) AND order_allowed             │
│          AND event_state==IDLE AND liquidity_ok            │
│          AND alignment policy by phase                     │
│  Output: Signal(direction, EV, P_cons,                     │
│          P_kalshi=VWAP, alignment_status)              │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │ Entry Signal       │ Existing Position
         ▼                    ▼
┌──────────────────┐  ┌──────────────────────────────────────┐
│  Step 4.3:       │  │  Step 4.4: Exit (directional formulas)│
│  Sizing          │  │                                  [v2]│
│                  │  │  Trigger 1: Edge decay (EV < 0.5¢)    │
│  • Directional   │  │  Trigger 2: Edge reversal             │
│    Kelly (W/L)   │  │    Yes: P_cons < P_bid - θ            │
│  • K_frac        │  │    No:  P_cons > P_bid + θ     [v2]   │
│    (0.25~0.50)   │  │  Trigger 3: Expiry eval (last 3 min)  │
│  • Alignment     │  │    Directional E_hold branch   [v2]   │
│    multiplier    │  │  Trigger 4: bet365 divergence warning │
│    (0.8/0.5/0.6) │  │    Yes: P_bet365 < entry - 5pp        │
│      [v2]        │  │    No:  P_bet365 > entry + 5pp [v2]   │
│  • Match cap     │  │       → logging first, then optional   │
│    pro-rata      │  │         auto-exit after data           │
│  • 3-layer risk  │  │  Trigger 5: Position trim       [v3]  │
│    (3%/5%/20%)   │  │    f_optimal < existing * 0.5 → trim  │
│  • [EXT] Liq.    │  │  Trigger 6: Opportunity cost    [v3]  │
│    gate (30%     │  │    Opposite EV > θ_entry AND           │
│    depth cap)    │  │    current EV < 2× θ_exit → exit      │
│  • [EXT] Cross-  │  │                                        │
│    market cov    │  │                                        │
│    optimization  │  │                                        │
└────────┬─────────┘  └──────────┬─────────────────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4.5: Order Execution                                  │
│                                                             │
│  • Normal entry: Limit Order (Ask + 0~1¢)                  │
│  • Urgent exit: Limit Order (Bid - 1¢)                     │
│  • Rapid Entry: Ask + 1¢ (conditional)                     │
│    + 5s VAR safety wait                            [v2]     │
│    + P_cons z-adjustment                           [v2]     │
│  • Partial fill: 5s timeout → cancel unfilled remainder    │
│                                                             │
│  Paper mode (via ExecutionRouter):                   [v3]    │
│  • Directional slippage + 1-3s fill delay                  │
│  • ob_freeze check during wait + partial fills             │
│  • All upstream logic (P_true, signals, Kelly) identical   │
│                                                             │
│  Trade log: P_true, P_cons, P_kalshi(VWAP),            │
│    P_kalshi_best, P_bet365, alignment_status,      [v2] │
│    kelly_multiplier, event_state, paper_slippage           │
│                                                             │
│  Real-time feedback: position DB + bankroll + risk refresh │
└──────────────────┬──────────────────────────────────────────┘
                   ▼ (after match end)
┌─────────────────────────────────────────────────────────────┐
│  Step 4.6: Settlement & Post-Analysis                       │
│                                                             │
│  Settlement (directional):                           [v2]   │
│  • Buy Yes: (Settlement - Entry) × Qty - Fee              │
│  • Buy No:  (Entry - Settlement) × Qty - Fee              │
│                                                             │
│  Original metrics (1~6):                                    │
│  1. Match-level P&L (directional settlement)        [v2]    │
│  2. Brier Score (vs Betfair Exchange baseline)                     │
│  3. Edge realization                                        │
│  4. Slippage performance (adds VWAP-vs-fill view)   [v2]    │
│  5. Cooldown effect                                         │
│  6. ob_freeze effect                                        │
│                                                             │
│  New metrics (7~11):                                        │
│  7. Market alignment value                           [v2]    │
│     (ALIGNED vs DIVERGENT return gap)                      │
│  8. Directional P_cons analysis (Yes vs No realization)     │
│  9. Preliminary accuracy (for rapid entry decisions)        │
│ 10. Rapid Entry hypothetical P&L (slippage-adjusted) [v2]   │
│ 11. bet365 divergence warning value (auto-exit decision)    │
│ 12. Paper realism score (Phase 0 only)               [v3]   │
│                                                             │
│  Adaptive tuning (8 parameters):                            │
│  1. K_frac (0.25~0.50)                                      │
│  2. DIVERGENT multiplier                            [v2]    │
│  3. Rapid entry on/off                                      │
│  4. z (directional conservativeness)                        │
│  5. Phase 1 retraining trigger                              │
│  6. Cooldown length (15s~8s)                                │
│  7. bet365 divergence auto-exit on/off                      │
│  8. THETA_ENTRY (breakeven + margin)                 [v2]   │
│                                                             │
│  All changes logged to parameter_change_log table           │
│                                                             │
│  System evolution: Phase 0 → A → B → C roadmap             │
│                                                             │
│  Output: P&L report, health dashboard, parameter updates,   │
│          retraining decisions                                │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
   [Phase 1 retraining (when triggered)]
```

---

## System Evolution Roadmap

```
Phase 0 — Paper Trading:
│  • TRADING_MODE = "paper"
│  • K_frac = 0.25, z = 1.645
│  • Paper fills: directional slippage + 1-3s delay + ob_freeze check [v3]
│  • Block entry on DIVERGENT
│  • Rapid entry disabled
│  • Calibrate THETA_ENTRY from observed breakeven edge
│  • Paper bankroll: virtual $10,000 (isolated from real balance)
│  • All real-time data feeds are LIVE (Odds-API, Goalserve, Kalshi)
│  • Only order submission is simulated
│  Period: 2~4 weeks (minimum 50 paper trades)
│
│  ┌─ Phase 0 → A Graduation Criteria (ALL must pass): ─────────┐
│  │  1. Paper trades ≥ 50                                       │
│  │  2. Edge realization ∈ [0.6, 1.5]                           │
│  │  3. Paper Brier Score within Phase 1.5 ± 0.03               │
│  │  4. Paper max drawdown < 15%                                 │
│  │  5. Directional correctness = 100% (no delta/gamma bugs)    │
│  │  6. Paper realism score > 0.85 (see Step 4.6 metric 12)     │
│  │  7. No system crashes or unrecoverable failures              │
│  │  8. THETA_ENTRY calibrated from paper data                   │
│  └──────────────────────────────────────────────────────────────┘
│
▼
Phase A — Conservative Live:
│  • TRADING_MODE = "live"
│  • Keep blocking DIVERGENT entries
│  • Rapid entry disabled
│  • Start with real bankroll × 0.5 (half allocation)
│  Period: 1~2 months
│
▼
Phase B — Adaptive Live:
│  • K_frac → 0.25~0.50 (based on Step 4.6)
│  • DIVERGENT entries allowed with multiplier 0.5
│  • Directional optimization of z
│  • Full bankroll allocation
│  Period: 2~4 months
│
▼
Phase C — Mature Live:
│  • Conditional rapid entry enabled (with VAR safety wait) [v2]
│  • bet365 divergence → auto-exit (if supported by data)
│  • Auto-tuning loop enabled
│
▼
(Every season: mandatory Phase 1 retraining)
```

---

## Full System Feedback Loop

```
Phase 1 (Offline Calibration)
│  Parameters: b[], γ^H, γ^A, δ_H, δ_A, Q, XGBoost weights
│
▼
Phase 2 (Pre-Match Initialization)
│  Initialize: a_H, a_A, P_grid, Q_off_normalized, C_time, T_exp
│
▼
Phase 3 (Live Trading Engine)
│  Real-time: P_true(dict), σ_MC(dict), order_allowed, P_bet365(dict)
│  2-Layer: Odds-API Live Odds WS + Goalserve Live Score REST
│  Wall-clock model.t with halftime exclusion
│  *** Identical in paper and live mode ***
│
▼
Phase 4 (Arbitrage & Execution) [v3]
│  • signal_generator: decompose P_true dict → per-market floats
│  • Per market: EV → Kelly → risk → execute (multi-market loop)
│  • Kalshi order book sync (Step 4.1) — LIVE data in both modes
│  • ExecutionRouter: paper fills (simulated) or live fills (real)
│  • VWAP-connected EV (2-pass), per-market σ_MC for P_cons
│  • Directional P_cons (Yes→lower bound, No→upper bound)
│  • Incremental Kelly (existing exposure aware)
│  • Directional exit triggers (edge reversal, expiry, settlement)
│  • Market alignment check (not independent validation, multiplier 0.8)
│  • Paper: directional slippage + fill delay + ob_freeze check [v3]
│  • Rapid Entry: VAR safety wait + P_cons adjustment
│
▼
Step 4.6 (Post-Match Analytics)
│  Analysis: 12 metrics
│  Tuning: 8 parameters (including THETA_ENTRY)
│
└──▶ Phase 1 retraining (when triggered)
```

---

## v2 Change Tracking

| # | Location | Before | After |
|---|------|--------|--------|
| 1 | Step 4.4 Trigger 2 | `P_cons > (1-P_bid) + θ` | `P_cons > P_bid + θ` |
| 2 | Step 4.4 Trigger 3 | Only Buy Yes `E_hold` | Directional `E_hold` branch |
| 3 | Step 4.4 Trigger 4 | `P_bet365 > (1-entry) + 0.05` | `P_bet365 > entry + 0.05` |
| 4 | Step 4.1~4.2 | "Independent validation", mult 1.0 | "Market alignment", mult 0.8 |
| 5 | Step 4.2 | EV with best ask/bid | EV with 2-pass VWAP |
| 6 | Step 4.5 Paper | Full instant fill at best ask | VWAP + 1 tick + partial fill |
| 7 | Step 4.5 Rapid | No VAR wait, no P_cons adjustment | 5s wait + z-adjustment + stricter conditions |
| 8 | Step 4.6 settlement | `Qty × (Sett - Entry)` | Directional `BuyYes: Sett-Entry`, `BuyNo: Entry-Sett` |

## v3 Change Tracking (Paper Trading Infrastructure)

| # | Location | Before | After |
|---|------|--------|--------|
| 9 | Step 4.5 Paper | Slippage always `+ ticks` | Directional: BUY_YES `+`, BUY_NO `−` |
| 10 | Step 4.5 Paper | Instant fill (0ms) | 1-3s delay + ob_freeze abort check |
| 11 | Step 4.5 | Live/Paper functions separate | `ExecutionRouter` unified interface |
| 12 | Step 4.6 | 11 metrics | 12 metrics: added Paper Realism Score |
| 13 | Roadmap Phase 0 | "Period: 2-4 weeks" only | 8 explicit graduation criteria for Phase 0 → A |
| 14 | Step 4.3 Kelly | Static Kelly (ignores existing position) | Incremental Kelly: `f_incremental = f_optimal - existing_fraction` |
| 15 | Step 4.2 Filters | No position check | Block opposite-direction + block pending orders for same market |
| 16 | Step 4.6 Settlement | Instant P&L after match end | Poll Kalshi resolution (up to 6h), AWAITING_SETTLEMENT state |
| 17 | Phase 3 tick_loop | `model.t = wall_clock / 60` (includes halftime) | `model.t = (wall_clock - halftime_accumulated) / 60` (play time only) |
| 18 | Phase 3→4 interface | P_true: float, σ_MC: float | P_true: dict, σ_MC: dict (per-market), decomposed in signal_generator |
| 19 | Phase 4 | No signal_generator; no multi-market loop | `signal_generator()` coroutine: per-ticker decomposition + execution loop |
| 20 | Phase 3 σ_MC | Single float for all markets | Per-market `math.sqrt(p*(1-p)/N)` via `compute_mc_stderr()` |

## v4 Change Tracking (Review Gap Fixes)

| # | Gap | Location | Before | After |
|---|-----|------|--------|--------|
| 21 | #1 σ_MC analytical | Phase 3 `step_3_4_async` | σ_MC = 0.0 in analytical mode | σ_MC = max(sqrt(p*(1-p)/N_MC), 0.005) — synthetic floor |
| 22 | #2 Partial exit | Phase 4 Step 4.4 | 4 exit triggers, binary only | 6 triggers: added Trigger 5 (position trim when f_optimal < existing * 0.5) |
| 23 | #3 Direction flip | Phase 4 Step 4.4 | No opportunity cost exit | Added Trigger 6: exit if opposite EV > θ_entry AND current EV < 2× θ_exit |
| 24 | #4 Within-container race | Phase 4 signal_generator | Sequential but not explicitly awaited | Explicit: await execute_with_reservation completes before next ticker |
| 25 | #5 Bankroll staleness | Phase 4 signal_generator | model.bankroll static from startup | model.bankroll decremented by fill_cost after each fill |
| 26 | #6 Multi-goal same poll | Phase 3 GoalserveLiveScoreSource | All goals in poll use final score tuple | Intermediate score tracking: running_home/away incremented per goal |
| 27 | #7 Stale bet365 | Phase 4 OrderBookSync | No timestamp on bet365_implied | bet365_last_update + 30s threshold → UNAVAILABLE if stale |
| 28 | #8 Kalshi rejection | Phase 4 execute_order | No error handling on submit_order | KalshiApiError catch: market_closed, insufficient_balance, price_out_of_range |
| 29 | #10 Kalshi WS stale | Phase 4 OrderBookSync | No timestamp on Kalshi WS data | kalshi_last_update + 5s threshold → skip trading if stale |