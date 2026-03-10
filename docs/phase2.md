# Phase 2: Pre-Match Initialization — Goalserve + Odds-API

## Overview

Before kickoff, this stage initializes the parameters learned in Phase 1 to match today's match reality,
setting the **Initial Condition** for the Live Trading Engine.

If Phase 1 formulates the "wisdom of the past,"
Phase 2 is the process of fine-tuning that wisdom to "today's lineup and condition".

From lineup announcement (about one hour before kickoff) until just before kickoff,
we break down the internal mathematical operations and data pipeline
into five steps.

### Unified Data Source

As in Phase 1, maintain a **consistent data source architecture**:
Goalserve for fixtures, results, and match stats; Odds-API for all odds data.
Because Phase 1 training data and Phase 2 inference data share identical schemas per source,
feature-mapping errors are eliminated.

| Data Source | Role in Phase 2 | Core Data |
|------------------|-----------------|-----------|
| **Goalserve Live Game Stats** | Lineups + formation (60 min before kickoff) | `teams.{team}.player[]`, `formation` |
| **Goalserve Live Game Stats** (historical) | Player-level rolling stats | `player_stats.{team}.player[]` |
| **Goalserve Fixtures/Results** (historical) | Team rolling stats, rest days, H2H | `stats.{team}`, match dates |
| **Odds-API Pre-Match Odds** | Odds features + sanity check | 5 bookmakers, 50+ markets |
| **Goalserve Live Score** | Prepare live event ingestion | REST polling every 3s (consumed in Phase 3) |
| **Odds-API Live Odds** | ob_freeze sensor + primary event-detection prep | WebSocket PUSH <1s (consumed in Phase 3) |

---

## Input Data

**Phase 1 production parameters:**

| Parameter | Source |
|---------|------|
| XGBoost weights + `feature_mask.json` | Step 1.3 |
| $\mathbf{b} = [b_1, \ldots, b_6]$ | Step 1.4 |
| $\gamma^H_1, \gamma^H_2$ (home red-card penalty) | Step 1.4 |
| $\gamma^A_1, \gamma^A_2$ (away red-card penalty) | Step 1.4 |
| $(\beta_H, \kappa_H, \tau_H)$, $(\beta_A, \kappa_A, \tau_A)$ (parametric delta) | Step 1.4 |
| $\boldsymbol{\delta}_H[5], \boldsymbol{\delta}_A[5]$ (generated lookup tables) | Step 1.4 |
| Q (4x4 matrix) | Step 1.2 |
| $\mathbb{E}[\alpha_1], \mathbb{E}[\alpha_2]$ (average stoppage time) | Step 1.1 |
| `DELTA_SIGNIFICANT` (delta significance flag) | Step 1.5 LRT |
| Sanity check thresholds (`go_threshold`, `hold_threshold`, `ou_threshold`) | Step 1.5 |

**Endpoints:**

```
# Goalserve: Lineups + player stats (60 min before kickoff)
GET /getfeed/{api_key}/soccerstats/match/{match_id}?json=1

# Odds-API: Same-day pre-match odds
GET https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={key}&regions=eu&markets=h2h,totals

# Goalserve: Live score (for Phase 3, only connectivity validation here)
GET /getfeed/{api_key}/soccerlive/home?json=1

# Odds-API: Live odds (for Phase 3, only connectivity validation here)
WebSocket wss://api.odds-api.io/v3/ws?apiKey={key}&markets=ML,Totals&status=live
```

---

## Step 2.1: Pre-Match Context Data Collection (Data Ingestion)

### Timing

About **60 minutes before kickoff** — Goalserve Live Game Stats provides lineups at this time.

### 2.1.1: Collect Lineups + Formation

**Goalserve Live Game Stats -> `teams.{team}`:**

```json
{
  "formation": "4-3-3",
  "player": [
    {
      "formation_pos": "1",
      "id": "102587",
      "name": "Emiliano Martínez",
      "number": "23",
      "pos": "G"
    },
    {
      "formation_pos": "9",
      "id": "119",
      "name": "Lionel Messi",
      "number": "10",
      "pos": "F"
    }
  ]
}
```

| Extracted Data | Field | Usage |
|------------|------|------|
| Starting XI IDs | `player[].id` | key for player-level rolling stat lookup |
| Formation | `formation` ("4-3-3") | formation feature (optional) |
| Position | `player[].pos` (G/D/M/F) | position-weighted aggregation |
| Position in formation | `player[].formation_pos` | infer detailed role (CB vs FB, etc.) |

**Bench — `substitutes.{team}.player[]`:**

```json
{
  "id": "404462",
  "name": "Lautaro Martínez",
  "number": "22",
  "pos": "F"
}
```

Bench data is not used directly in the model, but is required to detect **pre-kickoff lineup changes**.
If a starter/bench swap occurs, Step 2.1 must be re-run.

> **Core advantage of unified IDs:** player ID `119` (Messi) is the same in Phase 1 `player_stats`
> and Phase 2 `teams` lineup data. No separate ID-mapping table is needed.

### 2.1.2: Compute Rolling Stats for Each Starter

Compute rolling means **directly** from historical `player_stats` already stored in DB during Phase 1.
No separate player database is required.

**Data flow:**

```
Get today's starting-11 player_ids (Step 2.1.1)
        |
        v
Fetch each player's latest 5 matches in player_stats
(data loaded into DB from historical Goalserve Live Game Stats during Phase 1)
        |
        v
Compute per-player per-90 metrics
        |
        v
Position-weighted aggregation -> team-level feature vector
```

**Per-player per-90 metrics:**

| Metric | Calculation | Goalserve Field |
|------|------|---------------|
| goals_per_90 | goals / minutes_played * 90 | `goals`, `minutes_played` |
| shots_on_target_per_90 | shots_on_goal / minutes_played * 90 | `shots_on_goal` |
| key_passes_per_90 | keyPasses / minutes_played * 90 | `keyPasses` |
| pass_accuracy | passes_acc / passes | `passes_acc`, `passes` |
| dribble_success_rate | dribbleSucc / dribbleAttempts | `dribbleSucc`, `dribbleAttempts` |
| tackles_per_90 | tackles / minutes_played * 90 | `tackles` |
| interceptions_per_90 | interceptions / minutes_played * 90 | `interceptions` |
| rating_avg | rolling mean of rating | `rating` |

> **minutes_played caution:** in Goalserve, unused bench players have empty `minutes_played`.
> Exclude such matches from rolling averages. Also exclude short substitute appearances (`minutes_played < 10`).

**Missing-value handling:**

```python
def safe_per90(stat_value: str, minutes: str) -> Optional[float]:
    """Handle Goalserve empty string as None."""
    mp = float(minutes) if minutes else 0
    val = float(stat_value) if stat_value else 0
    if mp < 10:
        return None  # Statistically unstable -> exclude from rolling
    return val / mp * 90
```

### 2.1.3: Position-Based Team-Level Aggregation

Aggregate starter stats by position groups:

```python
def aggregate_team_features(starting_11_stats: List[PlayerRolling]) -> dict:
    """
    Starter rolling stats -> team-level feature vector.
    Same structure as Tier 2 features in Phase 1 Step 1.3.
    """
    forwards = [p for p in starting_11_stats if p.pos == "F"]
    midfielders = [p for p in starting_11_stats if p.pos == "M"]
    defenders = [p for p in starting_11_stats if p.pos == "D"]
    goalkeeper = [p for p in starting_11_stats if p.pos == "G"]

    return {
        # Attack features (FW)
        "fw_avg_rating": safe_mean([p.rating_avg for p in forwards]),
        "fw_goals_p90": safe_sum([p.goals_per_90 for p in forwards]),
        "fw_shots_on_target_p90": safe_sum([p.shots_on_target_per_90 for p in forwards]),

        # Creativity features (MF)
        "mf_avg_rating": safe_mean([p.rating_avg for p in midfielders]),
        "mf_key_passes_p90": safe_sum([p.key_passes_per_90 for p in midfielders]),
        "mf_pass_accuracy": safe_mean([p.pass_accuracy for p in midfielders]),

        # Defense features (DF)
        "df_avg_rating": safe_mean([p.rating_avg for p in defenders]),
        "df_tackles_p90": safe_sum([p.tackles_per_90 for p in defenders]),
        "df_interceptions_p90": safe_sum([p.interceptions_per_90 for p in defenders]),

        # GK features
        "gk_rating": goalkeeper[0].rating_avg if goalkeeper else None,
        "gk_save_rate": goalkeeper[0].save_rate if goalkeeper else None,

        # Team-wide features (minutes-weighted)
        "team_avg_rating": weighted_mean(
            [p.rating_avg for p in starting_11_stats],
            weights=[p.minutes_played_avg for p in starting_11_stats]
        ),
    }
```

> **Use of `formation_pos` — future extension:**
> Goalserve `formation_pos` can distinguish CB (3,4) vs FB (2,5),
> but consistency requires the same aggregation in Phase 1 training.
> Start with 4 classes (G/D/M/F), then expand to detailed positions after system stabilization.

### 2.1.4: Team-Level Rolling Stats

In addition to player-level features, compute **team-level rolling metrics** from
historical `stats.{team}` already loaded in Phase 1:

| Feature | Goalserve Field | Rolling Window |
|------|---------------|------|
| xG_per_90 | Live Game Stats xG field | 5 matches |
| xGA_per_90 | opponent xG | 5 matches |
| possession_avg | `possestiontime.total` | 5 matches |
| shots_per_90 | `shots.total` | 5 matches |
| shots_insidebox_ratio | `shots.insidebox / shots.total` | 5 matches |
| pass_accuracy | `passes.accurate / passes.total` | 5 matches |
| corners_per_90 | `corners.total` | 5 matches |
| fouls_per_90 | `fouls.total` | 5 matches |

### 2.1.5: Collect Odds

**Odds-API Pre-Match Odds — 5 bookmakers, 50+ markets:**

```python
def extract_odds_features(bookmakers: List[dict]) -> dict:
    """
    5 bookmaker odds (Odds-API format) -> feature vector + sanity-check baseline values.
    Same structure as Phase 1 Step 1.3 Tier 3.

    Odds-API bookmaker format:
    {
        "key": "betfair_exchange",
        "title": "Betfair Exchange",
        "markets": [{
            "key": "h2h",
            "outcomes": [
                {"name": "Home Team", "price": 1.85},
                {"name": "Draw", "price": 3.40},
                {"name": "Away Team", "price": 4.50}
            ]
        }]
    }
    """
    def remove_overround(h, d, a):
        total = 1/h + 1/d + 1/a
        return (1/h)/total, (1/d)/total, (1/a)/total

    all_probs = []
    exchange_prob = None

    for bm in bookmakers:
        h2h = next((m for m in bm["markets"] if m["key"] == "h2h"), None)
        if not h2h:
            continue
        outcomes = {o["name"]: float(o["price"]) for o in h2h["outcomes"]}
        h = outcomes.get("Home Team", outcomes.get(list(outcomes.keys())[0]))
        d = outcomes.get("Draw", 0)
        a = outcomes.get("Away Team", outcomes.get(list(outcomes.keys())[-1]))
        if not all([h, d, a]):
            continue
        prob = remove_overround(h, d, a)
        all_probs.append(prob)

        if bm["key"] == "betfair_exchange":
            exchange_prob = prob

    if exchange_prob is None:
        exchange_prob = tuple(np.mean(all_probs, axis=0))

    return {
        # For features (Step 2.2 -> XGBoost input)
        "exchange_home_prob": exchange_prob[0],
        "exchange_draw_prob": exchange_prob[1],
        "exchange_away_prob": exchange_prob[2],
        "market_avg_home_prob": np.mean([p[0] for p in all_probs]),
        "market_avg_draw_prob": np.mean([p[1] for p in all_probs]),
        "bookmaker_odds_std": np.std([p[0] for p in all_probs]),

        # For sanity check (Step 2.4)
        "_exchange_raw": exchange_prob,
        "_market_avg_raw": tuple(np.mean(all_probs, axis=0)),
        "_all_bookmakers": bookmakers,  # For O/U cross-validation
    }
```

### 2.1.6: Context Features

| Feature | Source | Calculation |
|------|------|------|
| home_away_flag | Fixtures metadata | localteam = 1, visitorteam = 0 |
| rest_days | Fixture date difference | days since each team's previous match |
| h2h_goal_diff | Fixtures H2H | average goal difference over last 5 H2H matches |

### Step 2.1 Output

```python
@dataclass
class PreMatchData:
    # Lineups
    home_starting_11: List[str]     # Goalserve player IDs
    away_starting_11: List[str]
    home_formation: str             # "4-3-3"
    away_formation: str

    # Tier 2: player aggregate features
    home_player_agg: dict           # output of aggregate_team_features()
    away_player_agg: dict

    # Tier 1: team rolling stats
    home_team_rolling: dict         # based on historical stats.{team}
    away_team_rolling: dict

    # Tier 3: odds features
    odds_features: dict             # output of extract_odds_features()

    # Tier 4: context
    home_rest_days: int
    away_rest_days: int
    h2h_goal_diff: float

    # Metadata
    match_id: str                   # Goalserve match ID (unified across all phases)
    kickoff_time: str
```

---

## Step 2.2: Feature Selection

### Goal

Extract only validated features selected in Phase 1 from high-dimensional raw features,
reducing noise.

### Apply Feature Mask

Apply `feature_mask.json` saved in Phase 1 Step 1.3.
Since Phase 1 and Phase 2 use the **same Goalserve schema**,
no separate feature-name mapping logic is needed.

```python
def apply_feature_mask(pre_match: PreMatchData,
                       feature_mask: List[str],
                       median_values: Dict[str, float]) -> np.ndarray:
    """
    Extract only features listed in Phase 1 feature_mask.json.
    Replace missing values with medians from Phase 1 training data.

    Because feature names share the same Goalserve schema as Phase 1,
    no manual mapping layer is required -> blocks silent bugs.
    """
    # Build full feature vector
    full_vec = {}

    # Tier 1: team rolling (home_/away_ prefixes)
    for prefix, rolling in [("home_", pre_match.home_team_rolling),
                            ("away_", pre_match.away_team_rolling)]:
        for k, v in rolling.items():
            full_vec[prefix + k] = v

    # Tier 2: player aggregates
    for prefix, agg in [("home_", pre_match.home_player_agg),
                        ("away_", pre_match.away_player_agg)]:
        for k, v in agg.items():
            full_vec[prefix + k] = v

    # Tier 3: odds (not team-specific)
    for k, v in pre_match.odds_features.items():
        if not k.startswith("_"):  # Exclude internal fields
            full_vec[k] = v

    # Tier 4: context
    full_vec["home_away_flag"] = 1  # Always in home perspective
    full_vec["home_rest_days"] = pre_match.home_rest_days
    full_vec["away_rest_days"] = pre_match.away_rest_days
    full_vec["h2h_goal_diff"] = pre_match.h2h_goal_diff

    # Apply mask
    selected = []
    for feat_name in feature_mask:
        val = full_vec.get(feat_name)
        if val is not None and not np.isnan(val):
            selected.append(val)
        else:
            selected.append(median_values[feat_name])  # Missing-value replacement

    return np.array(selected)
```

### Output

$$X_{match} \in \mathbb{R}^{d'}$$

A feature vector with identical dimensionality and feature order as Phase 1.

---

## Step 2.3: Back-Solving Baseline Intensity Parameter a (Prior Inference)

### Goal

Convert ML model predictions (expected goals) into intensity-function parameter a for the live engine.

### ML Inference

Feed Step 2.2's $X_{match}$ into the XGBoost Poisson model trained in Phase 1:

```python
import xgboost as xgb

def predict_expected_goals(X_match: np.ndarray, model_path: str) -> Tuple[float, float]:
    """
    Predict home/away expected goals with XGBoost Poisson model.
    If separate home/away models are used, call each model;
    if a single model is used, switch by home/away flag.
    """
    model = xgb.Booster()
    model.load_model(model_path)

    dmat = xgb.DMatrix(X_match.reshape(1, -1))
    mu_hat = model.predict(dmat)[0]  # Poisson expectation

    return mu_hat  # μ̂_H or μ̂_A
```

- $\hat{\mu}_H$: full-match expected goals for home team
- $\hat{\mu}_A$: full-match expected goals for away team

### Mathematical Back-Solving — Piecewise Basis Version

At kickoff, X = 0 and ΔS = 0:

$$\lambda_H(t \mid X=0, \Delta S=0) = \exp\!\left(a_H + b_{i(t)}\right)$$

(because $\gamma^H_0 = 0$ and $\delta_H(0) = 0$)

Full-match expected goals:

$$\hat{\mu}_H = \exp(a_H) \sum_{i=1}^{K} \exp(b_i) \cdot \Delta t_i = \exp(a_H) \cdot C_{time}$$

$$C_{time} \equiv \sum_{i=1}^{K} \exp(b_i) \cdot \Delta t_i$$

$$\boxed{a_H = \ln(\hat{\mu}_H) - \ln(C_{time})}$$

$$\boxed{a_A = \ln(\hat{\mu}_A) - \ln(C_{time})}$$

### Expected Match Duration $T_{exp}$

$$T_{exp} = 90 + \mathbb{E}[\alpha_1] + \mathbb{E}[\alpha_2]$$

$\mathbb{E}[\alpha_1], \mathbb{E}[\alpha_2]$ are league-level means of Goalserve `addedTime_period1/2`
computed in Phase 1 Step 1.1.

| Interval i | Coverage | $\Delta t_i$ |
|--------|----------|-------------|
| 1 | 1H 0-15 min | 15 |
| 2 | 1H 15-30 min | 15 |
| 3 | 1H 30-45 min + stoppage | $15 + \mathbb{E}[\alpha_1]$ |
| 4 | 2H 0-15 min | 15 |
| 5 | 2H 15-30 min | 15 |
| 6 | 2H 30-45 min + stoppage | $15 + \mathbb{E}[\alpha_2]$ |

### Relationship with delta

At kickoff, ΔS = 0 so delta(0) = 0.
**delta does not affect the back-solving formula.**
delta is activated only after goals occur in Phase 3.

### Output

$a_H$, $a_A$, $C_{time}$

---

## Step 2.4: Pre-Match Sanity Check

### Goal

Verify that pre-match model probabilities do not deviate excessively from market consensus.
Using Odds-API Pre-Match Odds (5 bookmakers, 50+ markets),
perform a more precise multidimensional validation than the original design.

### Primary Check: Match Winner (Betfair Exchange Baseline)

```python
def primary_sanity_check(mu_H: float, mu_A: float,
                          exchange_prob: Tuple[float, float, float],
                          market_avg: Tuple[float, float, float],
                          sanity_thresholds: dict) -> str:
    """
    Compare model probabilities vs Betfair Exchange + market average.
    Use Betfair Exchange as primary benchmark due to market efficiency.

    sanity_thresholds: calibrated in Phase 1 Step 1.5 from validation data.
        - go_threshold: 90th percentile of model-Betfair Exchange discrepancy
        - hold_threshold: 99th percentile of model-Betfair Exchange discrepancy
    """
    # Model probabilities (independent Poisson)
    P_model = compute_match_odds_poisson(mu_H, mu_A)  # {H, D, A}

    # Discrepancy vs Betfair Exchange
    delta_pin = max(
        abs(P_model[o] - exchange_prob[i])
        for i, o in enumerate(["H", "D", "A"])
    )

    # Discrepancy vs market average
    delta_mkt = max(
        abs(P_model[o] - market_avg[i])
        for i, o in enumerate(["H", "D", "A"])
    )

    go_thresh = sanity_thresholds["go_threshold"]      # calibrated 90th pct
    hold_thresh = sanity_thresholds["hold_threshold"]   # calibrated 99th pct

    if delta_pin < go_thresh:
        return "GO"
    elif delta_pin < hold_thresh:
        # Deviates from Betfair Exchange but close to market average
        # -> Betfair Exchange may be a temporary outlier
        if delta_mkt < go_thresh * 0.67:  # scaled from go_threshold
            return "GO_WITH_CAUTION"
        return "HOLD"
    else:
        return "SKIP"
```

### Secondary Check: Over/Under Cross-Validation (Goalserve-Specific)

Because Odds-API provides 50+ markets,
cross-check whether model μ_H + μ_A is also consistent with Over/Under market.

```python
def secondary_sanity_check(mu_H: float, mu_A: float,
                            ou_odds: dict,
                            sanity_thresholds: dict) -> dict:
    """
    Check whether model total expected goals aligns with O/U market.
    Detect cases where Match Winner alone misses "right total, wrong split".

    sanity_thresholds["ou_threshold"]: calibrated in Phase 1 Step 1.5
        from 90th percentile of model-market O/U discrepancy.
    """
    mu_total = mu_H + mu_A

    # Model Over 2.5 probability
    from scipy.stats import poisson
    P_model_over25 = 1 - poisson.cdf(2, mu_total)

    # Market implied Over 2.5 probability
    over_odds = float(ou_odds["Over"]["value"])
    under_odds = float(ou_odds["Under"]["value"])
    ou_sum = 1/over_odds + 1/under_odds
    P_market_over25 = (1/over_odds) / ou_sum

    delta_ou = abs(P_model_over25 - P_market_over25)
    ou_thresh = sanity_thresholds["ou_threshold"]  # calibrated 90th pct

    return {
        "P_model_over25": P_model_over25,
        "P_market_over25": P_market_over25,
        "delta_ou": delta_ou,
        "ou_consistent": delta_ou < ou_thresh,
    }
```

**Meaning of cross-validation:**

| Primary (Match Winner) | Secondary (Over/Under) | Diagnosis |
|-------------------|-------------------|------|
| GO | Match | ✅ both μ_H and μ_A are likely accurate |
| GO | Mismatch | ⚠️ total μ may be right but μ split may be wrong |
| HOLD | Match | ⚠️ possible Match Winner market anomaly (e.g., cup context) |
| SKIP | — | ❌ skip this match |

### Combined Final Verdict

```python
def combined_sanity_check(mu_H, mu_A, odds_data,
                          sanity_thresholds: dict) -> SanityResult:
    """
    sanity_thresholds: from Phase 1 Step 1.5 calibration.
        Keys: go_threshold, hold_threshold, ou_threshold
    """
    primary = primary_sanity_check(mu_H, mu_A, ..., sanity_thresholds)
    secondary = secondary_sanity_check(mu_H, mu_A, ..., sanity_thresholds)

    if primary == "SKIP":
        return SanityResult(verdict="SKIP")

    if primary == "GO" and secondary["ou_consistent"]:
        return SanityResult(verdict="GO")

    if primary == "GO" and not secondary["ou_consistent"]:
        return SanityResult(
            verdict="GO_WITH_CAUTION",
            warning="O/U mismatch — μ ratio may be off"
        )

    if primary == "HOLD":
        return SanityResult(verdict="HOLD")

    return SanityResult(verdict="GO_WITH_CAUTION")
```

### Output

```python
@dataclass
class SanityResult:
    verdict: str            # GO | GO_WITH_CAUTION | HOLD | SKIP
    delta_match_winner: float   # Discrepancy vs Betfair Exchange
    delta_over_under: float     # O/U discrepancy
    warning: Optional[str]      # Warning message
```

---

## Step 2.5: Live Engine Initialization and Connectivity Establishment (System Initialization)

### Load and Instantiate Parameters

```
Initial state of LiveFootballQuantModel:
|
+-- Time state
|   +-- current_time        = 0
|   +-- engine_phase        = WAITING_FOR_KICKOFF
|   +-- T_exp               <- Step 2.3
|
+-- Match state
|   +-- current_state (X)   = 0  (11v11)
|   +-- current_score (S)   = (0, 0)
|   +-- delta_S             = 0
|
+-- Intensity-function parameters
|   +-- a_H, a_A            <- Step 2.3
|   +-- b[1..6]             <- Phase 1
|   +-- gamma^H[0..3]       <- Phase 1 (gamma^H_0=0, gamma^H_1, gamma^H_2, gamma^H_1+gamma^H_2)
|   +-- gamma^A[0..3]       <- Phase 1 (gamma^A_0=0, gamma^A_1, gamma^A_2, gamma^A_1+gamma^A_2)
|   +-- delta_H_params[3]    <- Phase 1 (β, κ, τ parametric coefficients)
|   +-- delta_A_params[3]    <- Phase 1
|   +-- delta_H[5], delta_A[5] <- Phase 1 (generated lookup tables)
|   +-- C_time              <- Step 2.3
|
+-- Markov model
|   +-- Q (4x4)             <- Phase 1
|   +-- Q_deltaS[5] (4x4)   <- Phase 1 (ext: score-conditioned Q matrices)
|   +-- Q_off_normalized    <- computed below (single matrix, team-independent)
|   +-- P_grid[0..100]      <- precomputed matrix exponentials
|   +-- P_grid_by_ds[5][0..100] <- [EXT] per-ΔS-bin matrix exponentials
|
+-- Stoppage time distribution (extension)
|   +-- stoppage_dist_1     <- Phase 1 (LogNormal params for 1H stoppage)
|   +-- stoppage_dist_2     <- Phase 1 (LogNormal params for 2H stoppage)
|
+-- Phase 3 mode controls
|   +-- DELTA_SIGNIFICANT   <- result from Phase 1 Step 1.5 LRT
|   +-- preliminary_cache   = {}  (Phase 3 precompute cache)
|
+-- Event state machine (for Phase 3)
|   +-- event_state         = IDLE
|   +-- cooldown            = False
|   +-- ob_freeze           = False
|
+-- Quote anomaly detection (ob_freeze sensors)
|   +-- bet365_odds_prev    = None  (Odds-API bet365 odds sensor — Phase 3)
|   +-- live_score_prev     = None  (Goalserve Live Score sensor — Phase 3)
|   +-- P_kalshi_prev   = None  (Kalshi quote sensor — Phase 4)
|
+-- Goalserve connectivity
|   +-- match_id            <- Step 2.1 (unified ID across all phases)
|   +-- live_score_ready    = False (True after REST polling validation)
|
+-- Odds-API connectivity
|   +-- live_odds_ws        = None  (assigned after Odds-API WebSocket connect)
|   +-- live_odds_healthy   = False (True after connectivity validation)
|
+-- Kalshi connectivity
|   +-- kalshi_ws           = None  (assigned after WebSocket connect)
|   +-- kalshi_healthy      = False
|
+-- Sanity-check result
|   +-- sanity_thresholds   <- Phase 1 Step 1.5 (go_threshold, hold_threshold, ou_threshold)
|   +-- verdict             <- Step 2.4
|   +-- delta_match_winner  <- Step 2.4
|   +-- delta_over_under    <- Step 2.4
|
+-- Risk parameters
    +-- bankroll            <- current Kalshi account balance
    +-- f_order_cap  = 0.03
    +-- f_match_cap  = 0.05
    +-- f_total_cap  = 0.20
```

### Precompute Matrix Exponentials

In Phase 3 Step 3.2 analytic μ calculation, query P_grid for O(1) operations:

```python
import scipy.linalg

P_grid = {}
for dt_min in range(0, 101):
    P_grid[dt_min] = scipy.linalg.expm(Q * dt_min)

# [EXT] Q_ΔS: precompute per-ΔS-bin P_grids (5 bins × 101 steps = 505 matrices, ~64KB)
P_grid_by_ds = {}
if Q_deltaS is not None:
    for ds_bin in range(5):
        P_grid_by_ds[ds_bin] = {}
        for dt_min in range(0, 101):
            P_grid_by_ds[ds_bin][dt_min] = scipy.linalg.expm(Q_deltaS[ds_bin] * dt_min)
```

> **Near-end resolution limit:** if P_grid is minute-level only,
> relative error can increase when remaining time is under one minute.
> MC mode in Phase 3 simulates dismissals directly, so this issue does not apply there.
> In analytic mode, it is recommended to add a 10-second fine grid for the final 5 minutes:

```python
# Fine grid: final 5 minutes (0.0~5.0 min, 0.167-min steps)
P_fine_grid = {}
for dt_10sec in range(0, 31):  # 0~30 (= 0~5 min, 10-sec increments)
    dt_min = dt_10sec / 6.0
    P_fine_grid[dt_10sec] = scipy.linalg.expm(Q * dt_min)

# [EXT] Q_ΔS fine grid
P_fine_grid_by_ds = {}
if Q_deltaS is not None:
    for ds_bin in range(5):
        P_fine_grid_by_ds[ds_bin] = {}
        for dt_10sec in range(0, 31):
            dt_min = dt_10sec / 6.0
            P_fine_grid_by_ds[ds_bin][dt_10sec] = scipy.linalg.expm(
                Q_deltaS[ds_bin] * dt_min
            )
```

### Normalize Q_off Transition Probabilities

Normalize off-diagonal entries of Q from Phase 1 into transition probabilities for MC simulation:

```python
Q_off_normalized = np.zeros((4, 4))
for i in range(4):
    total_off_diag = -Q[i, i]
    if total_off_diag > 0:
        for j in range(4):
            if i != j:
                Q_off_normalized[i, j] = Q[i, j] / total_off_diag
```

> **Q_off is team-independent:** Q is transition rate for Markov state X(t), independent of team.
> Unlike gamma^H and gamma^A, no home/away split is needed. Create only **one** Q_off_normalized.

### Establish Connections — 3-Source Architecture

| Target | Protocol | Latency | Role in Phase 3 |
|----------|---------|------|-------------|
| **Odds-API Live Odds** | **WebSocket PUSH** | **<1s** | **primary event detection + ob_freeze** |
| Kalshi API | WebSocket | 1-2s | quote ingestion + execution |
| **Goalserve Live Score** | **REST polling every 3s** | 3-8s | **authoritative confirmation (VAR, scorer)** |

**Odds-API Live Odds WebSocket connection:**

```python
async def connect_live_odds(api_key: str, event_ids: List[str]):
    """
    Connect Odds-API Live Odds WebSocket.
    Receive bet365 in-play odds (ML, Totals, Spread markets).

    Note: Odds-API provides odds data only (no score, minute, period).
    Score/event confirmation comes from Goalserve Live Score REST.

    Role in Phase 3:
    1. Odds-jump detection (<1s) -> ob_freeze (early warning)
    2. Market-movement cross-check
    """
    ws = await websockets.connect(
        f"wss://api.odds-api.io/v3/ws?apiKey={api_key}&markets=ML,Totals&status=live"
    )

    # Connectivity validation: confirm welcome message
    first_msg = await asyncio.wait_for(ws.recv(), timeout=10)
    parsed = json.loads(first_msg)

    if parsed.get("type") != "welcome":
        raise ConnectionError("Odds-API WS: unexpected message format")

    log.info(f"Odds-API Live Odds WS connected: "
             f"bookmakers={parsed.get('bookmakers')}")
    return ws
```

**Goalserve Live Score REST validation:**

```python
async def verify_live_score(api_key: str, match_id: str):
    """
    Verify accessibility of Goalserve Live Score REST endpoint.
    IP whitelist authentication applies.
    """
    url = f"http://www.goalserve.com/getfeed/{api_key}/soccerlive/home"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params={"json": 1})
        response.raise_for_status()
        data = response.json()
        match_found = find_match_in_feed(data, match_id)
        return match_found
```

> **IP whitelist note:** Goalserve uses IP-based authentication.
> In cloud deployment, server IP must be pre-registered with Goalserve.

**Kalshi WebSocket connection:**

```python
async def connect_kalshi(api_key: str, market_tickers: List[str]):
    """Connect Kalshi WebSocket — quote ingestion + trading."""
    ws = await kalshi_api.connect_ws(api_key)
    for ticker in market_tickers:
        await ws.subscribe_orderbook(ticker)
    return ws
```

### Numba JIT Warm-Up

Phase 3 Step 3.4 MC simulation is JIT-compiled by Numba.
The first call incurs ~2s compile latency, so run a dummy call for warm-up.

**Both code paths must be warmed up:** if `mc_simulate_remaining` has a
`DELTA_SIGNIFICANT` branch that skips delta computation, Numba compiles
each branch on first entry. Warming up only one path leaves the other
uncompiled, causing a ~2s stall if that branch is hit live (e.g., right after a goal).

```python
from phase3.mc_core import mc_simulate_remaining

# Warm-up 1: delta_significant=False path (delta ignored)
_ = mc_simulate_remaining(
    t_now=0, T_end=1,
    S_H=0, S_A=0,
    state=0, score_diff=0,
    a_H=0.0, a_A=0.0,
    b=np.zeros(6),
    gamma_H=np.zeros(4), gamma_A=np.zeros(4),
    delta_H=np.zeros(5), delta_A=np.zeros(5),
    Q_diag=np.zeros(4), Q_off=Q_off_normalized,
    basis_bounds=np.zeros(7),
    N=10, seed=42,
    delta_significant=False
)

# Warm-up 2: delta_significant=True path (delta applied)
_ = mc_simulate_remaining(
    t_now=0, T_end=1,
    S_H=0, S_A=0,
    state=0, score_diff=1,       # non-zero score_diff to exercise delta lookup
    a_H=0.0, a_A=0.0,
    b=np.zeros(6),
    gamma_H=np.zeros(4), gamma_A=np.zeros(4),
    delta_H=np.array([0.1, 0.05, 0.0, -0.05, -0.1]),  # non-zero delta
    delta_A=np.array([-0.1, -0.05, 0.0, 0.05, 0.1]),
    Q_diag=np.zeros(4), Q_off=Q_off_normalized,
    basis_bounds=np.zeros(7),
    N=10, seed=42,
    delta_significant=True
)
log.info("Numba JIT warmup complete (both delta paths)")
```

### Event Sources -> Phase 3 Engine Mapping

**Odds-API Live Odds WebSocket (early warning, <1s):**

| Odds-API Data Change | Detection Method | Phase 3 Action |
|----------------------|----------|-------------|
| abrupt odds jump (>10%) | compare with previous push | **ob_freeze (goal or red card — type unknown)** |
| odds frozen / market deleted | message type "deleted" or "no_markets" | possible halftime or match end |

> **Note:** Odds-API provides odds data only — no score, minute, or period info.
> Score changes, period transitions, and stoppage-time entry are detected
> exclusively via Goalserve Live Score REST.

**Goalserve Live Score REST (authoritative confirmation, 3-8s):**

| Live Score Data Change | Detection Method | Phase 3 Action |
|----------------------|----------|-------------|
| score increase | diff vs previous poll | **CONFIRMED goal + scorer + VAR status** |
| redcards list added | diff vs previous poll | **CONFIRMED red card** |
| period change | "1st" -> "Half" -> "2nd" | halftime enter/exit handling |
| minute > 45 or > 90 | monitor field values | stoppage-time T rolling |
| status "Finished" | field change | final settlement |

### Circuit Breaker

| Failure Type | Detection | Response |
|----------|----------|------|
| **Odds-API Live Odds WS disconnected** | **no heartbeat for 5s** | **disable primary detection, fallback to Live Score + Kalshi** |
| Goalserve Live Score polling failure | HTTP failure 3 times in a row | stop new orders, skip match after 5 failures |
| Kalshi WS disconnected | no heartbeat for 10s | cancel working orders, attempt reconnect |
| lineup change | recheck before kickoff | rerun Steps 2.1-2.4 |
| match postponed/cancelled | Goalserve status update | full shutdown, flatten positions |

> **Graceful degradation:** Odds-API Live Odds WebSocket has been promoted from supplementary to core,
> but if it fails, operation remains possible via Live Score REST + Kalshi ob_freeze.
> Performance drops to original-design level, but safety is preserved.

### Final Pre-Kickoff Check (5 Minutes Before Kickoff)

```python
async def pre_kickoff_final_check(model: LiveFootballQuantModel):
    """5 minutes before kickoff — final validation of all conditions."""

    # 1. Re-verify lineup
    current_lineup = await fetch_lineup(model.match_id)
    if current_lineup != model.home_starting_11 + model.away_starting_11:
        log.warning("Lineup changed — re-running Steps 2.1~2.4")
        await re_initialize(model, current_lineup)

    # 2. Check connectivity health
    assert model.live_odds_healthy or model.live_score_ready, \
        "At least one Goalserve source must be healthy"
    assert model.kalshi_healthy, "Kalshi WS must be connected"

    # 3. Check sanity result
    if model.sanity_verdict == "SKIP":
        log.info(f"Match {model.match_id} SKIPPED by sanity check")
        return False

    # 4. Confirm Numba warm-up (already done)
    log.info(f"Match {model.match_id} ready for kickoff")
    return True
```

### Output

An operational `LiveFootballQuantModel` instance ready to run.

---

## Phase 2 -> Phase 3 Handoff

| Item | Value | Source |
|------|---|------|
| $a_H, a_A$ | initial scoring intensity | Step 2.3 |
| $\mathbf{b}[1..6]$ | time-interval profile | Phase 1 |
| $\gamma^H[0..3], \gamma^A[0..3]$ | team-specific dismissal penalties | Phase 1 |
| $(\beta, \kappa, \tau)_H$, $(\beta, \kappa, \tau)_A$ | parametric delta coefficients | Phase 1 |
| $\boldsymbol{\delta}_H[5], \boldsymbol{\delta}_A[5]$ | generated delta lookup tables | Phase 1 |
| Q (4x4) | Markov transition matrix | Phase 1 |
| $\{Q_{\Delta S}\}$ (5 x 4x4) | score-conditioned Q (extension) | Phase 1 |
| Stoppage time dist params | per-league LogNormal (extension) | Phase 1 |
| $C_{time}$, $T_{exp}$ | time constants | Step 2.3 |
| $P_{grid}[0..100]$ + $P_{fine\_grid}$ | matrix-exponential grids | Step 2.5 |
| $P_{grid\_by\_ds}[5][0..100]$ + $P_{fine\_grid\_by\_ds}$ | [EXT] per-ΔS-bin matrix-exponential grids | Step 2.5 |
| $Q_{off\_normalized}$ (4x4) | normalized transition probabilities for MC (single, team-independent) | Step 2.5 |
| `DELTA_SIGNIFICANT` | delta significance -> analytic/MC mode selection | Phase 1 Step 1.5 |
| system state | t=0, X=0, S=(0,0), ΔS=0 | Step 2.5 |
| event_state | IDLE (initial) | Step 2.5 |
| **Goalserve match_id** | **unified match ID across all phases** | Step 2.1 |
| **Odds-API Live Odds WS** | **primary event detection + ob_freeze** | Step 2.5 |
| Goalserve Live Score | REST polling ready | Step 2.5 |
| Kalshi WS | quote + trading ready | Step 2.5 |
| ob_freeze | False (ob_freeze sensors initialized) | Step 2.5 |
| cooldown | False | Step 2.5 |
| sanity result | GO / GO_WITH_CAUTION / HOLD / SKIP | Step 2.4 |
| risk limits | f_order=0.03, f_match=0.05, f_total=0.20 | Step 2.5 |

---

## Phase 2 Pipeline Summary

```
[60 Minutes Before Kickoff: Lineup Announcement]
              |
              v
+--------------------------------------------------------------+
|  Step 2.1: Data Collection (Goalserve Unified Source)         |
|                                                              |
|  2.1.1: Live Game Stats -> Starting XI + formation           |
|  2.1.2: Historical player_stats -> player rolling per-90     |
|  2.1.3: Position-based team aggregation (G/D/M/F)            |
|  2.1.4: Historical stats.{team} -> team rolling (incl. xG)   |
|  2.1.5: Odds-API -> 5 bookmaker odds features               |
|  2.1.6: Fixtures -> context (rest days, H2H)                 |
|                                                              |
|  Output: PreMatchData (Tier 1-4 features + lineups + odds)   |
+------------------+-------------------------------------------+
                   v
+--------------------------------------------------------------+
|  Step 2.2: Feature Selection                                  |
|  • Apply feature_mask.json (same Goalserve schema as Phase 1)|
|  • Missing values -> Phase 1 training medians                |
|  • No manual mapping layer required (unification effect)     |
|  Output: X_match ∈ ℝ^{d'}                                    |
+------------------+-------------------------------------------+
                   v
+--------------------------------------------------------------+
|  Step 2.3: Back-solve a parameters                            |
|  • XGBoost Poisson -> μ̂_H, μ̂_A                              |
|  • a = ln(μ̂) − ln(C_time)                                    |
|  Output: a_H, a_A, C_time                                     |
+------------------+-------------------------------------------+
                   v
+--------------------------------------------------------------+
|  Step 2.4: Sanity Check (Multidimensional)                    |
|  • Primary: Match Winner vs Betfair Exchange + market average         |
|  • Secondary: Over/Under cross-check (using 50+ markets)      |
|  Output: GO / GO_WITH_CAUTION / HOLD / SKIP                  |
+------------------+-------------------------------------------+
                   | [GO or GO_WITH_CAUTION]
                   v
+--------------------------------------------------------------+
|  Step 2.5: System Initialization                              |
|                                                              |
|  Parameter loading:                                           |
|  • Phase 1 params (b, gamma^H/A, delta_params, delta_lookup,Q)|
|  • [EXT] Q_ΔS, stoppage dist, position gamma modifier         |
|  • Precompute P_grid[0..100] + P_fine_grid                   |
|  • [EXT] P_grid_by_ds per ΔS bin (505 matrices, ~64KB)       |
|  • Normalize Q_off_normalized (single, team-independent)      |
|                                                              |
|  Connectivity (3-source):                                     |
|  • Odds-API Live Odds WebSocket (<1s) <- primary detection    |
|  • Kalshi WebSocket (1-2s) <- quotes + execution             |
|  • Goalserve Live Score REST (3-8s) <- authoritative confirm |
|                                                              |
|  Safety mechanisms:                                           |
|  • Numba JIT warm-up (both delta paths)                       |
|  • Circuit breaker enabled                                    |
|  • ob_freeze sensors initialized                              |
|  • Final check 5 min before kickoff                           |
|                                                              |
|  Output: LiveFootballQuantModel instance                      |
+--------------------------------------------------------------+
              |
              v
        [Phase 3: Live Trading Engine Starts]
        (2-layer event detection: Odds-API Live Odds WS + Goalserve Live Score REST)
        (Kalshi order book sync handled by Phase 4)
```
