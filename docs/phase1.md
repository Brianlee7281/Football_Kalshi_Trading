# Phase 1: Offline Calibration — Goalserve + Odds-API

## Overview

This is the stage where all MMPP (Markov-Modulated Poisson Process) parameters are learned from historical data.
If this stage is weak, every downstream live-trading calculation becomes Garbage In, Garbage Out.

Using tens of thousands of matches accumulated from Goalserve (fixtures + stats) and Odds-API (odds),
we break the process of extracting model parameters into five linear steps.

### Unified Data Source

Unify all of Phases 1-4 under a **consistent data source architecture**.
Goalserve handles fixtures, results, and match stats; Odds-API handles all odds data.
This removes schema mismatches and ID-mapping errors.

| Data Source | Role in Phase 1 | Core Data |
|------------------|-----------------|-----------|
| **Goalserve Commentaries** | Interval segmentation + event timeline | goals (minute+VAR), red cards (minute), stoppage time, halftime score, lineups, substitutions |
| **Goalserve Fixtures** | Match discovery + season schedule | match dates, teams, league IDs (used to discover which dates to fetch commentaries for) |
| **Goalserve Live Game Stats** | Team/player stats + xG | per-half team stats, detailed player stats (rating, passes, shots, etc.), xG |
| **Odds-API Historical Events** | Odds features + market baseline | 5 bookmaker odds, ML/Totals/Spread markets. **Available from Dec 2025 onwards only.** |

### Scoring Intensity Function (Final Form)

Home and away teams use **separate gamma and delta parameters**.

$$\lambda_H(t \mid X, \Delta S) = \exp\!\left(a_H + b_{i(t)} + \gamma^H_{X(t)} + \delta_H(\Delta S(t))\right)$$

$$\lambda_A(t \mid X, \Delta S) = \exp\!\left(a_A + b_{i(t)} + \gamma^A_{X(t)} + \delta_A(\Delta S(t))\right)$$

| Symbol | Meaning | Estimated in Step |
|------|------|----------|
| $a_H, a_A$ | Match-level baseline scoring intensity (team strength) | Step 1.3 init -> Step 1.4 correction |
| $b_{i(t)}$ | Time-interval scoring profile | Step 1.4 |
| $\gamma^H_{X(t)}$ | Red-card state -> **home-team** scoring penalty | Step 1.4 |
| $\gamma^A_{X(t)}$ | Red-card state -> **away-team** scoring penalty | Step 1.4 |
| $\delta_H(\Delta S)$ | Score-difference -> home tactical effect | Step 1.4 |
| $\delta_A(\Delta S)$ | Score-difference -> away tactical effect | Step 1.4 |

---

## Input Data

From Goalserve (commentaries + fixtures + stats) and Odds-API (odds):

**1. Commentaries — historical match events (minute-level detail):**

```
GET /getfeed/{api_key}/commentaries/{league_id}?date={DD.MM.YYYY}&json=1
```

- Goals: minute, scorer, team, penalty, own_goal, VAR cancelled
- Red cards: minute, player, team (critical for Q matrix estimation)
- Substitutions: minute, player_in, player_out, team
- Lineups: formation, starting 11
- Stoppage time: addedTime_period1/2
- Match metadata: ht_score, ft_score, status

> **Why commentaries, not fixtures:** The fixtures endpoint provides scores and basic events,
> but does NOT include red card data. The commentaries endpoint provides minute-level events
> including red cards, which are essential for Q matrix estimation (Step 1.2).
> Phase 1 uses `fetch_season_commentaries()` which discovers matchday dates from fixtures,
> then fetches commentaries for each date.

**1b. Fixtures — match discovery and season schedule:**

```
GET /getfeed/{api_key}/soccerfixtures/leagueid/{league_id}?json=1
```

- Used to discover which dates have matches (for commentaries fetching)
- Match metadata: teams, kickoff times, league IDs
- NOT used for interval segmentation (commentaries is the source)

**2. Live Game Stats — detailed historical match stats (100+ leagues):**

```
GET /getfeed/{api_key}/soccerstats/match/{match_id}?json=1
```

- Team stats: `stats.{team}` — shots, passes, possession, corners, fouls, saves (per-half)
- Player stats: `player_stats.{team}.player[]` — rating, goals, assists, shots, passes, tackles, interceptions, minutes_played, etc.
- xG: Expected Goals (included in the Live Game Stats package)

**3. Odds-API Historical Events — 5 bookmakers (Dec 2025+ only):**

```
GET https://api.odds-api.io/v3/events?apiKey={key}&sport=football&league={league_slug}&status=settled&from={date}&to={date}
GET https://api.odds-api.io/v3/odds?apiKey={key}&eventId={id}&bookmakers=Bet365,Betfair Exchange,Sbobet,1xBet,DraftKings
```

- Bookmaker odds: `bookmakers.{name}[].odds[]` (home, draw, away)
- Markets: ML (Match Winner), Totals (Over/Under), Asian Handicap
- 5 bookmakers: Bet365, Betfair Exchange, Sbobet, 1xBet, DraftKings
- **Historical data available from December 2025 onwards only.**
  Pre-Dec-2025 seasons train without odds features (XGBoost handles NaN natively).
  `scripts/odds_backfill.py` accumulates data daily for future retraining.

---

## Step 1.1: Time-Series Event Segmentation and Intervalization (Data Engineering)

### Goal

Convert point events in historical matches into **continuous intervals where lambda is constant**.

Because intensity function lambda depends on $(X(t), \Delta S(t))$,
the interval must be split whenever either variable changes.

### Goalserve Data Mapping

**Goal events — `summary.{team}.goals.player[]`:**

```json
{
  "id": "119",
  "minute": "23",
  "extra_min": "",
  "name": "Lionel Messi",
  "penalty": "True",
  "owngoal": "False",
  "var_cancelled": "False"
}
```

| Field | Usage |
|------|------|
| `minute` + `extra_min` | Goal timestamp (stoppage-time goal: `minute`=90, `extra_min`=3 -> 93rd minute) |
| `{team}` key (localteam/visitorteam) | Identify scoring team -> branch ln lambda_H vs ln lambda_A in NLL |
| `owngoal` | If True, flip scoring team (own goal increments opponent score) |
| **`var_cancelled`** | **If True, fully exclude from interval splitting** |
| `penalty` | Logging/analysis purpose (whether penalty kick) |

> **VAR-cancelled goal handling — an important addition not in the original design:**
> Since goals with `var_cancelled = "True"` did not actually change ΔS,
> they must be completely excluded from interval splitting.
> Ignoring this contaminates ΔS and introduces systematic bias in delta estimation.

**Red card events — `summary.{team}.redcards.player[]`:**

```json
{
  "id": "...",
  "minute": "35",
  "extra_min": "",
  "name": "Player Name"
}
```

| Field | Usage |
|------|------|
| `minute` + `extra_min` | Dismissal timestamp |
| `{team}` key | Team dismissed -> determines X(t) transition direction |

> **Second-yellow dismissal check:**
> During trial period, verify whether `summary.redcards` includes second-yellow dismissals.
> If not, complement via cross-check with `player_stats.{team}.player[].redcards`.

**Stoppage time — `matchinfo.time`:**

```json
{
  "addedTime_period1": "7",
  "addedTime_period2": "8"
}
```

Actual match end time per match:

$$T_m = 90 + \alpha_1 + \alpha_2$$

Since first/second-half stoppage time is provided directly,
ambiguous methods such as "estimating the final play timestamp" are unnecessary.

**Halftime — `{team}.ht_score`:**

```json
"localteam": { "ht_score": "2", "ft_score": "3" },
"visitorteam": { "ht_score": "0", "ft_score": "3" }
```

Because the exact halftime score is known,
ΔS at the halftime boundary can be determined unambiguously.

### Interval Boundary (Split Point) Rules

| Event | Split? | Reason |
|--------|----------|------|
| Goal (`var_cancelled=False`) | Yes | ΔS changes -> delta changes |
| Goal (`var_cancelled=True`) | **No** | ΔS unchanged — cancelled goal |
| Red card | Yes | X(t) changes -> gamma changes |
| Halftime start | Yes | Excluded from integration |
| Halftime end | Yes | Resume integration |
| Match end | Yes | Close interval |
| Yellow card | No | Not included in current state variables |
| **Substitution** | **Yes (extension)** | **Intensity drop detection — see Substitution Effects below** |

### Halftime Handling

During halftime, lambda(t) = 0. If this segment is included in integration,
"a long period with no events" distorts estimation of time profile b_i.

**Effective play-time transform:**

$$t_{eff} = \begin{cases} t & \text{if } t < 45 + \alpha_1 \\ t - \delta_{HT} & \text{if } t \geq 45 + \alpha_1 + \delta_{HT} \end{cases}$$

$\alpha_1$: first-half stoppage time (`addedTime_period1`),
$\delta_{HT}$: halftime break length (about 15 min).

Mark halftime as a separate flag and fully exclude it from NLL integration.

### Dual Role of Goals

due to the introduction of delta(ΔS), goals have two simultaneous roles:

1. **Interval boundary:** a new interval starts right after the goal; new interval delta uses post-goal ΔS.
2. **Point event:** contributes to the NLL term Σ ln lambda(t_i). Here, delta must use **pre-goal** ΔS.

> **Causality caution:** when home scores from 0-0,
> applying delta(+1) to that goal's lambda contribution reflects "scoring power while already ahead,"
> which reverses causality. For the goal-time NLL contribution, always use **pre-goal** ΔS.

### Own Goal Handling

For `owngoal = "True"`, the recorded team and actual scoring team are opposite:

```python
def resolve_scoring_team(goal_event, recorded_team):
    """Flip scoring team for own goals."""
    if goal_event["owngoal"] == "True":
        return "visitorteam" if recorded_team == "localteam" else "localteam"
    return recorded_team
```

Because own goals are exogenous stochastic events rather than intentional attacking outcomes,
it is ambiguous which team's ln lambda should receive point-event credit in NLL.

**Policy:**
- Exclude own goals from point-event term (Σ ln lambda).
- Keep them in interval integration term (Σ mu_k), since ΔS still changes in reality.
- In short, own goals are treated as "events that change score but prove neither team's scoring intensity."

> **Rationale:** lambda_H models "intensity of intentional home scoring."
> An away defender's own goal is not part of this intensity.
> Including own goals in ln lambda_H biases lambda_H upward and inflates a_H.

### Data Transformation Example

**Goalserve raw data (2022 World Cup Final):**

```json
"matchinfo": {
  "time": { "addedTime_period1": "7", "addedTime_period2": "8" }
},
"localteam": { "name": "Argentina", "ht_score": "2", "ft_score": "3" },
"visitorteam": { "name": "France", "ht_score": "0", "ft_score": "3" },
"summary": {
  "localteam": {
    "goals": {
      "player": [
        {"minute": "23", "name": "Messi", "penalty": "True", "var_cancelled": "False"},
        {"minute": "36", "name": "Di María", "penalty": "False", "var_cancelled": "False"},
        {"minute": "108", "name": "Messi", "penalty": "False", "var_cancelled": "False"}
      ]
    },
    "redcards": null
  },
  "visitorteam": {
    "goals": {
      "player": [
        {"minute": "80", "name": "Mbappé", "penalty": "True", "var_cancelled": "False"},
        {"minute": "81", "name": "Mbappé", "penalty": "False", "var_cancelled": "False"},
        {"minute": "118", "name": "Mbappé", "penalty": "True", "var_cancelled": "False"}
      ]
    },
    "redcards": null
  }
}
```

**Transformed output (T_m = 90 + 7 + 8 = 105, with extra time extending to 120+):**

| Interval | Time Range | X | ΔS | delta | Point Event | Scoring Team |
|------|----------|---|-----|---|----------|---------|
| 1 | [0, 23) | 0 | 0 | delta(0)=0 | — | — |
| 2 | [23, 36) | 0 | +1 | delta(+1) | t=23, delta_before=delta(0) | **Home** |
| 3 | [36, 45+7) | 0 | +2 | delta(+2) | t=36, delta_before=delta(+1) | **Home** |
| — | HT | — | — | — | **Halftime: excluded from integration** | — |
| 4 | [HT_end, 80) | 0 | +2 | delta(+2) | — | — |
| 5 | [80, 81) | 0 | +1 | delta(+1) | t=80, delta_before=delta(+2) | **Away** |
| 6 | [81, 90+8) | 0 | 0 | delta(0)=0 | t=81, delta_before=delta(+1) | **Away** |
| ... | Extra time continues | | | | | |

### Interval Record Schema

```python
@dataclass
class RedCardTransition:
    minute: float           # Dismissal timestamp
    team: str               # "localteam" / "visitorteam"
    from_state: int         # Markov state X before this red card
    to_state: int           # Markov state X after this red card

@dataclass
class IntervalRecord:
    match_id: str           # Goalserve match ID (unified across all phases)
    t_start: float          # Interval start (effective play time)
    t_end: float            # Interval end
    state_X: int            # Markov state {0,1,2,3}
    delta_S: int            # Score difference (home - away)
    home_goal_times: list   # Home goal timestamps in this interval
    away_goal_times: list   # Away goal timestamps in this interval
    goal_delta_before: list # Pre-goal ΔS for each goal
    T_m: float              # Actual match end time
    is_halftime: bool       # Whether this interval is halftime
    alpha_1: float          # First-half stoppage time (addedTime_period1)
    alpha_2: float          # Second-half stoppage time (addedTime_period2)
    red_card_transitions: list  # List[RedCardTransition] — red cards occurring in this interval
```

### ETL Pipeline

```python
def build_intervals_from_goalserve(match_data: dict) -> List[IntervalRecord]:
    """Goalserve Commentaries -> interval record transformation."""

    # 1. Extract stoppage time
    alpha_1 = float(match_data["matchinfo"]["time"]["addedTime_period1"] or 0)
    alpha_2 = float(match_data["matchinfo"]["time"]["addedTime_period2"] or 0)
    T_m = 90 + alpha_1 + alpha_2

    # 2. Collect events + VAR-cancelled filtering
    events = []

    for team_key in ["localteam", "visitorteam"]:
        goals = match_data["summary"][team_key].get("goals", {})
        if goals:
            for g in ensure_list(goals.get("player", [])):
                if g.get("var_cancelled") == "True":
                    continue  # Exclude VAR-cancelled goals

                scoring_team = resolve_scoring_team(g, team_key)
                minute = parse_minute(g["minute"], g.get("extra_min", ""))
                events.append(Event("goal", minute, scoring_team, g))

        redcards = match_data["summary"][team_key].get("redcards", {})
        if redcards:
            for r in ensure_list(redcards.get("player", [])):
                minute = parse_minute(r["minute"], r.get("extra_min", ""))
                events.append(Event("red_card", minute, team_key, r))

    # Add halftime boundaries
    events.append(Event("halftime_start", 45 + alpha_1, None, None))
    events.append(Event("halftime_end", 45 + alpha_1 + 15, None, None))  # ~15 min break
    events.append(Event("match_end", T_m, None, None))

    # 3. Sort by time and split into intervals
    events.sort(key=lambda e: e.minute)
    intervals = split_into_intervals(events, T_m, alpha_1, alpha_2)

    return intervals
```

### Substitution Effects (Extension)

Goalserve provides substitution data in `substitutions.{team}.substitution[]`:

```json
{
  "player_in": { "id": "404462", "name": "Lautaro Martínez" },
  "player_out": { "id": "119", "name": "Lionel Messi" },
  "minute": "72",
  "extra_min": ""
}
```

**Approach A — Empirical intensity drop detection (feasible now):**

Split intervals on substitution events and estimate a pre/post-substitution intensity ratio.
Substitutions cluster in the 55-75 minute window and often coincide with
tactical shifts (e.g., parking the bus when ahead). By splitting intervals at
substitution times, we can measure whether intensity drops systematically
after substitution windows.

```python
# Add substitution events to interval splitting
for team_key in ["localteam", "visitorteam"]:
    subs = match_data.get("substitutions", {}).get(team_key, {})
    if subs:
        for s in ensure_list(subs.get("substitution", [])):
            minute = parse_minute(s["minute"], s.get("extra_min", ""))
            events.append(Event("substitution", minute, team_key, s))
```

The `IntervalRecord` gains a field `n_subs_so_far: int` tracking cumulative
substitutions at each interval boundary. In Step 1.4, an optional
intensity modifier $\psi_{sub}$ can be estimated:

$$\lambda(t) \to \lambda(t) \cdot \exp(\psi_{sub} \cdot n_{sub}(t))$$

where $\psi_{sub} < 0$ captures average intensity decline per substitution.

> **Approach B (quality drop)** requires external player ratings (Wyscout, Understat)
> to weight the substitution impact by quality difference between outgoing and incoming
> players. This is **not feasible** with current Goalserve data alone, since post-match
> player ratings are only available after the match, not pre-match for bench players.

### Stoppage Time Distribution (Extension)

Goalserve provides `matchinfo.time.addedTime_period1` and `addedTime_period2`
for every historical match. These fields enable fitting a per-league stoppage
time distribution for use in Phase 3 MC simulation.

```python
def fit_stoppage_distribution(matches: List[dict], league_id: str) -> dict:
    """Fit per-league stoppage time distribution from historical data.

    Collect addedTime_period1/2 across all matches in the league,
    fit a Log-Normal or Gamma distribution.
    """
    alpha_1s = []
    alpha_2s = []
    for m in matches:
        if m.get("league_id") != league_id:
            continue
        a1 = float(m["matchinfo"]["time"].get("addedTime_period1") or 0)
        a2 = float(m["matchinfo"]["time"].get("addedTime_period2") or 0)
        if a1 > 0:
            alpha_1s.append(a1)
        if a2 > 0:
            alpha_2s.append(a2)

    from scipy.stats import lognorm
    # Fit log-normal (better fit for right-skewed stoppage times)
    shape_1, loc_1, scale_1 = lognorm.fit(alpha_1s, floc=0)
    shape_2, loc_2, scale_2 = lognorm.fit(alpha_2s, floc=0)

    return {
        "period_1": {"shape": shape_1, "scale": scale_1},
        "period_2": {"shape": shape_2, "scale": scale_2},
        "E_alpha_1": np.mean(alpha_1s),
        "E_alpha_2": np.mean(alpha_2s),
    }
```

**Output of stoppage time fitting:**

| Parameter | Shape | Usage |
|-----------|-------|-------|
| $\alpha_1 \sim \text{LogNormal}(\mu_1, \sigma_1)$ | Per-league | MC simulation: sample $T_{end}$ per path |
| $\alpha_2 \sim \text{LogNormal}(\mu_2, \sigma_2)$ | Per-league | MC simulation: sample $T_{end}$ per path |
| $\mathbb{E}[\alpha_1], \mathbb{E}[\alpha_2]$ | Per-league | Phase 2: deterministic $T_{exp}$ |

In Phase 3 Step 3.4 MC, instead of using a fixed $T_{end} = 90 + \mathbb{E}[\alpha_1] + \mathbb{E}[\alpha_2]$,
each MC path samples its own stoppage time from the fitted distribution,
naturally propagating uncertainty about match duration into probability estimates.

### Output

Tens of thousands of matches are transformed into hundreds of thousands of interval records.
Every record is tagged with Goalserve `match_id`, enabling lookup by the same ID in later phases.

Additionally, per-league stoppage time distributions and (optionally)
substitution-interval splits are produced.

---

## Step 1.2: Estimating Markov Chain Generator Matrix Q (Empirical + Shrinkage)

### Goal

Estimate red-card transition rates from historical data and construct a 4x4 generator matrix Q.

### State Space

| State | Meaning |
|------|------|
| 0 | 11v11 (normal) |
| 1 | 10v11 (home sent off) |
| 2 | 11v10 (away sent off) |
| 3 | 10v10 (both teams sent off) |

### Goalserve Data Mapping

**Red-card timeline — Fixtures/Results `summary.{team}.redcards`:**

Since each red card has exact minute and team (localteam/visitorteam),
the match-level Markov state path can be fully reconstructed:

```python
def reconstruct_markov_path(match_data: dict) -> List[Tuple[float, int]]:
    """Reconstruct match Markov state path: [(time, state), ...]."""
    path = [(0, 0)]  # Kickoff: state 0 (11v11)
    current_state = 0

    red_events = []
    for team_key in ["localteam", "visitorteam"]:
        redcards = match_data["summary"][team_key].get("redcards", {})
        if redcards:
            for r in ensure_list(redcards.get("player", [])):
                minute = parse_minute(r["minute"], r.get("extra_min", ""))
                red_events.append((minute, team_key))

    red_events.sort(key=lambda x: x[0])

    for minute, team in red_events:
        if team == "localteam":
            if current_state == 0: current_state = 1      # 11v11 -> 10v11
            elif current_state == 2: current_state = 3    # 11v10 -> 10v10
        else:
            if current_state == 0: current_state = 2      # 11v11 -> 11v10
            elif current_state == 1: current_state = 3    # 10v11 -> 10v10
        path.append((minute, current_state))

    return path
```

**Cross-check — Live Game Stats `player_stats.{team}.player[].redcards`:**

Compare `summary.redcards` from Fixtures/Results against player-level redcard fields from Live Game Stats
to detect missing data (especially second-yellow dismissals).

### Baseline Estimator

$$q_{ij} = \frac{N_{ij}}{\sum_m \int_0^{T_m} \mathbb{1}_{\{X_m(t) = i\}}\, dt}$$

- Numerator $N_{ij}$: number of observed i -> j transitions across all data
- Denominator: total **effective play time** spent in state i across all matches
  - Halftime excluded
  - Match-specific $T_m = 90 + \alpha_1 + \alpha_2$ (from Goalserve `addedTime`)
- Diagonal terms: $q_{ii} = -\sum_{j \neq i} q_{ij}$

### Sparse-State Handling (State 3: 10v10)

Additivity assumption:

$$q_{1 \to 3} \approx q_{0 \to 2}, \quad q_{2 \to 3} \approx q_{0 \to 1}$$

Scoring penalties are also additive by team:

$$\gamma^H_3 = \gamma^H_1 + \gamma^H_2, \quad \gamma^A_3 = \gamma^A_1 + \gamma^A_2$$

### League-Stratified Estimation

Because Goalserve Commentaries covers major leagues with minute-level event data, there is enough data to estimate league-specific Q.

- **Option A — independent Q per league:** independent estimates for Kalshi-tradable leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Brasileirão, Liga Argentina)
- **Option B — hierarchical Bayesian:** use all leagues as a prior pool and update each league posterior; better for low-data leagues

### Q_off Normalization (for MC simulation)

In Phase 3 Step 3.4 Monte Carlo, to decide "which state transition occurs when a dismissal event happens,"
the off-diagonal entries of Q must be normalized into **transition probabilities**:

```python
Q_off_normalized = np.zeros((4, 4))
for i in range(4):
    total_off_diag = -Q[i, i]  # = Σ_{j≠i} Q[i,j]
    if total_off_diag > 0:
        for j in range(4):
            if i != j:
                Q_off_normalized[i, j] = Q[i, j] / total_off_diag
```

This normalization is executed in Phase 2 Step 2.5,
but documented as a Phase 1 deliverable alongside Q.

### State-Dependent Q(ΔS) — Score-Conditioned Transition Rates (Extension)

The interval records from Step 1.1 already carry `delta_S` (score difference).
Instead of computing a single global Q, stratify by ΔS bin before aggregation
to capture the empirical observation that teams behave differently when leading vs trailing
(e.g., a trailing team may play more aggressively, increasing red card risk).

**ΔS bins for Q stratification:**

| Bin | ΔS Range | Interpretation |
|-----|----------|----------------|
| 0 | ΔS ≤ -2 | Home trailing heavily |
| 1 | ΔS = -1 | Home trailing slightly |
| 2 | ΔS = 0 | Level |
| 3 | ΔS = +1 | Home slightly ahead |
| 4 | ΔS ≥ +2 | Home comfortably ahead |

```python
def estimate_Q_by_delta_S(intervals: List[IntervalRecord]) -> Dict[int, np.ndarray]:
    """Estimate separate Q matrices per ΔS bin.

    Returns {ds_bin: Q_4x4} where ds_bin ∈ {0,1,2,3,4}.
    """
    DS_BINS = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}

    def ds_to_bin(ds: int) -> int:
        if ds <= -2: return 0
        if ds >= 2: return 4
        return DS_BINS[ds]

    # Accumulate per-bin
    N = {b: np.zeros((4, 4)) for b in range(5)}
    T = {b: np.zeros(4) for b in range(5)}

    for iv in intervals:
        if iv.is_halftime:
            continue
        b = ds_to_bin(iv.delta_S)
        state = iv.state_X
        duration = iv.t_end - iv.t_start
        T[b][state] += duration

        # Count transitions (red cards in this interval)
        for rc in iv.red_card_transitions:
            N[b][rc.from_state, rc.to_state] += 1

    Q_by_ds = {}
    for b in range(5):
        Q = np.zeros((4, 4))
        for i in range(4):
            if T[b][i] > 0:
                for j in range(4):
                    if i != j:
                        Q[i, j] = N[b][i, j] / T[b][i]
            Q[i, i] = -sum(Q[i, j] for j in range(4) if j != i)
        Q_by_ds[b] = Q

    return Q_by_ds
```

**Sparsity concern and hierarchical shrinkage:**

Red cards are already rare (~0.04 per match). Stratifying by 5 ΔS bins means
~1/5 the data per Q matrix. To prevent noisy estimates, apply hierarchical
shrinkage toward the global (pooled) Q:

$$Q_{\Delta S}^{shrunk} = w \cdot Q_{\Delta S}^{empirical} + (1 - w) \cdot Q^{global}$$

where $w = \min\!\left(1,\; \frac{T_{\Delta S}}{T_{threshold}}\right)$ and
$T_{threshold}$ is a minimum dwell-time threshold (e.g., 5000 match-minutes).

**In Phase 3:** the live engine selects the appropriate $Q_{\Delta S}$ based on the
current score difference, using it for matrix exponential and MC transition probabilities.
This requires storing all 5 Q matrices (or the global Q + per-bin deltas) in the
production parameter set.

### Output

Generator matrix Q (4x4) satisfying diagonal condition $q_{ii} = -\sum_{j \neq i} q_{ij}$.
League-specific or pooled. Used in Phase 3 for matrix exponential $e^{Q \cdot \Delta t}$.

**Extension:** optionally, a dict of 5 Q matrices keyed by ΔS bin
$\{Q_{\Delta S}\}$ for state-dependent transition modeling.

---

## Step 1.3: Learning Prematch Prior Parameter a (Machine Learning)

### Goal

Provide **initial estimates** of baseline intensity that reflect match-level strength difference.
These values initialize Step 1.4 joint optimization; final a is determined by NLL.

### Data Prerequisite: Stats & Odds Backfill

Step 1.3 requires per-match feature data (team stats, player stats, pregame odds) stored
in the `historical_matches` table columns: `stats`, `player_stats`, `odds`.

**If these columns are empty, the pipeline falls back to league-average `a_H`/`a_A` for all matches,
which eliminates team-strength differentiation and prevents the model from beating market baselines.**

Before running the full ML prior, backfill feature data using the data collector:

```bash
# Backfill match stats (shots, possession, xG, player stats via Goalserve commentaries endpoint)
python -m src.data.collector --backfill-stats --config config/system.yaml

# Backfill pregame odds (5 bookmakers per match via Odds-API historical endpoint)
python -m src.data.collector --backfill-odds --config config/system.yaml
```

| Data | Source | DB Column | Estimated Time (7,000 matches) |
|------|--------|-----------|-------------------------------|
| Match stats | Goalserve `commentaries/match?id={id}&league={league}` | `stats`, `player_stats` | ~2 hours (1 req/sec) |
| Pregame odds | Odds-API `v4/historical/sports/{sport}/odds/?date={date}` | `odds` | ~1 hour (grouped by date) |

**Verify after backfill:**
```sql
SELECT
  COUNT(*) FILTER (WHERE stats IS NOT NULL AND stats != '{}'::jsonb) AS has_stats,
  COUNT(*) FILTER (WHERE odds IS NOT NULL AND odds != '{}'::jsonb) AS has_odds
FROM historical_matches;
```

Coverage targets: **>80%** of matches should have stats, **>70%** should have odds.
Older matches (pre-2021) may have limited coverage from Odds-API historical endpoint.

### Feature Architecture — 3-Tier Structure

Build features with four tiers from Goalserve (Tiers 1-2) and Odds-API (Tier 3):

#### Tier 1: Team-Level Rolling Stats

**Source: Goalserve Live Game Stats — `stats.{team}`**

Aggregate rolling averages over each team's last five matches:

| Feature | Goalserve Field | Calculation |
|------|---------------|------|
| xG_per_90 | Live Game Stats xG field | xG / number of matches |
| xGA_per_90 | opponent xG | conceded-threat proxy |
| shots_per_90 | `stats.shots.total` | total shot frequency |
| shots_on_target_per_90 | `stats.shots.ongoal` | on-target shot frequency |
| shots_insidebox_ratio | `stats.shots.insidebox / stats.shots.total` | box penetration rate |
| possession_avg | `stats.possestiontime.total` | possession |
| pass_accuracy | `stats.passes.accurate / stats.passes.total` | passing accuracy |
| corners_per_90 | `stats.corners.total` | corner frequency |
| fouls_per_90 | `stats.fouls.total` | foul frequency (aggression proxy) |
| saves_per_90 | `stats.saves.total` | GK save frequency |

**Per-half split features (optional extension):**

Goalserve provides first/second-half split stats via `_h1`, `_h2` suffix:

| Feature | Meaning |
|------|------|
| shots_h2_ratio | second-half shot share -> stamina/tactical-change proxy |
| possession_h1_vs_h2 | first-vs-second-half possession gap -> game-management pattern |

#### Tier 2: Player-Level Aggregated Features

**Source: Goalserve Live Game Stats — `player_stats.{team}.player[]`**

Aggregate recent five-match stats of today's starting XI (confirmed in Phase 2) by position:

```python
def build_player_tier_features(starting_11_ids: List[str],
                                player_history: Dict) -> dict:
    """
    Historical stats of starting XI -> team-level aggregates.
    player_history: {player_id: [recent 5 matches of player_stats]}
    """
    features = {}

    for pos_group, pos_codes in [
        ("fw", ["F"]),
        ("mf", ["M"]),
        ("df", ["D"]),
        ("gk", ["G"])
    ]:
        players_in_group = [
            pid for pid in starting_11_ids
            if player_history[pid][0]["pos"] in pos_codes
        ]

        if not players_in_group:
            continue

        # Rolling metrics by position group
        ratings = []
        goals_p90 = []
        key_passes_p90 = []
        tackles_p90 = []

        for pid in players_in_group:
            for game_stats in player_history[pid]:
                mp = float(game_stats.get("minutes_played") or 0)
                if mp < 10:
                    continue  # Exclude too-short appearances

                ratings.append(float(game_stats.get("rating") or 0))
                goals_p90.append(
                    safe_float(game_stats.get("goals")) / mp * 90
                )
                key_passes_p90.append(
                    safe_float(game_stats.get("keyPasses")) / mp * 90
                )
                tackles_p90.append(
                    safe_float(game_stats.get("tackles")) / mp * 90
                )

        features[f"{pos_group}_avg_rating"] = safe_mean(ratings)
        features[f"{pos_group}_goals_p90"] = safe_sum(goals_p90)
        features[f"{pos_group}_key_passes_p90"] = safe_sum(key_passes_p90)
        features[f"{pos_group}_tackles_p90"] = safe_sum(tackles_p90)

    return features
```

**Core player aggregate features:**

| Feature | Position | Calculation | Meaning |
|------|--------|------|------|
| fw_avg_rating | FW | rolling mean rating | current attacking form |
| fw_goals_p90 | FW | sum(goals / minutes * 90) | attack scoring productivity |
| mf_key_passes_p90 | MF | sum(keyPasses / minutes * 90) | creativity |
| mf_pass_accuracy | MF | mean(passes_acc / passes) | build-up quality |
| df_tackles_p90 | DF | sum(tackles / minutes * 90) | defensive intensity |
| df_interceptions_p90 | DF | sum(interceptions / minutes * 90) | defensive positioning |
| gk_save_rate | GK | saves / (saves + goals_conceded) | GK performance |
| team_avg_rating | all players | minutes-weighted mean rating | overall team form |

> **minutes_played caution:** in Goalserve, unused bench players have empty `minutes_played`.
> Those entries should be excluded from rolling averages. Short substitute appearances (`mp < 10`) are also excluded as statistically unstable.

#### Tier 3: Odds Features

**Source: Odds-API Historical Odds — 5 bookmakers**

```python
def build_odds_features(bookmakers: List[dict]) -> dict:
    """5 bookmaker odds (Odds-API format) -> feature vector.

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

    # If Betfair Exchange is unavailable, use market average
    if exchange_prob is None:
        exchange_prob = tuple(np.mean(all_probs, axis=0))

    return {
        "exchange_home_prob": exchange_prob[0],
        "exchange_draw_prob": exchange_prob[1],
        "exchange_away_prob": exchange_prob[2],
        "market_avg_home_prob": np.mean([p[0] for p in all_probs]),
        "market_avg_draw_prob": np.mean([p[1] for p in all_probs]),
        "bookmaker_odds_std": np.std([p[0] for p in all_probs]),
    }
```

| Feature | Meaning |
|------|------|
| exchange_home/draw/away_prob | implied probabilities from most efficient market |
| market_avg_home_prob | consensus probability across 5 bookmakers |
| bookmaker_odds_std | market uncertainty (higher std = harder match) |
| odds_movement | pre-kickoff information-flow direction |

#### Tier 4: Context Features

| Feature | Source | Calculation |
|------|------|------|
| home_away_flag | Fixtures | localteam/visitorteam indicator |
| rest_days | Fixtures date diff | days since previous match |
| h2h_goal_diff | Fixtures H2H | mean goal difference over last 5 H2H |

### Feature Selection

Using XGBoost built-in feature importance (gain):

$$\text{Importance}(f) = \sum_{\text{splits on } f} \Delta \mathcal{L}_{\text{Poisson}}$$

Select top d' features that reach 95% cumulative importance,
and store in `feature_mask.json`. Apply the same mask in Phase 2 inference.

> **Why not PCA:** PCA is linear projection and can mismatch with nonlinear models (XGBoost).
> XGBoost's Poisson-deviance importance aligns better with objective.

### Target (y)

Each team's **total goals** in that match (including stoppage-time goals, excluding VAR-cancelled goals).
Use either separate home/away models or a single model with home/away flag.

### Modeling

- XGBoost / LightGBM, objective: `count:poisson`
- Output: full-match expected goals $\hat{\mu}_H, \hat{\mu}_A$ for each team

### Converting to Initial a

$$a_H^{(init)} = \ln\!\left(\frac{\hat{\mu}_H}{T_m}\right), \quad a_A^{(init)} = \ln\!\left(\frac{\hat{\mu}_A}{T_m}\right)$$

This is an initial estimate under constant-intensity assumption; corrected jointly with b in Step 1.4.

### Output

- Trained XGBoost weights (`.xgb`)
- `feature_mask.json`
- Predictor that outputs $\hat{\mu}_H, \hat{\mu}_A$ for a new match feature vector

### Fallback by Feature Availability

Depending on Live Game Stats historical coverage:

| Coverage | Tier 2 (player-level) strategy |
|----------|----------------------|
| 5+ seasons | Apply Tier 2 over full period |
| 2-4 seasons | Tier 2 only for recent period, Tier 1 only elsewhere |
| <=1 season | Disable Tier 2, train with Tier 1 + Tier 3 only |

During trial period, verify backfill coverage of `player_stats`.

---

## Step 1.4: Joint NLL Optimization (MMPP Calibration)

### Goal

Jointly optimize time profile, red-card penalty, score-difference effect, and match-level baseline intensity.

### Resolving Circular Dependency

Correct conversion for a needs b:

$$a = \ln\!\left(\frac{\hat{\mu} \cdot b}{e^{bT} - 1}\right) \quad \leftarrow \text{requires b (circular reference)}$$

**Solution:** treat a as learnable parameters instead of fixed constants,
and jointly minimize NLL with b, gamma, delta.
Add regularization pulling a toward ML predictions to prevent overfitting.

### Time Basis Functions (Piecewise Basis)

$$\sum_{i=1}^{K} b_i \cdot B_i(t), \quad K = 6$$

| i | $B_i(t)$ | Covered interval |
|---|----------|----------|
| 1 | $\mathbb{1}_{[0, 15)}(t)$ | early first half |
| 2 | $\mathbb{1}_{[15, 30)}(t)$ | mid first half |
| 3 | $\mathbb{1}_{[30, 45+\alpha_1)}(t)$ | late first half + first-half stoppage |
| 4 | $\mathbb{1}_{[HT_{end}, HT_{end}+15)}(t)$ | early second half |
| 5 | $\mathbb{1}_{[HT_{end}+15, HT_{end}+30)}(t)$ | mid second half |
| 6 | $\mathbb{1}_{[HT_{end}+30, T_m)}(t)$ | late second half + second-half stoppage |

Since halftime is included in no basis function, it is automatically excluded from integration.

> **If using $t_{eff}$ transform:** basis can be simplified to $B_i = \mathbb{1}_{[15(i-1), 15i)}$.

**Sanity check for b via per-half stats (Goalserve-specific advantage):**

Goalserve Live Game Stats provides `_h1`, `_h2` split shots/passes.
Check whether learned first-vs-second-half weight in b[1..6] roughly matches empirical shot split:

$$\frac{\exp(b_1) + \exp(b_2) + \exp(b_3)}{\sum_{i=1}^{6} \exp(b_i)} \approx \frac{\text{shots.total\_h1}}{\text{shots.total}} \quad \text{(league average)}$$

### Red-Card Penalty gamma — Team-Specific Separation

Home and away teams use **separate gamma**.
A red card harms one side and benefits the other, so shared gamma is invalid.

**Home-team gamma^H:**

$$\gamma^H = [0,\; \gamma^H_1,\; \gamma^H_2,\; \gamma^H_1 + \gamma^H_2]$$

| State | $\gamma^H$ | Physical interpretation |
|------|-----------|------------|
| 0 (11v11) | 0 | reference point |
| 1 (home dismissed) | $\gamma^H_1 < 0$ | home numerical disadvantage -> home scoring **decreases** |
| 2 (away dismissed) | $\gamma^H_2 > 0$ | home numerical advantage -> home scoring **increases** |
| 3 (both dismissed) | $\gamma^H_1 + \gamma^H_2$ | additive composition |

**Away-team gamma^A:**

$$\gamma^A = [0,\; \gamma^A_1,\; \gamma^A_2,\; \gamma^A_1 + \gamma^A_2]$$

| State | $\gamma^A$ | Physical interpretation |
|------|-----------|------------|
| 0 (11v11) | 0 | reference point |
| 1 (home dismissed) | $\gamma^A_1 > 0$ | away numerical advantage -> away scoring **increases** |
| 2 (away dismissed) | $\gamma^A_2 < 0$ | away numerical disadvantage -> away scoring **decreases** |
| 3 (both dismissed) | $\gamma^A_1 + \gamma^A_2$ | additive composition |

**Optional symmetry constraints:**

$$\gamma^A_1 = -\gamma^H_2, \quad \gamma^A_2 = -\gamma^H_1$$

Free parameters: 4 (asymmetric) or 2 (symmetric). Compare empirically in Step 1.5.

### Position-Specific Gamma (Extension)

The impact of a red card depends on the dismissed player's position.
Losing a defender (DEF/GK) likely increases opponent scoring more than losing
a forward (MID/FWD). Goalserve provides the data to estimate this:

1. **Red card player name:** `summary.{team}.redcards.player[].name`
2. **Lineup positions:** `teams.{team}.player[].pos` (G/D/M/F)

**Name join:** match red card player name to lineup player name to get position.
May require fuzzy matching (e.g., accent normalization, short name variants).

**Position buckets:** collapse to 2 groups to keep parameters estimable:

| Bucket | Positions | Rationale |
|--------|-----------|-----------|
| DEF+GK | G, D | Structural loss — defensive shape compromised |
| MID+FWD | M, F | Tactical loss — attacking threat reduced |

**Multiplicative modifier:** instead of expanding the 4-state Markov chain,
apply a position modifier to gamma:

$$\gamma^H_{X,pos} = \gamma^H_X \cdot (1 + \phi_{pos})$$

where $\phi_{DEF} > 0$ (amplifies penalty for losing a defender) and
$\phi_{MID} < 0$ (reduces penalty for losing an attacker).

Free parameters: 2 additional ($\phi_{DEF}, \phi_{MID}$).

> **Sparsity warning:** GK red cards are very rare (<1% of all reds).
> Collapsing GK with DEF is essential. Even with 10K+ matches,
> the DEF+GK bucket will have ~60% of reds and MID+FWD ~40%.

### Opponent Quality in Gamma and Delta (Extension)

Goalserve Live Game Stats provides `xG` per match. The rolling
`xG_per_90` from Step 1.3 Tier 1 features can stratify gamma and delta
by opponent pre-match quality:

| Quality Bucket | Criterion | Matches |
|----------------|-----------|---------|
| Strong | Opponent xG_per_90 ≥ top 33% | ~33% |
| Average | Middle 33% | ~33% |
| Weak | Opponent xG_per_90 ≤ bottom 33% | ~33% |

**For gamma:** a red card against a strong opponent has larger impact
(they exploit numerical advantage better). Apply a quality modifier:

$$\gamma_{adj} = \gamma \cdot (1 + \eta_{quality})$$

**For delta:** trailing against a strong team forces different urgency
than trailing against a weak team. Apply a quality-conditioned shift:

$$\delta_{adj}(s) = \delta(s) + \rho_{quality} \cdot s$$

> **Sparsity warning:** adding a 3-level quality stratification on top of
> existing parameters multiplies effective parameter count. Start with a
> single multiplicative modifier per quality bucket (2 additional parameters)
> rather than fully independent gamma/delta per quality tier.

### Score-Dependent Intensity delta(ΔS) — Parametric Smoothing

Instead of a raw lookup table with 4 free parameters per team (which suffers from
noise at extreme ΔS values due to data sparsity), delta is parameterized as a
smooth curve with 3 coefficients per team:

$$\delta(s) = \beta \cdot s + \kappa \cdot \text{sign}(s) \cdot \left(1 - e^{-|s|/\tau}\right)$$

**Properties:**
- $\delta(0) = 0$ by construction (reference point preserved)
- Linear at small $|s|$: $\delta \approx (\beta + \kappa/\tau) \cdot s$
- Saturates at large $|s|$: $\delta \to \beta \cdot s + \kappa \cdot \text{sign}(s)$
- $\tau$ controls how quickly saturation kicks in

| Parameter | Range | Meaning |
|-----------|-------|---------|
| $\beta_H, \beta_A$ | [-0.5, 0.5] | Linear slope (score sensitivity) |
| $\kappa_H, \kappa_A$ | [-1.0, 1.0] | Saturation magnitude |
| $\tau_H, \tau_A$ | [0.1, 5.0] | Saturation rate (must be positive) |

Free parameters: home 3 + away 3 = **6** (down from 8 in the lookup approach).

**The parametric curve generates a 5-element lookup table** at the boundary
between calibration and runtime:

| Index | ΔS | Value |
|-------|-----|-------|
| 0 | ≤ -2 | $\delta(-2)$ |
| 1 | -1 | $\delta(-1)$ |
| 2 | 0 | 0 (fixed) |
| 3 | +1 | $\delta(+1)$ |
| 4 | ≥ +2 | $\delta(+2)$ |

This lookup is consumed by `mc_core.py` (Numba JIT) and pricing modules
**without any changes** — they receive the same `np.ndarray` shape (5,).

**Advantages over raw lookup:**
- Eliminates noise at ΔS = ±2 (few training samples)
- Automatically enforces monotonicity if $\beta$ and $\kappa$ have consistent signs
- Reduces overfitting risk with fewer parameters
- L2 regularization on $\beta$ and $\kappa$ (not $\tau$) shrinks toward zero effect

### Interval Integral (Closed-Form) — Home/Away Separate

In interval k where $(X_k, \Delta S_k)$ is constant and basis index is $i_k$:

$$\mu^H_k = \exp\!\left(a^m_H + b_{i_k} + \gamma^H_{X_k} + \delta_H(\Delta S_k)\right) \cdot (t_k - t_{k-1})$$

$$\mu^A_k = \exp\!\left(a^m_A + b_{i_k} + \gamma^A_{X_k} + \delta_A(\Delta S_k)\right) \cdot (t_k - t_{k-1})$$

### Point-Event Contribution (Goal Times) — Home/Away Separate

Home goal: $\ln \lambda_H(t_g) = a^m_H + b_{i(t_g)} + \gamma^H_{X(t_g)} + \delta_H(\Delta S_{before,g})$

Away goal: $\ln \lambda_A(t_g) = a^m_A + b_{i(t_g)} + \gamma^A_{X(t_g)} + \delta_A(\Delta S_{before,g})$

> **Own goals:** excluded from point-event terms (per Step 1.1 policy).
> Included in interval integral through ΔS updates.

### Loss Function (Final NLL)

$$\mathcal{L} = \underbrace{-\sum_{m=1}^{M}\Bigg[\sum_{g \in \text{HomeGoals}_m} \ln \lambda_H(t_g) + \sum_{g \in \text{AwayGoals}_m} \ln \lambda_A(t_g) - \sum_{k \in \text{Intervals}_m} \left(\mu^H_k + \mu^A_k\right)\Bigg]}_{\text{Negative Log-Likelihood}}$$

$$+ \underbrace{\frac{1}{2\sigma_a^2}\sum_{m=1}^M \left[(a^m_H - a^{m,(init)}_H)^2 + (a^m_A - a^{m,(init)}_A)^2\right]}_{\text{ML Prior Regularization}}$$

$$+ \underbrace{\lambda_{reg}\left(\|\mathbf{b}\|^2 + \|\boldsymbol{\gamma}^H\|^2 + \|\boldsymbol{\gamma}^A\|^2 + \|\boldsymbol{\delta}_H\|^2 + \|\boldsymbol{\delta}_A\|^2\right)}_{\text{L2 Regularization}}$$

> **Exclude own goals from HomeGoals/AwayGoals.**
> **Exclude VAR-cancelled goals.** `var_cancelled=True` goals are already filtered in Step 1.1.

> **σ_a tuning:** $\sigma_a$ controls the balance between ML prior fidelity and NLL fit.
> Too large $\sigma_a$ lets 2M match-level parameters overfit to sparse goal counts
> (avg 2-3 goals per match), starving structural parameters (b, gamma, delta) of learning signal.
> Too small $\sigma_a$ prevents correction of XGBoost errors.
> Treat $\sigma_a$ as a hyperparameter and grid-search over $\sigma_a \in \{0.1, 0.3, 0.5, 1.0\}$
> in Step 1.5 walk-forward CV, selecting the value that minimizes validation Brier Score.

### Learnable Parameters (PyTorch `nn.Parameter`)

| Parameter | Dimension | Init | Note |
|---------|------|--------|------|
| $a^m_H$ | M x 1 | $\ln(\hat{\mu}^m_H / T_m)$ | match-level home baseline intensity |
| $a^m_A$ | M x 1 | $\ln(\hat{\mu}^m_A / T_m)$ | match-level away baseline intensity |
| **b** | 6 x 1 | **0** | time-profile by interval |
| $\gamma^H_1, \gamma^H_2$ | 2 scalars | -0.05, 0.05 | home-team red-card penalty |
| $\gamma^A_1, \gamma^A_2$ | 2 scalars | 0.05, -0.05 | away-team red-card penalty |
| $\beta_H, \kappa_H, \tau_H$ | 3 scalars | 0, 0, 1.0 | home parametric delta coefficients |
| $\beta_A, \kappa_A, \tau_A$ | 3 scalars | 0, 0, 1.0 | away parametric delta coefficients |

Total free parameters: $2M + 6 + 4 + 6 = 2M + 16$
(with gamma symmetry: $2M + 14$)

### Parameter Clamping

| Parameter | Allowed Range | Physical rationale |
|---------|----------|------------|
| $b_i$ | [-0.5, 0.5] | interval intensity ratio above x1.65 is unrealistic |
| $\gamma^H_1$ | [-1.5, 0] | home dismissal -> home scoring down |
| $\gamma^H_2$ | [0, 1.5] | away dismissal -> home scoring up |
| $\gamma^A_1$ | [0, 1.5] | home dismissal -> away scoring up |
| $\gamma^A_2$ | [-1.5, 0] | away dismissal -> away scoring down |
| $\beta_H, \beta_A$ | [-0.5, 0.5] | linear slope of score sensitivity |
| $\kappa_H, \kappa_A$ | [-1.0, 1.0] | saturation magnitude |
| $\tau_H, \tau_A$ | [0.1, 5.0] | saturation rate (positive) |

### Optimization Strategy

**1. Multi-start:**
Since NLL is non-convex, initialize b, gamma, delta from
5-10 random seeds and choose best local minimum.

**2. Two-stage optimizer:**
Adam (lr=1e-3, 1000 epochs) -> L-BFGS (fine-tuning).

**3. Numerical stability:**
With piecewise basis intervals, b -> 0 singularity does not arise.

**4. τ in log-space:**
Optimize $\phi = \ln \tau$ instead of $\tau$ directly.
The gradient $\partial\delta/\partial\tau$ vanishes at both boundaries
($\tau \to 0.1$: $e^{-|s|/\tau} \to 0$; $\tau \to 5.0$: $|s|/\tau^2 \to 0$).
Reparameterizing as $\tau = e^{\phi}$ rescales the gradient by $\tau$,
compensating for vanishing gradients and stabilizing Adam/L-BFGS convergence:

```python
self.log_tau_H = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
tau_H = torch.exp(self.log_tau_H).clamp(0.1, 5.0)
```

### Output

- Time-interval scoring profile $\mathbf{b} = [b_1, \ldots, b_6]$
- Home red-card penalty $\gamma^H_1, \gamma^H_2$ (+ $\gamma^H_3 = \gamma^H_1 + \gamma^H_2$)
- Away red-card penalty $\gamma^A_1, \gamma^A_2$ (+ $\gamma^A_3 = \gamma^A_1 + \gamma^A_2$)
- Parametric delta coefficients $(\beta_H, \kappa_H, \tau_H)$, $(\beta_A, \kappa_A, \tau_A)$
- Generated 5-element lookup tables $\boldsymbol{\delta}_H, \boldsymbol{\delta}_A$ (for mc_core/pricing)
- Corrected match-level baselines $\{a^m_H, a^m_A\}$

---

## Step 1.5: Time-Series Cross-Validation and Model Diagnostics (Validation)

### Goal

Detect overfitting and quantify probabilistic prediction quality.
**If this step fails, do not deploy to live.**

### Walk-Forward Validation (with Seasonal Holdout Extension)

**In-season folds (standard walk-forward):**

| Fold | Train Period | Validation Period | Type |
|------|----------|----------|------|
| 1 | Seasons 1-3 | Season 4 | in_season |
| 2 | Seasons 1-4 | Season 5 | in_season |
| 3 | Seasons 2-5 | Season 6 | in_season |

**Cross-season folds (seasonal holdout extension):**

When 5+ seasons are available, add cross-season folds that train on 4 seasons
and validate on the held-out 5th to detect seasonal drift:

| Fold | Train Period | Validation Period | Type |
|------|----------|----------|------|
| CS-1 | Seasons 2-5 | Season 1 | cross_season |
| CS-2 | Seasons 1,3-5 | Season 2 | cross_season |
| CS-3 | Seasons 1-2,4-5 | Season 3 | cross_season |
| CS-4 | Seasons 1-3,5 | Season 4 | cross_season |
| CS-5 | Seasons 1-4 | Season 5 | cross_season |

**Seasonal drift detection:**

$$\text{drift} = \overline{\Delta BS_{cross\_season}} - \overline{\Delta BS_{in\_season}}$$

If drift > 0.01 (cross-season performance significantly worse), the model may be
overfitting to temporal patterns that don't generalize across seasons.

For each fold, run Step 1.3 (ML) and Step 1.4 (NLL) using train period only,
then measure metrics below on validation period.

### League-Stratified Brier Score (Extension)

Compute Brier Scores broken down by league tier to detect whether model
performance is concentrated in certain leagues:

| Tier | Leagues | Expected |
|------|---------|----------|
| tier_1 | EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Brasileirão, Liga Argentina | Tradable — must validate |
| tier_2 | Eredivisie, Liga Portugal, Championship, etc. | Non-tradable — used for training only |

```python
LEAGUE_TIERS = {
    "tier_1": {1204, 1399, 1229, 1269, 1221,   # Europe: EPL, La Liga, BuLi, SerA, L1
               1440, 1141, 1081},                # Americas: MLS, Brasileirão, Liga Argentina
    "tier_2": {1007, 1352, ...},                 # non-tradable (training data only)
    "tier_3": set(),  # catch-all
}

def compute_league_stratified_bs(val_matches, model_preds, baseline_preds, outcomes):
    """Compute ΔBS per league tier."""
    results = {}
    for tier, league_ids in LEAGUE_TIERS.items():
        mask = [m["league_id"] in league_ids for m in val_matches]
        if sum(mask) < 10:
            continue
        bs_model = brier_score(model_preds[mask], outcomes[mask])
        bs_base = brier_score(baseline_preds[mask], outcomes[mask])
        results[tier] = {"bs_model": bs_model, "bs_baseline": bs_base,
                         "delta_bs": bs_model - bs_base, "n": sum(mask)}
    return results
```

This enables detecting cases where the model beats Betfair Exchange in EPL/La Liga/MLS
but underperforms in lower-quality leagues where data is sparser.

### Core Diagnostic Metrics

**1. Calibration Plot (Reliability Diagram):**

Visualize whether events predicted at "P = 0.6" actually occur near 60%.

**2. Brier Score — Betfair Exchange baseline:**

Use Odds-API Historical Odds **Betfair Exchange close line** as precise market benchmark:

$$BS_{model} = \frac{1}{N}\sum_n (P_{model,n} - O_n)^2$$

$$BS_{exchange} = \frac{1}{N}\sum_n (P_{exchange\_close,n} - O_n)^2$$

$$\Delta BS = BS_{model} - BS_{exchange}$$

If $\Delta BS < 0$, model beats the most efficient market (Betfair Exchange).

> **Odds-API advantage:** historical close odds from 5 bookmakers,
> enabling baselines beyond Betfair Exchange (Bet365, Sbobet, 1xBet, DraftKings).

**3. Log Loss (validation NLL):**

$$\text{Log Loss} = -\frac{1}{N}\sum_{n=1}^N [O_n \ln P_n + (1-O_n)\ln(1-P_n)]$$

**4. Simulation P&L:**

Because historical Odds-API odds are available,
proxy Kalshi quotes with Betfair Exchange odds and backtest through Phase 4 Kelly logic.

**5. Multi-market cross-validation (Odds-API-specific):**

Since Odds-API provides 50+ markets,
validate probabilities implied by model μ_H, μ_A **across multiple markets at once**:

| Market | Derived from Model | Derived from Market | Comparison |
|------|-------------|-------------|------|
| Match Winner | Poisson(μ_H) vs Poisson(μ_A) | Pregame Odds 1X2 | Brier score |
| Over/Under 2.5 | 1 - CDF(2, μ_H + μ_A) | Pregame Odds O/U | Brier score |
| Both Teams to Score | composite Poisson | Pregame Odds BTTS | Brier score |

If model is good on 1X2 but poor on O/U, total μ may be right but split μ_H vs μ_A may be wrong.
μ_H and μ_A should perform well jointly across all markets.

### gamma Sign Validation

| Expected Sign | Validation |
|----------|------|
| $\gamma^H_1 < 0$ | home dismissal -> home scoring down |
| $\gamma^H_2 > 0$ | away dismissal -> home scoring up |
| $\gamma^A_1 > 0$ | home dismissal -> away scoring up |
| $\gamma^A_2 < 0$ | away dismissal -> away scoring down |

**Symmetric vs asymmetric gamma comparison:**

- Symmetric model: $\gamma^A_1 = -\gamma^H_2$, $\gamma^A_2 = -\gamma^H_1$ (2 parameters)
- Asymmetric model: independent 4 parameters

Compare validation Log Loss to justify asymmetry.

### delta Sign Validation

With parametric delta, sign validation uses the generated lookup values:

| Expected Sign | Validation |
|----------|------|
| $\delta_H(-1) > 0$ | trailing home attacks more |
| $\delta_H(+1) < 0$ | leading home shifts defensive |
| $\delta_A(-1) < 0$ | leading away shifts defensive |
| $\delta_A(+1) > 0$ | trailing away attacks more |

**Test rejection of delta = 0 (Likelihood Ratio Test):**

$$LR = -2(\mathcal{L}_{\delta=0} - \mathcal{L}_{\delta \neq 0}) \sim \chi^2(df=6)$$

With parametric delta, $df = 6$ (3 params per team: $\beta, \kappa, \tau$).

If p < 0.05, inclusion of delta is justified. If not rejected, keep delta-free model.

**Symmetric vs asymmetric delta comparison:**

- Symmetric: $(\beta_A, \kappa_A, \tau_A) = (-\beta_H, -\kappa_H, \tau_H)$ (3 parameters)
- Asymmetric: independent 6 parameters

### Closed-Line Validation (Extension)

Odds-API Historical Odds provides **open and close** lines from 5 bookmakers.
The Betfair Exchange **closing line** is the most efficient pre-match probability estimate
and serves as a strong benchmark for in-play model quality.

**Pre-match validation (already in place):**

$$\Delta BS_{prematch} = BS_{model} - BS_{exchange\_close}$$

**In-play validation (extension):**

For each in-play pricing snapshot (from Step 3.6 backtest), compare model $P_{true}(t)$
against the Betfair Exchange closing line implied probability. While the closing line is
a pre-match estimate, it serves as a calibration anchor:

- At kickoff ($t=0$), $P_{true}$ should approximately match Betfair Exchange close
- After events, $P_{true}$ should deviate appropriately from Betfair Exchange close
- At settlement, the model's cumulative edge over Betfair Exchange closing line is the
  true measure of in-play information capture

```python
def closed_line_validation(backtest_results: List[dict],
                           exchange_close: Dict[str, float]) -> dict:
    """Compare in-play model P_true against pre-match Betfair Exchange close.

    Returns edge statistics:
    - CLV (Closing Line Value): average P_true at entry vs Betfair Exchange
    - Win rate at entry: does P_true predict outcomes better than Betfair Exchange?
    """
    clv_edges = []
    for trade in backtest_results:
        match_id = trade["match_id"]
        if match_id not in exchange_close:
            continue
        p_close = exchange_close[match_id]
        p_entry = trade["P_true_at_entry"]

        if trade["direction"] == "BUY_YES":
            clv = p_entry - p_close  # positive = bought cheaper than close
        else:
            clv = p_close - p_entry  # positive = sold higher than close

        clv_edges.append(clv)

    return {
        "mean_clv": np.mean(clv_edges),
        "median_clv": np.median(clv_edges),
        "pct_positive_clv": np.mean([e > 0 for e in clv_edges]),
        "n_trades": len(clv_edges),
    }
```

A positive mean CLV indicates the model's in-play information genuinely
improves upon efficient pre-match markets — the strongest evidence of edge.

### b Validation — per-half stats cross-check (Goalserve-specific)

Compare learned first/second-half weight in b[1..6] with actual shot split from Goalserve:

```python
def validate_b_with_half_stats(b, stats_db):
    """
    Cross-validate learned half split in b against
    shots.total_h1 and shots.total_h2 from Goalserve Live Game Stats.
    """
    # Model first-half weight
    model_h1_weight = sum(np.exp(b[i]) for i in range(3))
    model_h2_weight = sum(np.exp(b[i]) for i in range(3, 6))
    model_h1_ratio = model_h1_weight / (model_h1_weight + model_h2_weight)

    # Empirical first-half shot share (league average)
    actual_h1_ratio = stats_db["shots_h1_total"] / stats_db["shots_total"]

    discrepancy = abs(model_h1_ratio - actual_h1_ratio)
    if discrepancy > 0.10:
        log.warning(f"b half-ratio mismatch: model={model_h1_ratio:.2f}, "
                    f"actual={actual_h1_ratio:.2f}")
```

### Sanity Check Threshold Calibration (for Phase 2 Step 2.4)

Phase 2 Step 2.4 compares model-predicted Match Winner probabilities against
Betfair Exchange closing lines and triggers GO / HOLD / SKIP based on discrepancy thresholds.
These thresholds must be calibrated from data, not hardcoded.

**Procedure:** on each validation fold, compute the per-match maximum discrepancy
between model probability and Betfair Exchange closing line:

$$\Delta_{pin}^{(m)} = \max_{o \in \{H, D, A\}} |P_{model}^{(m)}(o) - P_{exchange}^{(m)}(o)|$$

Collect $\{\Delta_{pin}^{(m)}\}$ across all validation matches, then set thresholds
at empirical quantiles:

```python
def calibrate_sanity_thresholds(val_matches, model_preds, exchange_close):
    """Calibrate Phase 2 sanity check thresholds from validation data."""
    deltas = []
    for m, pred, pin in zip(val_matches, model_preds, exchange_close):
        delta = max(abs(pred[o] - pin[o]) for o in ["H", "D", "A"])
        deltas.append(delta)

    deltas = np.array(deltas)
    return {
        "go_threshold": np.percentile(deltas, 90),       # ~90th pct -> GO
        "hold_threshold": np.percentile(deltas, 99),     # ~99th pct -> HOLD
        "median_delta": np.median(deltas),                # diagnostic
        "n_matches": len(deltas),
    }
```

| Threshold | Quantile | Meaning |
|-----------|----------|---------|
| `go_threshold` | 90th percentile | below this = normal model-market disagreement -> GO |
| `hold_threshold` | 99th percentile | above this = extreme outlier -> SKIP; between = HOLD |

**Similarly for O/U cross-check:** compute $\Delta_{OU}^{(m)} = |P_{model}^{Over2.5} - P_{market}^{Over2.5}|$
and set the O/U consistency threshold at its 90th percentile.

These thresholds are stored alongside production parameters and consumed by Phase 2 Step 2.4.

### Pass Criteria (Go/No-Go)

| Criterion | Threshold |
|------|--------|
| Calibration plot | within +/-5% of diagonal |
| Brier Score | $\Delta BS < 0$ (improve vs Betfair Exchange) |
| Multi-market BS | improve vs market in 1X2, O/U, BTTS |
| Simulated Max Drawdown | <= 20% of capital |
| All folds | positive simulated return in all 3 folds |
| gamma signs | all 4 align with football intuition |
| delta signs | align with football intuition |
| delta LRT | $p < 0.05$ with $df = 6$ (parametric delta) |
| b half split | within +/-10% of empirical shot split |
| League-stratified BS | tier_1 $\Delta BS < 0$ (must beat Betfair Exchange in tradable leagues) |
| Seasonal drift | $\overline{\Delta BS_{cross}} - \overline{\Delta BS_{in}} < 0.01$ |
| Closed-line value | mean CLV > 0 (model beats closing lines) |

### Output

Fix the final parameter set that passes all criteria as **production parameters** and hand off to Phase 2.

---

## Phase 1 -> Phase 2 Handoff

| Parameter | Source | Usage |
|---------|------|------|
| XGBoost weights + `feature_mask.json` | Step 1.3 | predict $\hat{\mu}_H, \hat{\mu}_A$ for new matches |
| $\mathbf{b} = [b_1, \ldots, b_6]$ | Step 1.4 | time-interval scoring profile |
| $\gamma^H_1, \gamma^H_2$ | Step 1.4 | home intensity jump under dismissals |
| $\gamma^A_1, \gamma^A_2$ | Step 1.4 | away intensity jump under dismissals |
| $(\beta_H, \kappa_H, \tau_H)$, $(\beta_A, \kappa_A, \tau_A)$ | Step 1.4 | parametric delta coefficients |
| $\boldsymbol{\delta}_H[5], \boldsymbol{\delta}_A[5]$ | Step 1.4 | generated 5-element lookup tables (for mc_core) |
| Q (4x4 matrix) | Step 1.2 | future dismissal probabilities (matrix exponential) |
| $\{Q_{\Delta S}\}$ (5 x 4x4) | Step 1.2 (extension) | score-conditioned Q matrices |
| $\mathbb{E}[\alpha_1], \mathbb{E}[\alpha_2]$ | Step 1.1 | compute $T_{exp}$ in Phase 2 |
| Stoppage time distribution params | Step 1.1 (extension) | MC path-level $T$ sampling in Phase 3 |
| delta significance flag (`DELTA_SIGNIFICANT`) | Step 1.5 LRT | choose analytic/MC mode in Phase 3 |
| Betfair Exchange BS baseline | Step 1.5 | market benchmark for Phase 4 post-analysis |
| League-stratified BS | Step 1.5 (extension) | per-tier model quality assessment |
| CLV statistics | Step 1.5 (extension) | in-play edge vs closing lines |
| Sanity check thresholds (`go_threshold`, `hold_threshold`, `ou_threshold`) | Step 1.5 | Phase 2 Step 2.4 GO/HOLD/SKIP cutoffs |

> **delta and Phase 2:** at kickoff, ΔS = 0 so delta(0) = 0.
> Therefore delta does not affect a back-solving in Phase 2.
> delta activates only after goals occur in Phase 3.

---

## Phase 1 Pipeline Summary

```
[Goalserve (Commentaries + Stats) + Odds-API (Odds, Dec 2025+ only)]
              |
              v
+---------------------------------------------------------------+
|  Step 1.1: Interval Segmentation (Data Engineering)            |
|  • Commentaries -> goals (VAR filtering) + red-card events     |
|  • addedTime_period1/2 -> match-level T_m                      |
|  • var_cancelled=True -> exclude, owngoal -> exclude point term|
|  • Tag each interval with (X, ΔS), store ΔS_before + scorer    |
|  • [EXT] Substitution events -> split intervals for ψ_sub      |
|  • [EXT] Stoppage time distribution -> LogNormal per league     |
|  Output: intervals[], stoppage_dist, sub_intervals             |
+------------------+--------------------------------------------+
                   |
        +----------+----------+
        v                     v
+--------------+    +------------------------------------------+
|  Step 1.2:   |    |  Step 1.3: ML Prior (XGBoost)            |
|  Estimate Q  |    |  • Tier 1: team rolling stats (incl. xG) |
|  • Fixtures  |    |  • Tier 2: player aggregates (rating...) |
|    redcards  |    |  • Tier 3: odds (5 bookmakers)         |
|  • Empirical |    |  • Tier 4: context (H/A, rest, H2H)      |
|    rates     |    |  • Feature selection via importance       |
|  • gamma^H/A |    |  • Poisson regression -> μ̂_H, μ̂_A      |
|    additive  |    |  Output: â_H^(init), â_A^(init), .xgb     |
|  • League    |    +--------------+---------------------------+
|    stratified|                   |
|  • [EXT] Q_ΔS|                  |
|    by score  |                   |
+------+-------+                   |
       |                           |
       +-----------+---------------+
                   v
+---------------------------------------------------------------+
|  Step 1.4: Joint NLL Optimization (PyTorch)                    |
|  • Jointly learn a^m_H, a^m_A, b[1..6], gamma^H_1/2,         |
|    gamma^A_1/2, delta_H(β,κ,τ), delta_A(β,κ,τ)               |
|  • Parametric delta: δ(s) = β·s + κ·sign(s)·(1-exp(-|s|/τ)) |
|  • Home/away separated goal NLL                                |
|  • Exclude own goals from point-event term                     |
|  • Multi-start + L2 regularization + clamping                  |
|  • [EXT] Position-specific gamma modifier φ_pos                |
|  • [EXT] Opponent quality modifier η_quality                   |
|  Output: b[], gamma^H/A, delta_params, delta_lookup[5],       |
|          {a^m_H, a^m_A}                                        |
+------------------+--------------------------------------------+
                   v
+---------------------------------------------------------------+
|  Step 1.5: Time-Series Cross-Validation (Validation)           |
|  • Walk-forward CV (prevent temporal leakage)                  |
|  • [EXT] Cross-season folds (seasonal holdout)                 |
|  • Brier Score vs Betfair Exchange close line               |
|  • [EXT] League-stratified BS (tier_1/tier_2/tier_3)           |
|  • Multi-market checks (1X2 + O/U + BTTS)                      |
|  • b half split vs empirical shot split (per-half stats)       |
|  • gamma sign checks (4), delta sign checks, LRT (df=6)       |
|  • [EXT] Closed-line validation (CLV vs Betfair Exchange close)        |
|  • Simulated P&L + Max Drawdown                                |
|  • Go/No-Go decision                                            |
|  Output: Production Parameters + DELTA_SIGNIFICANT flag        |
|          + league_stratified_bs + seasonal_drift + CLV          |
|          + sanity_check_thresholds (go/hold/ou)                 |
+---------------------------------------------------------------+
```