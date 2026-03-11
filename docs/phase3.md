# Phase 3: Live Trading Engine — Odds-API + Goalserve

## Overview

A dynamic pricing engine that runs every second from kickoff to full time.

As option time value decays over time (Theta Decay),
it continuously decays expected goals by remaining match time,
while instantly re-adjusting the probability space whenever
jump events such as goals or red cards occur.

Using parameters learned in Phase 1 and initial conditions set in Phase 2,
it repeats the cycle of
**remaining expected goals μ calculation -> true probability P_true estimation**
every second.

This process is decomposed into five steps.

### Architecture Paradigm Shift: 2-Layer Event Detection

Odds-API Live Odds WebSocket (<1s)
fundamentally changes the Phase 3 architecture.

**Original design (single-layer):**
```
Defense 1: Goalserve Live Score REST (3-8s) -> event confirmation
```

**Current design (2-layer):**
```
Layer 1: Odds-API Live Odds WebSocket (<1s) -> early warning + ob_freeze
Layer 2: Goalserve Live Score REST (3-8s) -> authoritative confirmation
```

Phase 3 event detection sources:

| Source | Protocol | Latency | Data Provided | Role |
|------|---------|------|-----------|------|
| **Odds-API Live Odds** | **WebSocket PUSH** | **<1s** | bet365 ML/Totals/Spread odds, market updates | **early warning + ob_freeze** |
| **Goalserve Live Score** | **REST polling every 3s** | 3-8s | goal scorer, card detail, substitutions, VAR | **authoritative confirmation + details** |

> **Kalshi order book** ingestion and execution are handled by **Phase 4** (Step 4.1).
> Phase 3 focuses exclusively on pricing (P_true computation) and emits results to Phase 4.

Core insight:
- **Odds-API Live Odds** tells us **first** that "something happened" (abrupt odds move)
- **Live Score** tells us "what exactly happened" (who scored, whether VAR cancelled)

Why both are required: Odds-API detects market movement instantly,
but only Goalserve Live Score can confirm whether it was a regular goal, own goal,
or later VAR cancellation.

### Intensity Function (Fixed in Phase 1)

$$\lambda_H(t \mid X, \Delta S) = \exp\!\left(a_H + b_{i(t)} + \gamma^H_{X(t)} + \delta_H(\Delta S(t))\right)$$

$$\lambda_A(t \mid X, \Delta S) = \exp\!\left(a_A + b_{i(t)} + \gamma^A_{X(t)} + \delta_A(\Delta S(t))\right)$$

| Symbol | Meaning | Change Trigger |
|------|------|-----------|
| $a_H, a_A$ | match-level baseline intensity | invariant during match |
| $b_{i(t)}$ | time-interval profile | when basis boundary is crossed |
| $\gamma^H_{X(t)}$ | red card -> home penalty | on red card (jump) |
| $\gamma^A_{X(t)}$ | red card -> away penalty | on red card (jump) |
| $\delta_H(\Delta S)$ | score-diff home tactical effect | on goal (jump) |
| $\delta_A(\Delta S)$ | score-diff away tactical effect | on goal (jump) |

---

## Input Data

**Phase 2 outputs:**

| Item | Usage |
|------|------|
| `LiveFootballQuantModel` instance | all parameters + initial state |
| $P_{grid}[0..100]$ + $P_{fine\_grid}$ | precomputed matrix exponentials |
| $Q_{off\_normalized}$ (4x4) | normalized transition probabilities for MC |
| $C_{time}$, $T_{exp}$ | time constants |
| `DELTA_SIGNIFICANT` | analytic/MC mode-selection flag |

**Real-time data streams (2 sources — Phase 3 scope):**

| Source | Endpoint | Data |
|------|---------------------|--------|
| Odds-API Live Odds WS | `wss://api.odds-api.io/v3/ws?apiKey={key}&markets=ML,Totals&status=live` | bet365 ML/Totals/Spread odds, market updates |
| Goalserve Live Score REST | `GET /getfeed/{api_key}/soccerlive/home?json=1` | goal scorer, cards, substitutions, VAR |

> **Kalshi WebSocket** (order book Bid/Ask + depth) is consumed by **Phase 4 Step 4.1**,
> not by Phase 3. Phase 3 emits P_true to Phase 4, which independently manages Kalshi connectivity.

---

## Step 3.1: Asynchronous Real-Time Data Ingestion and State Machine (Event Loop & State Machine)

### Goal

Track physical time (seconds) and match state simultaneously,
and implement a 2-layer event detection framework with two-stage preliminary -> confirmed handling.

### Engine State Machine (Engine Phase)

```
FIRST_HALF --(first-half end)--> HALFTIME --(second-half kickoff)--> SECOND_HALF --(full time)--> FINISHED
```

| Engine State | Time Range | Pricing | Orders |
|----------|----------|---------|------|
| `FIRST_HALF` | $[0,\; 45+\alpha_1]$ | active | active |
| `HALFTIME` | about 15 min | **frozen** | **stopped** |
| `SECOND_HALF` | $[45+\alpha_1+\delta_{HT},\; T_m]$ | active | active |
| `FINISHED` | — | final settlement | — |

Since lambda(t) = 0 during halftime, continuing pricing creates
fictional decay where "time passes but no goals can occur."
In `HALFTIME`, freeze both pricing and orders.

### Event State Machine (Event State)

Event handling states for 2-layer detection:

```
IDLE --(Odds-API spike or Live Score score change)--> PRELIMINARY_DETECTED
  |                                                        |
  |                                                   ob_freeze = True
  |                                                   μ pre-compute (provisional)
  |                                                        |
  |                                        +---------------+---------------+
  |                                        v                               v
  |                                  CONFIRMED                       FALSE_ALARM
  |                            (Live Score confirms)      (3-tick stabilization or 10s timeout)
  |                                        |                               |
  |                                  +-----+-----+                    ob_freeze = False
  |                                  v           v                    keep state
  |                           not VAR-cancelled  VAR-cancelled             |
  |                                  |           |                         |
  |                             commit S,ΔS,X    rollback                  |
  |                             cooldown 15s     ob_freeze = False         |
  |                             ob_freeze=F           |                    |
  |                                  |                |                    |
  +----------------------------------+----------------+--------------------+
                                     v
                                   IDLE (return)
```

### Mathematical State Variables

```
t           : current effective play time (halftime excluded)
S(t)        : current score (S_H, S_A)
X(t)        : Markov state ∈ {0, 1, 2, 3}
ΔS(t)       : current score difference = S_H - S_A
engine_phase: {FIRST_HALF, HALFTIME, SECOND_HALF, FINISHED}
event_state : {IDLE, PRELIMINARY_DETECTED, CONFIRMED}
cooldown    : bool (15s order block after event)
ob_freeze   : bool (order block on anomaly detection)
T           : currently applied expected match end time
```

### EventSource Abstraction

```python
class EventSource(ABC):
    """Abstract layer decoupling engine and data sources."""
    async def connect(self, match_id: str) -> None: ...
    async def listen(self) -> AsyncIterator[NormalizedEvent]: ...
    async def disconnect(self) -> None: ...

@dataclass
class NormalizedEvent:
    type: str           # goal_detected, goal_confirmed, red_card,
                        # period_change, odds_spike, stoppage_entered
    source: str         # "live_odds" or "live_score"
    confidence: str     # "preliminary" or "confirmed"
    timestamp: float
    # Additional event-specific fields
    score: Optional[Tuple[int, int]] = None
    team: Optional[str] = None
    minute: Optional[float] = None
    period: Optional[str] = None
    var_cancelled: Optional[bool] = None
    scorer_id: Optional[str] = None
    delta: Optional[float] = None
```

### Source 1: Odds-API Live Odds WebSocket (Primary Detection, <1s)

```python
class OddsApiLiveOddsSource(EventSource):
    """
    WebSocket PUSH - <1s latency.
    bet365 in-play odds via Odds-API.io WebSocket.

    Connection: wss://api.odds-api.io/v3/ws?apiKey={key}&markets=ML,Totals&status=live
    (Bookmakers pre-selected via PUT /bookmakers/selected/select?bookmakers=Bet365)

    Odds-API WebSocket message format:
    {
      "type": "updated",
      "timestamp": "2025-08-18T14:52:52.983Z",
      "id": "63017989",
      "bookie": "Bet365",
      "url": "https://www.bet365.com/...",
      "markets": [
        {
          "name": "ML",
          "updatedAt": "2025-08-18T14:52:52Z",
          "odds": [{"home": "1.44", "draw": "3.50", "away": "12.00", "max": 500}]
        },
        {
          "name": "Totals",
          "updatedAt": "2025-08-18T14:52:52Z",
          "odds": [{"hdp": 2.5, "over": "1.90", "under": "1.90"}]
        }
      ]
    }

    Message types: "welcome", "created", "updated", "deleted", "no_markets"
    """

    def __init__(self, odds_threshold_pct: float = 0.10):
        self.ODDS_THRESHOLD = odds_threshold_pct
        self._last_home_odds = None
        self._event_ids: set = set()  # tracked live event IDs

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        async for msg in self.ws:
            parsed = json.loads(msg)
            msg_type = parsed.get("type", "")

            if msg_type == "welcome":
                log.info(f"Odds-API WS connected: bookmakers={parsed.get('bookmakers')}")
                continue

            if msg_type == "deleted":
                # Match removed (possibly finished)
                yield NormalizedEvent(
                    type="match_removed",
                    source="live_odds",
                    confidence="preliminary",
                    timestamp=time.time()
                )
                continue

            if msg_type not in ("updated", "created"):
                continue

            # Filter to our tracked event IDs
            event_id = str(parsed.get("id", ""))
            if self._event_ids and event_id not in self._event_ids:
                continue

            # --- Abrupt odds-move detection ---
            markets = parsed.get("markets", [])
            odds_delta = self._compute_odds_delta(markets)
            if odds_delta >= self.ODDS_THRESHOLD:
                yield NormalizedEvent(
                    type="odds_spike",
                    source="live_odds",
                    confidence="preliminary",
                    delta=odds_delta,
                    timestamp=time.time()
                )

    def _compute_odds_delta(self, markets: list) -> float:
        """Compute home-odds change rate from ML market."""
        try:
            for market in markets:
                if market.get("name") == "ML":
                    odds = market["odds"][0]
                    current = float(odds["home"])
                    if self._last_home_odds is not None and self._last_home_odds > 0:
                        delta = abs(current - self._last_home_odds) / self._last_home_odds
                        self._last_home_odds = current
                        return delta
                    self._last_home_odds = current
        except (KeyError, ValueError, IndexError):
            pass
        return 0.0
```

> **Note:** Odds-API WebSocket provides **odds data only** (no score, minute, period,
> or ball position). Score changes, period transitions, and stoppage-time entry
> are detected via **Goalserve Live Score REST** (Source 2). The Odds-API WebSocket
> serves as an **early-warning system** for events via abrupt odds movements,
> which typically arrive faster than score updates.

### Connection Resilience — Reconnect & Fallback

Both data sources implement automatic reconnection with exponential backoff.
If reconnection fails, the system degrades gracefully rather than crashing.

```python
class ResilientWebSocket:
    """
    Wrapper for WebSocket connections with exponential backoff reconnect.
    Used by OddsApiLiveOddsSource.
    """
    BACKOFF_BASE = 1.0      # initial retry delay (seconds)
    BACKOFF_MAX = 30.0      # max retry delay
    MAX_RETRIES = 5         # consecutive failures before fallback mode

    async def connect_with_retry(self, url: str) -> websockets.WebSocketClientProtocol:
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                ws = await asyncio.wait_for(
                    websockets.connect(url), timeout=10
                )
                retries = 0  # reset on success
                return ws
            except (websockets.ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                retries += 1
                delay = min(self.BACKOFF_BASE * (2 ** retries), self.BACKOFF_MAX)
                log.warning(f"WS reconnect attempt {retries}/{self.MAX_RETRIES}, "
                           f"backoff {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

        # All retries exhausted → enter fallback mode
        raise ConnectionError(f"WebSocket reconnect failed after {self.MAX_RETRIES} attempts")

    async def listen_with_reconnect(self, url: str) -> AsyncIterator:
        """
        Auto-reconnect wrapper. On disconnect:
        1. Attempt reconnect with exponential backoff (1s → 2s → 4s → 8s → 16s → 30s)
        2. After MAX_RETRIES failures, yield a source_failure event
        3. Caller (Phase 3 engine) enters fallback mode
        """
        while True:
            try:
                ws = await self.connect_with_retry(url)
                async for msg in ws:
                    yield msg
            except ConnectionError:
                yield NormalizedEvent(
                    type="source_failure",
                    source="live_odds",
                    confidence="confirmed",
                    timestamp=time.time()
                )
                return  # exit to fallback mode
            except websockets.ConnectionClosed:
                log.warning("Odds-API WS closed — reconnecting")
                continue  # retry loop
```

> **Fallback hierarchy:**
> - Odds-API WS healthy → 2-layer detection (normal operation)
> - Odds-API WS down, Goalserve REST healthy → 1-layer detection (slower, but safe)
> - Both down → freeze all orders, alert, await manual intervention

### Source 2: Goalserve Live Score REST (Authoritative Confirmation, 3-8s)

```python
class GoalserveLiveScoreSource(EventSource):
    """
    REST polling every 3s - 3~8s latency.
    Goal scorer, card details, substitutions, VAR information.

    Goalserve Live Score response format:
    {
      "scores": {
        "category": {
          "matches": {
            "match": [{
              "id": "5035743",
              "localteam": {"goals": "1", "name": "..."},
              "visitorteam": {"goals": "0", "name": "..."},
              "status": "28",
              "timer": "28",
              "events": {
                "event": [{
                  "type": "goal",
                  "team": "localteam",
                  "player": "Lu Pin",
                  "minute": "21",
                  "result": "[1 - 0]"
                }]
              },
              "live_stats": { ... }
            }]
          }
        }
      }
    }
    """

    def __init__(self, api_key: str, match_id: str, poll_interval: float = 3.0):
        self.api_key = api_key
        self.match_id = match_id
        self.poll_interval = poll_interval
        self._last_score = {"home": 0, "away": 0}
        self._last_cards = set()
        self._last_period = None

    async def listen(self) -> AsyncIterator[NormalizedEvent]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self.running:
                try:
                    data = await self._poll(client)
                    match = self._find_match(data)
                    if match:
                        async for event in self._diff(match):
                            yield event
                except httpx.HTTPError as e:
                    log.error(f"Live Score poll failed: {e}")
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 5:
                        yield NormalizedEvent(
                            type="source_failure",
                            source="live_score",
                            confidence="confirmed",
                            timestamp=time.time()
                        )

                await asyncio.sleep(self.poll_interval)

    async def _diff(self, match: dict) -> AsyncIterator[NormalizedEvent]:
        """Detect changes by comparing with previous poll result."""

        # --- Score change (goal confirmed) ---
        home_goals = int(match["localteam"]["goals"] or 0)
        away_goals = int(match["visitorteam"]["goals"] or 0)

        # Multi-goal same poll: if 2 goals scored in same 3s interval,
        # yield them one at a time with INTERMEDIATE scores so handlers
        # commit state correctly between goals (ΔS transitions step-by-step).
        running_home = self._last_score["home"]
        running_away = self._last_score["away"]

        if home_goals > running_home:
            for i in range(home_goals - running_home):
                running_home += 1
                yield NormalizedEvent(
                    type="goal_confirmed",
                    source="live_score",
                    confidence="confirmed",
                    score=(running_home, running_away),  # intermediate, not final
                    team="localteam",
                    var_cancelled=False,
                    timestamp=time.time()
                )

        if away_goals > running_away:
            for i in range(away_goals - running_away):
                running_away += 1
                yield NormalizedEvent(
                    type="goal_confirmed",
                    source="live_score",
                    confidence="confirmed",
                    score=(running_home, running_away),  # intermediate, not final
                    team="visitorteam",
                    var_cancelled=False,
                    timestamp=time.time()
                )

        self._last_score = {"home": home_goals, "away": away_goals}

        # --- Red card detection ---
        live_stats = match.get("live_stats", {}).get("value", "")
        home_reds = self._extract_stat(live_stats, "IRedCard", "home")
        away_reds = self._extract_stat(live_stats, "IRedCard", "away")

        current_cards = {("home", home_reds), ("away", away_reds)}
        new_cards = current_cards - self._last_cards
        for team, count in new_cards:
            if count > 0:
                yield NormalizedEvent(
                    type="red_card",
                    source="live_score",
                    confidence="confirmed",
                    team="localteam" if team == "home" else "visitorteam",
                    timestamp=time.time()
                )
        self._last_cards = current_cards

        # --- Period change ---
        status = match.get("status", "")
        if status != self._last_period:
            if status == "HT":
                yield NormalizedEvent(
                    type="period_change",
                    source="live_score",
                    confidence="confirmed",
                    period="Halftime",
                    timestamp=time.time()
                )
            elif status == "Finished":
                yield NormalizedEvent(
                    type="match_finished",
                    source="live_score",
                    confidence="confirmed",
                    timestamp=time.time()
                )
            self._last_period = status
```

### Asynchronous Loop Structure (2 Sources + Tick)

```python
async def run_engine(model: LiveFootballQuantModel):
    """Run two event source coroutines + tick loop concurrently.

    Note: Kalshi order book sync runs in Phase 4, not here.
    Phase 3 emits P_true via emit_to_phase4() each tick.
    """
    await asyncio.gather(
        tick_loop(model),                # Every 1s: μ recompute + P_true output
        live_odds_listener(model),       # Odds-API WebSocket: <1s odds detection
        live_score_poller(model),        # Goalserve REST every 3s: score/event confirmation
    )

async def tick_loop(model):
    """Coroutine 1: every-1s tick, wall-clock synchronized.

    model.t is EFFECTIVE PLAY TIME (halftime excluded), derived from wall clock.
    This prevents drift when MC computation or GC takes longer than 1s,
    while correctly excluding halftime from the time basis.

    Wall clock:  0 ──── 47min ──── 62min ──── 110min
    Play clock:  0 ──── 47min ┃HT┃ 47min ──── 95min
                              (excluded)
    """
    model.kickoff_wall_clock = time.monotonic()
    model.halftime_accumulated = 0.0    # total seconds spent in halftime
    model.halftime_start = None         # set when HALFTIME entered
    tick_count = 0

    while model.engine_phase != FINISHED:
        tick_start = time.monotonic()

        if model.engine_phase in (FIRST_HALF, SECOND_HALF):
            # Effective play time = wall clock elapsed - halftime duration
            wall_elapsed = time.monotonic() - model.kickoff_wall_clock
            model.t = (wall_elapsed - model.halftime_accumulated) / 60  # minutes

            # Step 3.2: remaining expected goals
            μ_H, μ_A = compute_remaining_mu(model)

            # Step 3.4: pricing → P_true is dict, σ_MC is dict (per market)
            P_true, σ_MC = await step_3_4_async(model, μ_H, μ_A)

            # Guard: MC may return None if result is stale or preliminary
            if P_true is None:
                tick_count += 1
                await _sleep_until_next_tick(model, tick_count)
                continue

            # Allow order only if all 3 conditions pass
            order_allowed = (
                not model.cooldown
                and not model.ob_freeze
                and model.event_state == IDLE
            )

            # Send to Phase 4 — P_true and σ_MC are dicts keyed by market
            emit_to_phase4(P_true, σ_MC, order_allowed, model)

        elif model.engine_phase == HALFTIME:
            # Track halftime start (set once on entry)
            if model.halftime_start is None:
                model.halftime_start = time.monotonic()


def emit_to_phase4(P_true: dict, σ_MC: dict, order_allowed: bool, model):
    """Push tick data to Phase 4 signal_generator via asyncio.Queue.

    Queue maxsize=1: if Phase 4 hasn't consumed the previous tick,
    the old tick is replaced (Phase 4 always sees the latest state).
    """
    tick_data = {
        "P_true": P_true,       # dict: {"home_win": 0.55, "over_25": 0.65, ...}
        "σ_MC": σ_MC,           # dict: {"home_win": 0.0022, "over_25": 0.0021, ...}
        "order_allowed": order_allowed,
    }
    # Replace stale tick (non-blocking)
    if model.phase4_queue.full():
        try:
            model.phase4_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    model.phase4_queue.put_nowait(tick_data)

        # ─── Backpressure monitoring ───
        tick_duration = time.monotonic() - tick_start
        tick_latency.observe(tick_duration)  # Prometheus histogram

        if tick_duration > 3.0:
            log.warning(f"Tick took {tick_duration:.2f}s — pricing may be delayed")
        if tick_duration > 1.0:
            log.info(f"Tick overrun: {tick_duration:.2f}s (tick {tick_count})")

        tick_count += 1
        await _sleep_until_next_tick(model, tick_count)


async def _sleep_until_next_tick(model, tick_count: int):
    """Sleep until the next absolute tick time. Skip if already past."""
    next_tick_time = model.kickoff_wall_clock + tick_count
    sleep_duration = next_tick_time - time.monotonic()
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)
    # else: tick was late, proceed immediately (no catch-up)

> **Paper/Live mode invariance:** Phase 3 (pricing) runs identically in paper and live mode.
> The same P_true, σ_MC, and order_allowed values are emitted regardless of trading mode.
> Paper vs live branching occurs exclusively in Phase 4 (execution layer).
> This ensures that paper trading validates the exact same pricing pipeline that live will use.

async def live_odds_listener(model):
    """Coroutine 2: Odds-API WebSocket (<1s) — odds-only detection."""
    async for event in model.live_odds_source.listen():
        if event.type == "odds_spike":
            handle_odds_spike(model, event)
        elif event.type == "match_removed":
            log.info("Match removed from Odds-API — awaiting Live Score confirmation")

async def live_score_poller(model):
    """Coroutine 3: Goalserve Live Score REST (3s polling) — score + event confirmation."""
    async for event in model.live_score_source.listen():
        if event.type == "goal_detected":
            handle_preliminary_goal(model, event)
        elif event.type == "goal_confirmed":
            handle_confirmed_goal(model, event)
        elif event.type == "score_rollback":
            handle_score_rollback(model, event)
        elif event.type == "red_card":
            handle_confirmed_red_card(model, event)
        elif event.type == "period_change":
            handle_confirmed_period(model, event)
        elif event.type == "stoppage_entered":
            model.stoppage_mgr.update_from_live_score(event.minute, event.period)
        elif event.type == "match_finished":
            model.engine_phase = FINISHED
        elif event.type == "source_failure":
            handle_live_score_failure(model)
```

### Event Handlers — Two-Stage Processing (Preliminary -> Confirmed)

#### Preliminary Handler (Goalserve Live Score REST, 3-8s)

```python
def handle_preliminary_goal(model, event: NormalizedEvent):
    """
    Detect score change from Goalserve Live Score.
    Still provisional before VAR confirmation.
    """
    # 1. Immediate ob_freeze + state transition
    model.ob_freeze = True
    model.event_state = PRELIMINARY_DETECTED

    # 2. Infer scoring side (from score difference)
    preliminary_score = event.score
    if preliminary_score[0] > model.S[0]:
        scoring_team = "home"
    elif preliminary_score[1] > model.S[1]:
        scoring_team = "away"
    else:
        log.warning("Score changed but neither team increased — ignoring")
        return

    # 3. Provisional ΔS
    preliminary_delta_S = preliminary_score[0] - preliminary_score[1]

    # 4. Precompute μ asynchronously (executor)
    #    for 0ms P_true output when Live Score confirms
    asyncio.create_task(precompute_preliminary_mu(
        model, preliminary_delta_S, scoring_team
    ))

    # 5. Cache provisional payload
    model.preliminary_cache = {
        "score": preliminary_score,
        "delta_S": preliminary_delta_S,
        "scoring_team": scoring_team,
        "timestamp": event.timestamp,
    }

    log.info(f"PRELIMINARY goal: {model.S} -> {preliminary_score} "
             f"(team={scoring_team})")

def handle_score_rollback(model, event: NormalizedEvent):
    """
    Score decreases in Goalserve Live Score -> likely VAR cancellation.
    If in preliminary state, rollback immediately.
    """
    if model.event_state == PRELIMINARY_DETECTED:
        log.warning(f"Score rollback: {model.preliminary_cache['score']} -> "
                    f"{event.score} — likely VAR cancellation")
        model.event_state = IDLE
        model.ob_freeze = False
        model.preliminary_cache = {}
    else:
        log.warning(f"Score rollback in state {model.event_state} — logging only")

def handle_odds_spike(model, event: NormalizedEvent):
    """
    Detect abrupt odds move from Odds-API WebSocket.
    Could be goal or red card -> set ob_freeze and wait for Goalserve Live Score.
    """
    model.ob_freeze = True
    log.warning(f"Odds-API spike: Δ={event.delta:.3f} — awaiting Live Score confirmation")

def handle_period_change(model, event: NormalizedEvent):
    """Detect period change from Goalserve Live Score.

    Halftime tracking: record start/end to exclude from model.t.
    model.t = effective play time (wall clock - halftime duration).
    """
    if event.period in ("Paused", "Half", "HT"):
        model.engine_phase = HALFTIME
        model.halftime_start = time.monotonic()  # start tracking halftime duration
        log.info(f"HALFTIME detected: model.t={model.t:.2f}min")

    elif event.period in ("2nd Half", "2nd"):
        # Accumulate halftime duration before resuming
        if model.halftime_start is not None:
            model.halftime_accumulated += (time.monotonic() - model.halftime_start)
            log.info(f"Halftime lasted {model.halftime_accumulated:.1f}s")
            model.halftime_start = None  # reset for safety
        model.engine_phase = SECOND_HALF
        log.info(f"SECOND HALF started: model.t={model.t:.2f}min (halftime excluded)")
```

#### Confirmed Handler (Goalserve Live Score REST, 3-8s)

```python
def handle_confirmed_goal(model, event: NormalizedEvent):
    """
    Goal confirmed by Live Score.
    VAR cancellation, scorer, and assist are now verified.
    """
    # 1. Check VAR cancellation
    if event.var_cancelled:
        model.event_state = IDLE
        model.ob_freeze = False
        model.preliminary_cache = {}
        log.info("Goal VAR cancelled — state rolled back")
        return

    # 2. Commit score
    if event.team == "localteam":
        model.S = (model.S[0] + 1, model.S[1])
    else:
        model.S = (model.S[0], model.S[1] + 1)
    model.delta_S = model.S[0] - model.S[1]

    # 3. Recompute μ — reuse preliminary precompute if available
    if (model.preliminary_cache
        and model.preliminary_cache.get("delta_S") == model.delta_S):
        # Reuse precomputed result -> 0ms
        model.μ_H = model.preliminary_cache["μ_H"]
        model.μ_A = model.preliminary_cache["μ_A"]
        log.info("Using pre-computed μ from preliminary stage")
    else:
        # No cache or mismatched delta_S -> recompute
        model.μ_H, model.μ_A = recompute_mu(model)

    # 4. State transition
    model.cooldown = True
    model.ob_freeze = False
    model.event_state = IDLE
    model.preliminary_cache = {}
    asyncio.create_task(cooldown_timer(model, duration=15))

    log.info(f"CONFIRMED goal: S={model.S}, ΔS={model.delta_S}, "
             f"team={event.team}, scorer={event.scorer_id}")

def handle_confirmed_red_card(model, event: NormalizedEvent):
    """
    Red cards can only be confirmed via Live Score.
    In Live Odds, only indirect detection via odds_spike is possible.
    """
    # 1. Markov state transition
    if event.team == "localteam":
        if model.X == 0: model.X = 1      # 11v11 -> 10v11
        elif model.X == 2: model.X = 3    # 11v10 -> 10v10
    else:  # visitorteam
        if model.X == 0: model.X = 2      # 11v11 -> 11v10
        elif model.X == 1: model.X = 3    # 10v11 -> 10v10

    # 2. Recompute μ — reflect gamma^H, gamma^A updates
    model.μ_H, model.μ_A = recompute_mu(model)

    # 3. State transition
    model.cooldown = True
    model.ob_freeze = False
    model.event_state = IDLE
    asyncio.create_task(cooldown_timer(model, duration=15))

    log.info(f"CONFIRMED red card: X={model.X}, team={event.team}")

def handle_confirmed_period(model, event: NormalizedEvent):
    """Period confirmed by Live Score — authoritative."""
    if event.period == "Halftime" and model.engine_phase != HALFTIME:
        log.warning("Halftime confirmed by Live Score")
        model.engine_phase = HALFTIME
        if model.halftime_start is None:
            model.halftime_start = time.monotonic()
    elif event.period == "Finished":
        model.engine_phase = FINISHED
```

#### Helper Functions

```python
async def cooldown_timer(model, duration: int = 15):
    """Cooldown timer: release cooldown after duration seconds."""
    await asyncio.sleep(duration)
    model.cooldown = False
    log.info(f"Cooldown expired after {duration}s")

def handle_live_score_failure(model):
    """5 consecutive Live Score polling failures -> stop new orders."""
    model.ob_freeze = True  # safe mode
    log.error("Live Score source failure — freezing all orders")
```

### ob_freeze Release Conditions

```python
def check_ob_freeze_release(model):
    """
    Called every tick. Check ob_freeze release conditions.

    Release if any one condition is met:
    1. Event detected and confirmed -> state update complete + entered cooldown
    2. 3 consecutive stable ticks (Odds-API move < threshold)
    3. 10-second timeout (false-positive protection)
    """
    if not model.ob_freeze:
        return

    # Condition 1: explained by event (cooldown takes over)
    if model.cooldown:
        model.ob_freeze = False
        return

    # Condition 2: 3-tick stabilization
    if model._ob_stable_ticks >= 3:
        model.ob_freeze = False
        model._ob_stable_ticks = 0
        log.info("ob_freeze released: 3-tick stabilization")
        return

    # Condition 3: 10-second timeout
    elapsed = time.time() - model._ob_freeze_start
    if elapsed >= 10:
        model.ob_freeze = False
        log.info("ob_freeze released: 10s timeout")
```

### Timeline Comparison

**Original design (REST only):**
```
t=0.0s  Goal occurs (on pitch)
t=3~6s  Live Score poll detects score change
t=6.0s  state update + 15s cooldown
t=21s   normal operation resumes
```
-> blind spot: **3-6 seconds**

**Odds-API + Goalserve (2-layer):**
```
t=0.0s  Goal occurs (on pitch)
t=0.5s  bet365 odds jump -> Odds-API WS received -> ob_freeze (Layer 1)
t=0.5s  μ precompute starts (executor)
t=1.5s  Kalshi MM quote reaction (observed by Phase 4, not Phase 3)
t=5.0s  Goalserve Live Score poll: goal confirmed + scorer + VAR status (Layer 2)
t=5.0s  CONFIRMED -> μ commit (reuse precompute, 0ms) + 15s cooldown
t=20s   normal operation resumes
```
-> blind spot: **~0.5 seconds**

### Output

State vector updated every tick:

$$\text{State}(t) = (t,\; S,\; X,\; \Delta S,\; \text{engine\_phase},\; \text{event\_state},\; \text{cooldown},\; \text{ob\_freeze},\; T)$$

---

## Step 3.2: Remaining Expected Goals Calculation

### Goal

From current time t to match end T,
compute remaining expected goals for home μ_H(t, T)
and away μ_A(t, T).

### Integral Structure

Split remaining interval [t, T] at basis-function boundaries into L subintervals:

$$[t, T] = [t, \tau_1) \cup [\tau_1, \tau_2) \cup \cdots \cup [\tau_{L-1}, T]$$

### Markov-Modulated Integral Formula

Apply team-specific gamma:

$$\boxed{\mu_H(t, T) = \sum_{\ell=1}^{L} \sum_{j=0}^{3} \overline{P}_{X(t),j}^{(\ell)} \cdot \exp\!\left(a_H + b_{i_\ell} + \gamma^H_j + \delta_H(\Delta S)\right) \cdot \Delta\tau_\ell}$$

$$\boxed{\mu_A(t, T) = \sum_{\ell=1}^{L} \sum_{j=0}^{3} \overline{P}_{X(t),j}^{(\ell)} \cdot \exp\!\left(a_A + b_{i_\ell} + \gamma^A_j + \delta_A(\Delta S)\right) \cdot \Delta\tau_\ell}$$

| Term | Meaning |
|----|------|
| $\overline{P}_{X(t),j}^{(\ell)}$ | average probability of being in state j during subinterval ℓ |
| $a_T + b_{i_\ell} + \gamma^T_j + \delta_T(\Delta S)$ | instantaneous scoring intensity for team T ∈ {H, A} |
| $\Delta\tau_\ell$ | subinterval length (minutes) |

> **Fix delta(ΔS) — analytic mode only:** hold ΔS at the **current** score difference.
> Future ΔS changes due to goals are not modeled in analytic mode.
> In MC mode (Step 3.4), goals within simulation paths dynamically update ΔS and delta,
> so this approximation does not apply.

### Matrix Exponential Lookup

```python
def get_transition_prob(model, dt_min: float) -> np.ndarray:
    """
    Lookup transition probability from P_grid or P_fine_grid.
    Use fine grid near match end.
    """
    if dt_min <= 5 and hasattr(model, 'P_fine_grid'):
        # Fine grid: 10-second increments (near match end)
        dt_10sec = int(round(dt_min * 6))
        dt_10sec = max(0, min(30, dt_10sec))
        return model.P_fine_grid[dt_10sec]
    else:
        # Standard grid: 1-minute increments
        dt_round = max(0, min(100, round(dt_min)))
        return model.P_grid[dt_round]
```

### Preliminary Precomputation

```python
async def precompute_preliminary_mu(model, preliminary_delta_S, scoring_team):
    """
    Called immediately when Goalserve Live Score detects score change.
    Precompute μ before full confirmation -> 0ms at confirmation.
    """
    loop = asyncio.get_event_loop()

    # delta index
    di = max(0, min(4, preliminary_delta_S + 2))

    # Compute μ_H, μ_A (analytic or MC, in executor)
    if model.X == 0 and preliminary_delta_S == 0 and not model.DELTA_SIGNIFICANT:
        # Analytic - immediate
        μ_H, μ_A = analytical_remaining_mu(model, preliminary_delta_S)
    else:
        # MC - executor
        final_scores = await loop.run_in_executor(
            mc_executor,
            mc_simulate_remaining,
            model.t, model.T, model.S[0], model.S[1],
            model.X, preliminary_delta_S,
            model.a_H, model.a_A, model.b,
            model.gamma_H, model.gamma_A,
            model.delta_H, model.delta_A,
            model.Q_diag, model.Q_off_normalized,
            model.basis_bounds, N_MC,
            int(time.time() * 1000) % (2**31)
        )
        μ_H = np.mean(final_scores[:, 0]) - model.S[0]
        μ_A = np.mean(final_scores[:, 1]) - model.S[1]

    # Store in cache
    model.preliminary_cache["μ_H"] = μ_H
    model.preliminary_cache["μ_A"] = μ_A

    log.info(f"Preliminary μ computed: μ_H={μ_H:.3f}, μ_A={μ_A:.3f}")
```

### Output

Every tick: μ_H(t, T), μ_A(t, T).

---

## Step 3.3: Discrete Shock Handling (Discrete Event Handler)

### Event-Source Role Matrix

| Event | Early Warning (Odds-API WS, <1s) | Detection + Confirmation (Goalserve Live Score REST, 3-8s) |
|--------|------------------------------|------------------------------|
| **Goal** | odds spike -> ob_freeze | score change -> PRELIMINARY, then goal scorer + VAR status -> CONFIRMED |
| **Red card** | odds jump -> ob_freeze (type unknown) | redcards diff -> CONFIRMED |
| **Halftime** | — | period "HT" -> engine_phase transition |
| **Stoppage time** | — | minute > 45 / > 90 -> T rolling |
| **VAR review** | odds oscillation (up then down) | var_cancelled field |
| **VAR cancellation** | — | score decrease -> score_rollback, var_cancelled=True |
| **Substitution** | — | substitutions diff -> **intensity modifier (extension)** |

### Event 1: Goal — Two-Stage Processing

**Stage 0 — Early Warning (Odds-API WS, <1s):**
- detect odds spike -> ob_freeze = True (no score info yet)

**Stage 1 — Preliminary (Goalserve Live Score, 3-8s):**
- detect score change -> `PRELIMINARY_DETECTED`
- μ precompute using provisional ΔS
- block Phase 4 orders

**Stage 2 — Confirmed (Goalserve Live Score, next poll):**
- check var_cancelled
- **if not cancelled:** commit S -> apply δ_H(ΔS), δ_A(ΔS) -> commit μ -> cooldown 15s
- **if cancelled:** rollback state -> release ob_freeze

### Event 2: Red Card — Confirmed Only via Goalserve Live Score

Odds-API can only detect "something happened" via abrupt odds movement.
Abrupt odds move without score change -> possible red card or VAR review.

| Transition | Trigger | gamma^H change | gamma^A change |
|------|--------|----------|----------|
| 0 -> 1 | home dismissal | 0 -> gamma^H₁ < 0 (home down) | 0 -> gamma^A₁ > 0 (away up) |
| 0 -> 2 | away dismissal | 0 -> gamma^H₂ > 0 (home up) | 0 -> gamma^A₂ < 0 (away down) |
| 1 -> 3 | additional away dismissal | gamma^H₁ -> gamma^H₁+gamma^H₂ | gamma^A₁ -> gamma^A₁+gamma^A₂ |
| 2 -> 3 | additional home dismissal | gamma^H₂ -> gamma^H₁+gamma^H₂ | gamma^A₂ -> gamma^A₁+gamma^A₂ |

When red card is confirmed, μ_H and μ_A move in **opposite directions**:
home dismissal -> μ_H down + μ_A up.

### Event 3: Halftime

| Action | Early Warning (Odds-API) | Authoritative (Goalserve Live Score) |
|------|-----------------|-------------------|
| first-half end | odds frozen / no updates | period="HT" -> HALFTIME |
| second-half start | odds resume | status change -> SECOND_HALF |

### Event 4: Stoppage Time

Goalserve Live Score provides minute values for stoppage time detection.
Handled in detail in Step 3.5.

### Cooldown

| Item | Value |
|------|---|
| detection -> confirmation delay | <1s (Odds-API spike) + 3-8s (Goalserve Live Score) |
| cooldown duration | **15s** (from confirmation time) |
| P_true calculation during cooldown | continues (monitoring) |
| orders during cooldown | **blocked** |

### Rapid Sequential Events (e.g., goals at 80' and 81')

When a new confirmed event arrives **during cooldown** from a previous event:

**Rule 1 — State updates are never blocked by cooldown.**
Cooldown blocks *orders*, not *state updates*. A goal at 81' must immediately update
S, ΔS, and μ even if the 80' cooldown is still active. Otherwise P_true would be
stale for 15+ seconds.

**Rule 2 — Cooldown timer resets on each confirmed event.**
If a goal at 80' starts a 15s cooldown (expires at 80'+15s), and a goal at 81' arrives
at 81'+5s (confirmation delay), the cooldown resets to expire at 81'+5s+15s.
This prevents orders from going out between two rapid events.

**Rule 3 — Events during PRELIMINARY state are queued.**
If the engine is in PRELIMINARY_DETECTED (waiting for Live Score confirmation of event A)
and a second event B arrives, event B is queued. When event A is confirmed or
times out, event B is processed from the queue.

```python
class EventQueue:
    """Handles events that arrive while another event is being processed."""

    def __init__(self):
        self._queue: List[NormalizedEvent] = []

    def enqueue(self, event: NormalizedEvent):
        self._queue.append(event)
        log.info(f"Event queued (state={event.type}): {len(self._queue)} pending")

    def drain(self, model) -> List[NormalizedEvent]:
        """Process all queued events after current event resolves."""
        events = self._queue.copy()
        self._queue.clear()
        return events

def handle_confirmed_goal_v2(model, event: NormalizedEvent):
    """
    Goal confirmed — always update state, reset cooldown.
    """
    if event.var_cancelled:
        model.event_state = IDLE
        model.ob_freeze = False
        model.preliminary_cache = {}
        return

    # Always update state (even during cooldown from previous event)
    if event.team == "localteam":
        model.S = (model.S[0] + 1, model.S[1])
    else:
        model.S = (model.S[0], model.S[1] + 1)
    model.delta_S = model.S[0] - model.S[1]

    # Recompute μ
    model.μ_H, model.μ_A = recompute_mu(model)

    # Reset cooldown timer (not extend — restart from now)
    model.cooldown = True
    model.ob_freeze = False
    model.event_state = IDLE
    model.preliminary_cache = {}

    if hasattr(model, '_cooldown_task') and model._cooldown_task:
        model._cooldown_task.cancel()  # cancel previous cooldown
    model._cooldown_task = asyncio.create_task(cooldown_timer(model, duration=15))

    # Drain event queue
    for queued in model.event_queue.drain(model):
        dispatch_event(model, queued)

    log.info(f"CONFIRMED goal (rapid seq ok): S={model.S}, ΔS={model.delta_S}")
```

### Substitution Effects (Extension)

Goalserve Live Score REST provides substitution data in `substitutions.{team}.substitution[]`.
When a substitution is detected via diff, apply the empirical intensity modifier
$\psi_{sub}$ from Phase 1:

```python
def handle_substitution(model, team: str, minute: float):
    """Apply intensity drop modifier on substitution detection."""
    model.n_subs_so_far += 1

    # Apply cumulative substitution intensity modifier
    # ψ_sub < 0 captures average intensity decline per substitution
    model.sub_modifier = math.exp(model.psi_sub * model.n_subs_so_far)

    log.info(f"Substitution detected: team={team}, minute={minute}, "
             f"total_subs={model.n_subs_so_far}, modifier={model.sub_modifier:.3f}")
```

The substitution modifier multiplies the base intensity:

$$\lambda_{adj}(t) = \lambda(t) \cdot \exp(\psi_{sub} \cdot n_{sub}(t))$$

This does **not** trigger cooldown or ob_freeze — substitutions are gradual tactical
changes, not discrete jumps like goals or red cards.

### Output

Re-adjusted state vector (S, X, ΔS, μ_H, μ_A, T) and event_state/cooldown/ob_freeze status.

---

## Step 3.4: Pricing — True Probability Estimation

### Goal

Using remaining expected goals μ_H and μ_A,
estimate true probabilities (P_true)
that can be compared against Kalshi order books.

### Independence Assumption Analysis

With delta(ΔS), when one team scores, both teams' lambda change simultaneously,
so home/away scoring independence breaks.

Even starting from X = 0, ΔS = 0, once first goal occurs, ΔS = ±1,
and if delta(±1) ≠ 0, subsequent intensities are coupled.
Therefore analytic Poisson/Skellam is only a **first-order approximation**.

### Hybrid Pricing

| Condition | Method | Accuracy |
|------|------|--------|
| X=0, ΔS=0, delta not significant | analytic Poisson/Skellam | **exact** |
| X=0, ΔS=0, delta significant | analytic (first-order approx) | **approximate** — ignores delta feedback |
| X≠0 or ΔS≠0 | Monte Carlo simulation | **exact** (with sufficient N) |

> **Practical guide:** if delta is small (|delta| < 0.1), analytic approximation error may be below MC standard error.
> If delta is larger (|delta| >= 0.15), MC is safer even at ΔS = 0.
> With Numba JIT + executor, MC overhead is ~0.5ms/match, so always-MC is also practical.

### Logic A: Analytic Pricing (X=0, ΔS=0)

Let $G = S_H + S_A$ be current total goals:

**Over/Under:**

$$P_{true}(\text{Over } N\text{.5}) = \begin{cases} 1 & \text{if } G > N \\ 1 - \sum_{k=0}^{N-G} \frac{\mu_{total}^k \cdot e^{-\mu_{total}}}{k!} & \text{if } G \leq N \end{cases}$$

**Match Odds (Skellam):**

$$P_{true}(\text{Home Win}) = \sum_{D=1}^{\infty} e^{-(\mu_H + \mu_A)} \left(\frac{\mu_H}{\mu_A}\right)^{D/2} I_{|D|}(2\sqrt{\mu_H \mu_A})$$

Analytic mode executes immediately on main thread (~0.1ms).

### Logic B: Monte Carlo Pricing (X≠0 or ΔS≠0)

#### Numba JIT-compiled MC Core

```python
@njit(cache=True)
def mc_simulate_remaining(
    t_now, T_end, S_H, S_A, state, score_diff,
    a_H, a_A,
    b,                  # shape (6,)
    gamma_H, gamma_A,   # shape (4,) each
    delta_H, delta_A,   # shape (5,) each
    Q_diag,             # shape (4,)
    Q_off,              # shape (4,4) — normalized transition probabilities
    basis_bounds,       # shape (7,)
    N, seed
):
    """
    Returns: final_scores — shape (N, 2)
    Uses team-specific gamma + normalized Q_off.
    """
    np.random.seed(seed)
    results = np.empty((N, 2), dtype=np.int32)

    for sim in range(N):
        s = t_now
        sh, sa = S_H, S_A
        st = state
        sd = score_diff

        while s < T_end:
            # current basis index
            bi = 0
            for k in range(6):
                if s >= basis_bounds[k] and s < basis_bounds[k + 1]:
                    bi = k
                    break

            # delta index: ΔS -> {0:<=-2, 1:-1, 2:0, 3:+1, 4:>=+2}
            di = max(0, min(4, sd + 2))

            # use team-specific gamma
            lam_H = np.exp(a_H + b[bi] + gamma_H[st] + delta_H[di])
            lam_A = np.exp(a_A + b[bi] + gamma_A[st] + delta_A[di])
            lam_red = -Q_diag[st]
            lam_total = lam_H + lam_A + lam_red

            if lam_total <= 0:
                break

            # waiting time to next event
            dt = -np.log(np.random.random()) / lam_total
            s_next = s + dt

            # basis boundary or match-end check
            next_bound = T_end
            for k in range(7):
                if basis_bounds[k] > s:
                    next_bound = min(next_bound, basis_bounds[k])
                    break

            if s_next >= next_bound:
                s = next_bound
                continue

            s = s_next

            # sample event
            u = np.random.random() * lam_total
            if u < lam_H:
                sh += 1
                sd += 1
            elif u < lam_H + lam_A:
                sa += 1
                sd -= 1
            else:
                # use normalized Q_off
                cum = 0.0
                r = np.random.random()
                for j in range(4):
                    if j == st:
                        continue
                    cum += Q_off[st, j]
                    if r < cum:
                        st = j
                        break

        results[sim, 0] = sh
        results[sim, 1] = sa

    return results
```

**Performance:**

| Implementation | Latency | 10 Matches in Parallel |
|------|----------|-----------|
| pure Python | ~50ms | ~500ms ❌ |
| **Numba @njit** | **~0.5ms** | **~5ms ✅** |

#### Executor Decoupling

```python
# MC configuration
N_MC = 50_000  # Number of Monte Carlo paths per pricing call
               # Yields σ_MC ≈ 0.2pp for typical probabilities
               # Latency: ~0.5ms with Numba @njit

mc_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mc")

async def step_3_4_async(model, μ_H, μ_A):
    """Non-blocking async pricing that does not block main event loop."""

    if model.X == 0 and model.delta_S == 0 and not model.DELTA_SIGNIFICANT:
        P_true = analytical_pricing(μ_H, μ_A, model.S)
        # Analytical mode has no sampling uncertainty, but model error still exists.
        # Use a synthetic σ floor so P_cons provides conservative protection
        # during the most common state (0-0 first half, no red cards).
        # Without this, P_cons = P_true and z-adjustment gives zero protection.
        σ_MC = {
            market: max(math.sqrt(p * (1 - p) / N_MC), 0.005)
            for market, p in P_true.items()
        }
        return P_true, σ_MC

    else:
        loop = asyncio.get_event_loop()
        model._mc_version += 1
        my_version = model._mc_version

        seed = hash((model.match_id, model.t, model.S[0],
                      model.S[1], model.X)) % (2**31)

        final_scores = await loop.run_in_executor(
            mc_executor,
            mc_simulate_remaining,
            model.t, model.T, model.S[0], model.S[1],
            model.X, model.delta_S,
            model.a_H, model.a_A, model.b,
            model.gamma_H, model.gamma_A,
            model.delta_H, model.delta_A,
            model.Q_diag, model.Q_off_normalized,
            model.basis_bounds, N_MC, seed
        )

        # stale check
        if my_version != model._mc_version:
            return None, None
        if model.event_state == PRELIMINARY_DETECTED:
            return None, None

        P_true = aggregate_markets(final_scores, model.S)
        σ_MC = compute_mc_stderr(P_true, N_MC)
        return P_true, σ_MC
```

> **Deterministic MC seeds:** use seeds based on `hash(match_id, t, S_H, S_A, X)`
> to guarantee reproducibility in identical states.
> This enables exact reproduction for debugging and backtesting.

### Market Probability Estimation (Aggregate MC Results)

A single MC batch yields probabilities for **all markets simultaneously**:

```python
def aggregate_markets(final_scores: np.ndarray, current_S: Tuple[int,int]) -> dict:
    N = len(final_scores)
    sh = final_scores[:, 0]
    sa = final_scores[:, 1]
    total = sh + sa

    return {
        # Over/Under
        "over_15": np.mean(total > 1),
        "over_25": np.mean(total > 2),
        "over_35": np.mean(total > 3),

        # Match Odds
        "home_win": np.mean(sh > sa),
        "draw": np.mean(sh == sa),
        "away_win": np.mean(sh < sa),

        # Both Teams to Score
        "btts_yes": np.mean((sh > 0) & (sa > 0)),

        # Correct Score (top-probability scores only)
        # ...
    }
```

### Per-Market Monte Carlo Standard Error

```python
def compute_mc_stderr(P_true: dict, N: int) -> dict:
    """
    Compute standard error for each market's probability estimate.

    σ_MC varies by market because bernoulli variance = p(1-p)/N.
    P(home_win)=0.50 → σ=0.0022, P(over_2.5)=0.90 → σ=0.0013 (at N=50,000).

    Phase 4 uses per-market σ_MC to compute P_cons:
    Buy Yes home_win: P_cons = 0.50 - 1.645 * 0.0022 = 0.4964
    Buy Yes over_2.5: P_cons = 0.90 - 1.645 * 0.0013 = 0.8979
    """
    import math
    return {
        market: math.sqrt(p * (1 - p) / N) if 0 < p < 1 else 0.0
        for market, p in P_true.items()
    }
```

### Output

Every second:
- `P_true(t): dict` — true probability by active market (e.g., `{"home_win": 0.55, "over_25": 0.65, ...}`)
- `σ_MC(t): dict` — Monte Carlo standard error per market (e.g., `{"home_win": 0.0022, "over_25": 0.0021, ...}`)
- `pricing_mode`: Analytical / Monte Carlo

> **Analytical mode:** σ_MC uses a synthetic floor `max(sqrt(p*(1-p)/N_MC), 0.005)` per market.
> This ensures P_cons always has conservative protection, even when there is no MC sampling uncertainty.
> Without this floor, the 0-0 first half (60-70% of match time) would have P_cons = P_true with zero protection.

---

## Step 3.5: Real-Time Stoppage-Time Handling

### Goal

Adjust $T_{exp}$ from Phase 2 in real time as match evolves.

### Stoppage Time Updates (Goalserve Live Score)

Goalserve Live Score REST provides minute/timer fields for stoppage time tracking.

```python
class StoppageTimeManager:
    def __init__(self, T_exp: float, rolling_horizon: float = 1.5):
        self.T_exp = T_exp
        self.rolling_horizon = rolling_horizon
        self.first_half_stoppage = False
        self.second_half_stoppage = False

    def update_from_live_score(self, minute: float, period: str) -> float:
        """Live Score REST — authoritative minute updates (3-8s)."""
        return self._compute_T(minute, period)

    def _compute_T(self, minute: float, period: str) -> float:
        # Phase B: first-half stoppage
        if period in ("1st Half", "1st") and minute > 45:
            if not self.first_half_stoppage:
                self.first_half_stoppage = True
            # Keep T_game unchanged; only adjust basis boundary
            # (first-half end is finalized by halftime event)
            return self.T_exp

        # Phase C: second-half stoppage
        if period in ("2nd Half", "2nd") and minute > 90:
            if not self.second_half_stoppage:
                self.second_half_stoppage = True
            # Rolling update for T_game
            return minute + self.rolling_horizon

        # Phase A: regular time
        return self.T_exp
```

> **Phase B vs Phase C distinction:**
> In Phase B (first-half stoppage), do not modify T_game.
> First-half end is determined by halftime entry event,
> so T_game rolling applies only in second-half stoppage (Phase C).

### Stoppage-Time Uncertainty Modeling (Extension — Implemented)

In Monte Carlo, each simulation path samples its own match end time from
the per-league stoppage time distribution fitted in Phase 1 Step 1.1:

$$T_{path} = 90 + \alpha_1^{(path)} + \alpha_2^{(path)}$$

where $\alpha_1^{(path)} \sim \text{LogNormal}(\mu_1, \sigma_1)$ and
$\alpha_2^{(path)} \sim \text{LogNormal}(\mu_2, \sigma_2)$.

```python
def mc_sample_T_end(stoppage_dist: dict, n_paths: int, rng) -> np.ndarray:
    """Sample path-level match end times from fitted stoppage distribution.

    Args:
        stoppage_dist: {"period_1": {"shape", "scale"}, "period_2": {...}}
        n_paths: Number of MC paths.
        rng: NumPy random generator.

    Returns:
        Array of T_end values, shape (n_paths,).
    """
    alpha_1 = rng.lognormal(
        mean=np.log(stoppage_dist["period_1"]["scale"]),
        sigma=stoppage_dist["period_1"]["shape"],
        size=n_paths,
    )
    alpha_2 = rng.lognormal(
        mean=np.log(stoppage_dist["period_2"]["scale"]),
        sigma=stoppage_dist["period_2"]["shape"],
        size=n_paths,
    )
    return 90.0 + alpha_1 + alpha_2
```

**Benefits:**
- Naturally propagates match-duration uncertainty into P_true
- Late-game probability estimates become less confident
  (wider MC spread when T_end is uncertain)
- Captures the asymmetric distribution of stoppage time (right-skewed)

**In second-half stoppage (Phase C):** once actual stoppage time is known
from live minute data, the distribution is truncated:
$\alpha_2 \mid \alpha_2 > \text{elapsed\_stoppage}$ — only sample from
the conditional distribution where stoppage exceeds what has already passed.

### Output

Real-time updated T.

---

## Phase 3 -> Phase 4 Handoff

| Item | Value | Update Frequency |
|------|---|-------------|
| P_true(t) | true probability by market | every 1s |
| σ_MC(t) | MC standard error | every 1s (analytic: 0) |
| **order_allowed** | **NOT cooldown AND NOT ob_freeze AND event_state == IDLE** | every 1s + on events |
| pricing_mode | Analytical / Monte Carlo | switches on events |
| μ_H, μ_A | remaining expected goals | every 1s (for logging) |
| engine_phase | current match phase | on period change |
| **event_state** | **IDLE / PRELIMINARY / CONFIRMED** | on events |
| **P_bet365(t)** | **bet365 in-play implied probability (via Odds-API WS)** | **every push (<1s)** |

---

## Phase 3 Pipeline Summary

```
[Kickoff - engine_phase: FIRST_HALF]
              |
              v
+-----------------------------------------------------------------+
|  Step 3.1: State Machine + 2-Layer Event Detection               |
|                                                                 |
|  +------------------------+  +-------------------------------+  |
|  | Odds-API WS            |  | Goalserve Live Score          |  |
|  | (<1s, PUSH)            |  | (3-8s, REST polling)          |  |
|  |                        |  |                               |  |
|  | odds spike:            |  | score change:                 |  |
|  | -> ob_freeze           |  | -> PRELIMINARY -> CONFIRMED   |  |
|  |                        |  | -> VAR check                  |  |
|  | bet365 odds:           |  |                               |  |
|  | -> P_bet365 (for Ph4)  |  | red card: -> CONFIRMED        |  |
|  |                        |  | period:   -> halftime/stoppage |  |
|  +-----------+------------+  +---------------+---------------+  |
|              |                                |                  |
|              +---------------+----------------+                  |
|                              v                                   |
|              Event State Machine                                 |
|              IDLE -> PRELIMINARY -> CONFIRMED -> COOLDOWN -> IDLE|
|                             \ FALSE_ALARM -> IDLE                |
|                             \ VAR_CANCELLED -> IDLE              |
|                                                                  |
|              order_allowed = NOT cooldown                        |
|                             AND NOT ob_freeze                    |
|                             AND event_state == IDLE              |
|                                                                  |
|  Note: Kalshi order book sync is handled by Phase 4 Step 4.1|
+------------------+----------------------------------------------+
                   |
        +----------+----------+
        | (every 1s tick)    | (on event detection)
        v                     v
+------------------+  +----------------------------------------+
|  Step 3.2:       |  |  Step 3.3: Discrete Shock Handling      |
|  Remaining μ     |  |                                        |
|                  |  |  • Goal: preliminary -> confirmed (VAR)|
|  • Piecewise int |  |  • Red card: X transition, gamma^H/A   |
|  • P_grid lookup |  |  • Halftime: freeze/resume             |
|  • Team gamma    |  |  • Stoppage: T rolling (dual source)   |
|  • delta(ΔS) adj |  |  • μ precompute (preliminary)          |
|  Output: μ_H,μ_A |  +--------+-------------------------------+
+--------+---------+           |
         |                     |
         +----------+----------+
                    v
+-----------------------------------------------------------------+
|  Step 3.4: Pricing (True Probability)                            |
|                                                                 |
|  +-----------------------+  +------------------------------+    |
|  | X=0, ΔS=0, delta not  |  | Otherwise                    |    |
|  | significant?          |  | -> Numba MC (ThreadPool)     |    |
|  | -> analytic (0.1ms)   |  |    N=50000, ~0.5ms/match     |    |
|  |    σ_MC = 0           |  |    deterministic seed         |    |
|  +-----------+-----------+  |    stale + preliminary check  |    |
|              |              +--------------+---------------+    |
|              +----------+-------------------+                    |
|                         v                                        |
|  Output: P_true(t), σ_MC(t), pricing_mode                        |
+------------------+----------------------------------------------+
                   |
                   v
+-----------------------------------------------------------------+
|  Step 3.5: Stoppage-Time Handling (Goalserve Live Score)         |
|  • Live Score timer (3-8s)                                      |
|  • Phase B (1H): keep T_game, confirm via halftime              |
|  • Phase C (2H): T = minute + rolling 1.5 min                  |
|  • [EXT] MC: sample T_end from LogNormal stoppage distribution  |
|  • [EXT] Substitution modifier: ψ_sub intensity drop            |
+------------------+----------------------------------------------+
                   |
                   v
         [Phase 4: Arbitrage & Execution]
         (order_allowed + P_true + σ_MC + P_bet365 from Odds-API)
```

---

## Step 3.6: In-Play Backtest — Live Pricing Validation

### Goal

Validate that the Phase 3 live pricing engine produces accurate, calibrated P_true
values at arbitrary match minutes — not just at kickoff or full time.

Phase 1 optimizes the MMPP parameters (b, gamma, delta, Q) using full-match interval data,
but never tests the live pricing pipeline that converts those parameters into
minute-by-minute P_true values. This step closes that gap.

### What This Validates (Gaps Filled)

| Gap | How It Is Tested |
|-----|-----------------|
| In-play P_true accuracy at arbitrary minutes | Brier score computed at every snapshot, not just pre-match |
| Time-decay trajectory | P_true monotonicity checked between events as remaining time shrinks |
| MC vs analytical agreement | At X=0, dS=0: both paths run, max divergence measured |
| Stoppage time modeling | T_end shifts applied; compare pricing with fixed T=90 vs dynamic T |
| State machine correctness under real event sequences | Every historical match exercises goal/red card/halftime/VAR paths |
| Profitability against recorded market prices | Simulated P&L using recorded P_kalshi from tick_snapshots |

### What This Cannot Validate

| Remaining Gap | Why |
|--------------|-----|
| Execution quality (latency, partial fills) | Backtest assumes instant fills; real trading has network delay and queue position |
| Market impact | Your orders move the Kalshi book; backtest ignores this |
| Liquidity and spread | Backtest uses recorded VWAP; real depth varies |
| Data feed reliability | Odds-API WebSocket drops and Goalserve delayed polls cannot be simulated from historical data |
| Regime shifts | Model trained on past seasons may not generalize to new leagues or rule changes |
| Concurrent position risk | Multiple simultaneous matches sharing a bankroll are tested independently |
| Odds-API spike → ob_freeze timing | Historical data has no recorded odds movements; odds_spike early warning cannot be replayed, only Goalserve Live Score events are reconstructed |

### Data Sources

**Primary: `historical_matches` table (always available)**

The `summary` JSONB column contains goal events with minutes, team, extra_min,
and `var_cancelled`. Red cards are in `summary.redcards`. This is sufficient to
reconstruct a NormalizedEvent sequence for any completed match.

```
historical_matches.summary -> goals[] -> {minute, extra_min, team, result, var_cancelled}
historical_matches.summary -> redcards[] -> {minute, team}
historical_matches -> ht_score_h, ht_score_a (halftime detection)
historical_matches -> added_time_1, added_time_2 (stoppage time)
```

**Secondary: `tick_snapshots` + `event_logs` tables (available after live/paper runs)**

Once the system has run on live matches (paper or real), these tables contain
recorded P_kalshi and P_bet365 at every tick. This enables profitability simulation
against actual market prices — not just model-vs-outcome validation.

### Architecture

```
src/calibration/step_3_6_backtest.py

Input:
  - Production params from data/parameters/production/
  - historical_matches table (or tick_snapshots for market-price backtest)

Output:
  - backtest_report.json (metrics + per-match details)
  - Per-minute calibration curve
  - Simulated P&L (if market prices available)
```

### Step 3.6.1: Event Reconstruction

Convert historical match data into ReplayEngine-compatible NormalizedEvent lists.

```python
def reconstruct_events(match: dict) -> list[NormalizedEvent]:
    """Build time-ordered event list from historical_matches row.

    Extracts from summary JSONB:
      - Goals: preliminary (live_odds, t) + confirmed (live_score, t+5s)
      - Red cards: confirmed (live_score)
      - Halftime: inferred from ht_score existence, placed at minute 45
      - Second half start: minute 45
      - Match finished: minute 90 + added_time_1 + added_time_2

    Returns events sorted by timestamp.
    """
```

Key details:
- Goals produce two events: preliminary at `minute`, confirmed at `minute + 5s`
- `var_cancelled=True` goals produce a VAR cancellation event instead of confirmation
- Own goals (`result == "Own Goal"`) invert the scoring team
- `extra_min` goals: effective minute = `minute + extra_min` (e.g., 90+3 = 93)
- Red cards produce a single confirmed event
- Halftime is inferred: if `ht_score_h` and `ht_score_a` exist, insert at minute 45
- Match end: `T_m = 90 + added_time_1 + added_time_2`

### Step 3.6.2: Replay Execution

For each match, run the ReplayEngine with production parameters and collect snapshots.

```python
def run_single_match_backtest(
    match: dict,
    params: ReplayModelParams,
    tick_interval: float = 1.0,
) -> list[Snapshot]:
    """Replay one match and return snapshots."""
    events = reconstruct_events(match)
    engine = ReplayEngine(params)
    return engine.replay_with_ticks(events, tick_interval=tick_interval)
```

Two modes:
- **Event-only** (`replay`): Fast; one snapshot per event. Use for large-scale validation.
- **Tick-based** (`replay_with_ticks`, 1-min intervals): Tests time-decay trajectory.
  Use 1-minute ticks (not 1-second) for backtest to keep runtime manageable
  (~98 snapshots per match vs ~5,880).

### Step 3.6.3: Metrics

#### Metric 1: In-Play Brier Score

At each snapshot where the match outcome is known (post-match),
compute Brier score for each market.

```
BS_inplay = (1/N) * sum over all snapshots [ (P_true_market - outcome)^2 ]
```

Segment by time bin (0-15, 15-30, 30-45, 45-60, 60-75, 75-90+) to detect
which periods have weak calibration.

Expected behavior: BS should decrease as match progresses
(more information = more accurate prediction).

#### Metric 2: Calibration Curve (In-Play)

Bin all snapshots by predicted probability (0.0-0.1, 0.1-0.2, ..., 0.9-1.0).
Within each bin, compute actual outcome frequency.
Plot reliability diagram.

Threshold: max deviation <= 7% (relaxed from Phase 1's 5% because
in-play pricing operates under more uncertainty).

#### Metric 3: Monotonicity Check

Between consecutive event-free snapshots (tick-only, no goals/cards),
verify that P_true moves smoothly in the expected direction:
- P(home_win) should not jump > 2pp between ticks with no event
- P(over 2.5) should change monotonically when score is fixed

Violations indicate numerical instability or stoppage-time artifacts.

#### Metric 4: MC vs Analytical Consistency

For snapshots where X=0 and dS=0 (analytical path is used),
also run MC pricing and compare.

```
max_divergence = max over snapshots | P_analytical - P_mc |
```

Threshold: max_divergence <= 1pp for N_MC=50,000.

#### Metric 5: Directional Correctness

After each goal event, verify:
- Home goal: P(home_win) increases, P(away_win) decreases
- Away goal: P(away_win) increases, P(home_win) decreases
- Any goal: P(over 2.5) increases (if total goals <= 3)

After each red card:
- Team with red card: scoring rate should decrease (mu decreases)

Threshold: 100% directional correctness (any violation is a bug).

#### Metric 6: Simulated P&L (requires tick_snapshots)

If recorded P_kalshi values are available from prior live/paper runs,
simulate trading decisions at each snapshot:

```
For each snapshot:
  edge = P_true - P_kalshi
  if edge > threshold:
    record simulated entry
  apply Phase 4 exit logic at subsequent snapshots
  settle at match end
```

Compute: total P&L, Sharpe ratio, max drawdown, edge realization ratio.

### Step 3.6.4: Go/No-Go Criteria

All criteria are evaluated across the full backtest set (minimum 200 matches).

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| In-play BS (home_win) | < 0.20 | Reasonable in-play accuracy |
| In-play BS by time bin | Decreasing trend across bins | More info = better prediction |
| Calibration max deviation | <= 7% | In-play tolerance |
| Monotonicity violations | < 1% of tick-only snapshots | Numerical stability |
| MC vs analytical max divergence | <= 1pp | Pricing consistency |
| Directional correctness | 100% | Any failure is a bug |
| Simulated P&L (if available) | > 0 over full period | Edge exists against market |
| Simulated max drawdown (if available) | < 25% | Risk is bounded |

**GO**: All criteria pass. Phase 3 pricing engine is validated for live use.

**NO-GO**: Any criterion fails. Investigate and fix before proceeding to live trading.
Common failure modes:
- BS not decreasing by time bin: b parameter shape may be wrong (revisit Step 1.4)
- Calibration off: a_H/a_A initialization from Phase 2 may be biased
- Monotonicity violations: stoppage time jumps or bin-boundary discontinuities
- MC/analytical divergence: bug in analytical pricing formulas
- Directional failure: event handler or delta/gamma sign error (critical bug)

### Step 3.6.5: Orchestration

```python
async def run_phase3_backtest(
    config: SystemConfig,
    output_dir: str = "output/backtest_phase3",
    tick_interval: float = 1.0,
    max_matches: int | None = None,
) -> BacktestReport:
    """Run full Phase 3 in-play backtest.

    1. Load production params from data/parameters/production/
    2. Load completed matches from historical_matches
    3. Reconstruct events and replay each match
    4. Compute all metrics
    5. Evaluate Go/No-Go
    6. Save report
    """
```

```bash
# Run backtest
python -m src.calibration.step_3_6_backtest

# With options
python -m src.calibration.step_3_6_backtest --config config/system.yaml \
    --output output/backtest_phase3 \
    --tick-interval 1.0 \
    --max-matches 500
```

Output:
```
output/backtest_phase3/
├── backtest_report.json          # Full metrics + Go/No-Go verdict
├── per_match_details.json        # Per-match BS, directional checks
├── calibration_curve.json        # Binned reliability data
└── time_bin_brier.json           # BS by 15-min time bin
```

### Integration with Calibration Pipeline

Step 3.6 runs **after** Phase 1 calibration (Step 1.5) and **before** paper trading.
It uses the same production parameters that Phase 1 produces.

```
Phase 1 (Steps 1.1-1.5)
  |
  v
Production params saved
  |
  v
Step 3.6: In-Play Backtest    <-- validates live pricing with those params
  |
  v
GO? --> Phase 2 + 3 + 4 paper trading
NO-GO? --> Revisit Phase 1 or fix Phase 3 engine bugs
```

This step is re-run whenever:
- Phase 1 parameters are retrained (CRON 1 recalibration)
- Phase 3 engine logic changes (event handler, pricing, state machine)
- New historical data is backfilled