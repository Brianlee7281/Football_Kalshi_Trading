#!/usr/bin/env python3
"""Historical Match Replay — Phase 2 → Phase 3 end-to-end validation tool.

Loads a completed EPL match from Goalserve commentaries, simulates the
event sequence minute-by-minute using Phase 3 pricing, and produces a
P_true(t) chart for all markets.

Match: Newcastle United 4-1 Chelsea (25.11.2023, EPL)
  Goals: Isak 13', Sterling 23', Lascelles 60', Joelinton 61',
         Anthony Gordon 83'
  Red card: Reece James (Chelsea) 73'
  Halftime: 1-1

Phase 1 params are loaded from outputs/params_1204.json if cached,
otherwise computed from EPL 2023-24 data (~60s on first run).

Output:
  outputs/replay_plot.png — P_true(t) chart for 4 markets
  Console — ticks, P_true before/after each goal, halftime duration

Usage:
  python scripts/replay_match.py

Reference: docs/implementation_roadmap.md §Task 4.6
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.calibration.phase1_worker import (
    fetch_season_commentaries,
    params_to_json,
    run_phase1,
)
from src.clients.goalserve import GoalserveClient
from src.common.logging import setup_logging
from src.common.types import Phase2Result
from src.engine.compute_mu import compute_remaining_mu
from src.engine.event_handlers import handle_confirmed_goal, handle_confirmed_red_card
from src.engine.mc_pricing import step_3_4_async
from src.engine.model import (
    FIRST_HALF,
    HALFTIME,
    SECOND_HALF,
    LiveFootballQuantModel,
)
from src.engine.period_handler import _enter_halftime, _enter_second_half
from src.prematch.step_2_3_backsolve import backsolve_from_mle
from src.prematch.step_2_5_initialization import precompute_P_fine_grid, precompute_P_grid

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
_PARAMS_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "params_1204.json",
)
_PLOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "replay_plot.png",
)

EPL_LEAGUE_ID = 1204
EPL_SEASON = "2023-2024"

# Match: Newcastle United 4-1 Chelsea, 25.11.2023
MATCH_DATE = "25.11.2023"
MATCH_ID = "3837869"

# Known events (minute → (team, event_type))
_HOME_GOALS = [13, 60, 61, 83]   # Newcastle goals
_AWAY_GOALS = [23]                # Chelsea goals
_RED_CARDS = [(73, "visitorteam")]  # Reece James

# Timing
_HT_MINUTE = 48   # end of first half (45 + 3 added)
_HT_RESUME = 63   # second half start (HT = 15 min)
_ADDEDTIME_1 = 3
_ADDEDTIME_2 = 5
_T_EXP = 90.0 + _ADDEDTIME_1 + _ADDEDTIME_2  # 98

# Pre-match expected goals (EPL average with home advantage for this fixture)
_PRE_MATCH_MU_H = 1.65
_PRE_MATCH_MU_A = 1.10


# ---------------------------------------------------------------------------
# Environment + API key loading
# ---------------------------------------------------------------------------


def _load_env() -> None:
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Phase 1 params: load from cache or compute
# ---------------------------------------------------------------------------


async def _load_or_compute_params(api_key: str) -> dict:  # type: ignore[type-arg]
    """Load EPL Phase 1 params from cache file or compute from scratch."""
    os.makedirs(os.path.dirname(_PARAMS_CACHE), exist_ok=True)

    if os.path.exists(_PARAMS_CACHE):
        print(f"[Phase 1] Loading cached EPL params from {_PARAMS_CACHE}")
        with open(_PARAMS_CACHE) as f:
            return json.load(f)  # type: ignore[no-any-return]

    print(f"[Phase 1] Running EPL 2023-24 calibration (~60s, caching to {_PARAMS_CACHE}) ...")
    t0 = time.monotonic()

    async with GoalserveClient(api_key) as client:
        matches = await fetch_season_commentaries(
            client, EPL_LEAGUE_ID, EPL_SEASON
        )

    print(f"[Phase 1]   Fetched {len(matches)} EPL matches")
    result = run_phase1(matches, league_id=EPL_LEAGUE_ID, skip_validation=True)
    params = params_to_json(result)

    with open(_PARAMS_CACHE, "w") as f:
        json.dump(params, f, indent=2)

    elapsed = time.monotonic() - t0
    print(f"[Phase 1]   Done in {elapsed:.1f}s → saved to {_PARAMS_CACHE}")
    return params


# ---------------------------------------------------------------------------
# Match data loading
# ---------------------------------------------------------------------------


async def _load_match(api_key: str) -> dict:  # type: ignore[type-arg]
    """Load Newcastle vs Chelsea commentary data from Goalserve."""
    print(f"[Match]   Loading Newcastle vs Chelsea ({MATCH_DATE}) ...")
    async with GoalserveClient(api_key) as client:
        matches = await client.get_commentaries_by_league(
            league_id=EPL_LEAGUE_ID, date=MATCH_DATE
        )
    for m in matches:
        if str(m.get("@id", "")) == MATCH_ID:
            print(
                f"[Match]   Found: {m['localteam']['@name']} "
                f"{m['localteam']['@goals']}-{m['visitorteam']['@goals']} "
                f"{m['visitorteam']['@name']}"
            )
            return m
    # Fallback: use first match with goals
    for m in matches:
        try:
            total = int(m.get("localteam", {}).get("@goals", 0) or 0) + int(
                m.get("visitorteam", {}).get("@goals", 0) or 0
            )
            if total >= 2:
                print(f"[Match]   Using fallback match: {m.get('@id')}")
                return m
        except (ValueError, TypeError):
            pass
    raise RuntimeError(f"No suitable match found for {MATCH_DATE}")


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------


def _build_model(params: dict) -> LiveFootballQuantModel:  # type: ignore[type-arg]
    """Build LiveFootballQuantModel from Phase 1 params + MLE backsolve."""
    b = np.array(params["b"], dtype=np.float64)

    # Backsolve a_H, a_A from pre-match expected goals (MLE path)
    bs = backsolve_from_mle(
        _PRE_MATCH_MU_H,
        _PRE_MATCH_MU_A,
        b,
        alpha_1_mean=float(_ADDEDTIME_1),
        alpha_2_mean=float(_ADDEDTIME_2),
    )

    phase2_result = Phase2Result(
        a_H=bs.a_H,
        a_A=bs.a_A,
        C_time=bs.C_time,
        verdict="GO",
    )

    model = LiveFootballQuantModel.from_phase2(
        phase2_result,
        params,
        match_id=MATCH_ID,
        league_id=EPL_LEAGUE_ID,
        trading_mode="paper",
    )

    model.T_exp = _T_EXP
    model.delta_significant = False  # use analytical mode when possible

    # Precompute P_grid + P_fine_grid for compute_remaining_mu
    Q = model.Q
    model.P_grid = precompute_P_grid(Q)
    model.P_fine_grid = precompute_P_fine_grid(Q)

    print(
        f"[Model]   a_H={model.a_H:.4f}, a_A={model.a_A:.4f}, "
        f"C_time={model.C_time:.4f}, T_exp={model.T_exp}"
    )
    return model


# ---------------------------------------------------------------------------
# Simulation loop (minute-by-minute replay)
# ---------------------------------------------------------------------------


async def _simulate(model: LiveFootballQuantModel) -> list[dict]:  # type: ignore[type-arg]
    """Run minute-by-minute simulation from kickoff to T_exp.

    Returns list of tick records with minute, score, P_true, pricing_mode.
    """
    ticks: list[dict] = []  # type: ignore[type-arg]
    model.engine_phase = FIRST_HALF

    effective_t = 0.0  # play minutes (halftime excluded)
    ht_duration = float(_HT_RESUME - _HT_MINUTE)

    for real_minute in range(0, int(_T_EXP) + 1):
        minute_f = float(real_minute)

        # ── Period transitions ────────────────────────────────────────────
        if minute_f == _HT_MINUTE and model.engine_phase == FIRST_HALF:
            _enter_halftime(model)

        if minute_f == _HT_RESUME and model.engine_phase == HALFTIME:
            # Accumulate halftime duration and resume
            model.halftime_accumulated = ht_duration * 60.0  # seconds
            model.halftime_start = None
            _enter_second_half(model)

        # ── Compute effective play time ───────────────────────────────────
        if model.engine_phase == FIRST_HALF:
            effective_t = minute_f
        elif model.engine_phase == SECOND_HALF:
            effective_t = _HT_MINUTE + (minute_f - _HT_RESUME)

        model.t = effective_t

        # ── Skip halftime ticks ───────────────────────────────────────────
        if model.engine_phase == HALFTIME:
            continue

        # ── Apply goal events at this minute (keyed by effective play-min) ──
        eff_min = int(effective_t)
        for goal_min in _HOME_GOALS:
            if eff_min == goal_min:
                _apply_goal(model, "localteam")

        for goal_min in _AWAY_GOALS:
            if eff_min == goal_min:
                _apply_goal(model, "visitorteam")

        for rc_min, rc_team in _RED_CARDS:
            if eff_min == rc_min:
                _apply_red_card(model, rc_team)

        # ── Price ─────────────────────────────────────────────────────────
        mu_H, mu_A = compute_remaining_mu(model)
        P_true, sigma_MC = await step_3_4_async(model, mu_H, mu_A)

        if P_true is None:
            # Stale — use previous tick's P_true if available
            P_true = ticks[-1]["P_true"] if ticks else {
                "home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3,
                "over_15": 0.5, "over_25": 0.5, "over_35": 0.5, "btts_yes": 0.5,
            }
            sigma_MC = {}

        ticks.append({
            "real_minute": real_minute,
            "effective_t": effective_t,
            "score": model.score,
            "X": model.current_state_X,
            "P_true": dict(P_true),
            "sigma_MC": dict(sigma_MC) if sigma_MC else {},
            "mu_H": mu_H,
            "mu_A": mu_A,
            "pricing_mode": model.pricing_mode
            if model.current_state_X == 0 and model.delta_S == 0
            else "mc",
        })

    return ticks


def _apply_goal(model: LiveFootballQuantModel, team: str) -> None:
    """Apply a goal directly (bypass cooldown for replay)."""
    if team == "localteam":
        model.update_score(model.score_home + 1, model.score_away)
    else:
        model.update_score(model.score_home, model.score_away + 1)
    # Force reset cooldown state so pricing continues
    model.cooldown = False
    model.event_state = "IDLE"


def _apply_red_card(model: LiveFootballQuantModel, team: str) -> None:
    """Apply a red card (Markov state transition for replay)."""
    from src.common.types import NormalizedEvent

    event = NormalizedEvent(
        type="red_card",
        source="live_score",
        confidence="confirmed",
        team=team,
        timestamp=float(model.t),
    )
    handle_confirmed_red_card(model, event)
    # Reset cooldown so pricing continues in replay
    model.cooldown = False


# ---------------------------------------------------------------------------
# Reporting + plotting
# ---------------------------------------------------------------------------


def _print_summary(ticks: list[dict]) -> None:  # type: ignore[type-arg]
    """Print tick count, P_true before/after each goal, halftime duration."""
    print(f"\n{'='*70}")
    print("REPLAY SUMMARY")
    print(f"{'='*70}")
    print(f"Total ticks computed: {len(ticks)}")
    print(f"Halftime duration (simulated): {_HT_RESUME - _HT_MINUTE} min")
    print(f"Final score: Newcastle {ticks[-1]['score'][0]}-{ticks[-1]['score'][1]} Chelsea")
    print()

    all_events = (
        [(m, "home_goal", "Newcastle") for m in _HOME_GOALS]
        + [(m, "away_goal", "Chelsea") for m in _AWAY_GOALS]
        + [(m, "red_card", "Chelsea (Reece James)") for m, _ in _RED_CARDS]
    )
    all_events.sort(key=lambda x: x[0])

    print("Goal-by-goal P_true(home_win) / P_true(over_25):")
    print(f"  {'Min':>4}  {'Event':<30}  {'Score':>7}  {'P(HW)':>7}  {'P(O2.5)':>8}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*8}")

    for ev_min, ev_type, ev_desc in all_events:
        # Find tick just before event (compare against effective play-time)
        before_ticks = [t for t in ticks if t["effective_t"] < ev_min]
        after_ticks  = [t for t in ticks if t["effective_t"] >= ev_min]

        if before_ticks and after_ticks:
            before = before_ticks[-1]
            after  = after_ticks[0]
            score_str = f"{after['score'][0]}-{after['score'][1]}"
            p_hw_before = before["P_true"].get("home_win", 0.0)
            p_hw_after  = after["P_true"].get("home_win", 0.0)
            p_o25_before = before["P_true"].get("over_25", 0.0)
            p_o25_after  = after["P_true"].get("over_25", 0.0)
            arrow_hw  = "↑" if p_hw_after > p_hw_before + 0.001 else ("↓" if p_hw_after < p_hw_before - 0.001 else "→")
            arrow_o25 = "↑" if p_o25_after > p_o25_before + 0.001 else ("↓" if p_o25_after < p_o25_before - 0.001 else "→")
            print(
                f"  {ev_min:>4}'  {ev_desc:<30}  {score_str:>7}  "
                f"{p_hw_before:.3f}→{p_hw_after:.3f}{arrow_hw}  "
                f"{p_o25_before:.3f}→{p_o25_after:.3f}{arrow_o25}"
            )

    print()


def _plot(ticks: list[dict]) -> None:  # type: ignore[type-arg]
    """Plot P_true(t) for home_win, draw, away_win, over_25."""
    minutes = [t["effective_t"] for t in ticks]
    p_hw  = [t["P_true"].get("home_win", 0.0) for t in ticks]
    p_dr  = [t["P_true"].get("draw", 0.0) for t in ticks]
    p_aw  = [t["P_true"].get("away_win", 0.0) for t in ticks]
    p_o25 = [t["P_true"].get("over_25", 0.0) for t in ticks]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "P_true(t) — Newcastle United 4-1 Chelsea (25 Nov 2023)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Match odds ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(minutes, p_hw, color="#1f77b4", linewidth=1.8, label="P(Home Win)")
    ax.plot(minutes, p_dr, color="#aec7e8", linewidth=1.5, linestyle="--", label="P(Draw)")
    ax.plot(minutes, p_aw, color="#ff7f0e", linewidth=1.8, label="P(Away Win)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title("Match Result Probabilities")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    _add_event_lines(ax, ticks)

    # ── Over 2.5 ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(minutes, p_o25, color="#2ca02c", linewidth=1.8, label="P(Over 2.5)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Effective Play Time (minutes)")
    ax2.set_title("Over 2.5 Goals")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    _add_event_lines(ax2, ticks)

    plt.tight_layout()
    os.makedirs(os.path.dirname(_PLOT_PATH), exist_ok=True)
    plt.savefig(_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"[Plot]    Saved → {_PLOT_PATH}")


def _add_event_lines(ax: plt.Axes, ticks: list[dict]) -> None:  # type: ignore[type-arg]
    """Add vertical lines for goals and red card events."""
    # Convert real minutes to effective time for event lines
    def real_to_eff(m: int) -> float:
        if m < _HT_MINUTE:
            return float(m)
        return _HT_MINUTE + (m - _HT_RESUME)

    for gm in _HOME_GOALS:
        ax.axvline(x=real_to_eff(gm), color="#1f77b4", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(real_to_eff(gm) + 0.3, 0.97, f"⚽{gm}'", fontsize=7, color="#1f77b4", va="top")

    for gm in _AWAY_GOALS:
        ax.axvline(x=real_to_eff(gm), color="#ff7f0e", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(real_to_eff(gm) + 0.3, 0.91, f"⚽{gm}'", fontsize=7, color="#ff7f0e", va="top")

    for rc_min, _ in _RED_CARDS:
        ax.axvline(x=real_to_eff(rc_min), color="red", linewidth=1.0, linestyle="-.", alpha=0.6)
        ax.text(real_to_eff(rc_min) + 0.3, 0.85, f"🟥{rc_min}'", fontsize=7, color="red", va="top")

    # Halftime band
    ax.axvspan(float(_HT_MINUTE), float(_HT_MINUTE), alpha=0.1, color="gray")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    setup_logging(log_level="WARNING")
    _load_env()

    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set in .env or environment")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("MMPP Soccer Replay: Newcastle United 4-1 Chelsea (25.11.2023)")
    print(f"{'='*70}\n")

    # 1. Phase 1 params
    params = await _load_or_compute_params(api_key)

    # 2. Numba warm-up (first compilation)
    print("[Numba]   Warming up JIT compilation ...")
    t0 = time.monotonic()
    from src.engine.mc_core import mc_simulate_remaining

    _b = np.zeros(6)
    _g = np.zeros(4)
    _d = np.zeros(5)
    _Q_diag = np.array([-0.01, -0.01, -0.01, -0.01])
    _Q_off = np.eye(4) * 0.0
    _bb = np.array([0.0, 15.0, 30.0, 47.0, 62.0, 77.0, 96.0])
    mc_simulate_remaining(0.0, 96.0, 0, 0, 0, 0, -4.0, -4.0, _b, _g, _g, _d, _d, _Q_diag, _Q_off, _bb, 10, 42)
    print(f"[Numba]   Warm-up complete in {time.monotonic() - t0:.2f}s")

    # 3. Build model
    model = _build_model(params)

    # 4. Simulate
    print("[Replay]  Simulating match minute by minute ...")
    t1 = time.monotonic()
    ticks = await _simulate(model)
    elapsed = time.monotonic() - t1
    print(f"[Replay]  {len(ticks)} ticks in {elapsed:.2f}s ({elapsed / len(ticks) * 1000:.1f}ms/tick)")

    # 5. Summary
    _print_summary(ticks)

    # 6. Plot
    _plot(ticks)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
