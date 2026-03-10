#!/usr/bin/env python3
"""Integration test: Run Step 1.4 NLL optimization on 1 season of EPL data.

Usage: python scripts/run_step_1_4_epl.py
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.step_1_1_intervals import build_intervals_from_goalserve
from src.calibration.step_1_2_Q_estimation import estimate_Q_global, estimate_Q_by_delta_S
from src.calibration.step_1_4_nll_optimize import (
    IntervalData,
    GoalEvent,
    MatchData,
    MMPPModel,
    _extract_result,
    compute_nll,
    optimize_nll,
    prepare_match_data,
)
from src.clients.goalserve import GoalserveClient
from src.common.types import IntervalRecord

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPL_LEAGUE_ID = 1204
BRASILEIRAO_LEAGUE_ID = 1572
LIGA_ARGENTINA_LEAGUE_ID = 1300

SIGMA_A = 0.3
NUM_EPOCHS = 1000
LR = 1e-3


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


async def fetch_league(client: GoalserveClient, league_id: int, name: str) -> list[dict]:
    """Fetch fixtures for a league, trying current season then historical."""
    print(f"  Fetching {name} (ID={league_id})...")
    try:
        matches = await client.get_fixtures(league_id)
        if matches:
            return matches
    except Exception as e:
        print(f"    Current season failed: {e}")

    for season in ["2024-2025", "2023-2024"]:
        print(f"    Trying historical {season}...")
        try:
            matches = await client.get_historical_fixtures(league_id, season)
            if matches:
                return matches
        except Exception as e:
            print(f"    {season} failed: {e}")

    return []


def match_has_score(match: dict) -> bool:
    for team in ("localteam", "visitorteam"):
        td = match.get(team, {})
        if isinstance(td, dict):
            score = td.get("@score", td.get("score"))
            if score is not None and str(score).strip() not in ("", "?"):
                return True
    return False


def build_intervals_safe(match: dict) -> list[IntervalRecord]:
    try:
        return build_intervals_from_goalserve(match)
    except Exception:
        return []


def estimate_initial_a(
    intervals_by_match: dict[str, list[IntervalRecord]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Estimate initial log-intensities from observed goals."""
    match_ids: list[str] = []
    a_H_list: list[float] = []
    a_A_list: list[float] = []
    LEAGUE_AVG = 1.3

    for mid, intervals in intervals_by_match.items():
        if not intervals:
            continue
        T_m = intervals[0].T_m if intervals[0].T_m > 0 else 90.0
        home_goals = sum(len(iv.home_goal_times) for iv in intervals)
        away_goals = sum(len(iv.away_goal_times) for iv in intervals)
        mu_H = max(home_goals, 0.1) if (home_goals + away_goals) > 0 else LEAGUE_AVG
        mu_A = max(away_goals, 0.1) if (home_goals + away_goals) > 0 else LEAGUE_AVG
        match_ids.append(mid)
        a_H_list.append(math.log(mu_H / T_m))
        a_A_list.append(math.log(mu_A / T_m))

    return match_ids, np.array(a_H_list), np.array(a_A_list)


def q_condition_number(Q: np.ndarray) -> float:
    active = [i for i in range(Q.shape[0]) if Q[i, i] != 0.0]
    if len(active) < 2:
        return float("inf")
    sub = Q[np.ix_(active, active)]
    svs = np.linalg.svd(sub, compute_uv=False)
    if svs[-1] == 0:
        return float("inf")
    return float(svs[0] / svs[-1])


async def main() -> None:
    load_env()
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        return

    async with GoalserveClient(api_key, timeout=60.0) as client:
        # ==============================================================
        # 1. Fetch EPL fixtures
        # ==============================================================
        print("=" * 70)
        print("STEP 1: Fetch EPL fixtures")
        print("=" * 70)
        epl_matches = await fetch_league(client, EPL_LEAGUE_ID, "EPL")
        completed = [m for m in epl_matches if match_has_score(m)]
        print(f"  Total: {len(epl_matches)}, Completed: {len(completed)}")

        if len(completed) < 10:
            print("ERROR: Not enough matches")
            return

        # ==============================================================
        # 2. Build intervals (Step 1.1)
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 2: Build intervals (Step 1.1)")
        print("=" * 70)
        intervals_by_match: dict[str, list[IntervalRecord]] = {}
        all_intervals: list[IntervalRecord] = []
        total_goals = 0

        for m in completed:
            ivs = build_intervals_safe(m)
            if ivs:
                mid = ivs[0].match_id
                intervals_by_match[mid] = ivs
                all_intervals.extend(ivs)
                for iv in ivs:
                    total_goals += len(iv.home_goal_times) + len(iv.away_goal_times)

        print(f"  Matches with intervals: {len(intervals_by_match)}")
        print(f"  Total intervals: {len(all_intervals)}")
        print(f"  Total goals extracted: {total_goals}")

        # ==============================================================
        # 3. Q matrix (Step 1.2) — EPL
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 3: Q matrix (Step 1.2) — EPL")
        print("=" * 70)
        Q_epl = estimate_Q_global(all_intervals)
        cond_epl = q_condition_number(Q_epl)
        n_red = sum(
            len(iv.red_card_transitions) for iv in all_intervals
        )
        print(f"  Red card events: {n_red}")
        print(f"  Q matrix:\n{Q_epl}")
        print(f"  Condition number: {cond_epl:.2f}")

        # ==============================================================
        # 4. Initial a_H, a_A
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 4: Initial log-intensities")
        print("=" * 70)
        match_ids, a_H_init, a_A_init = estimate_initial_a(intervals_by_match)
        print(f"  Matches: {len(match_ids)}")
        print(f"  a_H range: [{a_H_init.min():.4f}, {a_H_init.max():.4f}]")
        print(f"  a_A range: [{a_A_init.min():.4f}, {a_A_init.max():.4f}]")
        mean_mu_H = np.exp(a_H_init).mean() * 90
        mean_mu_A = np.exp(a_A_init).mean() * 90
        print(f"  Mean μ_H = {mean_mu_H:.2f}, μ_A = {mean_mu_A:.2f} goals/match")

        # ==============================================================
        # 5. Prepare match data
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 5: Prepare match data")
        print("=" * 70)
        match_data = prepare_match_data(intervals_by_match, match_ids)
        n_home = sum(len(md.home_goal_log_lambdas) for md in match_data)
        n_away = sum(len(md.away_goal_log_lambdas) for md in match_data)
        n_ivs = sum(len(md.intervals) for md in match_data)
        print(f"  Intervals: {n_ivs}, Home goals: {n_home}, Away goals: {n_away}")
        print(f"  Goals/match: {(n_home + n_away) / len(match_data):.2f}")

        # ==============================================================
        # 6. NLL optimization (1000 steps, σ_a=0.3)
        # ==============================================================
        print("\n" + "=" * 70)
        print(f"STEP 6: NLL optimization (σ_a={SIGMA_A}, {NUM_EPOCHS} epochs)")
        print("=" * 70)

        # First try without gradient clipping
        result = optimize_nll(
            match_data, a_H_init, a_A_init,
            sigma_a=SIGMA_A, lr=LR, num_epochs=NUM_EPOCHS,
        )
        losses = result.loss_history

        # Check monotonicity
        violations = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i - 1] + 1e-6)
        max_inc = max((losses[i] - losses[i - 1] for i in range(1, len(losses))), default=0.0)

        print(f"\n  Without clipping: {violations} monotonicity violations (max increase: {max_inc:.6f})")

        if violations > 0:
            print("  → Re-running with gradient clipping (max_norm=1.0)...")
            n_matches = len(a_H_init)
            model = MMPPModel(
                n_matches=n_matches,
                a_H_init=torch.tensor(a_H_init, dtype=torch.float32),
                a_A_init=torch.tensor(a_A_init, dtype=torch.float32),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            clipped_losses: list[float] = []

            for epoch in range(NUM_EPOCHS):
                optimizer.zero_grad()
                loss = compute_nll(model, match_data, sigma_a=SIGMA_A)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                clipped_losses.append(float(loss.item()))

            result = _extract_result(model, clipped_losses)
            losses = clipped_losses

            violations2 = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i - 1] + 1e-6)
            max_inc2 = max((losses[i] - losses[i - 1] for i in range(1, len(losses))), default=0.0)
            print(f"  With clipping: {violations2} violations (max increase: {max_inc2:.6f})")

        # Print NLL every 100 steps
        print("\n  Step | NLL")
        print("  " + "-" * 35)
        for step in range(0, NUM_EPOCHS, 100):
            print(f"  {step:5d} | {losses[step]:12.4f}")
        print(f"  {NUM_EPOCHS - 1:5d} | {losses[-1]:12.4f}")

        # ==============================================================
        # 7. Verification
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 7: Verification")
        print("=" * 70)

        has_nan = any(math.isnan(x) for x in losses)
        has_inf = any(math.isinf(x) for x in losses)
        print(f"  NaN in losses: {'✗ YES' if has_nan else '✓ No'}")
        print(f"  Inf in losses: {'✗ YES' if has_inf else '✓ No'}")
        print(f"  tau_H = {result.tau_H:.4f} ∈ [0.1, 5.0]: {'✓' if 0.1 <= result.tau_H <= 5.0 else '✗'}")
        print(f"  tau_A = {result.tau_A:.4f} ∈ [0.1, 5.0]: {'✓' if 0.1 <= result.tau_A <= 5.0 else '✗'}")
        print(f"  b finite: {'✓' if np.all(np.isfinite(result.b)) else '✗'}")
        print(f"  a_H finite: {'✓' if np.all(np.isfinite(result.a_H)) else '✗'}")
        print(f"  a_A finite: {'✓' if np.all(np.isfinite(result.a_A)) else '✗'}")

        print("\n  --- Fitted Parameters ---")
        print(f"  b (time profile):  {np.array2string(result.b, precision=4)}")
        print(f"  gamma_H:           {np.array2string(result.gamma_H, precision=4)}")
        print(f"  gamma_A:           {np.array2string(result.gamma_A, precision=4)}")
        print(f"  delta_H (lookup):  {np.array2string(result.delta_H, precision=4)}")
        print(f"  delta_A (lookup):  {np.array2string(result.delta_A, precision=4)}")
        print(f"  beta_H={result.beta_H:.4f}, kappa_H={result.kappa_H:.4f}, tau_H={result.tau_H:.4f}")
        print(f"  beta_A={result.beta_A:.4f}, kappa_A={result.kappa_A:.4f}, tau_A={result.tau_A:.4f}")
        print(f"  a_H: mean={result.a_H.mean():.4f}, std={result.a_H.std():.4f}")
        print(f"  a_A: mean={result.a_A.mean():.4f}, std={result.a_A.std():.4f}")
        print(f"  NLL: {losses[0]:.2f} → {losses[-1]:.2f} ({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")

        # ==============================================================
        # 8. Q matrices for Liga Argentina and Brasileirão
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 8: Q matrices — Liga Argentina & Brasileirão")
        print("=" * 70)

        for name, lid in [("Liga Argentina", LIGA_ARGENTINA_LEAGUE_ID), ("Brasileirão", BRASILEIRAO_LEAGUE_ID)]:
            print(f"\n  --- {name} (ID={lid}) ---")
            try:
                lm = await fetch_league(client, lid, name)
            except Exception as e:
                print(f"  Failed to fetch: {e}")
                continue

            lc = [m for m in lm if match_has_score(m)]
            print(f"  Completed: {len(lc)}")

            if not lc:
                print(f"  No data available for {name}")
                continue

            league_ivs: list[IntervalRecord] = []
            for m in lc:
                league_ivs.extend(build_intervals_safe(m))

            n_rc = sum(len(iv.red_card_transitions) for iv in league_ivs)
            print(f"  Intervals: {len(league_ivs)}, Red cards: {n_rc}")

            Q = estimate_Q_global(league_ivs)
            cond = q_condition_number(Q)
            print(f"  Q:\n{Q}")
            print(f"  Condition number: {cond:.2f}")

            if cond > 1000:
                print(f"  ⚠ cond > 1000 → applying stronger shrinkage (T=50000)...")
                Q_ds = estimate_Q_by_delta_S(league_ivs, T_threshold=50000.0)
                for b_idx, Q_b in Q_ds.items():
                    c = q_condition_number(Q_b)
                    print(f"    Bin {b_idx}: cond={c:.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
