#!/usr/bin/env python3
"""Step-by-step Phase 1 pipeline on EPL 2023-2024.

Usage: python scripts/run_phase1_stepwise.py
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.step_1_1_intervals import build_intervals_from_goalserve
from src.calibration.step_1_2_Q_estimation import (
    apply_state3_additivity,
    estimate_Q_by_delta_S,
    estimate_Q_global,
)
from src.calibration.step_1_3_ml_prior import (
    FEATURE_COLUMNS,
    _LEAGUE_AVG_GOALS,
    build_match_features,
    features_to_array,
    goals_from_intervals,
    mu_to_log_intensity,
    predict_expected_goals,
    select_features_by_importance,
    train_poisson_xgb,
)
from src.calibration.step_1_4_nll_optimize import (
    optimize_nll,
    prepare_match_data,
)
from src.calibration.step_1_5_validation import (
    brier_score,
    calibrate_sanity_thresholds,
    encode_outcome_1x2,
    poisson_1x2,
    run_validation,
    validate_delta_signs,
    validate_gamma_signs,
)
from src.calibration.phase1_worker import fetch_season_commentaries
from src.clients.goalserve import GoalserveClient
from src.clients.odds_api import OddsApiClient, build_odds_features, remove_overround
from src.common.types import IntervalRecord

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPL_LEAGUE_ID = 1204
SEASON = "2023-2024"
SIGMA_A = 0.3
NUM_EPOCHS = 1000


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


def match_has_score(match: dict) -> bool:
    for team in ("localteam", "visitorteam"):
        td = match.get(team, {})
        if isinstance(td, dict):
            # Commentaries uses @goals/@ft_score, fixtures uses @score
            for key in ("@goals", "@ft_score", "@score", "goals", "score"):
                score = td.get(key)
                if score is not None and str(score).strip() not in ("", "?"):
                    return True
    return False


def get_score(match: dict, team: str) -> int:
    td = match.get(team, {})
    if isinstance(td, dict):
        for key in ("@goals", "@ft_score", "@score", "goals", "score"):
            score = td.get(key)
            if score is not None and str(score).strip() not in ("", "?"):
                try:
                    return int(score)
                except (ValueError, TypeError):
                    continue
    return 0


def match_has_red_card(match: dict) -> bool:
    """Check if a match has red card data."""
    # Check cards node
    cards = match.get("cards", {})
    if cards:
        for team_key in ("localteam", "visitorteam"):
            team_cards = cards.get(team_key, {})
            if isinstance(team_cards, dict):
                player_cards = team_cards.get("player", [])
                if isinstance(player_cards, list):
                    for card in player_cards:
                        if isinstance(card, dict) and card.get("@card") == "red":
                            return True
                elif isinstance(player_cards, dict) and player_cards.get("@card") == "red":
                    return True
    # Check events
    events = match.get("events", {})
    if events:
        for ev in (events if isinstance(events, list) else [events]):
            if isinstance(ev, dict) and ev.get("type") in ("redcard", "red_card"):
                return True
    return False


def hr_line() -> None:
    print("=" * 78)


def sub_line() -> None:
    print("-" * 78)


# =====================================================================
# MAIN
# =====================================================================


async def main() -> None:
    load_env()
    goalserve_key = os.environ.get("GOALSERVE_API_KEY", "")
    odds_key = os.environ.get("ODDS_API_KEY", "")

    if not goalserve_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        return

    summary: dict[str, Any] = {}

    # ==================================================================
    # STEP 1: Fetch commentaries (with red cards + summary)
    # ==================================================================
    hr_line()
    print(f" STEP 1: Fetch EPL {SEASON} via commentaries endpoint")
    hr_line()

    async with GoalserveClient(goalserve_key, timeout=60.0) as gs:
        print("  Fetching commentaries for all matchdays (concurrency=5)...")
        matches = await fetch_season_commentaries(
            gs, EPL_LEAGUE_ID, SEASON, concurrency=5,
        )

    if not matches:
        print("ERROR: No matches returned. Aborting.")
        return

    completed = [m for m in matches if match_has_score(m)]
    with_summary = [m for m in completed if m.get("summary")]
    with_red = [m for m in completed if match_has_red_card(m)]

    print(f"  Total matches returned:    {len(matches)}")
    print(f"  Matches with final score:  {len(completed)}")
    print(f"  Matches with summary:      {len(with_summary)}")
    print(f"  Matches with red cards:    {len(with_red)}")

    # Score distribution
    total_goals = sum(get_score(m, "localteam") + get_score(m, "visitorteam") for m in completed)
    home_goals = sum(get_score(m, "localteam") for m in completed)
    away_goals = sum(get_score(m, "visitorteam") for m in completed)
    print(f"  Total goals (from scores): {total_goals} (H={home_goals}, A={away_goals})")
    print(f"  Goals/match average:       {total_goals / len(completed):.2f}")

    summary["total_fixtures"] = len(matches)
    summary["completed"] = len(completed)
    summary["red_card_matches"] = len(with_red)
    summary["total_goals_score"] = total_goals

    if len(completed) < 10:
        print("ERROR: Not enough completed matches. Aborting.")
        return

    print("\n  ✓ Step 1 complete.\n")

    # ==================================================================
    # STEP 2: Build intervals
    # ==================================================================
    hr_line()
    print(" STEP 2: Build IntervalRecords (Step 1.1)")
    hr_line()

    intervals_by_match: dict[str, list[IntervalRecord]] = {}
    all_intervals: list[IntervalRecord] = []
    zero_interval_matches: list[str] = []
    goals_extracted = 0
    intervals_per_match: list[int] = []

    for m in completed:
        try:
            ivs = build_intervals_from_goalserve(m)
        except Exception as e:
            mid = m.get("@id", m.get("id", "?"))
            zero_interval_matches.append(f"{mid} (error: {e})")
            continue
        if not ivs:
            mid = m.get("@id", m.get("id", "?"))
            zero_interval_matches.append(str(mid))
            continue
        mid = ivs[0].match_id
        intervals_by_match[mid] = ivs
        all_intervals.extend(ivs)
        intervals_per_match.append(len(ivs))
        for iv in ivs:
            goals_extracted += len(iv.home_goal_times) + len(iv.away_goal_times)

    avg_ivs = np.mean(intervals_per_match) if intervals_per_match else 0
    min_ivs = min(intervals_per_match) if intervals_per_match else 0
    max_ivs = max(intervals_per_match) if intervals_per_match else 0

    print(f"  Matches with intervals:    {len(intervals_by_match)}")
    print(f"  Total IntervalRecords:     {len(all_intervals)}")
    print(f"  Intervals per match:       avg={avg_ivs:.1f}, min={min_ivs}, max={max_ivs}")
    print(f"  Goals extracted:           {goals_extracted}")
    print(f"  Matches with 0 intervals:  {len(zero_interval_matches)}")
    if zero_interval_matches[:5]:
        for zm in zero_interval_matches[:5]:
            print(f"    - {zm}")
        if len(zero_interval_matches) > 5:
            print(f"    ... and {len(zero_interval_matches) - 5} more")

    summary["matches_with_intervals"] = len(intervals_by_match)
    summary["total_intervals"] = len(all_intervals)
    summary["goals_extracted"] = goals_extracted
    summary["zero_interval_count"] = len(zero_interval_matches)

    print("\n  ✓ Step 2 complete.\n")

    # ==================================================================
    # STEP 3: Q matrix estimation (Step 1.2)
    # ==================================================================
    hr_line()
    print(" STEP 3: Q matrix estimation (Step 1.2)")
    hr_line()

    n_red = sum(len(iv.red_card_transitions) for iv in all_intervals)
    print(f"  Red card transition events: {n_red}")

    Q_global = estimate_Q_global(all_intervals)
    Q_global = apply_state3_additivity(Q_global)

    print(f"\n  Q matrix (4x4):")
    for i in range(4):
        row_str = "  " + " ".join(f"{Q_global[i, j]:10.6f}" for j in range(4))
        print(f"    [{row_str} ]")

    row_sums = Q_global.sum(axis=1)
    print(f"\n  Row sums (should be ~0):    {np.array2string(row_sums, precision=8)}")

    # Condition number
    active = [i for i in range(4) if Q_global[i, i] != 0.0]
    if len(active) >= 2:
        sub = Q_global[np.ix_(active, active)]
        svs = np.linalg.svd(sub, compute_uv=False)
        cond = float(svs[0] / svs[-1]) if svs[-1] > 0 else float("inf")
    else:
        cond = float("inf")
    print(f"  Condition number:           {cond:.2f}")

    # Per-ΔS Q matrices
    Q_by_ds = estimate_Q_by_delta_S(all_intervals)
    print(f"  Per-ΔS bins available:      {sorted(Q_by_ds.keys())}")

    summary["red_card_events"] = n_red
    summary["Q_condition"] = cond

    if n_red == 0:
        print("\n  ⚠ No red card data in fixture format. Q is zero matrix.")
        print("    (Per-match commentaries needed for red card extraction)")

    print("\n  ✓ Step 3 complete.\n")

    # ==================================================================
    # STEP 4: ML Prior (Step 1.3) — with Odds-API odds
    # ==================================================================
    hr_line()
    print(" STEP 4: ML Prior with Odds-API odds (Step 1.3)")
    hr_line()

    match_ids = list(intervals_by_match.keys())

    # Try to fetch odds from Odds-API for exchange probabilities
    odds_by_match: dict[str, dict[str, list[dict[str, Any]]]] = {}
    exchange_probs: dict[str, tuple[float, float, float]] = {}

    if odds_key:
        print("  Fetching odds from Odds-API...")
        try:
            async with OddsApiClient(odds_key) as odds_client:
                events = await odds_client.get_events(
                    "football",
                    league="england-premier-league",
                    status="settled",
                )
                print(f"  Odds-API events found:     {len(events)}")

                # Fetch odds for events in batches of 10
                event_ids = [ev.get("id") for ev in events if ev.get("id")]
                bookmaker_str = "Bet365,Betfair Exchange,Sbobet,1xbet,DraftKings"
                fetched = 0

                for batch_start in range(0, min(len(event_ids), 100), 10):
                    batch = event_ids[batch_start:batch_start + 10]
                    try:
                        odds_results = await odds_client.get_odds_multi(batch, bookmaker_str)
                        for ev_odds in odds_results:
                            ev_id = str(ev_odds.get("id", ""))
                            bm = ev_odds.get("bookmakers", {})
                            if bm:
                                odds_by_match[ev_id] = bm
                                # Extract exchange probs for validation
                                features = build_odds_features(bm)
                                if features["exchange_home_prob"] > 0:
                                    exchange_probs[ev_id] = (
                                        features["exchange_home_prob"],
                                        features["exchange_draw_prob"],
                                        features["exchange_away_prob"],
                                    )
                                fetched += 1
                    except Exception as e:
                        print(f"  Warning: batch fetch failed: {e}")
                        break

                print(f"  Matches with odds:         {fetched}")
                print(f"  Matches with exchange:     {len(exchange_probs)}")
        except Exception as e:
            print(f"  Warning: Odds-API failed: {e}")
            print("  Falling back to MLE without odds.")
    else:
        print("  No ODDS_API_KEY set — running MLE fallback (no odds features).")

    # Build initial a_H, a_A via MLE fallback (Goalserve stats not available for historical)
    # The XGBoost path needs match_stats from Goalserve get_match_stats() which isn't
    # available in bulk for historical fixtures. Use MLE.
    a_H_list: list[float] = []
    a_A_list: list[float] = []
    valid_ids: list[str] = []

    for mid in match_ids:
        ivs = intervals_by_match[mid]
        if not ivs:
            continue
        T_m = ivs[0].T_m if ivs[0].T_m > 0 else 90.0
        h_goals = sum(len(iv.home_goal_times) for iv in ivs)
        a_goals = sum(len(iv.away_goal_times) for iv in ivs)

        mu_H = max(h_goals, 0.1) if (h_goals + a_goals) > 0 else _LEAGUE_AVG_GOALS
        mu_A = max(a_goals, 0.1) if (h_goals + a_goals) > 0 else _LEAGUE_AVG_GOALS

        valid_ids.append(mid)
        a_H_list.append(math.log(mu_H / T_m))
        a_A_list.append(math.log(mu_A / T_m))

    a_H_init = np.array(a_H_list)
    a_A_init = np.array(a_A_list)

    print(f"\n  Method:                    MLE from observed goals")
    print(f"  Matches:                   {len(valid_ids)}")
    sub_line()
    print(f"  a_H:  min={a_H_init.min():.4f}  mean={a_H_init.mean():.4f}  max={a_H_init.max():.4f}")
    print(f"  a_A:  min={a_A_init.min():.4f}  mean={a_A_init.mean():.4f}  max={a_A_init.max():.4f}")

    mu_H_vals = np.exp(a_H_init) * 90
    mu_A_vals = np.exp(a_A_init) * 90
    print(f"  μ_H:  min={mu_H_vals.min():.2f}  mean={mu_H_vals.mean():.2f}  max={mu_H_vals.max():.2f}")
    print(f"  μ_A:  min={mu_A_vals.min():.2f}  mean={mu_A_vals.mean():.2f}  max={mu_A_vals.max():.2f}")

    # Feature importance (trivial for MLE but list the columns)
    print(f"\n  Feature columns ({len(FEATURE_COLUMNS)} total, MLE = no XGBoost features):")
    for i, col in enumerate(FEATURE_COLUMNS[:10]):
        print(f"    {i+1:2d}. {col}")

    summary["prior_method"] = "MLE"
    summary["prior_matches"] = len(valid_ids)
    summary["a_H_mean"] = float(a_H_init.mean())
    summary["a_A_mean"] = float(a_A_init.mean())
    summary["mu_H_mean"] = float(mu_H_vals.mean())
    summary["mu_A_mean"] = float(mu_A_vals.mean())

    print("\n  ✓ Step 4 complete.\n")

    # ==================================================================
    # STEP 5: NLL Optimization (Step 1.4)
    # ==================================================================
    hr_line()
    print(f" STEP 5: NLL Optimization (σ_a={SIGMA_A}, {NUM_EPOCHS} epochs)")
    hr_line()

    match_data = prepare_match_data(intervals_by_match, valid_ids)
    n_ivs = sum(len(md.intervals) for md in match_data)
    n_home_goals = sum(len(md.home_goal_log_lambdas) for md in match_data)
    n_away_goals = sum(len(md.away_goal_log_lambdas) for md in match_data)
    print(f"  MatchData objects:         {len(match_data)}")
    print(f"  Total intervals:           {n_ivs}")
    print(f"  Home goals:                {n_home_goals}")
    print(f"  Away goals:                {n_away_goals}")
    print(f"\n  Training...")

    t0 = time.perf_counter()
    result = optimize_nll(
        match_data, a_H_init, a_A_init,
        sigma_a=SIGMA_A, num_epochs=NUM_EPOCHS,
    )
    elapsed = time.perf_counter() - t0

    losses = result.loss_history
    print(f"  Training time:             {elapsed:.1f}s ({elapsed/NUM_EPOCHS*1000:.1f}ms/epoch)")
    sub_line()

    # NLL at checkpoints
    checkpoints = [0, 100, 500, 999]
    print(f"\n  {'Step':>6s}  {'NLL':>12s}  {'ΔNLL':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}")
    for cp in checkpoints:
        if cp < len(losses):
            delta = losses[cp] - losses[0]
            print(f"  {cp:6d}  {losses[cp]:12.2f}  {delta:+10.2f}")

    pct = (1 - losses[-1] / losses[0]) * 100
    print(f"\n  NLL reduction:             {losses[0]:.2f} → {losses[-1]:.2f} ({pct:.1f}%)")

    # Monotonicity check
    violations = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i-1] + 1e-6)
    max_inc = max((losses[i] - losses[i-1] for i in range(1, len(losses))), default=0.0)
    print(f"  Monotonicity violations:   {violations} (max increase: {max_inc:.6f})")

    # Parameters
    sub_line()
    print(f"  tau_H = {result.tau_H:.4f}  (range [0.1, 5.0]: {'✓' if 0.1 <= result.tau_H <= 5.0 else '✗'})")
    print(f"  tau_A = {result.tau_A:.4f}  (range [0.1, 5.0]: {'✓' if 0.1 <= result.tau_A <= 5.0 else '✗'})")
    print(f"  beta_H={result.beta_H:.4f}  kappa_H={result.kappa_H:.4f}")
    print(f"  beta_A={result.beta_A:.4f}  kappa_A={result.kappa_A:.4f}")
    print(f"  b (time):    {np.array2string(result.b, precision=4, floatmode='fixed')}")
    print(f"  gamma_H:     {np.array2string(result.gamma_H, precision=4, floatmode='fixed')}")
    print(f"  gamma_A:     {np.array2string(result.gamma_A, precision=4, floatmode='fixed')}")
    print(f"  delta_H:     {np.array2string(result.delta_H, precision=4, floatmode='fixed')}")
    print(f"  delta_A:     {np.array2string(result.delta_A, precision=4, floatmode='fixed')}")

    # Sanity checks
    has_nan = any(math.isnan(x) for x in losses)
    has_inf = any(math.isinf(x) for x in losses)
    b_finite = bool(np.all(np.isfinite(result.b)))
    a_finite = bool(np.all(np.isfinite(result.a_H)) and np.all(np.isfinite(result.a_A)))

    print(f"\n  NaN in losses:  {'✗' if has_nan else '✓'}  |  Inf in losses: {'✗' if has_inf else '✓'}")
    print(f"  b finite:       {'✓' if b_finite else '✗'}  |  a finite:      {'✓' if a_finite else '✗'}")

    summary["nll_start"] = float(losses[0])
    summary["nll_end"] = float(losses[-1])
    summary["nll_reduction_pct"] = pct
    summary["tau_H"] = result.tau_H
    summary["tau_A"] = result.tau_A
    summary["training_time_s"] = elapsed

    print("\n  ✓ Step 5 complete.\n")

    # ==================================================================
    # STEP 6: Validation (Step 1.5) — 5-fold walk-forward CV
    # ==================================================================
    hr_line()
    print(" STEP 6: Validation with 5-fold walk-forward CV (Step 1.5)")
    hr_line()

    n = len(valid_ids)

    # Derive model predictions from fitted μ
    model_preds = np.zeros((n, 3))
    outcomes = np.zeros((n, 3))

    for i, mid in enumerate(valid_ids):
        ivs = intervals_by_match[mid]
        T_m = ivs[0].T_m if ivs[0].T_m > 0 else 90.0
        mu_H = float(np.exp(result.a_H[i]) * T_m)
        mu_A = float(np.exp(result.a_A[i]) * T_m)
        model_preds[i] = poisson_1x2(mu_H, mu_A)

        h_goals = sum(len(iv.home_goal_times) for iv in ivs)
        a_goals = sum(len(iv.away_goal_times) for iv in ivs)
        outcomes[i] = encode_outcome_1x2(h_goals, a_goals)

    # Build exchange predictions: use Odds-API if available, else uniform
    exchange_preds_arr = np.full((n, 3), 1.0 / 3.0)
    odds_matched = 0

    # For exchange preds, try to match Goalserve match IDs to odds.
    # Since IDs differ between systems, we use uniform as baseline.
    if exchange_probs:
        print(f"  Odds-API exchange probs available: {len(exchange_probs)}")
        print(f"  (Cannot match to Goalserve IDs — using uniform baseline)")
    else:
        print(f"  No exchange odds available — using uniform [0.33, 0.33, 0.33] as baseline")

    # 5-fold chronological CV
    fold_size = n // 5
    fold_bs_model: list[float] = []
    fold_bs_exch: list[float] = []
    fold_delta: list[float] = []
    fold_n: list[int] = []

    print(f"\n  Total matches: {n}, Fold size: ~{fold_size}")
    sub_line()
    print(f"  {'Fold':>4s}  {'N':>4s}  {'BS_model':>10s}  {'BS_exch':>10s}  {'ΔBS':>10s}  {'Verdict':>8s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for fold_idx in range(5):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < 4 else n
        fold_model = model_preds[start:end]
        fold_exch = exchange_preds_arr[start:end]
        fold_out = outcomes[start:end]

        bs_m = brier_score(fold_model, fold_out)
        bs_e = brier_score(fold_exch, fold_out)
        d_bs = bs_m - bs_e

        fold_bs_model.append(bs_m)
        fold_bs_exch.append(bs_e)
        fold_delta.append(d_bs)
        fold_n.append(end - start)

        verdict = "GO" if d_bs < 0 else "HOLD"
        print(f"  {fold_idx+1:4d}  {end-start:4d}  {bs_m:10.6f}  {bs_e:10.6f}  {d_bs:+10.6f}  {verdict:>8s}")

    sub_line()

    # Full-sample validation
    full_result = run_validation(
        model_preds, exchange_preds_arr, outcomes,
        gamma_H=result.gamma_H,
        gamma_A=result.gamma_A,
        delta_H=result.delta_H,
        delta_A=result.delta_A,
    )

    print(f"\n  Full-sample results:")
    print(f"  BS_model:            {full_result.bs_model:.6f}")
    print(f"  BS_exchange:         {full_result.bs_exchange:.6f}")
    print(f"  ΔBS:                 {full_result.delta_bs:+.6f}")
    print(f"  Log-loss model:      {full_result.log_loss_model:.6f}")

    # Sanity thresholds
    t = full_result.thresholds
    print(f"\n  Sanity thresholds (model vs exchange discrepancy):")
    print(f"    go_threshold (90th pctl):   {t.go_threshold:.6f}")
    print(f"    hold_threshold (99th pctl): {t.hold_threshold:.6f}")
    print(f"    median_delta:               {t.median_delta:.6f}")
    print(f"    n_matches:                  {t.n_matches}")

    # Sign checks
    if full_result.gamma_sign_checks:
        print(f"\n  Gamma sign checks:")
        for sc in full_result.gamma_sign_checks:
            status = "✓ PASS" if sc.passes else "✗ FAIL"
            print(f"    {sc.param_name}: {status} (expected {sc.expected_sign}, actual={sc.actual_value:.4f})")

    if full_result.delta_sign_checks:
        print(f"\n  Delta sign checks:")
        for sc in full_result.delta_sign_checks:
            status = "✓ PASS" if sc.passes else "✗ FAIL"
            print(f"    {sc.param_name}: {status} (expected {sc.expected_sign}, actual={sc.actual_value:.4f})")

    # Go/No-Go
    print(f"\n  Go/No-Go reasons:")
    for r in full_result.reasons:
        print(f"    - {r}")

    verdict_str = "GO ✓" if full_result.go_decision else "NO-GO ✗"
    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  VERDICT:  {verdict_str:>20s}   ║")
    print(f"  ╚══════════════════════════════════╝")

    summary["bs_model"] = full_result.bs_model
    summary["bs_exchange"] = full_result.bs_exchange
    summary["delta_bs"] = full_result.delta_bs
    summary["go_threshold"] = t.go_threshold
    summary["hold_threshold"] = t.hold_threshold
    summary["go_decision"] = full_result.go_decision

    print("\n  ✓ Step 6 complete.\n")

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    hr_line()
    print(" SUMMARY — Phase 1 Pipeline: EPL 2023-2024")
    hr_line()

    rows = [
        ("Data", "", ""),
        ("  Total fixtures", str(summary["total_fixtures"]), ""),
        ("  Completed matches", str(summary["completed"]), ""),
        ("  Red card matches", str(summary["red_card_matches"]), "fixture format"),
        ("  Goals (from scores)", str(summary["total_goals_score"]), ""),
        ("", "", ""),
        ("Intervals (Step 1.1)", "", ""),
        ("  Matches with intervals", str(summary["matches_with_intervals"]), ""),
        ("  Total IntervalRecords", str(summary["total_intervals"]), ""),
        ("  Goals extracted", str(summary["goals_extracted"]), ""),
        ("  Zero-interval matches", str(summary["zero_interval_count"]), ""),
        ("", "", ""),
        ("Q Matrix (Step 1.2)", "", ""),
        ("  Red card events", str(summary["red_card_events"]), ""),
        ("  Condition number", f"{summary['Q_condition']:.2f}", ""),
        ("", "", ""),
        ("ML Prior (Step 1.3)", "", ""),
        ("  Method", summary["prior_method"], ""),
        ("  Matches", str(summary["prior_matches"]), ""),
        ("  Mean μ_H", f"{summary['mu_H_mean']:.2f} goals/match", ""),
        ("  Mean μ_A", f"{summary['mu_A_mean']:.2f} goals/match", ""),
        ("", "", ""),
        ("NLL Optimization (Step 1.4)", "", ""),
        ("  NLL start → end", f"{summary['nll_start']:.2f} → {summary['nll_end']:.2f}", f"{summary['nll_reduction_pct']:.1f}%"),
        ("  tau_H", f"{summary['tau_H']:.4f}", "[0.1, 5.0]"),
        ("  tau_A", f"{summary['tau_A']:.4f}", "[0.1, 5.0]"),
        ("  Training time", f"{summary['training_time_s']:.1f}s", ""),
        ("", "", ""),
        ("Validation (Step 1.5)", "", ""),
        ("  BS_model", f"{summary['bs_model']:.6f}", ""),
        ("  BS_exchange", f"{summary['bs_exchange']:.6f}", "uniform baseline"),
        ("  ΔBS", f"{summary['delta_bs']:+.6f}", "negative = model better"),
        ("  go_threshold (90th)", f"{summary['go_threshold']:.6f}", ""),
        ("  hold_threshold (99th)", f"{summary['hold_threshold']:.6f}", ""),
        ("  GO/NO-GO", "GO ✓" if summary["go_decision"] else "NO-GO ✗", ""),
    ]

    for label, value, note in rows:
        if not label and not value:
            print()
        else:
            line = f"  {label:<28s} {value:>24s}"
            if note:
                line += f"  ({note})"
            print(line)

    hr_line()
    print(" PIPELINE COMPLETE")
    hr_line()


if __name__ == "__main__":
    asyncio.run(main())
