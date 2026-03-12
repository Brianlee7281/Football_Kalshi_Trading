#!/usr/bin/env python3
"""Full Phase 1 retrain on all 8 leagues × all collected seasons.

Loads match data from data/commentaries/, runs Steps 1.1-1.5 with
σ_a grid search and 5-fold chronological CV, saves to production_params.

Usage:
    python scripts/retrain_phase1_full.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.phase1_worker import (
    Phase1Result,
    params_to_json,
    step_1_1_intervalize,
    step_1_2_estimate_Q,
    step_1_3_ml_prior,
    step_1_4_optimize,
    step_1_5_validate,
    thresholds_to_json,
    validation_to_json,
)
from src.calibration.step_1_4_nll_optimize import OptimizationResult, prepare_match_data
from src.calibration.step_1_5_validation import (
    brier_score,
    encode_outcome_1x2,
    poisson_1x2,
    run_validation,
)
from src.common.types import IntervalRecord

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/commentaries")
OUTPUT_DIR = Path("data/production_params")

LEAGUE_NAMES: dict[int, str] = {
    1204: "EPL",
    1399: "La Liga",
    1269: "Serie A",
    1229: "Bundesliga",
    1221: "Ligue 1",
    1440: "MLS",
    1141: "Brasileirão",
    1081: "Liga Argentina",
}

SIGMA_A_GRID = [0.1, 0.3, 0.5, 1.0]
NUM_EPOCHS = 1000
CV_EPOCHS = 300  # fewer epochs for CV (sufficient for ranking σ_a)
N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_matches() -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Load all match data from data/commentaries/.

    Returns:
        (all_matches, league_match_counts)
    """
    all_matches: list[dict[str, Any]] = []
    league_counts: dict[int, int] = {}

    for league_dir in sorted(DATA_DIR.iterdir()):
        if not league_dir.is_dir():
            continue
        league_id = int(league_dir.name)
        count = 0
        for season_file in sorted(league_dir.glob("*.json")):
            with open(season_file) as f:
                matches = json.load(f)
            # Tag each match with league_id for stratification
            for m in matches:
                m["_league_id"] = league_id
                m["_season_file"] = season_file.stem
            all_matches.extend(matches)
            count += len(matches)
        league_counts[league_id] = count

    return all_matches, league_counts


def sort_matches_chronologically(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort matches by date (best-effort parsing)."""
    def date_key(m: dict[str, Any]) -> str:
        raw = m.get("@formatted_date", m.get("@date", m.get("date", "")))
        if not raw:
            return "9999.99.9999"
        # Try DD.MM.YYYY -> YYYY.MM.DD for sorting
        parts = str(raw).split(".")
        if len(parts) == 3 and len(parts[2]) == 4:
            return f"{parts[2]}.{parts[1]}.{parts[0]}"
        return str(raw)
    return sorted(matches, key=date_key)


def q_condition_number(Q: np.ndarray) -> float:
    """Compute condition number of active submatrix of Q."""
    active = [i for i in range(Q.shape[0]) if Q[i, i] != 0.0]
    if len(active) < 2:
        return float("inf")
    sub = Q[np.ix_(active, active)]
    svs = np.linalg.svd(sub, compute_uv=False)
    if svs[-1] == 0:
        return float("inf")
    return float(svs[0] / svs[-1])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def chronological_cv_split(
    matches: list[dict[str, Any]],
    n_folds: int = N_CV_FOLDS,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split chronologically sorted matches into expanding-window CV folds.

    Fold k: train on first (k+1)/(n_folds+1) of data, validate on next chunk.
    """
    n = len(matches)
    chunk_size = n // (n_folds + 1)
    folds: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    for k in range(n_folds):
        train_end = chunk_size * (k + 1)
        val_start = train_end
        val_end = min(train_end + chunk_size, n)
        if val_end <= val_start:
            continue
        folds.append((matches[:train_end], matches[val_start:val_end]))

    return folds


def evaluate_fold(
    train_matches: list[dict[str, Any]],
    val_matches: list[dict[str, Any]],
    sigma_a: float,
) -> float:
    """Train on train_matches, evaluate Brier Score on val_matches."""
    # Step 1.1 on train
    train_ivm, train_all_iv = step_1_1_intervalize(train_matches)
    if len(train_ivm) < 10:
        return 1.0  # worst possible

    # Step 1.3: MLE fallback (no XGBoost stats for historical)
    match_ids, a_H_init, a_A_init, _ = step_1_3_ml_prior(train_ivm)

    # Step 1.4: optimize (fewer epochs for CV)
    opt = step_1_4_optimize(
        train_ivm, match_ids, a_H_init, a_A_init,
        sigma_a=sigma_a, num_epochs=CV_EPOCHS,
    )

    # Evaluate on validation set
    val_ivm, _ = step_1_1_intervalize(val_matches)
    if len(val_ivm) < 5:
        return 1.0

    # For each val match, compute Poisson 1x2 using league-average params
    # (since val matches don't have individual a_H/a_A from training)
    # Use trained structural params (b, gamma, delta) + MLE baseline per val match
    val_ids = list(val_ivm.keys())
    n_val = len(val_ids)
    model_preds = np.zeros((n_val, 3))
    outcomes = np.zeros((n_val, 3))

    for i, mid in enumerate(val_ids):
        ivs = val_ivm[mid]
        T_m = ivs[0].T_m if ivs[0].T_m > 0 else 90.0
        h_goals = sum(len(iv.home_goal_times) for iv in ivs)
        a_goals = sum(len(iv.away_goal_times) for iv in ivs)

        # MLE baseline for this match
        mu_H = max(h_goals, 0.1) if (h_goals + a_goals) > 0 else 1.3
        mu_A = max(a_goals, 0.1) if (h_goals + a_goals) > 0 else 1.3

        # Apply trained time profile correction (average over b)
        b_avg = float(np.mean(opt.b))
        mu_H *= math.exp(b_avg)
        mu_A *= math.exp(b_avg)

        model_preds[i] = poisson_1x2(mu_H, mu_A)
        outcomes[i] = encode_outcome_1x2(h_goals, a_goals)

    return brier_score(model_preds, outcomes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    # ================================================================
    # 1. Load all data
    # ================================================================
    print("=" * 80)
    print("PHASE 1 FULL RETRAIN — 8 Leagues")
    print("=" * 80)

    all_matches, league_counts = load_all_matches()
    total = len(all_matches)

    print(f"\n{'League':<18s} {'ID':>5s} {'Matches':>9s}")
    print("-" * 36)
    for lid in sorted(league_counts.keys()):
        name = LEAGUE_NAMES.get(lid, f"Unknown-{lid}")
        print(f"{name:<18s} {lid:>5d} {league_counts[lid]:>9d}")
    print("-" * 36)
    print(f"{'TOTAL':<18s} {'':>5s} {total:>9d}")

    # Sort chronologically
    all_matches = sort_matches_chronologically(all_matches)
    print(f"\nSorted {total} matches chronologically")

    # ================================================================
    # 2. Full dataset: Steps 1.1-1.2
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 1.1-1.2: Intervalize + Q Matrix (full dataset)")
    print("=" * 80)

    intervals_by_match, all_intervals = step_1_1_intervalize(all_matches)
    n_matches_iv = len(intervals_by_match)
    n_goals = sum(
        len(iv.home_goal_times) + len(iv.away_goal_times) for iv in all_intervals
    )
    n_red = sum(len(iv.red_card_transitions) for iv in all_intervals)
    print(f"  Matches with intervals: {n_matches_iv}")
    print(f"  Total goals: {n_goals}")
    print(f"  Total red cards: {n_red}")

    Q_global, Q_by_ds = step_1_2_estimate_Q(all_intervals)
    cond_global = q_condition_number(Q_global)
    print(f"\n  Q_global:\n{Q_global}")
    print(f"  Q condition number: {cond_global:.2f}")

    # Per-league stats
    print(f"\n  {'League':<18s} {'Matches':>9s} {'Goals':>7s} {'Reds':>6s} {'Q Cond':>8s}")
    print("  " + "-" * 52)

    for lid in sorted(league_counts.keys()):
        name = LEAGUE_NAMES.get(lid, f"?{lid}")
        league_ivs = [
            iv for mid, ivs in intervals_by_match.items()
            for iv in ivs
            if any(m.get("_league_id") == lid for m in all_matches if str(m.get("@id", m.get("id", ""))) == mid)
        ]
        # Simpler: count from original match tags
        league_match_ids = set()
        for m in all_matches:
            if m.get("_league_id") == lid:
                mid = str(m.get("@id", m.get("id", "")))
                if mid in intervals_by_match:
                    league_match_ids.add(mid)

        l_ivs: list[IntervalRecord] = []
        for mid in league_match_ids:
            l_ivs.extend(intervals_by_match[mid])

        l_goals = sum(len(iv.home_goal_times) + len(iv.away_goal_times) for iv in l_ivs)
        l_reds = sum(len(iv.red_card_transitions) for iv in l_ivs)

        from src.calibration.step_1_2_Q_estimation import estimate_Q_global as _est_Q
        if l_ivs:
            l_Q = _est_Q(l_ivs)
            l_cond = q_condition_number(l_Q)
            cond_str = f"{l_cond:.1f}" if l_cond < 1e10 else "inf"
        else:
            cond_str = "N/A"

        print(f"  {name:<18s} {len(league_match_ids):>9d} {l_goals:>7d} {l_reds:>6d} {cond_str:>8s}")

    # ================================================================
    # 3. σ_a Grid Search with 5-fold Chronological CV
    # ================================================================
    print("\n" + "=" * 80)
    print(f"STEP 1.4: σ_a Grid Search × {N_CV_FOLDS}-fold Chronological CV")
    print("=" * 80)

    folds = chronological_cv_split(all_matches, N_CV_FOLDS)
    print(f"  Folds: {len(folds)}")
    for i, (train, val) in enumerate(folds):
        print(f"    Fold {i+1}: train={len(train)}, val={len(val)}")

    grid_results: dict[float, list[float]] = {}
    for sigma_a in SIGMA_A_GRID:
        print(f"\n  σ_a = {sigma_a}:")
        fold_scores: list[float] = []
        for i, (train, val) in enumerate(folds):
            bs = evaluate_fold(train, val, sigma_a)
            fold_scores.append(bs)
            print(f"    Fold {i+1}: BS = {bs:.4f}")
        grid_results[sigma_a] = fold_scores
        avg = np.mean(fold_scores)
        print(f"    → Mean BS = {avg:.4f}")

    print(f"\n  {'σ_a':>6s} {'Mean BS':>9s} {'Std BS':>9s}")
    print("  " + "-" * 28)
    best_sigma_a = SIGMA_A_GRID[0]
    best_mean_bs = 1.0
    for sigma_a in SIGMA_A_GRID:
        scores = grid_results[sigma_a]
        m = np.mean(scores)
        s = np.std(scores)
        marker = ""
        if m < best_mean_bs:
            best_mean_bs = m
            best_sigma_a = sigma_a
            marker = " ← best"
        print(f"  {sigma_a:>6.1f} {m:>9.4f} {s:>9.4f}{marker}")

    print(f"\n  Best σ_a = {best_sigma_a} (CV Brier Score = {best_mean_bs:.4f})")

    # ================================================================
    # 4. Final training on ALL data with best σ_a
    # ================================================================
    print("\n" + "=" * 80)
    print(f"STEP 1.4 (final): NLL Optimization (σ_a={best_sigma_a}, {NUM_EPOCHS} epochs, ALL data)")
    print("=" * 80)

    match_ids, a_H_init, a_A_init, feature_mask = step_1_3_ml_prior(intervals_by_match)
    print(f"  Matches for optimization: {len(match_ids)}")
    print(f"  a_H range: [{a_H_init.min():.4f}, {a_H_init.max():.4f}]")
    print(f"  a_A range: [{a_A_init.min():.4f}, {a_A_init.max():.4f}]")

    opt_result = step_1_4_optimize(
        intervals_by_match, match_ids, a_H_init, a_A_init,
        sigma_a=best_sigma_a, num_epochs=NUM_EPOCHS,
    )

    losses = opt_result.loss_history
    print(f"\n  NLL: {losses[0]:.2f} → {losses[-1]:.2f} ({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")
    print(f"\n  Step | NLL")
    print("  " + "-" * 25)
    for step in [0, 100, 200, 500, 999]:
        if step < len(losses):
            print(f"  {step:5d} | {losses[step]:12.2f}")

    print(f"\n  --- Fitted Parameters ---")
    print(f"  b (time profile):  {np.array2string(opt_result.b, precision=4)}")
    print(f"  gamma_H:           {np.array2string(opt_result.gamma_H, precision=4)}")
    print(f"  gamma_A:           {np.array2string(opt_result.gamma_A, precision=4)}")
    print(f"  delta_H (lookup):  {np.array2string(opt_result.delta_H, precision=4)}")
    print(f"  delta_A (lookup):  {np.array2string(opt_result.delta_A, precision=4)}")
    print(f"  beta_H={opt_result.beta_H:.4f}, kappa_H={opt_result.kappa_H:.4f}, tau_H={opt_result.tau_H:.4f}")
    print(f"  beta_A={opt_result.beta_A:.4f}, kappa_A={opt_result.kappa_A:.4f}, tau_A={opt_result.tau_A:.4f}")

    # ================================================================
    # 5. Step 1.5: Validation
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 1.5: Validation + Go/No-Go")
    print("=" * 80)

    val_result = step_1_5_validate(opt_result, intervals_by_match, match_ids)

    print(f"\n  Brier Score (model):    {val_result.bs_model:.4f}")
    print(f"  Brier Score (baseline): {val_result.bs_exchange:.4f}")
    print(f"  ΔBS:                    {val_result.delta_bs:.4f}")
    print(f"  Log Loss:               {val_result.log_loss_model:.4f}")

    print(f"\n  Sanity Thresholds:")
    print(f"    go_threshold:   {val_result.thresholds.go_threshold:.4f}")
    print(f"    hold_threshold: {val_result.thresholds.hold_threshold:.4f}")
    print(f"    median_delta:   {val_result.thresholds.median_delta:.4f}")

    print(f"\n  Gamma sign checks:")
    for c in val_result.gamma_sign_checks:
        mark = "PASS" if c.passes else "FAIL"
        print(f"    {c.param_name}: {c.actual_value:+.4f} (expected {c.expected_sign}) → {mark}")

    print(f"\n  Delta sign checks:")
    for c in val_result.delta_sign_checks:
        mark = "PASS" if c.passes else "FAIL"
        print(f"    {c.param_name}: {c.actual_value:+.4f} (expected {c.expected_sign}) → {mark}")

    print(f"\n  Go/No-Go: {'GO' if val_result.go_decision else 'NO-GO'}")
    for r in val_result.reasons:
        print(f"    {r}")

    # ================================================================
    # 6. Save to production_params (JSON file + DB attempt)
    # ================================================================
    print("\n" + "=" * 80)
    print("SAVE: production_params")
    print("=" * 80)

    phase1_result = Phase1Result(
        Q_global=Q_global,
        Q_by_delta_S=Q_by_ds,
        feature_mask=feature_mask,
        n_train_matches=len(match_ids),
        optimization=opt_result,
        validation=val_result,
        league_id=0,  # all leagues
        n_matches=n_matches_iv,
        n_goals=n_goals,
    )

    params_json = params_to_json(phase1_result)
    validation_json = validation_to_json(phase1_result)
    thresholds_json_ = thresholds_to_json(phase1_result)

    production_record = {
        "params": params_json,
        "xgb_model_path": "mle_fallback",
        "feature_mask": feature_mask,
        "validation": validation_json,
        "sanity_thresholds": thresholds_json_,
        "is_active": True,
        "sigma_a": best_sigma_a,
        "cv_brier_score": best_mean_bs,
        "cv_results": {str(s): scores for s, scores in grid_results.items()},
        "n_leagues": len(league_counts),
        "n_total_matches": total,
        "n_intervalized_matches": n_matches_iv,
        "n_goals": n_goals,
        "n_red_cards": n_red,
        "Q_condition_number": cond_global,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "phase1_full_retrain.json"
    with open(out_path, "w") as f:
        json.dump(production_record, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    # Attempt DB insert
    try:
        import asyncio
        from dotenv import load_dotenv
        load_dotenv()
        from src.common.db import create_pool

        async def _save_db() -> None:
            pool = await create_pool(os.environ["DB_URL"])
            async with pool.acquire() as conn:
                # Deactivate old params
                await conn.execute("UPDATE production_params SET is_active = FALSE WHERE is_active = TRUE")
                # Insert new
                await conn.execute(
                    """INSERT INTO production_params
                       (params, xgb_model_path, feature_mask, validation, sanity_thresholds, is_active)
                       VALUES ($1, $2, $3, $4, $5, TRUE)""",
                    json.dumps(params_json),
                    "mle_fallback",
                    json.dumps(feature_mask),
                    json.dumps(validation_json),
                    json.dumps(thresholds_json_),
                )
            await pool.close()
            print("  Saved to PostgreSQL production_params (is_active=TRUE)")

        asyncio.run(_save_db())
    except Exception as e:
        print(f"  DB save skipped ({type(e).__name__}: {str(e)[:60]})")
        print(f"  JSON file at {out_path} is the authoritative record")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total matches loaded:      {total}")
    print(f"  Matches intervalized:      {n_matches_iv}")
    print(f"  Total goals:               {n_goals}")
    print(f"  Total red cards:           {n_red}")
    print(f"  Best σ_a:                  {best_sigma_a}")
    print(f"  CV Brier Score:            {best_mean_bs:.4f}")
    print(f"  Final ΔBS:                 {val_result.delta_bs:.4f}")
    print(f"  Q condition number:        {cond_global:.2f}")
    print(f"  Go/No-Go:                  {'GO' if val_result.go_decision else 'NO-GO'}")
    print(f"  Elapsed:                   {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
