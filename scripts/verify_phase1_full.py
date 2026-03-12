#!/usr/bin/env python3
"""Full Phase 1 pipeline verification.

1. Run end-to-end on EPL 2023-2024 via commentaries
2. Verify production_params JSON has all expected keys
3. Run Steps 1.1-1.2 on La Liga and MLS for Q matrix comparison
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.phase1_worker import (
    Phase1Result,
    fetch_season_commentaries,
    params_to_json,
    run_phase1,
    step_1_1_intervalize,
    step_1_2_estimate_Q,
    thresholds_to_json,
    validation_to_json,
)
from src.clients.goalserve import GoalserveClient

# ---------------------------------------------------------------------------

EPL_LEAGUE_ID = 1204
LA_LIGA_LEAGUE_ID = 1399
MLS_LEAGUE_ID = 1440


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


def q_condition_number(Q: np.ndarray) -> float:
    active = [i for i in range(Q.shape[0]) if Q[i, i] != 0.0]
    if len(active) < 2:
        return float("inf")
    sub = Q[np.ix_(active, active)]
    svs = np.linalg.svd(sub, compute_uv=False)
    if svs[-1] == 0:
        return float("inf")
    return float(svs[0] / svs[-1])


def hr() -> None:
    print("=" * 72)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


async def main() -> None:
    load_env()
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        return

    # ==================================================================
    # 1. EPL 2023-2024 full pipeline
    # ==================================================================
    hr()
    print(" 1. EPL 2023-2024 — Full Phase 1 Pipeline")
    hr()

    async with GoalserveClient(api_key, timeout=60.0) as gs:
        print("  Fetching commentaries...")
        epl_matches = await fetch_season_commentaries(gs, EPL_LEAGUE_ID, "2023-2024")

    print(f"  Matches fetched: {len(epl_matches)}")

    t0 = time.perf_counter()
    result = run_phase1(
        epl_matches,
        league_id=EPL_LEAGUE_ID,
        sigma_a=0.3,
        num_epochs=1000,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Pipeline completed in {elapsed:.1f}s")

    # Print key results
    opt = result.optimization
    print(f"\n  Matches: {result.n_matches}, Goals: {result.n_goals}")
    n_red = sum(
        len(iv.red_card_transitions)
        for ivs in _get_all_intervals(epl_matches)
        for iv in ivs
    )
    print(f"  Red card transitions: {n_red}")
    print(f"  Q condition number: {q_condition_number(result.Q_global):.2f}")
    print(f"  NLL: {opt.loss_history[0]:.2f} → {opt.loss_history[-1]:.2f}")
    print(f"  tau_H={opt.tau_H:.4f}, tau_A={opt.tau_A:.4f}")
    print(f"  gamma_H: {np.array2string(opt.gamma_H, precision=4)}")
    print(f"  gamma_A: {np.array2string(opt.gamma_A, precision=4)}")

    if result.validation:
        v = result.validation
        print(f"  BS_model={v.bs_model:.6f}, BS_exchange={v.bs_exchange:.6f}")
        print(f"  ΔBS={v.delta_bs:+.6f}, Go={v.go_decision}")
    print()

    # ==================================================================
    # 2. Verify production_params JSON schema
    # ==================================================================
    hr()
    print(" 2. Verify production_params JSON Keys")
    hr()

    pj = params_to_json(result)
    vj = validation_to_json(result)
    tj = thresholds_to_json(result)

    # Expected keys per schema.sql columns
    params_expected = {
        "b", "gamma_H", "gamma_A", "delta_H", "delta_A",
        "beta_H", "kappa_H", "tau_H", "beta_A", "kappa_A", "tau_A",
        "Q_global", "Q_by_delta_S", "n_matches", "n_goals", "league_id",
    }
    validation_expected = {
        "bs_model", "bs_exchange", "delta_bs",
        "log_loss_model", "go_decision", "reasons",
    }
    thresholds_expected = {
        "go_threshold", "hold_threshold", "median_delta", "n_matches",
    }

    all_pass = True

    def _check(name: str, actual: set[str], expected: set[str]) -> bool:
        missing = expected - actual
        extra = actual - expected
        ok = not missing
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
        if missing:
            print(f"    Missing keys: {missing}")
        if extra:
            print(f"    Extra keys: {extra}")
        return ok

    all_pass &= _check("params JSONB", set(pj.keys()), params_expected)
    all_pass &= _check("validation JSONB", set(vj.keys()), validation_expected)
    all_pass &= _check("sanity_thresholds JSONB", set(tj.keys()), thresholds_expected)

    # Verify JSON-serializability
    try:
        json.dumps(pj)
        json.dumps(vj)
        json.dumps(tj)
        print(f"  JSON serializable: ✓ PASS")
    except (TypeError, ValueError) as e:
        print(f"  JSON serializable: ✗ FAIL — {e}")
        all_pass = False

    # Verify value types
    checks = [
        ("params.b is list[float]", isinstance(pj["b"], list) and len(pj["b"]) == 6),
        ("params.gamma_H len=4", isinstance(pj["gamma_H"], list) and len(pj["gamma_H"]) == 4),
        ("params.delta_H len=5", isinstance(pj["delta_H"], list) and len(pj["delta_H"]) == 5),
        ("params.Q_global is 4x4", isinstance(pj["Q_global"], list) and len(pj["Q_global"]) == 4),
        ("params.tau_H is float", isinstance(pj["tau_H"], float)),
        ("params.league_id is int", isinstance(pj["league_id"], int)),
        ("validation.go_decision is bool", isinstance(vj["go_decision"], bool)),
        ("validation.reasons is list", isinstance(vj["reasons"], list)),
        ("thresholds.go < hold", tj["go_threshold"] < tj["hold_threshold"]),
        ("feature_mask is list[str]", isinstance(result.feature_mask, list)),
    ]

    for label, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {label}")
        if not ok:
            all_pass = False

    # Simulated production_params row
    print(f"\n  --- Simulated production_params row ---")
    print(f"  version:            (auto)")
    print(f"  params:             {len(json.dumps(pj))} bytes")
    print(f"  xgb_model_path:     (no XGBoost model — MLE fallback)")
    print(f"  feature_mask:       {json.dumps(result.feature_mask[:5])}... ({len(result.feature_mask)} features)")
    print(f"  validation:         {len(json.dumps(vj))} bytes")
    print(f"  sanity_thresholds:  {len(json.dumps(tj))} bytes")
    print(f"  is_active:          True")

    if all_pass:
        print(f"\n  ✓ All production_params checks PASS")
    else:
        print(f"\n  ✗ Some checks FAILED")
        return

    # ==================================================================
    # 3. Multi-league Q matrices (La Liga, MLS)
    # ==================================================================
    hr()
    print(" 3. Multi-League Q Matrix Comparison")
    hr()

    leagues = [
        ("La Liga", LA_LIGA_LEAGUE_ID, "2023-2024"),
        ("MLS", MLS_LEAGUE_ID, "2024"),
    ]

    async with GoalserveClient(api_key, timeout=60.0) as gs:
        for name, lid, season in leagues:
            print(f"\n  --- {name} (ID={lid}, season={season}) ---")
            try:
                matches = await fetch_season_commentaries(gs, lid, season)
            except Exception as e:
                print(f"  Failed to fetch: {e}")
                # Try alternate season formats
                alt_seasons = ["2023-2024", "2024-2025", "2023", "2024"]
                alt_seasons = [s for s in alt_seasons if s != season]
                fetched = False
                for alt in alt_seasons:
                    try:
                        matches = await fetch_season_commentaries(gs, lid, alt)
                        if matches:
                            print(f"  Used alternate season: {alt}")
                            fetched = True
                            break
                    except Exception:
                        continue
                if not fetched:
                    print(f"  Skipping {name} — no data available")
                    continue

            if not matches:
                print(f"  No matches returned — skipping")
                continue

            intervals_by_match, all_intervals = step_1_1_intervalize(matches)
            n_red = sum(len(iv.red_card_transitions) for iv in all_intervals)
            n_goals = sum(
                len(iv.home_goal_times) + len(iv.away_goal_times)
                for iv in all_intervals
            )

            print(f"  Matches: {len(intervals_by_match)}, Goals: {n_goals}, Red cards: {n_red}")

            if all_intervals:
                Q_global, Q_by_ds = step_1_2_estimate_Q(all_intervals)
                cond = q_condition_number(Q_global)
                print(f"  Q matrix:")
                for i in range(4):
                    row = " ".join(f"{Q_global[i,j]:10.6f}" for j in range(4))
                    print(f"    [{row}]")
                print(f"  Row sums: {np.array2string(Q_global.sum(axis=1), precision=8)}")
                print(f"  Condition number: {cond:.2f}")

    # ==================================================================
    # Summary
    # ==================================================================
    hr()
    print(" VERIFICATION COMPLETE — ALL CHECKS PASSED")
    hr()


def _get_all_intervals(matches: list[dict[str, Any]]) -> list[list[Any]]:
    """Quick helper to count red cards from raw matches."""
    from src.calibration.step_1_1_intervals import build_intervals_from_goalserve
    result = []
    for m in matches:
        try:
            ivs = build_intervals_from_goalserve(m)
            if ivs:
                result.append(ivs)
        except Exception:
            pass
    return result


if __name__ == "__main__":
    asyncio.run(main())
