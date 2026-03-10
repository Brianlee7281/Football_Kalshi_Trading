#!/usr/bin/env python3
"""Integration test: Run full Phase 1 pipeline on 1 season of EPL data.

Usage: python scripts/run_phase1_epl.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.phase1_worker import (
    params_to_json,
    run_phase1,
    thresholds_to_json,
    validation_to_json,
)
from src.clients.goalserve import GoalserveClient

EPL_LEAGUE_ID = 1204


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
            score = td.get("@score", td.get("score"))
            if score is not None and str(score).strip() not in ("", "?"):
                return True
    return False


async def main() -> None:
    load_env()
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        return

    # Fetch EPL fixtures
    print("=" * 70)
    print("Fetching EPL fixtures...")
    print("=" * 70)

    async with GoalserveClient(api_key, timeout=60.0) as client:
        try:
            matches = await client.get_fixtures(EPL_LEAGUE_ID)
        except Exception:
            matches = []
        if not matches:
            for season in ["2024-2025", "2023-2024"]:
                try:
                    matches = await client.get_historical_fixtures(EPL_LEAGUE_ID, season)
                    if matches:
                        break
                except Exception:
                    continue

    completed = [m for m in matches if match_has_score(m)]
    print(f"Total: {len(matches)}, Completed: {len(completed)}")

    if len(completed) < 10:
        print("ERROR: Not enough completed matches")
        return

    # Run Phase 1 pipeline
    print("\n" + "=" * 70)
    print("Running Phase 1 pipeline (σ_a=0.3, 500 epochs)...")
    print("=" * 70)

    result = run_phase1(
        completed,
        league_id=EPL_LEAGUE_ID,
        sigma_a=0.3,
        num_epochs=500,  # fewer epochs for integration test
    )

    # Print results
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS")
    print("=" * 70)
    print(f"  Matches: {result.n_matches}")
    print(f"  Goals: {result.n_goals}")
    print(f"  Train matches: {result.n_train_matches}")

    opt = result.optimization
    print(f"\n  --- Fitted Parameters ---")
    print(f"  b (time):     {opt.b}")
    print(f"  gamma_H:      {opt.gamma_H}")
    print(f"  gamma_A:      {opt.gamma_A}")
    print(f"  delta_H:      {opt.delta_H}")
    print(f"  delta_A:      {opt.delta_A}")
    print(f"  tau_H={opt.tau_H:.4f}, tau_A={opt.tau_A:.4f}")
    print(f"  beta_H={opt.beta_H:.4f}, kappa_H={opt.kappa_H:.4f}")
    print(f"  beta_A={opt.beta_A:.4f}, kappa_A={opt.kappa_A:.4f}")

    losses = opt.loss_history
    print(f"\n  NLL: {losses[0]:.2f} → {losses[-1]:.2f} ({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")

    if result.validation:
        v = result.validation
        print(f"\n  --- Validation ---")
        print(f"  BS_model:    {v.bs_model:.6f}")
        print(f"  BS_exchange: {v.bs_exchange:.6f}")
        print(f"  ΔBS:         {v.delta_bs:.6f}")
        print(f"  Go decision: {v.go_decision}")
        if v.reasons:
            for r in v.reasons:
                print(f"    - {r}")

    # Test serialization
    pj = params_to_json(result)
    vj = validation_to_json(result)
    tj = thresholds_to_json(result)
    print(f"\n  --- Serialization ---")
    print(f"  params keys:      {list(pj.keys())}")
    print(f"  validation keys:  {list(vj.keys())}")
    print(f"  thresholds keys:  {list(tj.keys())}")

    print("\n" + "=" * 70)
    print("PHASE 1 PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
