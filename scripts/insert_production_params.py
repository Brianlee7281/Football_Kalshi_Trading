#!/usr/bin/env python3
"""Insert production_params from JSON file into PostgreSQL.

Reads data/production_params/phase1_full_retrain.json and inserts into
the production_params table (deactivating any existing active row first).

Usage:
    DB_URL=postgresql://postgres:postgres@localhost:5432/soccer_trading \
    python scripts/insert_production_params.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.db import create_pool

PARAMS_PATH = Path("data/production_params/phase1_full_retrain.json")


async def main() -> None:
    if not PARAMS_PATH.exists():
        print(f"ERROR: {PARAMS_PATH} not found")
        sys.exit(1)

    db_url = os.environ.get(
        "DB_URL",
        "postgresql://postgres:postgres@localhost:5432/soccer_trading",
    )

    record = json.loads(PARAMS_PATH.read_text())
    params = json.dumps(record["params"])
    xgb_model_path = record.get("xgb_model_path", "mle_fallback")
    feature_mask = json.dumps(record.get("feature_mask", []))
    validation = json.dumps(record.get("validation", {}))
    sanity_thresholds = json.dumps(record.get("sanity_thresholds", {}))

    print(f"Connecting to {db_url.split('@')[1] if '@' in db_url else db_url}...")
    pool = await create_pool(db_url)

    async with pool.acquire() as conn:
        # Deactivate existing
        n = await conn.execute(
            "UPDATE production_params SET is_active = FALSE WHERE is_active = TRUE"
        )
        print(f"  Deactivated existing active params: {n}")

        # Insert new
        row = await conn.fetchrow(
            """INSERT INTO production_params
               (params, xgb_model_path, feature_mask, validation, sanity_thresholds, is_active)
               VALUES ($1::jsonb, $2, $3::jsonb, $4::jsonb, $5::jsonb, TRUE)
               RETURNING version""",
            params, xgb_model_path, feature_mask, validation, sanity_thresholds,
        )
        version = row["version"] if row else "unknown"
        print(f"  Inserted production_params version={version} (is_active=TRUE)")

        # Verify
        check = await conn.fetchrow(
            "SELECT version, is_active, created_at FROM production_params WHERE is_active = TRUE"
        )
        if check:
            print(f"  Verified: version={check['version']}, active={check['is_active']}, created={check['created_at']}")
        else:
            print("  WARNING: No active row found after insert!")

    await pool.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
