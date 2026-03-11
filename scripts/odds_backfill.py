#!/usr/bin/env python3
"""Daily job: accumulate Odds-API data for future Phase 1 calibration.

Odds-API historical data is only available from ~Dec 2025 onwards.
This script fetches and stores closing odds for settled EPL matches,
building up a local cache. By the time paper trading starts, we'll
have enough data for meaningful Betfair Exchange Brier Score comparisons.

Storage: JSON Lines file (one event per line) at data/odds_cache/{league}.jsonl

Usage:
    # Fetch today's settled events
    python scripts/odds_backfill.py

    # Backfill from a specific date
    python scripts/odds_backfill.py --since 2025-12-01

    # Cron: run daily at 06:00 UTC
    # 0 6 * * * cd /path/to/FKT_v3 && python scripts/odds_backfill.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.odds_api import OddsApiClient, build_odds_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEAGUES = {
    "epl": "england-premier-league",
    "la_liga": "spain-la-liga",
    "serie_a": "italy-serie-a",
    "bundesliga": "germany-bundesliga",
    "ligue_1": "france-ligue-1",
}
BOOKMAKERS = "Bet365,Betfair Exchange,Sbobet,1xbet,DraftKings"
CACHE_DIR = Path("data/odds_cache")


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()


def load_existing_ids(path: Path) -> set[str]:
    """Load already-cached event IDs to avoid duplicates."""
    ids: set[str] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    ids.add(str(ev.get("id", "")))
                except json.JSONDecodeError:
                    continue
    return ids


async def backfill_league(
    client: OddsApiClient,
    league_slug: str,
    league_name: str,
    cache_path: Path,
) -> int:
    """Fetch settled events for a league and append new ones to cache."""
    existing = load_existing_ids(cache_path)

    events = await client.get_events(
        "football",
        league=league_slug,
        status="settled",
    )

    if not events:
        print(f"  {league_name}: no settled events found")
        return 0

    new_events: list[dict[str, Any]] = []
    for ev in events:
        ev_id = str(ev.get("id", ""))
        if ev_id in existing:
            continue

        # Fetch odds for this event
        try:
            odds = await client.get_odds(ev_id, BOOKMAKERS)
            bookmakers = odds.get("bookmakers", {})
            if not bookmakers:
                continue
        except Exception:
            continue

        # Build the cache record
        features = build_odds_features(bookmakers)
        record = {
            "id": ev_id,
            "home": ev.get("home", ""),
            "away": ev.get("away", ""),
            "date": ev.get("date", ""),
            "status": ev.get("status", ""),
            "league": league_slug,
            "bookmakers": bookmakers,
            "exchange_probs": {
                "home": features["exchange_home_prob"],
                "draw": features["exchange_draw_prob"],
                "away": features["exchange_away_prob"],
            },
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        new_events.append(record)

    if new_events:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f:
            for rec in new_events:
                f.write(json.dumps(rec) + "\n")

    print(f"  {league_name}: {len(new_events)} new events (total cached: {len(existing) + len(new_events)})")
    return len(new_events)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Odds-API data")
    parser.add_argument("--leagues", nargs="*", default=["epl"],
                        help="League keys to backfill (default: epl)")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("ERROR: ODDS_API_KEY not set in .env")
        return

    print(f"Odds backfill — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("-" * 50)

    total_new = 0
    async with OddsApiClient(api_key) as client:
        for key in args.leagues:
            slug = LEAGUES.get(key)
            if not slug:
                print(f"  Unknown league: {key}")
                continue
            cache_path = CACHE_DIR / f"{key}.jsonl"
            total_new += await backfill_league(client, slug, key, cache_path)

    print(f"\nTotal new events cached: {total_new}")


if __name__ == "__main__":
    asyncio.run(main())
