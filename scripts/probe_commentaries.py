#!/usr/bin/env python3
"""Probe Goalserve commentaries availability across leagues and seasons.

One date per season per league — quick availability check.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.clients.goalserve import GoalserveClient, extract_red_cards

# 8 leagues
LEAGUES = {
    1204: "EPL",
    1399: "La Liga",
    1229: "Serie A",
    1269: "Bundesliga",
    1221: "Ligue 1",
    1440: "MLS",
    1141: "Brasileirão",
    1081: "Liga Argentina",
}

# Seasons with a representative mid-season date (DD.MM.YYYY)
SEASONS = {
    "2020-21": "15.01.2021",
    "2021-22": "15.01.2022",
    "2022-23": "14.01.2023",
    "2023-24": "13.01.2024",
    "2024-25": "11.01.2025",
    "2025-26": "17.01.2026",
}


async def probe_one(
    client: GoalserveClient,
    league_id: int,
    league_name: str,
    season: str,
    date: str,
) -> dict:
    """Probe a single league+date combo."""
    result = {
        "league_id": league_id,
        "league": league_name,
        "season": season,
        "date": date,
        "status": "FAIL",
        "match_count": 0,
        "red_cards_present": False,
        "error": "",
    }
    try:
        matches = await client.get_commentaries_by_league(league_id, date)
        result["match_count"] = len(matches)
        if len(matches) > 0:
            result["status"] = "OK"
            # Check for red cards in any match
            for m in matches:
                summary = m.get("summary", {})
                if summary:
                    for team_key in ("localteam", "visitorteam"):
                        cards = extract_red_cards(summary, team_key)
                        if cards:
                            result["red_cards_present"] = True
                            break
                if result["red_cards_present"]:
                    break
        else:
            result["status"] = "EMPTY"
    except Exception as e:
        result["error"] = str(e)[:80]
    return result


async def main() -> None:
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        sys.exit(1)

    client = GoalserveClient(api_key=api_key, timeout=30.0)

    results: list[dict] = []
    total = len(LEAGUES) * len(SEASONS)
    done = 0

    print(f"Probing {total} league+season combos...\n")

    for league_id, league_name in LEAGUES.items():
        for season, date in SEASONS.items():
            r = await probe_one(client, league_id, league_name, season, date)
            results.append(r)
            done += 1
            status_char = "✓" if r["status"] == "OK" else ("∅" if r["status"] == "EMPTY" else "✗")
            print(f"  [{done}/{total}] {league_name:14s} {season} {date} → {status_char} ({r['match_count']} matches)")
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.3)

    await client.close()

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'League':<16s}", end="")
    for season in SEASONS:
        print(f"  {season:>12s}", end="")
    print()
    print("-" * 110)

    for league_id, league_name in LEAGUES.items():
        print(f"{league_name:<16s}", end="")
        for season in SEASONS:
            r = next(x for x in results if x["league_id"] == league_id and x["season"] == season)
            if r["status"] == "OK":
                rc = "R" if r["red_cards_present"] else " "
                cell = f"{r['match_count']}m {rc}"
            elif r["status"] == "EMPTY":
                cell = "EMPTY"
            else:
                cell = "FAIL"
            print(f"  {cell:>12s}", end="")
        print()

    print("=" * 110)
    print("Legend: Nm = N matches, R = red cards present, EMPTY = 0 matches, FAIL = API error")

    # Failures
    fails = [r for r in results if r["status"] == "FAIL"]
    empties = [r for r in results if r["status"] == "EMPTY"]
    oks = [r for r in results if r["status"] == "OK"]
    rc_count = sum(1 for r in results if r["red_cards_present"])

    print(f"\nTotals: {len(oks)} OK, {len(empties)} EMPTY, {len(fails)} FAIL, {rc_count} with red cards")

    if fails:
        print("\nFailed probes:")
        for r in fails:
            print(f"  {r['league']} {r['season']}: {r['error']}")

    if empties:
        print("\nEmpty probes (no matches on that date):")
        for r in empties:
            print(f"  {r['league']} {r['season']} ({r['date']})")


if __name__ == "__main__":
    asyncio.run(main())
