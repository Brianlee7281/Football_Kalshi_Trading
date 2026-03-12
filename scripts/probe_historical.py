#!/usr/bin/env python3
"""Probe Goalserve historical data availability across leagues and seasons.

Part 1: Fetch raw EPL 2023-2024, print first match JSON structure.
Part 2: For all 8 leagues × 6 seasons, report match count + data availability.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()

from src.clients.goalserve import GoalserveClient, extract_goals, extract_red_cards

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

# Goalserve season format: "YYYY-YYYY"
SEASONS = [
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]


def inspect_match(m: dict) -> None:
    """Print structural inspection of one match dict."""
    print("\n--- RAW FIRST MATCH (top-level keys) ---")
    for k in sorted(m.keys()):
        v = m[k]
        if isinstance(v, dict):
            print(f"  {k}: dict({len(v)} keys) → {list(v.keys())[:8]}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
        else:
            vstr = str(v)[:80]
            print(f"  {k}: {type(v).__name__} = {vstr}")

    # Goal minutes?
    summary = m.get("summary", {})
    has_goals = False
    has_red_cards = False
    if summary:
        for team_key in ("localteam", "visitorteam"):
            goals = extract_goals(summary, team_key)
            if goals:
                has_goals = True
                print(f"\n  Goals ({team_key}): {len(goals)}")
                for g in goals[:3]:
                    print(f"    min={g.get('parsed_minute','?')}, scorer={g.get('@name', g.get('name', '?'))}")
            cards = extract_red_cards(summary, team_key)
            if cards:
                has_red_cards = True
                print(f"  Red cards ({team_key}): {len(cards)}")
                for c in cards[:2]:
                    print(f"    min={c.get('parsed_minute','?')}, player={c.get('@name', c.get('name', '?'))}")
    else:
        print("\n  summary: MISSING")

    # Lineups?
    for lk in ("lineup", "lineups", "teams"):
        if lk in m:
            val = m[lk]
            if isinstance(val, dict):
                print(f"\n  {lk} keys: {list(val.keys())[:10]}")
            else:
                print(f"\n  {lk}: {type(val).__name__}")
            break
    else:
        print("\n  lineups: NOT FOUND in top-level keys")

    # Stoppage time?
    matchinfo = m.get("matchinfo", {})
    if matchinfo:
        time_info = matchinfo.get("time", matchinfo.get("@time", ""))
        print(f"  matchinfo.time: {time_info}")
        print(f"  matchinfo keys: {list(matchinfo.keys())[:10]}")
    else:
        print("  matchinfo: MISSING")

    # Score?
    for sk in ("@localteam_score", "localteam_score", "@ft_score", "ft_score"):
        if sk in m:
            print(f"  {sk}: {m[sk]}")

    print(f"\n  has_goal_minutes={has_goals}, has_red_cards={has_red_cards}")
    print(f"  has_summary={'summary' in m}, has_matchinfo={'matchinfo' in m}")


def check_match_data(m: dict) -> dict:
    """Check what data is available in a match."""
    summary = m.get("summary", {})
    has_goals = False
    has_red_cards = False
    if summary:
        for tk in ("localteam", "visitorteam"):
            if extract_goals(summary, tk):
                has_goals = True
            if extract_red_cards(summary, tk):
                has_red_cards = True
    has_matchinfo = bool(m.get("matchinfo"))
    has_lineups = any(k in m for k in ("lineup", "lineups", "teams"))
    return {
        "has_goals": has_goals,
        "has_red_cards": has_red_cards,
        "has_matchinfo": has_matchinfo,
        "has_lineups": has_lineups,
        "has_summary": bool(summary),
    }


async def main() -> None:
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set")
        sys.exit(1)

    client = GoalserveClient(api_key=api_key, timeout=30.0)

    # ── Part 1: Raw inspection of EPL 2023-2024 ──
    print("=" * 80)
    print("PART 1: Raw JSON structure — EPL (1204) 2023-2024")
    print("=" * 80)

    try:
        matches = await client.get_historical_fixtures(1204, "2023-2024")
        print(f"\nTotal matches returned: {len(matches)}")
        if matches:
            # Find a match with goals for a better inspection
            picked = matches[0]
            for m in matches[:20]:
                s = m.get("summary", {})
                if s and (s.get("localteam") or s.get("visitorteam")):
                    picked = m
                    break
            inspect_match(picked)
        else:
            print("NO MATCHES RETURNED")
    except Exception as e:
        print(f"ERROR: {e}")

    # ── Part 2: All leagues × all seasons ──
    print("\n\n" + "=" * 80)
    print("PART 2: Availability probe — 8 leagues × 6 seasons")
    print("=" * 80 + "\n")

    results: list[dict] = []
    total = len(LEAGUES) * len(SEASONS)
    done = 0

    for league_id, league_name in LEAGUES.items():
        for season in SEASONS:
            done += 1
            row = {
                "league_id": league_id,
                "league": league_name,
                "season": season,
                "status": "FAIL",
                "match_count": 0,
                "goals_present": False,
                "red_cards_present": False,
                "matchinfo_present": False,
                "error": "",
            }
            try:
                matches = await client.get_historical_fixtures(league_id, season)
                row["match_count"] = len(matches)
                if matches:
                    row["status"] = "OK"
                    # Sample up to 20 matches for data checks
                    for m in matches[:20]:
                        info = check_match_data(m)
                        if info["has_goals"]:
                            row["goals_present"] = True
                        if info["has_red_cards"]:
                            row["red_cards_present"] = True
                        if info["has_matchinfo"]:
                            row["matchinfo_present"] = True
                else:
                    row["status"] = "EMPTY"
            except Exception as e:
                row["error"] = str(e)[:80]

            tag = "OK" if row["status"] == "OK" else row["status"]
            print(f"  [{done:2d}/{total}] {league_name:14s} {season}  {tag:5s}  {row['match_count']:3d} matches"
                  f"  goals={row['goals_present']}  reds={row['red_cards_present']}  matchinfo={row['matchinfo_present']}"
                  + (f"  err={row['error']}" if row['error'] else ""))
            results.append(row)
            await asyncio.sleep(0.4)

    await client.close()

    # ── Summary table ──
    short_seasons = ["20-21", "21-22", "22-23", "23-24", "24-25", "25-26"]

    print("\n\n" + "=" * 120)
    print(f"{'League':<16s}", end="")
    for s in short_seasons:
        print(f" | {s:>14s}", end="")
    print()
    print("-" * 120)

    for league_id, league_name in LEAGUES.items():
        print(f"{league_name:<16s}", end="")
        for season in SEASONS:
            r = next(x for x in results if x["league_id"] == league_id and x["season"] == season)
            if r["status"] == "OK":
                flags = ""
                flags += "G" if r["goals_present"] else "-"
                flags += "R" if r["red_cards_present"] else "-"
                flags += "M" if r["matchinfo_present"] else "-"
                cell = f"{r['match_count']:3d}m [{flags}]"
            elif r["status"] == "EMPTY":
                cell = "EMPTY"
            else:
                cell = "FAIL"
            print(f" | {cell:>14s}", end="")
        print()

    print("=" * 120)
    print("Flags: G=goal minutes  R=red cards  M=matchinfo/stoppage  -=absent")

    # Failures summary
    fails = [r for r in results if r["status"] == "FAIL"]
    empties = [r for r in results if r["status"] == "EMPTY"]
    oks = [r for r in results if r["status"] == "OK"]

    print(f"\nTotals: {len(oks)} OK, {len(empties)} EMPTY, {len(fails)} FAIL")

    if fails:
        print("\nFailed probes:")
        for r in fails:
            print(f"  {r['league']} {r['season']}: {r['error']}")

    if empties:
        print("\nEmpty probes (0 matches):")
        for r in empties:
            print(f"  {r['league']} {r['season']}")


if __name__ == "__main__":
    asyncio.run(main())
