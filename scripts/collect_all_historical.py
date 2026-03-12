#!/usr/bin/env python3
"""Collect 6-season × 8-league historical commentaries from Goalserve.

Fetches via the commentaries-by-league endpoint (which includes summary,
matchinfo, goals with minutes, red cards, and stoppage time — unlike
soccerhistory which only has fixtures + scores).

Strategy per league+season:
  1. Fetch soccerhistory to discover unique matchday dates
  2. For each date, fetch commentaries-by-league
  3. Deduplicate by match ID, save to data/commentaries/{league_id}/{season}/
  4. Track goals, red cards, stoppage time availability

Usage:
    python scripts/collect_all_historical.py
    python scripts/collect_all_historical.py --league 1204     # EPL only
    python scripts/collect_all_historical.py --resume           # skip already-saved seasons
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.clients.goalserve import GoalserveClient, extract_goals, extract_red_cards

# ---------------------------------------------------------------------------
# League + season config
# ---------------------------------------------------------------------------

# European leagues: Aug-May cycle → "YYYY-YYYY" format
EUROPEAN_LEAGUES: dict[int, str] = {
    1204: "EPL",
    1399: "La Liga",
    1269: "Serie A",
    1229: "Bundesliga",
    1221: "Ligue 1",
}
EUROPEAN_SEASONS = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

# Calendar-year leagues → "YYYY" format
CALENDAR_LEAGUES: dict[int, str] = {
    1440: "MLS",
    1141: "Brasileirão",
    1081: "Liga Argentina",
}
CALENDAR_SEASONS = [
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
]

SAVE_DIR = Path("data/commentaries")
MAX_CONCURRENT_DATES = 3  # conservative to avoid 429/500


# ---------------------------------------------------------------------------
# Data counting helpers
# ---------------------------------------------------------------------------


def count_goals(match: dict[str, Any]) -> int:
    """Count total goals in a match from summary."""
    total = 0
    summary = match.get("summary", {})
    if not summary:
        return 0
    for team_key in ("localteam", "visitorteam"):
        goals = extract_goals(summary, team_key)
        total += sum(1 for g in goals if not g.get("is_var_cancelled", False))
    return total


def count_red_cards(match: dict[str, Any]) -> int:
    total = 0
    summary = match.get("summary", {})
    if not summary:
        return 0
    for team_key in ("localteam", "visitorteam"):
        total += len(extract_red_cards(summary, team_key))
    return total


def has_stoppage(match: dict[str, Any]) -> bool:
    mi = match.get("matchinfo", {})
    if not mi:
        return False
    t = mi.get("time", {})
    if not isinstance(t, dict):
        return False
    return bool(t.get("@addedTime_period1") or t.get("@addedTime_period2"))


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------


async def fetch_season(
    client: GoalserveClient,
    league_id: int,
    league_name: str,
    season: str,
    pbar: tqdm,  # type: ignore[type-arg]
) -> list[dict[str, Any]]:
    """Fetch full season via fixtures → commentaries-by-date."""

    # Step 1: Discover matchday dates from fixtures
    try:
        fixtures = await client.get_historical_fixtures(league_id, season)
    except Exception as e:
        pbar.write(f"  ⚠ {league_name} {season}: fixtures failed — {e!s:.60s}")
        return []

    if not fixtures:
        pbar.write(f"  ⚠ {league_name} {season}: no fixtures returned")
        return []

    dates: set[str] = set()
    for m in fixtures:
        raw = m.get("@formatted_date", m.get("@date", m.get("date", "")))
        if raw:
            dates.add(str(raw))

    if not dates:
        pbar.write(f"  ⚠ {league_name} {season}: {len(fixtures)} fixtures but 0 parseable dates")
        return []

    # Step 2: Fetch commentaries for each date
    sem = asyncio.Semaphore(MAX_CONCURRENT_DATES)
    all_matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    failed_dates = 0

    async def _fetch_date(date: str) -> list[dict[str, Any]]:
        nonlocal failed_dates
        async with sem:
            for attempt in range(2):
                try:
                    result = await client.get_commentaries_by_league(league_id, date)
                    pbar.update(1)
                    return result
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(2.0)
                    else:
                        failed_dates += 1
                        pbar.update(1)
                        return []
            return []  # unreachable but satisfies type checker

    sorted_dates = sorted(dates)
    pbar.total = len(sorted_dates)
    pbar.reset()
    pbar.set_description(f"  {league_name} {season} ({len(sorted_dates)} dates)")

    tasks = [_fetch_date(d) for d in sorted_dates]
    results = await asyncio.gather(*tasks)

    for batch in results:
        for m in batch:
            mid = str(m.get("@id", m.get("id", "")))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)

    n_summary = sum(1 for m in all_matches if m.get("summary"))
    n_goals = sum(count_goals(m) for m in all_matches)
    n_reds = sum(count_red_cards(m) for m in all_matches)
    n_stoppage = sum(1 for m in all_matches if has_stoppage(m))

    extra = f" ({failed_dates} dates failed)" if failed_dates else ""
    pbar.write(
        f"  ✓ {league_name} {season}: {len(all_matches)} matches, "
        f"{n_goals} goals, {n_reds} reds, "
        f"{n_summary} w/summary, {n_stoppage} w/stoppage{extra}"
    )

    return all_matches


def save_season(
    league_id: int,
    season: str,
    matches: list[dict[str, Any]],
) -> Path:
    """Save matches to data/commentaries/{league_id}/{season}.json."""
    out_dir = SAVE_DIR / str(league_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{season}.json"
    with open(out_path, "w") as f:
        json.dump(matches, f, ensure_ascii=False)
    return out_path


def season_already_saved(league_id: int, season: str) -> bool:
    path = SAVE_DIR / str(league_id) / f"{season}.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(
    *,
    filter_league: int | None = None,
    resume: bool = False,
) -> None:
    api_key = os.environ.get("GOALSERVE_API_KEY", "")
    if not api_key:
        print("ERROR: GOALSERVE_API_KEY not set in .env")
        sys.exit(1)

    # Build work list: (league_id, league_name, season)
    work: list[tuple[int, str, str]] = []
    for lid, name in EUROPEAN_LEAGUES.items():
        if filter_league and lid != filter_league:
            continue
        for season in EUROPEAN_SEASONS:
            work.append((lid, name, season))
    for lid, name in CALENDAR_LEAGUES.items():
        if filter_league and lid != filter_league:
            continue
        for season in CALENDAR_SEASONS:
            work.append((lid, name, season))

    if resume:
        before = len(work)
        work = [(lid, name, s) for lid, name, s in work if not season_already_saved(lid, s)]
        print(f"Resume mode: {before - len(work)} seasons already saved, {len(work)} remaining\n")

    if not work:
        print("Nothing to collect!")
        return

    print(f"Collecting {len(work)} league+season combos ({len(EUROPEAN_LEAGUES) + len(CALENDAR_LEAGUES)} leagues)")
    print(f"Save dir: {SAVE_DIR.resolve()}\n")

    # Track totals per league
    totals: dict[int, dict[str, int]] = {}

    client = GoalserveClient(api_key=api_key, timeout=45.0)

    # Per-date progress bar
    date_pbar = tqdm(total=0, desc="  dates", unit="date", leave=False, position=1)
    # Per-season progress bar
    season_pbar = tqdm(total=len(work), desc="Seasons", unit="season", position=0)

    try:
        for lid, name, season in work:
            if lid not in totals:
                totals[lid] = {"matches": 0, "goals": 0, "reds": 0, "seasons": 0, "failed": 0}

            matches = await fetch_season(client, lid, name, season, date_pbar)

            if matches:
                save_season(lid, season, matches)
                n_goals = sum(count_goals(m) for m in matches)
                n_reds = sum(count_red_cards(m) for m in matches)
                totals[lid]["matches"] += len(matches)
                totals[lid]["goals"] += n_goals
                totals[lid]["reds"] += n_reds
                totals[lid]["seasons"] += 1
            else:
                totals[lid]["failed"] += 1

            season_pbar.update(1)

            # Small delay between seasons
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.\n")
    finally:
        date_pbar.close()
        season_pbar.close()
        await client.close()

    # Summary table
    all_leagues = {**EUROPEAN_LEAGUES, **CALENDAR_LEAGUES}
    print("\n" + "=" * 95)
    print(f"{'League':<18s} {'ID':>5s} {'Seasons':>8s} {'Matches':>9s} {'Goals':>7s} {'Red Cards':>10s} {'Status'}")
    print("-" * 95)

    grand_matches = grand_goals = grand_reds = 0
    for lid, name in all_leagues.items():
        if filter_league and lid != filter_league:
            continue
        t = totals.get(lid, {"matches": 0, "goals": 0, "reds": 0, "seasons": 0, "failed": 0})
        status = "OK" if t["failed"] == 0 and t["seasons"] > 0 else f"{t['failed']} failed"
        print(
            f"{name:<18s} {lid:>5d} {t['seasons']:>6d}/6 {t['matches']:>9d} "
            f"{t['goals']:>7d} {t['reds']:>10d}   {status}"
        )
        grand_matches += t["matches"]
        grand_goals += t["goals"]
        grand_reds += t["reds"]

    print("-" * 95)
    print(f"{'TOTAL':<18s} {'':>5s} {'':>8s} {grand_matches:>9d} {grand_goals:>7d} {grand_reds:>10d}")
    print("=" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical commentaries for all leagues")
    parser.add_argument("--league", type=int, default=None, help="Collect single league ID only")
    parser.add_argument("--resume", action="store_true", help="Skip seasons already saved")
    args = parser.parse_args()

    asyncio.run(run(filter_league=args.league, resume=args.resume))


if __name__ == "__main__":
    main()
