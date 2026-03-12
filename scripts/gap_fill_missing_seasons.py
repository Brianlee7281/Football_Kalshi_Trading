#!/usr/bin/env python3
"""Gap-fill missing seasons by probing commentaries every 3 days.

For seasons where soccerhistory returns empty (no fixtures), this script
bypasses fixture discovery entirely and iterates dates every 3 days across
the season window, calling commentaries-by-league directly.

Missing seasons identified from collect_all_historical.py:
  - Bundesliga (1229): 5 of 6 seasons missing
  - Ligue 1 (1221): 3 of 6 seasons missing
  - MLS (1440): all 6 seasons missing
  - Liga Argentina (1081): all 6 seasons missing

Usage:
    python scripts/gap_fill_missing_seasons.py
    python scripts/gap_fill_missing_seasons.py --league 1229   # Bundesliga only
    python scripts/gap_fill_missing_seasons.py --resume        # skip already-saved
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.clients.goalserve import GoalserveClient, extract_goals, extract_red_cards

# ---------------------------------------------------------------------------
# Season windows: (start_date, end_date) for date iteration
# ---------------------------------------------------------------------------

# European leagues: Aug-May cycle
EUROPEAN_SEASON_WINDOWS: dict[str, tuple[str, str]] = {
    "2019-2020": ("2019-08-01", "2020-07-31"),
    "2020-2021": ("2020-08-01", "2021-07-31"),
    "2021-2022": ("2021-08-01", "2022-07-31"),
    "2022-2023": ("2022-08-01", "2023-07-31"),
    "2023-2024": ("2023-08-01", "2024-07-31"),
    "2024-2025": ("2024-08-01", "2025-07-31"),
}

# Calendar-year leagues: Feb-Dec (MLS: Mar-Dec, Argentina: Feb-Dec)
CALENDAR_SEASON_WINDOWS: dict[str, tuple[str, str]] = {
    "2020": ("2020-02-01", "2020-12-31"),
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-02-01", "2022-12-31"),
    "2023": ("2023-02-01", "2023-12-31"),
    "2024": ("2024-02-01", "2024-12-31"),
    "2025": ("2025-02-01", "2025-12-31"),
}

# Missing seasons per league (from collect_all_historical.py run)
GAPS: dict[int, dict[str, list[str]]] = {
    # Bundesliga — European format
    1229: {
        "name": "Bundesliga",
        "seasons": ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"],
        "type": "european",
    },
    # Ligue 1 — European format
    1221: {
        "name": "Ligue 1",
        "seasons": ["2019-2020", "2021-2022", "2022-2023"],
        "type": "european",
    },
    # MLS — Calendar format
    1440: {
        "name": "MLS",
        "seasons": ["2020", "2021", "2022", "2023", "2024", "2025"],
        "type": "calendar",
    },
    # Liga Argentina — Calendar format
    1081: {
        "name": "Liga Argentina",
        "seasons": ["2020", "2021", "2022", "2023", "2024", "2025"],
        "type": "calendar",
    },
}

SAVE_DIR = Path("data/commentaries")
MAX_CONCURRENT_DATES = 3
DATE_STEP_DAYS = 3  # probe every 3 days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_dates(start: str, end: str, step_days: int = DATE_STEP_DAYS) -> list[str]:
    """Generate dates in DD.MM.YYYY format (Goalserve format) every step_days."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    dates: list[str] = []
    current = start_d
    while current <= end_d:
        dates.append(current.strftime("%d.%m.%Y"))
        current += timedelta(days=step_days)
    return dates


def count_goals(match: dict[str, Any]) -> int:
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


def save_season(
    league_id: int,
    season: str,
    matches: list[dict[str, Any]],
) -> Path:
    out_dir = SAVE_DIR / str(league_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{season}.json"
    with open(out_path, "w") as f:
        json.dump(matches, f, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Core gap-fill
# ---------------------------------------------------------------------------


async def gap_fill_season(
    client: GoalserveClient,
    league_id: int,
    league_name: str,
    season: str,
    start_date: str,
    end_date: str,
    pbar: tqdm,  # type: ignore[type-arg]
) -> list[dict[str, Any]]:
    """Probe every 3 days across the season window via commentaries-by-league."""
    dates = generate_dates(start_date, end_date)
    sem = asyncio.Semaphore(MAX_CONCURRENT_DATES)
    all_matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    failed_dates = 0

    async def _fetch_date(dt: str) -> list[dict[str, Any]]:
        nonlocal failed_dates
        async with sem:
            for attempt in range(2):
                try:
                    result = await client.get_commentaries_by_league(league_id, dt)
                    pbar.update(1)
                    return result
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(2.0)
                    else:
                        failed_dates += 1
                        pbar.update(1)
                        return []
            return []

    pbar.total = len(dates)
    pbar.reset()
    pbar.set_description(f"  {league_name} {season} ({len(dates)} dates)")

    tasks = [_fetch_date(d) for d in dates]
    results = await asyncio.gather(*tasks)

    for batch in results:
        for m in batch:
            mid = str(m.get("@id", m.get("id", "")))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                all_matches.append(m)

    n_goals = sum(count_goals(m) for m in all_matches)
    n_reds = sum(count_red_cards(m) for m in all_matches)
    n_summary = sum(1 for m in all_matches if m.get("summary"))

    extra = f" ({failed_dates} dates failed)" if failed_dates else ""
    pbar.write(
        f"  ✓ {league_name} {season}: {len(all_matches)} matches, "
        f"{n_goals} goals, {n_reds} reds, {n_summary} w/summary{extra}"
    )

    return all_matches


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

    # Build work list
    work: list[tuple[int, str, str, str, str]] = []  # (lid, name, season, start, end)
    for lid, info in GAPS.items():
        if filter_league and lid != filter_league:
            continue
        windows = EUROPEAN_SEASON_WINDOWS if info["type"] == "european" else CALENDAR_SEASON_WINDOWS
        for season in info["seasons"]:
            if season not in windows:
                continue
            start, end = windows[season]
            work.append((lid, info["name"], season, start, end))

    if resume:
        before = len(work)
        work = [(lid, name, s, st, en) for lid, name, s, st, en in work if not season_already_saved(lid, s)]
        print(f"Resume mode: {before - len(work)} seasons already saved, {len(work)} remaining\n")

    if not work:
        print("Nothing to gap-fill!")
        return

    total_dates = sum(len(generate_dates(st, en)) for _, _, _, st, en in work)
    print(f"Gap-filling {len(work)} league+season combos (~{total_dates} date probes)")
    print(f"Save dir: {SAVE_DIR.resolve()}\n")

    client = GoalserveClient(api_key=api_key, timeout=45.0)
    date_pbar = tqdm(total=0, desc="  dates", unit="date", leave=False, position=1)
    season_pbar = tqdm(total=len(work), desc="Seasons", unit="season", position=0)

    totals: dict[int, dict[str, int]] = {}

    try:
        for lid, name, season, start, end in work:
            if lid not in totals:
                totals[lid] = {"matches": 0, "goals": 0, "reds": 0, "seasons": 0, "failed": 0}

            matches = await gap_fill_season(client, lid, name, season, start, end, date_pbar)

            if matches:
                save_season(lid, season, matches)
                totals[lid]["matches"] += len(matches)
                totals[lid]["goals"] += sum(count_goals(m) for m in matches)
                totals[lid]["reds"] += sum(count_red_cards(m) for m in matches)
                totals[lid]["seasons"] += 1
            else:
                totals[lid]["failed"] += 1

            season_pbar.update(1)
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.\n")
    finally:
        date_pbar.close()
        season_pbar.close()
        await client.close()

    # Summary
    print("\n" + "=" * 80)
    print(f"{'League':<18s} {'ID':>5s} {'Seasons':>8s} {'Matches':>9s} {'Goals':>7s} {'Reds':>6s}")
    print("-" * 80)

    grand_matches = grand_goals = grand_reds = 0
    for lid, info in GAPS.items():
        if filter_league and lid != filter_league:
            continue
        t = totals.get(lid, {"matches": 0, "goals": 0, "reds": 0, "seasons": 0, "failed": 0})
        expected = len(info["seasons"])
        print(
            f"{info['name']:<18s} {lid:>5d} {t['seasons']:>5d}/{expected:<2d} "
            f"{t['matches']:>9d} {t['goals']:>7d} {t['reds']:>6d}"
        )
        grand_matches += t["matches"]
        grand_goals += t["goals"]
        grand_reds += t["reds"]

    print("-" * 80)
    print(f"{'TOTAL':<18s} {'':>5s} {'':>8s} {grand_matches:>9d} {grand_goals:>7d} {grand_reds:>6d}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gap-fill missing seasons via date probing")
    parser.add_argument("--league", type=int, default=None, help="Fill single league ID only")
    parser.add_argument("--resume", action="store_true", help="Skip seasons already saved")
    args = parser.parse_args()

    asyncio.run(run(filter_league=args.league, resume=args.resume))


if __name__ == "__main__":
    main()
