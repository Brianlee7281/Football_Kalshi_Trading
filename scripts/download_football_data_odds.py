#!/usr/bin/env python3
"""Download football-data.co.uk odds CSVs and benchmark against Pinnacle closing lines."""

import csv
import io
import json
import os
import sys
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# ── Project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = ROOT / "data" / "odds_historical"
COMMENTARIES_DIR = ROOT / "data" / "commentaries"
PARAMS_PATH = ROOT / "data" / "production_params" / "phase1_full_retrain.json"

# ── League config ─────────────────────────────────────────────────────
# European leagues: /mmz4281/{season_code}/{league_code}.csv
EURO_LEAGUES = {
    "E0":  {"name": "EPL",        "goalserve_id": "1204"},
    "SP1": {"name": "La Liga",    "goalserve_id": "1399"},
    "I1":  {"name": "Serie A",    "goalserve_id": "1269"},
    "D1":  {"name": "Bundesliga", "goalserve_id": "1229"},
    "F1":  {"name": "Ligue 1",    "goalserve_id": "1221"},
}

EURO_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]

# Americas leagues: /new/{code}.csv (all seasons in one file)
AMERICAS_LEAGUES = {
    "USA": {"name": "MLS",          "goalserve_id": "1440"},
    "BRA": {"name": "Brasileirão",  "goalserve_id": "1141"},
    "ARG": {"name": "Argentina",    "goalserve_id": "1081"},
}

# Season code → commentary filename mapping
SEASON_TO_FILENAME = {
    "1920": "2019-2020",
    "2021": "2020-2021",
    "2122": "2021-2022",
    "2223": "2022-2023",
    "2324": "2023-2024",
    "2425": "2024-2025",
}

BASE_URL = "https://www.football-data.co.uk"


def download_file(url: str, dest: Path) -> bool:
    """Download a URL to dest. Returns True on success."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except (URLError, OSError) as e:
        print(f"  FAILED {url}: {e}")
        return False


# ── Step 1: Download all CSVs ─────────────────────────────────────────
def download_all() -> dict[str, list[Path]]:
    """Download CSVs, return {league_name: [paths]}."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}

    # European leagues
    for code, info in EURO_LEAGUES.items():
        paths = []
        for season in EURO_SEASONS:
            url = f"{BASE_URL}/mmz4281/{season}/{code}.csv"
            dest = ODDS_DIR / f"{code}_{season}.csv"
            if dest.exists():
                print(f"  exists: {dest.name}")
                paths.append(dest)
            elif download_file(url, dest):
                rows = len(dest.read_text().strip().split("\n")) - 1
                print(f"  ✓ {dest.name} ({rows} matches)")
                paths.append(dest)
        result[info["name"]] = paths

    # Americas leagues (single file per league)
    for code, info in AMERICAS_LEAGUES.items():
        url = f"{BASE_URL}/new/{code}.csv"
        dest = ODDS_DIR / f"{code}.csv"
        if dest.exists():
            print(f"  exists: {dest.name}")
        elif download_file(url, dest):
            rows = len(dest.read_text().strip().split("\n")) - 1
            print(f"  ✓ {dest.name} ({rows} matches)")
        result[info["name"]] = [dest] if dest.exists() else []

    return result


# ── Step 2: Parse odds CSVs ──────────────────────────────────────────
def parse_euro_csv(path: Path) -> list[dict]:
    """Parse European-format CSV. Returns list of match dicts."""
    matches = []
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        date_str = row.get("Date", "").strip()
        if not date_str:
            continue
        # Date format: DD/MM/YYYY or DD/MM/YY
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        else:
            continue

        home_goals = row.get("FTHG", "").strip()
        away_goals = row.get("FTAG", "").strip()
        if not home_goals or not away_goals:
            continue

        match = {
            "date": date,
            "home": row.get("HomeTeam", "").strip(),
            "away": row.get("AwayTeam", "").strip(),
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "result": row.get("FTR", "").strip(),
            # Pinnacle closing odds
            "psch": _float(row.get("PSCH", "")),
            "pscd": _float(row.get("PSCD", "")),
            "psca": _float(row.get("PSCA", "")),
        }
        matches.append(match)
    return matches


def parse_americas_csv(path: Path, start_year: int = 2019) -> list[dict]:
    """Parse Americas-format CSV (/new/ format). Filter to seasons >= start_year."""
    matches = []
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        date_str = row.get("Date", "").strip()
        if not date_str:
            continue
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                date = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        else:
            continue

        if date.year < start_year:
            continue

        hg = row.get("HG", "").strip()
        ag = row.get("AG", "").strip()
        if not hg or not ag:
            continue

        match = {
            "date": date,
            "home": row.get("Home", "").strip(),
            "away": row.get("Away", "").strip(),
            "home_goals": int(hg),
            "away_goals": int(ag),
            "result": row.get("Res", "").strip(),
            "psch": _float(row.get("PSCH", "")),
            "pscd": _float(row.get("PSCD", "")),
            "psca": _float(row.get("PSCA", "")),
        }
        matches.append(match)
    return matches


def _float(s: str) -> float | None:
    try:
        v = float(s.strip())
        return v if v > 1.0 else None
    except (ValueError, AttributeError):
        return None


# ── Step 3: Parse commentary JSONs ────────────────────────────────────
def load_commentaries(league_id: str) -> list[dict]:
    """Load all commentary matches for a league."""
    league_dir = COMMENTARIES_DIR / league_id
    if not league_dir.exists():
        return []

    matches = []
    for f in sorted(league_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for m in data:
            date_str = m.get("@date", "")
            if not date_str:
                continue
            try:
                date = datetime.strptime(date_str, "%d.%m.%Y").date()
            except ValueError:
                continue

            lt = m.get("localteam", {})
            vt = m.get("visitorteam", {})
            ft_h = lt.get("@ft_score", "")
            ft_a = vt.get("@ft_score", "")
            if not ft_h or not ft_a:
                continue

            matches.append({
                "date": date,
                "home": lt.get("@name", ""),
                "away": vt.get("@name", ""),
                "home_goals": int(ft_h),
                "away_goals": int(ft_a),
                "match_id": m.get("@id", ""),
            })
    return matches


# ── Step 4: Match odds to commentaries ────────────────────────────────
def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_odds_to_commentaries(
    odds: list[dict], comms: list[dict]
) -> list[dict]:
    """Match odds rows to commentary rows by date(±1d) + score, verify team names."""
    # Build lookup: (date, home_goals, away_goals) → [commentary matches]
    comm_index: dict[tuple, list[dict]] = {}
    for c in comms:
        for delta in range(-1, 2):
            key = (c["date"] + timedelta(days=delta), c["home_goals"], c["away_goals"])
            comm_index.setdefault(key, []).append(c)

    matched = []
    for o in odds:
        key = (o["date"], o["home_goals"], o["away_goals"])
        candidates = comm_index.get(key, [])
        if not candidates:
            continue

        # Find best team name match
        best_score = 0.0
        best_comm = None
        for c in candidates:
            score = (fuzzy_ratio(o["home"], c["home"]) + fuzzy_ratio(o["away"], c["away"])) / 2
            if score > best_score:
                best_score = score
                best_comm = c

        if best_comm and best_score > 0.60:
            matched.append({**o, "comm": best_comm, "fuzzy_score": best_score})

    return matched


# ── Step 5: Pinnacle Brier score ──────────────────────────────────────
def pinnacle_implied_probs(psch: float, pscd: float, psca: float) -> tuple[float, float, float]:
    """Convert Pinnacle closing odds to vig-removed implied probabilities."""
    raw_h = 1.0 / psch
    raw_d = 1.0 / pscd
    raw_a = 1.0 / psca
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def brier_score_1x2(p_h: float, p_d: float, p_a: float, result: str) -> float:
    """Brier score for a single 1x2 prediction. result in {H, D, A}."""
    y_h = 1.0 if result == "H" else 0.0
    y_d = 1.0 if result == "D" else 0.0
    y_a = 1.0 if result == "A" else 0.0
    return (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2


def compute_pinnacle_bs(matched: list[dict]) -> tuple[float, int]:
    """Compute mean Brier score from Pinnacle closing odds. Returns (bs, n_valid)."""
    scores = []
    for m in matched:
        if m["psch"] and m["pscd"] and m["psca"]:
            p_h, p_d, p_a = pinnacle_implied_probs(m["psch"], m["pscd"], m["psca"])
            result = m["result"]
            if result not in ("H", "D", "A"):
                continue
            scores.append(brier_score_1x2(p_h, p_d, p_a, result))
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("football-data.co.uk Odds Downloader + Pinnacle Benchmark")
    print("=" * 70)

    # Step 1: Download
    print("\n[1/4] Downloading CSVs...")
    download_all()

    # Step 2+3: Parse and match per league
    print("\n[2/4] Parsing odds + commentaries and matching...\n")

    all_leagues = {**EURO_LEAGUES, **AMERICAS_LEAGUES}
    summary_rows = []
    all_matched_with_pinnacle = []

    for code, info in all_leagues.items():
        league_name = info["name"]
        gs_id = info["goalserve_id"]
        is_euro = code in EURO_LEAGUES

        # Parse odds
        odds: list[dict] = []
        if is_euro:
            for season in EURO_SEASONS:
                path = ODDS_DIR / f"{code}_{season}.csv"
                if path.exists():
                    odds.extend(parse_euro_csv(path))
        else:
            path = ODDS_DIR / f"{code}.csv"
            if path.exists():
                odds.extend(parse_americas_csv(path))

        # Parse commentaries
        comms = load_commentaries(gs_id)

        # Match
        matched = match_odds_to_commentaries(odds, comms)

        # Count Pinnacle available
        n_pinnacle = sum(1 for m in matched if m["psch"] and m["pscd"] and m["psca"])

        summary_rows.append({
            "league": league_name,
            "n_odds": len(odds),
            "n_comms": len(comms),
            "n_matched": len(matched),
            "n_pinnacle": n_pinnacle,
        })

        all_matched_with_pinnacle.extend(
            [m for m in matched if m["psch"] and m["pscd"] and m["psca"]]
        )

    # Step 3: Print summary table
    print("\n[3/4] Summary:")
    print(f"{'League':<15} {'Odds':>7} {'Comms':>7} {'Matched':>8} {'Pinnacle':>9}")
    print("-" * 50)
    for r in summary_rows:
        print(
            f"{r['league']:<15} {r['n_odds']:>7} {r['n_comms']:>7} "
            f"{r['n_matched']:>8} {r['n_pinnacle']:>9}"
        )
    totals = {k: sum(r[k] for r in summary_rows) for k in ["n_odds", "n_comms", "n_matched", "n_pinnacle"]}
    print("-" * 50)
    print(
        f"{'TOTAL':<15} {totals['n_odds']:>7} {totals['n_comms']:>7} "
        f"{totals['n_matched']:>8} {totals['n_pinnacle']:>9}"
    )

    # Step 4: Brier score benchmark
    print("\n[4/4] Pinnacle Brier Score Benchmark:")

    # Load model BS from production params
    model_bs = None
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text())
        model_bs = params.get("cv_brier_score")

    print(f"\n  Model CV Brier Score (from production_params): {model_bs:.6f}" if model_bs else "")

    # Per-league Pinnacle BS
    print(f"\n  {'League':<15} {'Pinnacle BS':>12} {'N':>6} {'ΔBS':>10}")
    print("  " + "-" * 47)

    for code, info in all_leagues.items():
        league_name = info["name"]
        gs_id = info["goalserve_id"]
        is_euro = code in EURO_LEAGUES

        odds = []
        if is_euro:
            for season in EURO_SEASONS:
                path = ODDS_DIR / f"{code}_{season}.csv"
                if path.exists():
                    odds.extend(parse_euro_csv(path))
        else:
            path = ODDS_DIR / f"{code}.csv"
            if path.exists():
                odds.extend(parse_americas_csv(path))

        comms = load_commentaries(gs_id)
        matched = match_odds_to_commentaries(odds, comms)
        bs, n = compute_pinnacle_bs(matched)

        if n > 0 and model_bs:
            delta = model_bs - bs
            sign = "" if delta < 0 else "+"
            print(f"  {league_name:<15} {bs:>12.6f} {n:>6} {sign}{delta:>9.6f}")
        elif n > 0:
            print(f"  {league_name:<15} {bs:>12.6f} {n:>6} {'N/A':>10}")
        else:
            print(f"  {league_name:<15} {'N/A':>12} {0:>6} {'N/A':>10}")

    # Overall Pinnacle BS
    overall_bs, overall_n = compute_pinnacle_bs(all_matched_with_pinnacle)
    if overall_n > 0:
        print("  " + "-" * 47)
        delta_str = ""
        if model_bs:
            delta = model_bs - overall_bs
            sign = "" if delta < 0 else "+"
            delta_str = f"{sign}{delta:.6f}"
        print(f"  {'OVERALL':<15} {overall_bs:>12.6f} {overall_n:>6} {delta_str:>10}")

    print()
    if model_bs and overall_n > 0:
        delta = model_bs - overall_bs
        if delta < 0:
            print(f"  → Model BEATS Pinnacle by {abs(delta):.6f} BS")
        else:
            print(f"  → Pinnacle beats model by {delta:.6f} BS")
    print()


if __name__ == "__main__":
    main()
