#!/usr/bin/env python3
"""Download football-data.co.uk odds CSVs and benchmark against Pinnacle closing lines.

Apples-to-apples comparison:
  - Model: poisson_1x2(mu_H, mu_A) where mu = MLE from actual goals
  - Pinnacle: vig-removed implied probabilities from closing odds
  - Same Brier Score formula on the same N matches
"""

import csv
import io
import json
import math
import os
import sys
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from scipy.stats import poisson as poisson_dist

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


# ── Step 5: Model + Pinnacle predictions ─────────────────────────────

def pinnacle_implied_probs(psch: float, pscd: float, psca: float) -> tuple[float, float, float]:
    """Convert Pinnacle closing odds to vig-removed implied probabilities."""
    raw_h = 1.0 / psch
    raw_d = 1.0 / pscd
    raw_a = 1.0 / psca
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def poisson_1x2(mu_H: float, mu_A: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Compute 1x2 probabilities from independent Poisson(mu_H), Poisson(mu_A)."""
    p_h = 0.0
    p_d = 0.0
    p_a = 0.0
    for i in range(max_goals + 1):
        p_i = poisson_dist.pmf(i, mu_H)
        for j in range(max_goals + 1):
            p_j = poisson_dist.pmf(j, mu_A)
            p_ij = p_i * p_j
            if i > j:
                p_h += p_ij
            elif i == j:
                p_d += p_ij
            else:
                p_a += p_ij
    total = p_h + p_d + p_a
    if total > 0:
        return p_h / total, p_d / total, p_a / total
    return 1.0 / 3, 1.0 / 3, 1.0 / 3


def brier_score_1x2(p_h: float, p_d: float, p_a: float, home_goals: int, away_goals: int) -> float:
    """Multi-class Brier score for a single 1x2 prediction."""
    y_h = 1.0 if home_goals > away_goals else 0.0
    y_d = 1.0 if home_goals == away_goals else 0.0
    y_a = 1.0 if home_goals < away_goals else 0.0
    return (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2


def compute_benchmark(matched: list[dict], b: list[float]) -> tuple[float, float, int]:
    """Apples-to-apples Model vs Pinnacle BS on same matches.

    Model: poisson_1x2(mu_H, mu_A) where mu from MLE (actual goals)
           with time-profile correction via C_time.
    Pinnacle: vig-removed implied probs from closing odds.

    Returns: (model_bs, pinnacle_bs, n_valid)
    """
    # Compute C_time = Σ exp(b_i) * Δt_i
    import numpy as np
    dt = np.array([15.0, 15.0, 17.0, 15.0, 15.0, 19.0])  # α₁=2, α₂=4
    b_arr = np.array(b[:6])
    C_time = float(np.sum(np.exp(b_arr) * dt))
    T_m = float(np.sum(dt))  # 96 minutes

    model_scores = []
    pinnacle_scores = []

    for m in matched:
        if not (m["psch"] and m["pscd"] and m["psca"]):
            continue
        hg = m["home_goals"]
        ag = m["away_goals"]

        # Model: MLE intensity → expected goals with time-profile correction
        mu_H_mle = max(hg, 0.1)
        mu_A_mle = max(ag, 0.1)
        # a = ln(mu_mle / T_m), then mu_model = exp(a) * C_time = mu_mle * C_time / T_m
        mu_H = mu_H_mle * C_time / T_m
        mu_A = mu_A_mle * C_time / T_m

        p_model = poisson_1x2(mu_H, mu_A)
        model_scores.append(brier_score_1x2(*p_model, hg, ag))

        # Pinnacle
        p_pin = pinnacle_implied_probs(m["psch"], m["pscd"], m["psca"])
        pinnacle_scores.append(brier_score_1x2(*p_pin, hg, ag))

    n = len(model_scores)
    if n == 0:
        return 0.0, 0.0, 0
    return sum(model_scores) / n, sum(pinnacle_scores) / n, n


# ── Step 6: Per-league matching helper ────────────────────────────────

def get_matched_for_league(code: str, info: dict) -> list[dict]:
    """Parse odds + commentaries for one league and return matched list."""
    gs_id = info["goalserve_id"]
    is_euro = code in EURO_LEAGUES

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

    comms = load_commentaries(gs_id)
    return match_odds_to_commentaries(odds, comms), len(odds), len(comms)


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 78)
    print("football-data.co.uk Odds — Apples-to-Apples Model vs Pinnacle Benchmark")
    print("=" * 78)

    # Step 1: Download
    print("\n[1/4] Downloading CSVs...")
    download_all()

    # Step 2: Parse + match
    print("\n[2/4] Parsing odds + commentaries and matching...")

    all_leagues = {**EURO_LEAGUES, **AMERICAS_LEAGUES}
    league_matched: dict[str, list[dict]] = {}
    summary_rows = []

    for code, info in all_leagues.items():
        matched, n_odds, n_comms = get_matched_for_league(code, info)
        league_name = info["name"]
        n_pinnacle = sum(1 for m in matched if m["psch"] and m["pscd"] and m["psca"])
        league_matched[league_name] = matched

        summary_rows.append({
            "league": league_name,
            "n_odds": n_odds,
            "n_comms": n_comms,
            "n_matched": len(matched),
            "n_pinnacle": n_pinnacle,
        })

    # Step 3: Summary table
    print(f"\n[3/4] Match Summary:")
    print(f"  {'League':<15} {'Odds':>7} {'Comms':>7} {'Matched':>8} {'Pinnacle':>9}")
    print("  " + "-" * 50)
    for r in summary_rows:
        print(
            f"  {r['league']:<15} {r['n_odds']:>7} {r['n_comms']:>7} "
            f"{r['n_matched']:>8} {r['n_pinnacle']:>9}"
        )
    totals = {k: sum(r[k] for r in summary_rows) for k in ["n_odds", "n_comms", "n_matched", "n_pinnacle"]}
    print("  " + "-" * 50)
    print(
        f"  {'TOTAL':<15} {totals['n_odds']:>7} {totals['n_comms']:>7} "
        f"{totals['n_matched']:>8} {totals['n_pinnacle']:>9}"
    )

    # Step 4: Apples-to-apples Brier Score benchmark
    print(f"\n[4/4] Apples-to-Apples Brier Score Benchmark")
    print("=" * 78)

    # Load trained parameters
    if not PARAMS_PATH.exists():
        print("  ERROR: production_params not found at", PARAMS_PATH)
        return
    params = json.loads(PARAMS_PATH.read_text())
    b = params["params"]["b"]
    print(f"  Loaded b (time profile): {[round(x, 3) for x in b]}")

    import numpy as np
    dt = np.array([15.0, 15.0, 17.0, 15.0, 15.0, 19.0])
    C_time = float(np.sum(np.exp(np.array(b[:6])) * dt))
    T_m = float(np.sum(dt))
    print(f"  C_time = {C_time:.2f},  T_m = {T_m:.0f},  C_time/T_m = {C_time/T_m:.4f}")
    print(f"  → MLE goals are scaled by {C_time/T_m:.4f}x before Poisson 1x2")
    print()

    # Per-league benchmark
    print(f"  {'League':<15} {'Model BS':>10} {'Pinnacle BS':>12} {'ΔBS':>10} {'N':>6}  {'Winner':>10}")
    print("  " + "-" * 68)

    all_matched = []
    for code, info in all_leagues.items():
        league_name = info["name"]
        matched = league_matched[league_name]
        m_bs, p_bs, n = compute_benchmark(matched, b)

        if n > 0:
            delta = m_bs - p_bs
            winner = "Model" if delta < 0 else "Pinnacle"
            sign = "" if delta < 0 else "+"
            print(f"  {league_name:<15} {m_bs:>10.6f} {p_bs:>12.6f} {sign}{delta:>9.6f} {n:>6}  {winner:>10}")
            all_matched.extend([m for m in matched if m["psch"] and m["pscd"] and m["psca"]])
        else:
            print(f"  {league_name:<15} {'N/A':>10} {'N/A':>12} {'N/A':>10} {0:>6}")

    # Overall
    m_bs_all, p_bs_all, n_all = compute_benchmark(all_matched, b)
    if n_all > 0:
        delta_all = m_bs_all - p_bs_all
        sign = "" if delta_all < 0 else "+"
        winner = "Model" if delta_all < 0 else "Pinnacle"
        print("  " + "-" * 68)
        print(f"  {'OVERALL':<15} {m_bs_all:>10.6f} {p_bs_all:>12.6f} {sign}{delta_all:>9.6f} {n_all:>6}  {winner:>10}")

        print(f"\n  Model BS:    {m_bs_all:.6f}  (Poisson MLE, same formula as retrain CV)")
        print(f"  Pinnacle BS: {p_bs_all:.6f}  (vig-removed closing odds)")
        print(f"  ΔBS:         {sign}{delta_all:.6f}")
        print()
        if delta_all < 0:
            print(f"  → Model beats Pinnacle by {abs(delta_all):.6f} BS")
        elif delta_all > 0:
            print(f"  → Pinnacle beats model by {delta_all:.6f} BS")
        else:
            print(f"  → Exact tie")

        # Context
        print(f"\n  Note: Model uses actual match goals as MLE input (hindsight).")
        print(f"  Pinnacle odds are genuine pre-match predictions.")
        print(f"  This benchmark validates the Poisson 1x2 formula, not predictive edge.")
    print()


if __name__ == "__main__":
    main()
