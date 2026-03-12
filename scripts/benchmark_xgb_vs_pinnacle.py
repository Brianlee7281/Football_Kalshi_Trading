#!/usr/bin/env python3
"""Benchmark: XGBoost pre-match predictions vs Pinnacle closing lines.

Proper train/test evaluation:
  - Train: 2019-2023 seasons
  - Test: 2023-2024 and 2024-2025 seasons
  - Features: team rolling stats (last 5 matches) + Pinnacle OPENING odds
  - Target: goals scored (Poisson regression)
  - Comparison: Model BS vs Pinnacle Closing BS on test set

Usage:
    python scripts/benchmark_xgb_vs_pinnacle.py
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.stats import poisson as poisson_dist

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ODDS_DIR = ROOT / "data" / "odds_historical"
COMMENTARIES_DIR = ROOT / "data" / "commentaries"
PARAMS_PATH = ROOT / "data" / "production_params" / "phase1_full_retrain.json"

# ── League config ─────────────────────────────────────────────────────
EURO_LEAGUES = {
    "E0":  {"name": "EPL",        "gs_id": "1204"},
    "SP1": {"name": "La Liga",    "gs_id": "1399"},
    "I1":  {"name": "Serie A",    "gs_id": "1269"},
    "D1":  {"name": "Bundesliga", "gs_id": "1229"},
    "F1":  {"name": "Ligue 1",    "gs_id": "1221"},
}
AMERICAS_LEAGUES = {
    "USA": {"name": "MLS",         "gs_id": "1440"},
    "BRA": {"name": "Brasileirão", "gs_id": "1141"},
    "ARG": {"name": "Argentina",   "gs_id": "1081"},
}
EURO_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]

# Goalserve ID → football-data code
GS_TO_FD = {v["gs_id"]: k for k, v in {**EURO_LEAGUES, **AMERICAS_LEAGUES}.items()}
GS_TO_NAME = {v["gs_id"]: v["name"] for v in {**EURO_LEAGUES, **AMERICAS_LEAGUES}.values()}

ROLLING_WINDOW = 5
TRAIN_CUTOFF = datetime(2023, 7, 1).date()  # train < this, test >= this

# ── Feature columns ──────────────────────────────────────────────────
ROLLING_FEATURES = [
    "r_xg", "r_shots", "r_shots_on_target", "r_possession",
    "r_corners", "r_fouls", "r_saves", "r_passes",
    "r_pass_accuracy", "r_insidebox_ratio",
    "r_goals_scored", "r_goals_conceded",
    "r_avg_rating",
]
ODDS_FEATURES = [
    "pin_open_home_prob", "pin_open_draw_prob", "pin_open_away_prob",
]
CONTEXT_FEATURES = ["home_away_flag"]
ALL_FEATURES = ROLLING_FEATURES + ODDS_FEATURES + CONTEXT_FEATURES


# ── Stats extraction from commentary ─────────────────────────────────

def _sf(val: Any) -> float:
    """Safe float."""
    if val is None or val == "":
        return 0.0
    try:
        s = str(val).replace("%", "")
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def extract_team_stats(match: dict, team_key: str) -> dict[str, float]:
    """Extract numeric stats for one team from a commentary match."""
    stats_root = match.get("stats")
    if not stats_root or not isinstance(stats_root, dict):
        return {}
    stats = stats_root.get(team_key, {})
    if not stats:
        return {}

    shots_node = stats.get("shots", {})
    poss_node = stats.get("possestiontime", {})
    corners_node = stats.get("corners", {})
    fouls_node = stats.get("fouls", {})
    saves_node = stats.get("saves", {})
    passes_node = stats.get("passes", {})
    xg_node = stats.get("expected_goals", {})

    shots_total = _sf(shots_node.get("@total", 0)) if isinstance(shots_node, dict) else _sf(shots_node)
    shots_on = _sf(shots_node.get("@ongoal", 0)) if isinstance(shots_node, dict) else 0.0
    insidebox = _sf(shots_node.get("@insidebox", 0)) if isinstance(shots_node, dict) else 0.0
    possession = _sf(poss_node.get("@total", 0)) if isinstance(poss_node, dict) else _sf(poss_node)
    corners = _sf(corners_node.get("@total", 0)) if isinstance(corners_node, dict) else _sf(corners_node)
    fouls = _sf(fouls_node.get("@total", 0)) if isinstance(fouls_node, dict) else _sf(fouls_node)
    saves = _sf(saves_node.get("@total", 0)) if isinstance(saves_node, dict) else _sf(saves_node)
    passes_total = _sf(passes_node.get("@total", 0)) if isinstance(passes_node, dict) else _sf(passes_node)
    passes_acc = _sf(passes_node.get("@accurate", 0)) if isinstance(passes_node, dict) else 0.0
    xg = _sf(xg_node.get("@total", 0)) if isinstance(xg_node, dict) else _sf(xg_node)

    pass_accuracy = (passes_acc / passes_total * 100.0) if passes_total > 0 else 0.0
    ib_ratio = (insidebox / shots_total) if shots_total > 0 else 0.0

    # Player avg rating
    ps_root = match.get("player_stats")
    ps = ps_root.get(team_key, {}) if isinstance(ps_root, dict) else {}
    players = ps.get("player", []) if ps else []
    if isinstance(players, dict):
        players = [players]
    ratings = [_sf(p.get("@rating", p.get("rating", 0))) for p in players]
    ratings = [r for r in ratings if r > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

    return {
        "xg": xg,
        "shots": shots_total,
        "shots_on_target": shots_on,
        "possession": possession,
        "corners": corners,
        "fouls": fouls,
        "saves": saves,
        "passes": passes_total,
        "pass_accuracy": pass_accuracy,
        "insidebox_ratio": ib_ratio,
        "avg_rating": avg_rating,
    }


# ── Load all commentary matches ──────────────────────────────────────

def load_all_commentaries() -> list[dict]:
    """Load all matches from all leagues, enriched with league_id and parsed date."""
    all_matches = []
    for league_dir in sorted(COMMENTARIES_DIR.iterdir()):
        if not league_dir.is_dir():
            continue
        league_id = league_dir.name
        for season_file in sorted(league_dir.glob("*.json")):
            try:
                data = json.loads(season_file.read_text())
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
                try:
                    hg, ag = int(ft_h), int(ft_a)
                except ValueError:
                    continue

                all_matches.append({
                    "league_id": league_id,
                    "date": date,
                    "home": lt.get("@name", ""),
                    "away": vt.get("@name", ""),
                    "home_goals": hg,
                    "away_goals": ag,
                    "match_id": m.get("@id", ""),
                    "_raw": m,  # keep raw for stats extraction
                })
    return all_matches


# ── Load odds CSVs ────────────────────────────────────────────────────

def _float_odds(s: str) -> float | None:
    try:
        v = float(s.strip())
        return v if v > 1.0 else None
    except (ValueError, AttributeError):
        return None


def load_euro_odds() -> list[dict]:
    """Load European odds CSVs with both opening and closing Pinnacle odds."""
    all_odds = []
    for code, info in EURO_LEAGUES.items():
        for season in EURO_SEASONS:
            path = ODDS_DIR / f"{code}_{season}.csv"
            if not path.exists():
                continue
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
                hg = row.get("FTHG", "").strip()
                ag = row.get("FTAG", "").strip()
                if not hg or not ag:
                    continue
                all_odds.append({
                    "league_code": code,
                    "gs_id": info["gs_id"],
                    "date": date,
                    "home": row.get("HomeTeam", "").strip(),
                    "away": row.get("AwayTeam", "").strip(),
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "result": row.get("FTR", "").strip(),
                    # Opening
                    "psh": _float_odds(row.get("PSH", "")),
                    "psd": _float_odds(row.get("PSD", "")),
                    "psa": _float_odds(row.get("PSA", "")),
                    # Closing
                    "psch": _float_odds(row.get("PSCH", "")),
                    "pscd": _float_odds(row.get("PSCD", "")),
                    "psca": _float_odds(row.get("PSCA", "")),
                })
    return all_odds


def load_americas_odds() -> list[dict]:
    """Load Americas odds CSVs (closing only)."""
    all_odds = []
    for code, info in AMERICAS_LEAGUES.items():
        path = ODDS_DIR / f"{code}.csv"
        if not path.exists():
            continue
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
            if date.year < 2019:
                continue
            hg = row.get("HG", "").strip()
            ag = row.get("AG", "").strip()
            if not hg or not ag:
                continue
            all_odds.append({
                "league_code": code,
                "gs_id": info["gs_id"],
                "date": date,
                "home": row.get("Home", "").strip(),
                "away": row.get("Away", "").strip(),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "result": row.get("Res", "").strip(),
                "psh": None, "psd": None, "psa": None,  # no opening
                "psch": _float_odds(row.get("PSCH", "")),
                "pscd": _float_odds(row.get("PSCD", "")),
                "psca": _float_odds(row.get("PSCA", "")),
            })
    return all_odds


# ── Match odds to commentaries ────────────────────────────────────────

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_odds_to_commentaries(
    matches: list[dict], odds: list[dict],
) -> list[dict]:
    """Merge odds data into commentary matches by date+score+fuzzy team name."""
    # Build odds index: (gs_id, date, hg, ag) → [odds rows]
    odds_idx: dict[tuple, list[dict]] = defaultdict(list)
    for o in odds:
        for delta in range(-1, 2):
            key = (o["gs_id"], o["date"] + timedelta(days=delta), o["home_goals"], o["away_goals"])
            odds_idx[key].append(o)

    merged = 0
    for m in matches:
        key = (m["league_id"], m["date"], m["home_goals"], m["away_goals"])
        candidates = odds_idx.get(key, [])
        if not candidates:
            m["odds"] = None
            continue
        best_score = 0.0
        best_odds = None
        for o in candidates:
            score = (fuzzy_ratio(m["home"], o["home"]) + fuzzy_ratio(m["away"], o["away"])) / 2
            if score > best_score:
                best_score = score
                best_odds = o
        if best_odds and best_score > 0.60:
            m["odds"] = best_odds
            merged += 1
        else:
            m["odds"] = None

    print(f"  Merged odds for {merged}/{len(matches)} matches")
    return matches


# ── Rolling stats computation ─────────────────────────────────────────

def compute_rolling_features(matches: list[dict]) -> None:
    """Add rolling team stats to each match (in-place). Must be sorted by date."""
    # team_key → list of (date, stats_dict, goals_scored, goals_conceded)
    team_history: dict[str, list[tuple]] = defaultdict(list)

    for m in matches:
        league_id = m["league_id"]
        home = m["home"]
        away = m["away"]
        hk = f"{league_id}_{home}"
        ak = f"{league_id}_{away}"

        # Compute rolling for home team
        m["home_rolling"] = _rolling_avg(team_history[hk])
        m["away_rolling"] = _rolling_avg(team_history[ak])

        # Extract stats and add to history
        raw = m["_raw"]
        h_stats = extract_team_stats(raw, "localteam")
        a_stats = extract_team_stats(raw, "visitorteam")

        if h_stats:
            h_stats["goals_scored"] = float(m["home_goals"])
            h_stats["goals_conceded"] = float(m["away_goals"])
            team_history[hk].append((m["date"], h_stats))
            if len(team_history[hk]) > ROLLING_WINDOW + 1:
                team_history[hk] = team_history[hk][-(ROLLING_WINDOW + 1):]

        if a_stats:
            a_stats["goals_scored"] = float(m["away_goals"])
            a_stats["goals_conceded"] = float(m["home_goals"])
            team_history[ak].append((m["date"], a_stats))
            if len(team_history[ak]) > ROLLING_WINDOW + 1:
                team_history[ak] = team_history[ak][-(ROLLING_WINDOW + 1):]


def _rolling_avg(history: list[tuple]) -> dict[str, float] | None:
    """Compute average of last ROLLING_WINDOW entries."""
    recent = history[-ROLLING_WINDOW:]
    if len(recent) < 3:  # need at least 3 matches for meaningful rolling
        return None
    n = len(recent)
    keys = [
        "xg", "shots", "shots_on_target", "possession", "corners",
        "fouls", "saves", "passes", "pass_accuracy", "insidebox_ratio",
        "goals_scored", "goals_conceded", "avg_rating",
    ]
    result = {}
    for k in keys:
        vals = [entry[1].get(k, 0.0) for entry in recent]
        result[f"r_{k}"] = sum(vals) / n
    return result


# ── Feature vector construction ───────────────────────────────────────

def odds_to_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Convert decimal odds to vig-removed probabilities."""
    raw_h, raw_d, raw_a = 1.0 / h, 1.0 / d, 1.0 / a
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def build_feature_row(
    rolling: dict[str, float] | None,
    odds: dict | None,
    is_home: bool,
    use_opening: bool = True,
) -> list[float] | None:
    """Build feature vector for one team-match observation."""
    if rolling is None:
        return None

    row = []
    # Rolling features
    for f in ROLLING_FEATURES:
        row.append(rolling.get(f, 0.0))

    # Odds features
    if odds and use_opening and odds.get("psh") and odds.get("psd") and odds.get("psa"):
        ph, pd, pa = odds_to_probs(odds["psh"], odds["psd"], odds["psa"])
        row.extend([ph, pd, pa])
    else:
        row.extend([float("nan"), float("nan"), float("nan")])

    # Context
    row.append(1.0 if is_home else 0.0)

    return row


# ── Poisson 1x2 ──────────────────────────────────────────────────────

def poisson_1x2(mu_H: float, mu_A: float, max_goals: int = 10) -> tuple[float, float, float]:
    p_h, p_d, p_a = 0.0, 0.0, 0.0
    for i in range(max_goals + 1):
        pi = poisson_dist.pmf(i, mu_H)
        for j in range(max_goals + 1):
            pj = poisson_dist.pmf(j, mu_A)
            pij = pi * pj
            if i > j:
                p_h += pij
            elif i == j:
                p_d += pij
            else:
                p_a += pij
    total = p_h + p_d + p_a
    if total > 0:
        return p_h / total, p_d / total, p_a / total
    return 1.0 / 3, 1.0 / 3, 1.0 / 3


def brier_1x2(p_h: float, p_d: float, p_a: float, hg: int, ag: int) -> float:
    y_h = 1.0 if hg > ag else 0.0
    y_d = 1.0 if hg == ag else 0.0
    y_a = 1.0 if hg < ag else 0.0
    return (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("XGBoost Pre-Match Predictions vs Pinnacle Closing Lines")
    print("=" * 78)

    # 1. Load data
    print("\n[1/5] Loading commentary data...")
    matches = load_all_commentaries()
    matches.sort(key=lambda m: m["date"])
    print(f"  Loaded {len(matches)} matches from {len(set(m['league_id'] for m in matches))} leagues")

    print("\n[2/5] Loading and merging odds...")
    odds = load_euro_odds() + load_americas_odds()
    print(f"  Loaded {len(odds)} odds rows")
    merge_odds_to_commentaries(matches, odds)

    # 2. Compute rolling features
    print("\n[3/5] Computing rolling team stats (window={})...".format(ROLLING_WINDOW))
    compute_rolling_features(matches)

    # 3. Build feature matrix
    print("\n[4/5] Building feature matrix...")

    # Each match → 2 rows (home team, away team)
    train_X, train_y = [], []
    test_matches_data = []  # (match, home_features, away_features)

    n_train = 0
    n_test = 0
    n_skip_rolling = 0
    n_skip_odds = 0

    for m in matches:
        is_test = m["date"] >= TRAIN_CUTOFF
        odds_data = m.get("odds")

        home_row = build_feature_row(m["home_rolling"], odds_data, is_home=True)
        away_row = build_feature_row(m["away_rolling"], odds_data, is_home=False)

        if home_row is None or away_row is None:
            n_skip_rolling += 1
            continue

        if is_test:
            # For test: need Pinnacle closing odds to compare against
            if not (odds_data and odds_data.get("psch") and odds_data.get("pscd") and odds_data.get("psca")):
                n_skip_odds += 1
                continue
            test_matches_data.append((m, home_row, away_row))
            n_test += 1
        else:
            train_X.append(home_row)
            train_y.append(m["home_goals"])
            train_X.append(away_row)
            train_y.append(m["away_goals"])
            n_train += 1

    train_X_arr = np.array(train_X, dtype=np.float64)
    train_y_arr = np.array(train_y, dtype=np.float64)

    print(f"  Train: {n_train} matches ({len(train_X)} rows)")
    print(f"  Test:  {n_test} matches (with Pinnacle closing odds)")
    print(f"  Skipped (no rolling): {n_skip_rolling}")
    print(f"  Skipped (no closing odds): {n_skip_odds}")

    # 4. Train XGBoost
    print("\n[5/5] Training XGBoost (count:poisson)...")

    xgb_params = {
        "objective": "count:poisson",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "eval_metric": "poisson-nloglik",
        "seed": 42,
    }
    dtrain = xgb.DMatrix(train_X_arr, label=train_y_arr, feature_names=ALL_FEATURES)
    model = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=False)

    # Feature importance
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature importance (top 10):")
    for name, score in sorted_imp[:10]:
        print(f"    {name:<28s} {score:.1f}")

    # 5. Evaluate on test set
    print("\n" + "=" * 78)
    print("TEST SET RESULTS")
    print("=" * 78)

    # Per-league accumulators
    league_model_bs: dict[str, list[float]] = defaultdict(list)
    league_pinnacle_bs: dict[str, list[float]] = defaultdict(list)

    for m, home_row, away_row in test_matches_data:
        odds_data = m["odds"]
        hg, ag = m["home_goals"], m["away_goals"]
        league_id = m["league_id"]
        league_name = GS_TO_NAME.get(league_id, league_id)

        # Model prediction
        X_home = np.array([home_row], dtype=np.float64)
        X_away = np.array([away_row], dtype=np.float64)
        mu_H = float(model.predict(xgb.DMatrix(X_home, feature_names=ALL_FEATURES))[0])
        mu_A = float(model.predict(xgb.DMatrix(X_away, feature_names=ALL_FEATURES))[0])

        # Clamp to avoid degenerate Poisson
        mu_H = max(mu_H, 0.05)
        mu_A = max(mu_A, 0.05)

        p_model = poisson_1x2(mu_H, mu_A)
        model_bs = brier_1x2(*p_model, hg, ag)

        # Pinnacle closing
        p_pin = odds_to_probs(odds_data["psch"], odds_data["pscd"], odds_data["psca"])
        pin_bs = brier_1x2(*p_pin, hg, ag)

        league_model_bs[league_name].append(model_bs)
        league_pinnacle_bs[league_name].append(pin_bs)

    # Print results
    print(f"\n  {'League':<15} {'Model BS':>10} {'Pinnacle BS':>12} {'ΔBS':>10} {'N':>6}  {'Winner':>10}")
    print("  " + "-" * 68)

    all_model, all_pin = [], []
    for league_name in [v["name"] for v in {**EURO_LEAGUES, **AMERICAS_LEAGUES}.values()]:
        m_scores = league_model_bs.get(league_name, [])
        p_scores = league_pinnacle_bs.get(league_name, [])
        if not m_scores:
            print(f"  {league_name:<15} {'N/A':>10} {'N/A':>12} {'N/A':>10} {0:>6}")
            continue
        m_bs = sum(m_scores) / len(m_scores)
        p_bs = sum(p_scores) / len(p_scores)
        delta = m_bs - p_bs
        n = len(m_scores)
        winner = "Model" if delta < 0 else "Pinnacle"
        sign = "" if delta < 0 else "+"
        print(f"  {league_name:<15} {m_bs:>10.6f} {p_bs:>12.6f} {sign}{delta:>9.6f} {n:>6}  {winner:>10}")
        all_model.extend(m_scores)
        all_pin.extend(p_scores)

    if all_model:
        m_bs_all = sum(all_model) / len(all_model)
        p_bs_all = sum(all_pin) / len(all_pin)
        delta_all = m_bs_all - p_bs_all
        sign = "" if delta_all < 0 else "+"
        winner = "Model" if delta_all < 0 else "Pinnacle"
        print("  " + "-" * 68)
        print(f"  {'OVERALL':<15} {m_bs_all:>10.6f} {p_bs_all:>12.6f} {sign}{delta_all:>9.6f} {len(all_model):>6}  {winner:>10}")

        print(f"\n  Model BS:    {m_bs_all:.6f}")
        print(f"  Pinnacle BS: {p_bs_all:.6f}")
        print(f"  ΔBS:         {sign}{delta_all:.6f}")
        print()

        random_bs = 2.0 / 3.0
        model_skill = 1.0 - m_bs_all / random_bs
        pin_skill = 1.0 - p_bs_all / random_bs
        print(f"  Skill score (1 - BS/BS_random):")
        print(f"    Model:    {model_skill:.4f}")
        print(f"    Pinnacle: {pin_skill:.4f}")
        print()

        if delta_all > 0:
            print(f"  → Pinnacle beats model by {delta_all:.6f} BS on held-out test set")
            print(f"    Model needs improvement before paper trading.")
        else:
            print(f"  → Model beats Pinnacle by {abs(delta_all):.6f} BS on held-out test set")
            print(f"    Edge detected — proceed to paper trading evaluation.")

    # Clean up _raw references to save memory
    for m in matches:
        m.pop("_raw", None)

    print()


if __name__ == "__main__":
    main()
