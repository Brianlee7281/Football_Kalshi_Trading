"""Step 2.1 — Pre-Match Context Data Collection.

Collects all pre-match data ~60 minutes before kickoff:
  2.1.1  Lineups + formation (Goalserve Live Game Stats)
  2.1.2  Player rolling stats (historical player_stats)
  2.1.3  Position-based team aggregation (G/D/M/F)
  2.1.4  Team-level rolling stats (historical stats)
  2.1.5  Odds features (Odds-API pre-match)
  2.1.6  Context features (rest days, H2H)

Reference: docs/phase2.md Step 2.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.clients.goalserve import GoalserveClient, ensure_list
from src.clients.odds_api import OddsApiClient, build_odds_features
from src.common.logging import get_logger

logger = get_logger("step_2_1")


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlayerRolling:
    """Per-player rolling stats (per-90 metrics over last 5 matches)."""

    player_id: str
    name: str
    pos: str  # G, D, M, F
    goals_per_90: float = 0.0
    shots_on_target_per_90: float = 0.0
    key_passes_per_90: float = 0.0
    pass_accuracy: float = 0.0
    dribble_success_rate: float = 0.0
    tackles_per_90: float = 0.0
    interceptions_per_90: float = 0.0
    rating_avg: float = 0.0
    save_rate: float = 0.0
    minutes_played_avg: float = 0.0


@dataclass
class PreMatchData:
    """Complete pre-match data collected in Step 2.1."""

    # Lineups
    home_starting_11: list[str] = field(default_factory=list)
    away_starting_11: list[str] = field(default_factory=list)
    home_formation: str = ""
    away_formation: str = ""

    # Tier 2: player aggregate features
    home_player_agg: dict[str, float] = field(default_factory=dict)
    away_player_agg: dict[str, float] = field(default_factory=dict)

    # Tier 1: team rolling stats
    home_team_rolling: dict[str, float] = field(default_factory=dict)
    away_team_rolling: dict[str, float] = field(default_factory=dict)

    # Tier 3: odds features
    odds_features: dict[str, float] = field(default_factory=dict)

    # Tier 4: context
    home_rest_days: int = 3
    away_rest_days: int = 3
    h2h_goal_diff: float = 0.0

    # Metadata
    match_id: str = ""
    kickoff_time: str = ""


# ---------------------------------------------------------------------------
# 2.1.1: Lineups + formation
# ---------------------------------------------------------------------------


def extract_lineups(
    match_stats: dict[str, Any],
) -> tuple[list[str], list[str], str, str]:
    """Extract starting XI player IDs and formations from Goalserve match stats.

    Args:
        match_stats: Goalserve Live Game Stats response for a match.

    Returns:
        (home_ids, away_ids, home_formation, away_formation)
    """
    home_ids: list[str] = []
    away_ids: list[str] = []
    home_formation = ""
    away_formation = ""

    teams = match_stats.get("teams", {})

    for team_key, id_list, formation_holder in [
        ("localteam", home_ids, "home"),
        ("visitorteam", away_ids, "away"),
    ]:
        team_data = teams.get(team_key, {})
        if not team_data:
            continue

        if formation_holder == "home":
            home_formation = str(team_data.get("formation", ""))
        else:
            away_formation = str(team_data.get("formation", ""))

        players = ensure_list(team_data.get("player", []))
        for p in players:
            pid = str(p.get("id", p.get("@id", "")))
            if pid:
                id_list.append(pid)

    return home_ids, away_ids, home_formation, away_formation


# ---------------------------------------------------------------------------
# 2.1.2: Player rolling stats
# ---------------------------------------------------------------------------


def safe_per90(stat_value: object, minutes: object) -> float | None:
    """Compute per-90 stat, returning None if minutes too low.

    Goalserve uses empty strings for missing values.
    """
    mp = _safe_float(minutes)
    val = _safe_float(stat_value)
    if mp < 10:
        return None
    return val / mp * 90.0


def compute_player_rolling(
    player_id: str,
    player_name: str,
    pos: str,
    match_stats_list: list[dict[str, Any]],
) -> PlayerRolling:
    """Compute rolling per-90 averages from a player's recent matches.

    Args:
        player_id: Goalserve player ID.
        player_name: Player name.
        pos: Position (G, D, M, F).
        match_stats_list: List of per-match player stat dicts from Goalserve.

    Returns:
        PlayerRolling with averaged metrics.
    """
    goals: list[float] = []
    shots_on: list[float] = []
    key_passes: list[float] = []
    pass_acc: list[float] = []
    dribble_succ: list[float] = []
    tackles: list[float] = []
    interceptions: list[float] = []
    ratings: list[float] = []
    save_rates: list[float] = []
    minutes_list: list[float] = []

    for ms in match_stats_list:
        mp = _safe_float(ms.get("minutes_played", 0))
        if mp < 10:
            continue

        minutes_list.append(mp)

        g = safe_per90(ms.get("goals", 0), mp)
        if g is not None:
            goals.append(g)

        s = safe_per90(ms.get("shots_on_goal", 0), mp)
        if s is not None:
            shots_on.append(s)

        kp = safe_per90(ms.get("keyPasses", ms.get("key_passes", 0)), mp)
        if kp is not None:
            key_passes.append(kp)

        p_total = _safe_float(ms.get("passes", 0))
        p_acc = _safe_float(ms.get("passes_acc", 0))
        if p_total > 0:
            pass_acc.append(p_acc / p_total)

        d_att = _safe_float(ms.get("dribbleAttempts", ms.get("dribble_attempts", 0)))
        d_succ = _safe_float(ms.get("dribbleSucc", ms.get("dribble_succ", 0)))
        if d_att > 0:
            dribble_succ.append(d_succ / d_att)

        t = safe_per90(ms.get("tackles", 0), mp)
        if t is not None:
            tackles.append(t)

        i = safe_per90(ms.get("interceptions", 0), mp)
        if i is not None:
            interceptions.append(i)

        r = _safe_float(ms.get("rating", 0))
        if r > 0:
            ratings.append(r)

        saves_val = _safe_float(ms.get("saves", 0))
        shots_faced = _safe_float(ms.get("shots_faced", 0))
        if shots_faced > 0:
            save_rates.append(saves_val / shots_faced)

    return PlayerRolling(
        player_id=player_id,
        name=player_name,
        pos=pos,
        goals_per_90=_safe_mean(goals),
        shots_on_target_per_90=_safe_mean(shots_on),
        key_passes_per_90=_safe_mean(key_passes),
        pass_accuracy=_safe_mean(pass_acc),
        dribble_success_rate=_safe_mean(dribble_succ),
        tackles_per_90=_safe_mean(tackles),
        interceptions_per_90=_safe_mean(interceptions),
        rating_avg=_safe_mean(ratings),
        save_rate=_safe_mean(save_rates),
        minutes_played_avg=_safe_mean(minutes_list),
    )


# ---------------------------------------------------------------------------
# 2.1.3: Position-based team aggregation
# ---------------------------------------------------------------------------


def aggregate_team_features(player_stats: list[PlayerRolling]) -> dict[str, float]:
    """Aggregate starter rolling stats into team-level features by position.

    Groups: F(orward), M(idfield), D(efense), G(oalkeeper).
    Same structure as Tier 2 features in Phase 1 Step 1.3.
    """
    forwards = [p for p in player_stats if p.pos == "F"]
    midfielders = [p for p in player_stats if p.pos == "M"]
    defenders = [p for p in player_stats if p.pos == "D"]
    goalkeepers = [p for p in player_stats if p.pos == "G"]

    return {
        # Attack features (FW)
        "fw_avg_rating": _safe_mean([p.rating_avg for p in forwards]),
        "fw_goals_p90": _safe_sum([p.goals_per_90 for p in forwards]),
        "fw_shots_on_target_p90": _safe_sum(
            [p.shots_on_target_per_90 for p in forwards],
        ),
        # Creativity features (MF)
        "mf_avg_rating": _safe_mean([p.rating_avg for p in midfielders]),
        "mf_key_passes_p90": _safe_sum(
            [p.key_passes_per_90 for p in midfielders],
        ),
        "mf_pass_accuracy": _safe_mean([p.pass_accuracy for p in midfielders]),
        # Defense features (DF)
        "df_avg_rating": _safe_mean([p.rating_avg for p in defenders]),
        "df_tackles_p90": _safe_sum([p.tackles_per_90 for p in defenders]),
        "df_interceptions_p90": _safe_sum(
            [p.interceptions_per_90 for p in defenders],
        ),
        # GK features
        "gk_rating": goalkeepers[0].rating_avg if goalkeepers else 0.0,
        "gk_save_rate": goalkeepers[0].save_rate if goalkeepers else 0.0,
        # Team-wide
        "team_avg_rating": _weighted_mean(
            [p.rating_avg for p in player_stats],
            [p.minutes_played_avg for p in player_stats],
        ),
    }


# ---------------------------------------------------------------------------
# 2.1.4: Team-level rolling stats
# ---------------------------------------------------------------------------


def extract_team_rolling(
    recent_match_stats: list[dict[str, Any]],
    team_key: str,
) -> dict[str, float]:
    """Compute team-level rolling stats from recent historical match stats.

    Args:
        recent_match_stats: Last 5 match stats dicts from Goalserve.
        team_key: "localteam" or "visitorteam".

    Returns:
        Dict of team-level rolling features.
    """
    xg_vals: list[float] = []
    possession_vals: list[float] = []
    shots_vals: list[float] = []
    insidebox_ratios: list[float] = []
    pass_acc_vals: list[float] = []
    corners_vals: list[float] = []
    fouls_vals: list[float] = []

    for ms in recent_match_stats:
        stats = ms.get("stats", {}).get(team_key, {})
        if not stats:
            continue

        xg = _safe_float(stats.get("expected_goals", 0))
        xg_vals.append(xg)

        poss = _safe_float(
            stats.get("possestiontime", stats.get("possession", 0)),
        )
        possession_vals.append(poss)

        shots_total = _safe_float(
            stats.get("shots", stats.get("shots_total", 0)),
        )
        shots_vals.append(shots_total)

        inside = _safe_float(stats.get("insidebox", 0))
        if shots_total > 0:
            insidebox_ratios.append(inside / shots_total)

        passes_node = stats.get("passes", {})
        if isinstance(passes_node, dict):
            p_total = _safe_float(passes_node.get("total", 0))
            p_acc = _safe_float(passes_node.get("accurate", 0))
        else:
            p_total = _safe_float(passes_node)
            p_acc = _safe_float(stats.get("passes_acc", 0))
        if p_total > 0:
            pass_acc_vals.append(p_acc / p_total)

        corners_vals.append(_safe_float(stats.get("corners", 0)))
        fouls_vals.append(_safe_float(stats.get("fouls", 0)))

    return {
        "xg_per_90": _safe_mean(xg_vals),
        "possession_avg": _safe_mean(possession_vals),
        "shots_per_90": _safe_mean(shots_vals),
        "shots_insidebox_ratio": _safe_mean(insidebox_ratios),
        "pass_accuracy": _safe_mean(pass_acc_vals),
        "corners_per_90": _safe_mean(corners_vals),
        "fouls_per_90": _safe_mean(fouls_vals),
    }


# ---------------------------------------------------------------------------
# 2.1.5: Odds features
# ---------------------------------------------------------------------------


def extract_odds_features(
    bookmakers: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Extract odds features from Odds-API bookmakers.

    Delegates to ``build_odds_features`` from the odds_api client,
    which handles overround removal and Betfair Exchange fallback.
    """
    return build_odds_features(bookmakers)


# ---------------------------------------------------------------------------
# 2.1.6: Context features
# ---------------------------------------------------------------------------


def compute_rest_days(
    team_fixtures: list[dict[str, Any]],
    kickoff_date: str,
) -> int:
    """Compute days since the team's most recent match.

    Args:
        team_fixtures: Recent fixtures for the team.
        kickoff_date: Today's match date (YYYY-MM-DD or DD.MM.YYYY).

    Returns:
        Rest days (default 3 if unknown).
    """
    from datetime import datetime

    # Parse kickoff date
    today: datetime | None = None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            today = datetime.strptime(kickoff_date, fmt)
            break
        except ValueError:
            continue
    if today is None:
        return 3

    # Find most recent match before today
    most_recent: datetime | None = None
    for fix in team_fixtures:
        raw_date = fix.get("@formatted_date", fix.get("@date", fix.get("date", "")))
        if not raw_date:
            continue
        match_date: datetime | None = None
        for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
            try:
                match_date = datetime.strptime(str(raw_date), fmt)
                break
            except ValueError:
                continue
        if match_date is None or match_date >= today:
            continue
        if most_recent is None or match_date > most_recent:
            most_recent = match_date

    if most_recent is None:
        return 3
    return max(1, (today - most_recent).days)


def compute_h2h_goal_diff(
    h2h_matches: list[dict[str, Any]],
    home_team: str,
) -> float:
    """Compute average goal difference from last 5 H2H matches.

    Args:
        h2h_matches: Recent H2H fixture dicts.
        home_team: Name of the home team (to determine sign).

    Returns:
        Average goal difference (positive = home team advantage).
    """
    diffs: list[float] = []
    for m in h2h_matches[:5]:
        local = str(m.get("@localteam", m.get("localteam", "")))
        score = str(
            m.get("@goals", m.get("@ft_score", m.get("@score", "")))
        )
        if not score or "-" not in score:
            continue
        parts = score.split("-")
        try:
            home_g = int(parts[0].strip())
            away_g = int(parts[1].strip())
        except (ValueError, IndexError):
            continue
        diff = home_g - away_g
        if local != home_team:
            diff = -diff
        diffs.append(float(diff))

    return _safe_mean(diffs)


# ---------------------------------------------------------------------------
# 2.1.7: Pinnacle pre-match odds (Goalserve getodds)
# ---------------------------------------------------------------------------


async def fetch_prematch_odds(
    gs_client: GoalserveClient,
    match_id: str,
    league_id: int,
    *,
    prefetched_odds_matches: list[dict[str, Any]] | None = None,
) -> tuple[float, float, float] | None:
    """Fetch Pinnacle 1x2 pre-match odds from Goalserve getodds endpoint.

    Searches for the match by fix_id, static_id, or id, then extracts
    Pinnacle (bookmaker id=82 "Pncl") Match Winner odds.

    Args:
        gs_client: Active Goalserve client.
        match_id: Goalserve match/fixture ID.
        league_id: Goalserve league ID.
        prefetched_odds_matches: Pre-fetched odds matches list (avoids
            duplicate API calls that trigger rate limiting).

    Returns:
        (odds_H, odds_D, odds_A) or None if unavailable.
    """
    if prefetched_odds_matches is not None:
        matches = prefetched_odds_matches
    else:
        try:
            matches = await gs_client.get_prematch_odds(league_id)
        except Exception as exc:
            logger.warning(
                "prematch_odds_fetch_failed",
                match_id=match_id,
                league_id=league_id,
                error=str(exc),
            )
            return None

    mid = str(match_id)
    target: dict | None = None
    for m in matches:
        if (
            str(m.get("fix_id", "")) == mid
            or str(m.get("static_id", "")) == mid
            or str(m.get("id", "")) == mid
        ):
            target = m
            break

    if target is None:
        logger.info(
            "prematch_odds_match_not_found",
            match_id=match_id,
            league_id=league_id,
            available_count=len(matches),
        )
        return None

    # Find "Match Winner" odds market
    odds_list = target.get("odds", [])
    if isinstance(odds_list, dict):
        odds_list = [odds_list]

    for market in odds_list:
        if market.get("value") != "Match Winner":
            continue
        bookmakers = market.get("bookmakers", [])
        if isinstance(bookmakers, dict):
            bookmakers = [bookmakers]
        for bm in bookmakers:
            if bm.get("name") == "Pncl" or bm.get("id") == "82":
                odds_entries = bm.get("odds", [])
                h_val = d_val = a_val = 0.0
                for entry in odds_entries:
                    name = entry.get("name", "")
                    val = _safe_float(entry.get("value", 0))
                    if name == "Home":
                        h_val = val
                    elif name == "Draw":
                        d_val = val
                    elif name == "Away":
                        a_val = val
                if h_val > 1.0 and d_val > 1.0 and a_val > 1.0:
                    logger.info(
                        "pinnacle_odds_found",
                        match_id=match_id,
                        odds_H=h_val,
                        odds_D=d_val,
                        odds_A=a_val,
                    )
                    return (h_val, d_val, a_val)

    logger.info(
        "pinnacle_odds_not_available",
        match_id=match_id,
        league_id=league_id,
    )
    return None


# ---------------------------------------------------------------------------
# Full data collection
# ---------------------------------------------------------------------------


async def collect_pre_match_data(
    gs_client: GoalserveClient,
    odds_client: OddsApiClient | None,
    match_id: str,
    league_id: int,
    *,
    odds_event_id: str | None = None,
    kickoff_date: str = "",
    recent_home_stats: list[dict[str, Any]] | None = None,
    recent_away_stats: list[dict[str, Any]] | None = None,
    home_fixtures: list[dict[str, Any]] | None = None,
    away_fixtures: list[dict[str, Any]] | None = None,
    h2h_matches: list[dict[str, Any]] | None = None,
) -> PreMatchData:
    """Collect all pre-match data for a single match.

    This is the main entry point for Step 2.1. It fetches lineups,
    computes player and team rolling stats, collects odds, and
    builds context features.

    Args:
        gs_client: Active Goalserve client.
        odds_client: Active Odds-API client (or None if unavailable).
        match_id: Goalserve match ID.
        league_id: Goalserve league ID.
        odds_event_id: Odds-API event ID (for odds lookup).
        kickoff_date: Match date string.
        recent_home_stats: Pre-fetched recent home match stats.
        recent_away_stats: Pre-fetched recent away match stats.
        home_fixtures: Pre-fetched home team fixtures (for rest days).
        away_fixtures: Pre-fetched away team fixtures (for rest days).
        h2h_matches: Pre-fetched H2H fixtures.

    Returns:
        Populated PreMatchData.
    """
    data = PreMatchData(match_id=match_id, kickoff_time=kickoff_date)

    # 2.1.1: Lineups
    try:
        match_stats = await gs_client.get_match_stats(match_id, league_id)
    except Exception as exc:
        logger.warning(
            "match_stats_fetch_failed",
            match_id=match_id,
            league_id=league_id,
            error=str(exc),
        )
        match_stats = {}

    if not match_stats:
        logger.warning(
            "no_match_stats_available",
            match_id=match_id,
            league_id=league_id,
            detail="Commentaries endpoint returned empty — match may not have started yet",
        )

    home_ids, away_ids, home_form, away_form = extract_lineups(match_stats)
    data.home_starting_11 = home_ids
    data.away_starting_11 = away_ids
    data.home_formation = home_form
    data.away_formation = away_form

    logger.info(
        "lineups_collected",
        match_id=match_id,
        home_count=len(home_ids),
        away_count=len(away_ids),
        home_formation=home_form,
        away_formation=away_form,
    )

    # 2.1.3: Team aggregation (requires player rolling stats from 2.1.2)
    # In a full system, we'd fetch player_stats for each starter.
    # For now, extract from the match_stats if available.
    home_player_stats = _extract_player_rolling_from_stats(
        match_stats, "localteam",
    )
    away_player_stats = _extract_player_rolling_from_stats(
        match_stats, "visitorteam",
    )
    data.home_player_agg = aggregate_team_features(home_player_stats)
    data.away_player_agg = aggregate_team_features(away_player_stats)

    # 2.1.4: Team rolling stats
    if recent_home_stats:
        data.home_team_rolling = extract_team_rolling(
            recent_home_stats, "localteam",
        )
    if recent_away_stats:
        data.away_team_rolling = extract_team_rolling(
            recent_away_stats, "visitorteam",
        )

    # 2.1.5: Odds features
    if odds_client and odds_event_id:
        try:
            from src.clients.odds_api import SELECTED_BOOKMAKERS

            odds_data = await odds_client.get_odds(
                odds_event_id,
                ",".join(SELECTED_BOOKMAKERS),
            )
            bookmakers = odds_data.get("bookmakers", {})
            if bookmakers:
                data.odds_features = extract_odds_features(bookmakers)
        except Exception as e:
            logger.warning("odds_fetch_failed", error=str(e))

    # 2.1.6: Context features
    home_team = str(
        match_stats.get("@localteam", match_stats.get("localteam", "")),
    )
    if home_fixtures:
        data.home_rest_days = compute_rest_days(home_fixtures, kickoff_date)
    if away_fixtures:
        data.away_rest_days = compute_rest_days(away_fixtures, kickoff_date)
    if h2h_matches:
        data.h2h_goal_diff = compute_h2h_goal_diff(h2h_matches, home_team)

    logger.info(
        "pre_match_data_collected",
        match_id=match_id,
        has_odds=bool(data.odds_features),
        home_rest=data.home_rest_days,
        away_rest=data.away_rest_days,
    )

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_player_rolling_from_stats(
    match_stats: dict[str, Any],
    team_key: str,
) -> list[PlayerRolling]:
    """Extract PlayerRolling from a single match's player_stats.

    This is a simplified version that uses the current match's stats
    as a proxy when historical rolling data is unavailable.
    """
    player_stats_node = match_stats.get("player_stats", {}).get(team_key, {})
    if not player_stats_node:
        return []

    players = ensure_list(player_stats_node.get("player", []))
    result: list[PlayerRolling] = []

    for p in players:
        pid = str(p.get("id", p.get("@id", "")))
        name = str(p.get("name", p.get("@name", "")))
        pos = str(p.get("pos", p.get("@pos", "M")))
        mp = _safe_float(p.get("minutes_played", 0))

        if mp < 10:
            continue

        rolling = PlayerRolling(
            player_id=pid,
            name=name,
            pos=pos,
            goals_per_90=_safe_float(p.get("goals", 0)) / mp * 90.0 if mp > 0 else 0.0,
            shots_on_target_per_90=_safe_float(p.get("shots_on_goal", 0)) / mp * 90.0 if mp > 0 else 0.0,
            key_passes_per_90=_safe_float(p.get("keyPasses", p.get("key_passes", 0))) / mp * 90.0 if mp > 0 else 0.0,
            tackles_per_90=_safe_float(p.get("tackles", 0)) / mp * 90.0 if mp > 0 else 0.0,
            interceptions_per_90=_safe_float(p.get("interceptions", 0)) / mp * 90.0 if mp > 0 else 0.0,
            rating_avg=_safe_float(p.get("rating", 0)),
            minutes_played_avg=mp,
        )

        # Pass accuracy
        passes_total = _safe_float(p.get("passes", 0))
        passes_acc = _safe_float(p.get("passes_acc", 0))
        if passes_total > 0:
            rolling.pass_accuracy = passes_acc / passes_total

        # Save rate (GK)
        saves = _safe_float(p.get("saves", 0))
        shots_faced = _safe_float(p.get("shots_faced", 0))
        if shots_faced > 0:
            rolling.save_rate = saves / shots_faced

        result.append(rolling)

    return result


def _safe_mean(values: list[float]) -> float:
    """Mean of a list, returning 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_sum(values: list[float]) -> float:
    """Sum of a list, returning 0.0 if empty."""
    return sum(values) if values else 0.0


def _weighted_mean(
    values: list[float],
    weights: list[float],
) -> float:
    """Weighted mean, falling back to simple mean if weights are zero."""
    if not values:
        return 0.0
    total_weight = sum(w for w in weights if w > 0)
    if total_weight <= 0:
        return _safe_mean(values)
    return sum(v * max(w, 0) for v, w in zip(values, weights, strict=False)) / total_weight
