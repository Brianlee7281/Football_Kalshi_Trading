"""Market mapping — model key ↔ Kalshi ticker type.

Provides:
    MODEL_TO_KALSHI_TYPE  — canonical model key → Kalshi market type dict
    KALSHI_TYPE_TO_MODEL  — reverse lookup
    classify_ticker()     — heuristic: Kalshi ticker string → model key
    build_ticker_mapping()  — batch classify a list of tickers for one match
    insert_ticker_mapping() — persist mapping to ticker_mapping DB table

Market key convention (Phase 3 output / Phase 4 input):
    "home_win"   — home team wins at full time
    "away_win"   — away team wins at full time
    "draw"       — draw at full time
    "over_25"    — total goals > 2.5
    "over_35"    — total goals > 3.5
    "btts_yes"   — both teams score

Kalshi ticker format (real-world examples from exploration):
    KXUCLGAME-26MAR10PSGCFC-PSG   → match winner PSG
    KXUCLGAME-26MAR10PSGCFC-CFC   → match winner CFC
    KXUCLGAME-26MAR10PSGCFC-DRAW  → draw
    KXUCLOU25-26MAR10PSGCFC-YES   → over 2.5 goals
    KXUCLBTTS-26MAR10PSGCFC-YES   → both teams score

Home/away classification for winner markets requires home_code / away_code
(3-4 letter team abbreviation as it appears in the ticker suffix).

Reference: docs/phase4.md §Market Key Mapping, sql/schema.sql ticker_mapping
"""

from __future__ import annotations

from typing import Any

import asyncpg

from src.common.logging import get_logger

logger = get_logger("market_mapping")

# ---------------------------------------------------------------------------
# Canonical mapping dicts
# ---------------------------------------------------------------------------

#: Model key → Kalshi market type string.
#: Used by Phase 2.5 when discovering available markets on Kalshi.
MODEL_TO_KALSHI_TYPE: dict[str, str] = {
    "home_win": "match_winner_home",
    "away_win": "match_winner_away",
    "draw": "match_winner_draw",
    "over_25": "over_under_2.5",
    "over_35": "over_under_3.5",
    "btts_yes": "btts",
}

#: Reverse lookup: Kalshi market type → model key.
KALSHI_TYPE_TO_MODEL: dict[str, str] = {v: k for k, v in MODEL_TO_KALSHI_TYPE.items()}

# All valid model market keys (for validation)
_VALID_MODEL_KEYS: frozenset[str] = frozenset(MODEL_TO_KALSHI_TYPE)

# ---------------------------------------------------------------------------
# Ticker classification
# ---------------------------------------------------------------------------

# Uppercase substrings that identify over/under and BTTS markets
_OU25_PATTERNS = ("OU25", "OU-25", "O2.5", "OVER2", "OVER25", "OVER-25", "OVER_25")
_OU35_PATTERNS = ("OU35", "OU-35", "O3.5", "OVER3", "OVER35", "OVER-35", "OVER_35")
_BTTS_PATTERNS = ("BTTS", "BOTHSCOR", "BOTH-SCOR")
_DRAW_PATTERNS = ("DRAW", "-D-", "_D_")


def classify_ticker(
    ticker: str,
    *,
    home_code: str | None = None,
    away_code: str | None = None,
) -> str | None:
    """Infer model key from a Kalshi ticker string.

    Classification priority (first match wins):
      1. Over 3.5 goals patterns
      2. Over 2.5 goals patterns
      3. BTTS patterns
      4. Draw patterns (in suffix position or explicit keyword)
      5. Home winner — home_code present in the ticker suffix
      6. Away winner — away_code present in the ticker suffix

    Both team codes are matched case-insensitively against the ticker.
    The suffix (last ``-`` segment) is checked first to reduce false positives
    when a team abbreviation could appear elsewhere in the ticker.

    Args:
        ticker: Kalshi market ticker, e.g. "KXUCLGAME-26MAR10PSGCFC-PSG".
        home_code: Short team code for the home team (e.g. "PSG", "ARS").
        away_code: Short team code for the away team (e.g. "CFC", "CHE").

    Returns:
        One of the MODEL_TO_KALSHI_TYPE keys, or None if unrecognised.
    """
    upper = ticker.upper()
    suffix = upper.rsplit("-", 1)[-1] if "-" in upper else upper

    # Over 3.5 (check before 2.5 to avoid substring collision)
    if any(pat in upper for pat in _OU35_PATTERNS):
        return "over_35"

    # Over 2.5
    if any(pat in upper for pat in _OU25_PATTERNS):
        return "over_25"

    # Both teams score
    if any(pat in upper for pat in _BTTS_PATTERNS):
        return "btts_yes"

    # Draw
    if any(pat in upper for pat in _DRAW_PATTERNS) or suffix in ("D", "DRAW", "TIE"):
        return "draw"

    # Winner markets — resolve home vs away via team code matching
    if home_code:
        hc = home_code.upper()
        # Prefer suffix match (most specific), fallback to any occurrence
        if suffix == hc or upper.endswith(f"-{hc}"):
            return "home_win"

    if away_code:
        ac = away_code.upper()
        if suffix == ac or upper.endswith(f"-{ac}"):
            return "away_win"

    # Looser team-code scan when suffix didn't match (e.g. longer codes)
    if home_code and home_code.upper() in upper:
        return "home_win"
    if away_code and away_code.upper() in upper:
        return "away_win"

    return None


# ---------------------------------------------------------------------------
# Batch mapping builder
# ---------------------------------------------------------------------------


def build_ticker_mapping(
    match_id: str,
    kalshi_tickers: list[str],
    *,
    home_code: str | None = None,
    away_code: str | None = None,
) -> dict[str, str]:
    """Classify a list of Kalshi tickers and return a ticker → model_key mapping.

    Unrecognised tickers are logged as warnings and excluded from the result.
    Tickers that resolve to the same model key (e.g. two "home_win" tickers)
    keep only the first occurrence and warn.

    Args:
        match_id: Goalserve match ID (used for logging only).
        kalshi_tickers: Kalshi ticker strings for this match.
        home_code: Short code for the home team (e.g. "PSG", "ARS").
        away_code: Short code for the away team (e.g. "CFC", "CHE").

    Returns:
        Dict mapping each successfully classified ticker to its model key.
    """
    log = logger.bind(match_id=match_id)
    result: dict[str, str] = {}
    seen_model_keys: dict[str, str] = {}  # model_key → first ticker that claimed it

    for ticker in kalshi_tickers:
        model_key = classify_ticker(ticker, home_code=home_code, away_code=away_code)

        if model_key is None:
            log.warning("ticker_unrecognised", ticker=ticker)
            continue

        if model_key in seen_model_keys:
            log.warning(
                "duplicate_model_key",
                model_key=model_key,
                first_ticker=seen_model_keys[model_key],
                skipped_ticker=ticker,
            )
            continue

        seen_model_keys[model_key] = ticker
        result[ticker] = model_key

    log.info(
        "ticker_mapping_built",
        total=len(kalshi_tickers),
        mapped=len(result),
        model_keys=list(result.values()),
    )
    return result


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


async def insert_ticker_mapping(
    pool: asyncpg.Pool,
    match_id: str,
    ticker_map: dict[str, str],
) -> None:
    """Insert ticker mapping rows into the ticker_mapping table.

    Uses INSERT … ON CONFLICT DO NOTHING so repeated calls (e.g. on container
    restart) are idempotent.

    Args:
        pool: asyncpg connection pool (from src.common.db in Sprint 6).
        match_id: Match ID to associate with the mapping rows.
        ticker_map: Dict of {kalshi_ticker: model_key} from build_ticker_mapping.
    """
    if not ticker_map:
        return

    rows: list[tuple[str, str, str]] = [
        (match_id, ticker, model_key)
        for ticker, model_key in ticker_map.items()
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO ticker_mapping (match_id, kalshi_ticker, model_key)
            VALUES ($1, $2, $3)
            ON CONFLICT (match_id, kalshi_ticker) DO NOTHING
            """,
            rows,
        )

    logger.info(
        "ticker_mapping_inserted",
        match_id=match_id,
        rows=len(rows),
    )


async def load_ticker_mapping(
    pool: asyncpg.Pool,
    match_id: str,
) -> dict[str, str]:
    """Load ticker → model_key mapping from DB for a given match.

    Args:
        pool: asyncpg connection pool.
        match_id: Match ID to load mappings for.

    Returns:
        Dict of {kalshi_ticker: model_key}.
    """
    async with pool.acquire() as conn:
        rows: list[Any] = await conn.fetch(
            "SELECT kalshi_ticker, model_key FROM ticker_mapping WHERE match_id = $1",
            match_id,
        )

    mapping = {row["kalshi_ticker"]: row["model_key"] for row in rows}
    logger.info(
        "ticker_mapping_loaded",
        match_id=match_id,
        count=len(mapping),
    )
    return mapping
