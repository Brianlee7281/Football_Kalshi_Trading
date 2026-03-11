"""Unit tests for market_mapping — MODEL_TO_KALSHI_TYPE, classify_ticker, build_ticker_mapping.

Tests:
  - MODEL_TO_KALSHI_TYPE completeness and reverse lookup consistency
  - classify_ticker: over_25, over_35, btts_yes, draw, home_win, away_win, None
  - build_ticker_mapping: batch classification, duplicate warning, unknown skipped

Reference: docs/phase4.md §Market Key Mapping
"""

from __future__ import annotations

import pytest

from src.execution.market_mapping import (
    KALSHI_TYPE_TO_MODEL,
    MODEL_TO_KALSHI_TYPE,
    _VALID_MODEL_KEYS,
    build_ticker_mapping,
    classify_ticker,
)


# ---------------------------------------------------------------------------
# MODEL_TO_KALSHI_TYPE / reverse dict
# ---------------------------------------------------------------------------


def test_model_to_kalshi_type_has_all_keys() -> None:
    """MODEL_TO_KALSHI_TYPE contains all 6 expected model keys."""
    expected = {"home_win", "away_win", "draw", "over_25", "over_35", "btts_yes"}
    assert set(MODEL_TO_KALSHI_TYPE.keys()) == expected


def test_kalshi_type_to_model_is_exact_reverse() -> None:
    """KALSHI_TYPE_TO_MODEL is the exact inverse of MODEL_TO_KALSHI_TYPE."""
    for model_key, kalshi_type in MODEL_TO_KALSHI_TYPE.items():
        assert KALSHI_TYPE_TO_MODEL[kalshi_type] == model_key


def test_valid_model_keys_matches_dict() -> None:
    """_VALID_MODEL_KEYS equals the set of MODEL_TO_KALSHI_TYPE keys."""
    assert _VALID_MODEL_KEYS == frozenset(MODEL_TO_KALSHI_TYPE)


# ---------------------------------------------------------------------------
# classify_ticker — over goals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", [
    "KXUCLOU25-26MAR10PSGCFC-YES",
    "SOCCER-EPL-ARS-v-CHE-OU25-YES",
    "MATCH-O2.5-GOALS",
    "OVER25-KXEPL-ARS",
    "SOCCER-OVER2-TOTAL",
])
def test_classify_over_25(ticker: str) -> None:
    """Tickers containing over-2.5 patterns classify as over_25."""
    assert classify_ticker(ticker) == "over_25"


@pytest.mark.parametrize("ticker", [
    "KXUCLOU35-26MAR10PSGCFC-YES",
    "SOCCER-EPL-ARS-v-CHE-OU35-YES",
    "MATCH-O3.5-GOALS",
    "OVER35-KXEPL-ARS",
    "SOCCER-OVER3-TOTAL",
])
def test_classify_over_35(ticker: str) -> None:
    """Tickers containing over-3.5 patterns classify as over_35."""
    assert classify_ticker(ticker) == "over_35"


def test_over35_takes_priority_over_over25() -> None:
    """OU35 is classified as over_35 not over_25 (check before 2.5 patterns)."""
    assert classify_ticker("KXUCLOU35-MATCH") == "over_35"


# ---------------------------------------------------------------------------
# classify_ticker — BTTS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", [
    "KXUCLBTTS-26MAR10PSGCFC-YES",
    "SOCCER-BTTS-MATCH",
    "MATCH-BOTHSCOR-YES",
])
def test_classify_btts_yes(ticker: str) -> None:
    """Tickers with BTTS/BOTHSCOR patterns classify as btts_yes."""
    assert classify_ticker(ticker) == "btts_yes"


# ---------------------------------------------------------------------------
# classify_ticker — draw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", [
    "KXUCLGAME-26MAR10PSGCFC-DRAW",
    "SOCCER-EPL-ARS-v-CHE-DRAW",
    "MATCH-DRAW-RESULT",
])
def test_classify_draw(ticker: str) -> None:
    """Tickers with DRAW in them classify as draw."""
    assert classify_ticker(ticker) == "draw"


def test_classify_draw_suffix_d() -> None:
    """Tickers ending in suffix 'D' or 'TIE' classify as draw."""
    assert classify_ticker("SOCCER-ARS-v-CHE-D") == "draw"
    assert classify_ticker("SOCCER-ARS-v-CHE-TIE") == "draw"


# ---------------------------------------------------------------------------
# classify_ticker — home_win / away_win
# ---------------------------------------------------------------------------


def test_classify_home_win_by_suffix() -> None:
    """Ticker suffix matching home_code → home_win."""
    assert classify_ticker("KXUCLGAME-26MAR10PSGCFC-PSG", home_code="PSG", away_code="CFC") == "home_win"


def test_classify_away_win_by_suffix() -> None:
    """Ticker suffix matching away_code → away_win."""
    assert classify_ticker("KXUCLGAME-26MAR10PSGCFC-CFC", home_code="PSG", away_code="CFC") == "away_win"


def test_classify_home_win_case_insensitive() -> None:
    """Team code matching is case-insensitive."""
    assert classify_ticker("KXUCLGAME-26MAR10PSGCFC-PSG", home_code="psg", away_code="cfc") == "home_win"


def test_classify_epl_style_ticker() -> None:
    """EPL-style SOCCER-EPL-ARS-v-CHE ticker with home_code ARS → home_win."""
    assert classify_ticker("SOCCER-EPL-ARS-v-CHE-ARS", home_code="ARS", away_code="CHE") == "home_win"


def test_classify_away_win_epl_style() -> None:
    """EPL-style ticker ending in away team code → away_win."""
    assert classify_ticker("SOCCER-EPL-ARS-v-CHE-CHE", home_code="ARS", away_code="CHE") == "away_win"


def test_classify_no_team_codes_returns_none_for_winner() -> None:
    """Winner ticker without team codes provided → None (can't distinguish home/away)."""
    result = classify_ticker("KXUCLGAME-26MAR10PSGCFC-PSG")
    assert result is None


# ---------------------------------------------------------------------------
# classify_ticker — unrecognised → None
# ---------------------------------------------------------------------------


def test_classify_unknown_ticker_returns_none() -> None:
    """Completely unrecognised ticker returns None."""
    assert classify_ticker("KXNASDAQ-AAPL-OVER100") is None


def test_classify_empty_ticker_returns_none() -> None:
    """Empty ticker string returns None."""
    assert classify_ticker("") is None


# ---------------------------------------------------------------------------
# build_ticker_mapping — happy path
# ---------------------------------------------------------------------------


def test_build_ticker_mapping_all_markets() -> None:
    """build_ticker_mapping correctly classifies all 6 market types."""
    tickers = [
        "KXUCLGAME-26MAR10PSGCFC-PSG",   # home_win
        "KXUCLGAME-26MAR10PSGCFC-CFC",   # away_win
        "KXUCLGAME-26MAR10PSGCFC-DRAW",  # draw
        "KXUCLOU25-26MAR10PSGCFC-YES",   # over_25
        "KXUCLOU35-26MAR10PSGCFC-YES",   # over_35
        "KXUCLBTTS-26MAR10PSGCFC-YES",   # btts_yes
    ]
    result = build_ticker_mapping(
        "match_001", tickers, home_code="PSG", away_code="CFC"
    )
    assert result["KXUCLGAME-26MAR10PSGCFC-PSG"] == "home_win"
    assert result["KXUCLGAME-26MAR10PSGCFC-CFC"] == "away_win"
    assert result["KXUCLGAME-26MAR10PSGCFC-DRAW"] == "draw"
    assert result["KXUCLOU25-26MAR10PSGCFC-YES"] == "over_25"
    assert result["KXUCLOU35-26MAR10PSGCFC-YES"] == "over_35"
    assert result["KXUCLBTTS-26MAR10PSGCFC-YES"] == "btts_yes"


def test_build_ticker_mapping_count() -> None:
    """build_ticker_mapping returns exactly as many entries as recognised tickers."""
    tickers = [
        "KXUCLOU25-MATCH-YES",
        "KXUCLOU35-MATCH-YES",
        "UNKNOWN-TICKER-XYZ",  # unrecognised
    ]
    result = build_ticker_mapping("match_002", tickers)
    assert len(result) == 2
    assert "UNKNOWN-TICKER-XYZ" not in result


def test_build_ticker_mapping_unknown_excluded() -> None:
    """Unrecognised tickers are excluded from the returned dict."""
    tickers = ["KXUCLBTTS-MATCH-YES", "GARBAGE-TICKER"]
    result = build_ticker_mapping("match_003", tickers)
    assert "GARBAGE-TICKER" not in result
    assert "KXUCLBTTS-MATCH-YES" in result


def test_build_ticker_mapping_duplicate_model_key_keeps_first() -> None:
    """When two tickers classify to the same model key, only the first is kept."""
    tickers = [
        "KXUCLOU25-MATCH1-YES",   # over_25
        "SOCCER-OU25-MATCH2-YES",  # also over_25
    ]
    result = build_ticker_mapping("match_004", tickers)
    assert len(result) == 1
    # First ticker wins
    assert "KXUCLOU25-MATCH1-YES" in result


def test_build_ticker_mapping_empty_list() -> None:
    """Empty ticker list returns empty dict without error."""
    result = build_ticker_mapping("match_005", [])
    assert result == {}


def test_build_ticker_mapping_values_are_valid_model_keys() -> None:
    """All values in the returned dict are valid model keys."""
    tickers = [
        "KXUCLOU25-M-YES",
        "KXUCLBTTS-M-YES",
        "KXUCLGAME-M-DRAW",
    ]
    result = build_ticker_mapping("match_006", tickers, home_code="PSG", away_code="CFC")
    for model_key in result.values():
        assert model_key in _VALID_MODEL_KEYS
