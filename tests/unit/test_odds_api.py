"""Tests for src/clients/odds_api.py — parsing, filtering, odds features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.clients.odds_api import (
    SELECTED_BOOKMAKERS,
    OddsApiClient,
    _compute_odds_delta,
    _filter_bookmakers,
    build_odds_features,
    extract_bet365_implied_probs,
    remove_overround,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def historical_data() -> dict[str, Any]:
    with open(FIXTURES_DIR / "odds_api_historical.json") as f:
        return json.load(f)


@pytest.fixture
def arsenal_bookmakers(historical_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Arsenal vs Chelsea bookmakers from the fixture."""
    return historical_data["data"][0]["bookmakers"]


# ---------------------------------------------------------------------------
# remove_overround
# ---------------------------------------------------------------------------


class TestRemoveOverround:
    def test_probabilities_sum_to_one(self) -> None:
        h, d, a = remove_overround(1.44, 3.50, 12.00)
        assert abs(h + d + a - 1.0) < 1e-10

    def test_betfair_exchange_example(self) -> None:
        """From docs/phase1.md Step 1.3: betfair_exchange home=1.44, draw=3.50, away=12.00."""
        h, d, a = remove_overround(1.44, 3.50, 12.00)
        # Expected: 1/1.44 / (1/1.44 + 1/3.50 + 1/12.00) ≈ 0.653
        assert abs(h - 0.653) < 0.001

    def test_even_odds(self) -> None:
        h, d, a = remove_overround(3.0, 3.0, 3.0)
        assert abs(h - 1 / 3) < 1e-10
        assert abs(d - 1 / 3) < 1e-10
        assert abs(a - 1 / 3) < 1e-10

    def test_heavy_favorite(self) -> None:
        h, d, a = remove_overround(1.10, 8.00, 30.00)
        assert h > 0.85
        assert h + d + a == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_odds_features
# ---------------------------------------------------------------------------


class TestBuildOddsFeatures:
    def test_with_betfair_exchange(self, arsenal_bookmakers: list[dict[str, Any]]) -> None:
        features = build_odds_features(arsenal_bookmakers)
        # Betfair Exchange: home=1.44, draw=3.50, away=12.00
        assert abs(features["exchange_home_prob"] - 0.653) < 0.001
        assert features["exchange_draw_prob"] > 0
        assert features["exchange_away_prob"] > 0
        # Probabilities should sum to ~1
        total = (
            features["exchange_home_prob"]
            + features["exchange_draw_prob"]
            + features["exchange_away_prob"]
        )
        assert abs(total - 1.0) < 1e-10

    def test_market_avg_populated(self, arsenal_bookmakers: list[dict[str, Any]]) -> None:
        features = build_odds_features(arsenal_bookmakers)
        assert features["market_avg_home_prob"] > 0
        assert features["market_avg_draw_prob"] > 0
        assert features["bookmaker_odds_std"] >= 0

    def test_without_betfair_exchange_falls_back(self) -> None:
        """When Betfair Exchange is absent, exchange_*_prob = market average."""
        bookmakers = [
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home Team", "price": 2.00},
                            {"name": "Draw", "price": 3.00},
                            {"name": "Away Team", "price": 4.00},
                        ],
                    }
                ],
            },
            {
                "key": "sbobet",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home Team", "price": 2.10},
                            {"name": "Draw", "price": 3.10},
                            {"name": "Away Team", "price": 3.80},
                        ],
                    }
                ],
            },
        ]
        features = build_odds_features(bookmakers)
        # Should be market average of bet365 + sbobet
        assert features["exchange_home_prob"] == pytest.approx(
            features["market_avg_home_prob"], abs=1e-10
        )

    def test_empty_bookmakers(self) -> None:
        features = build_odds_features([])
        assert features["exchange_home_prob"] == 0.0
        assert features["market_avg_home_prob"] == 0.0

    def test_bookmaker_without_h2h_skipped(self) -> None:
        bookmakers = [
            {
                "key": "betfair_exchange",
                "markets": [{"key": "totals", "outcomes": []}],
            },
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home Team", "price": 2.00},
                            {"name": "Draw", "price": 3.00},
                            {"name": "Away Team", "price": 4.00},
                        ],
                    }
                ],
            },
        ]
        features = build_odds_features(bookmakers)
        # Betfair Exchange had no h2h, so falls back to bet365 average
        assert features["exchange_home_prob"] > 0
        assert features["bookmaker_odds_std"] == 0.0  # only 1 bookmaker with h2h


# ---------------------------------------------------------------------------
# extract_bet365_implied_probs
# ---------------------------------------------------------------------------


class TestExtractBet365ImpliedProbs:
    def test_bet365_present(self, arsenal_bookmakers: list[dict[str, Any]]) -> None:
        probs = extract_bet365_implied_probs(arsenal_bookmakers)
        assert probs is not None
        assert probs["home_win"] > 0
        assert probs["draw"] > 0
        assert probs["away_win"] > 0
        assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-10

    def test_bet365_absent(self) -> None:
        bookmakers = [{"key": "betfair_exchange", "markets": []}]
        assert extract_bet365_implied_probs(bookmakers) is None

    def test_empty_bookmakers(self) -> None:
        assert extract_bet365_implied_probs([]) is None


# ---------------------------------------------------------------------------
# _filter_bookmakers
# ---------------------------------------------------------------------------


class TestFilterBookmakers:
    def test_filters_to_selected(self, historical_data: dict[str, Any]) -> None:
        event = historical_data["data"][0]
        filtered = _filter_bookmakers(event)
        for bm in filtered["bookmakers"]:
            assert bm["key"] in SELECTED_BOOKMAKERS

    def test_pinnacle_removed(self, historical_data: dict[str, Any]) -> None:
        event = historical_data["data"][0]
        original_keys = {bm["key"] for bm in event["bookmakers"]}
        assert "pinnacle" in original_keys

        filtered = _filter_bookmakers(event)
        filtered_keys = {bm["key"] for bm in filtered["bookmakers"]}
        assert "pinnacle" not in filtered_keys

    def test_five_bookmakers_kept(self, historical_data: dict[str, Any]) -> None:
        event = historical_data["data"][0]
        filtered = _filter_bookmakers(event)
        assert len(filtered["bookmakers"]) == 5


# ---------------------------------------------------------------------------
# _compute_odds_delta
# ---------------------------------------------------------------------------


class TestComputeOddsDelta:
    def test_first_message_returns_zero(self) -> None:
        markets = [{"name": "ML", "odds": [{"home": "1.50"}]}]
        delta, last = _compute_odds_delta(markets, None)
        assert delta == 0.0
        assert last == 1.50

    def test_subsequent_message_computes_delta(self) -> None:
        markets = [{"name": "ML", "odds": [{"home": "1.65"}]}]
        delta, last = _compute_odds_delta(markets, 1.50)
        assert delta == pytest.approx(0.10, abs=0.001)  # |1.65-1.50|/1.50 = 0.10
        assert last == 1.65

    def test_no_ml_market_returns_zero(self) -> None:
        markets = [{"name": "Totals", "odds": [{"over": "1.90"}]}]
        delta, last = _compute_odds_delta(markets, 1.50)
        assert delta == 0.0
        assert last == 1.50

    def test_empty_markets(self) -> None:
        delta, last = _compute_odds_delta([], 1.50)
        assert delta == 0.0

    def test_malformed_odds_handled(self) -> None:
        markets = [{"name": "ML", "odds": []}]
        delta, last = _compute_odds_delta(markets, 1.50)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# Client HTTP parsing (mock transport)
# ---------------------------------------------------------------------------


def _make_odds_client(handler: Any) -> OddsApiClient:
    client = OddsApiClient(api_key="test_key")
    client._http._client = httpx.AsyncClient(
        base_url="https://api.the-odds-api.com",
        transport=httpx.MockTransport(handler),
    )
    return client


class TestOddsApiClientHistorical:
    async def test_parses_historical_response(
        self, historical_data: dict[str, Any]
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=historical_data)

        client = _make_odds_client(handler)
        events = await client.get_historical_odds(
            sport="soccer_epl",
            date="2024-03-15T12:00:00Z",
        )
        assert len(events) == 2
        # First event: Arsenal vs Chelsea — Pinnacle filtered out, 5 kept
        assert len(events[0]["bookmakers"]) == 5
        # Second event: Liverpool vs Man City — only 2 of 5 selected present
        assert len(events[1]["bookmakers"]) == 2
        await client.close()

    async def test_api_key_in_params(self, historical_data: dict[str, Any]) -> None:
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.update(dict(request.url.params))
            return httpx.Response(200, json=historical_data)

        client = _make_odds_client(handler)
        await client.get_historical_odds(
            sport="soccer_epl",
            date="2024-03-15T12:00:00Z",
            event_id="event_001",
        )
        assert captured_params["apiKey"] == "test_key"
        assert captured_params["eventIds"] == "event_001"
        assert "h2h" in captured_params["markets"]
        await client.close()

    async def test_current_odds_list_format(self) -> None:
        """Current odds endpoint returns a plain list, not wrapped in 'data'."""
        plain_list = [
            {
                "id": "ev1",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home Team", "price": 2.0},
                                    {"name": "Draw", "price": 3.0},
                                    {"name": "Away Team", "price": 4.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=plain_list)

        client = _make_odds_client(handler)
        events = await client.get_odds(sport="soccer_epl")
        assert len(events) == 1
        assert events[0]["bookmakers"][0]["key"] == "bet365"
        await client.close()


# ---------------------------------------------------------------------------
# SELECTED_BOOKMAKERS constant
# ---------------------------------------------------------------------------


def test_selected_bookmakers_has_five() -> None:
    assert len(SELECTED_BOOKMAKERS) == 5
    assert "betfair_exchange" in SELECTED_BOOKMAKERS
    assert "bet365" in SELECTED_BOOKMAKERS
    assert "sbobet" in SELECTED_BOOKMAKERS
    assert "1xbet" in SELECTED_BOOKMAKERS
    assert "draftkings" in SELECTED_BOOKMAKERS
