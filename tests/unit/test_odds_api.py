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
    _extract_ml_odds,
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
def event_odds_data() -> dict[str, Any]:
    with open(FIXTURES_DIR / "odds_api_historical.json") as f:
        return json.load(f)


@pytest.fixture
def arsenal_bookmakers(event_odds_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Arsenal vs Chelsea bookmakers from the fixture."""
    return event_odds_data["bookmakers"]


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
# _extract_ml_odds
# ---------------------------------------------------------------------------


class TestExtractMlOdds:
    def test_extracts_from_ml_market(self) -> None:
        markets = [
            {"name": "ML", "odds": [{"home": "2.10", "draw": "3.40", "away": "3.20"}]}
        ]
        h, d, a = _extract_ml_odds(markets)
        assert h == pytest.approx(2.10)
        assert d == pytest.approx(3.40)
        assert a == pytest.approx(3.20)

    def test_ignores_non_ml(self) -> None:
        markets = [
            {"name": "Totals", "odds": [{"hdp": 2.5, "over": "1.90", "under": "1.90"}]}
        ]
        h, d, a = _extract_ml_odds(markets)
        assert h == 0.0

    def test_empty_markets(self) -> None:
        assert _extract_ml_odds([]) == (0.0, 0.0, 0.0)

    def test_empty_odds_list(self) -> None:
        markets = [{"name": "ML", "odds": []}]
        assert _extract_ml_odds(markets) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# build_odds_features
# ---------------------------------------------------------------------------


class TestBuildOddsFeatures:
    def test_with_betfair_exchange(
        self, arsenal_bookmakers: dict[str, list[dict[str, Any]]]
    ) -> None:
        features = build_odds_features(arsenal_bookmakers)
        # Betfair Exchange: home=1.44, draw=3.50, away=12.00
        assert abs(features["exchange_home_prob"] - 0.653) < 0.001
        assert features["exchange_draw_prob"] > 0
        assert features["exchange_away_prob"] > 0
        total = (
            features["exchange_home_prob"]
            + features["exchange_draw_prob"]
            + features["exchange_away_prob"]
        )
        assert abs(total - 1.0) < 1e-10

    def test_market_avg_populated(
        self, arsenal_bookmakers: dict[str, list[dict[str, Any]]]
    ) -> None:
        features = build_odds_features(arsenal_bookmakers)
        assert features["market_avg_home_prob"] > 0
        assert features["market_avg_draw_prob"] > 0
        assert features["bookmaker_odds_std"] >= 0

    def test_without_betfair_exchange_falls_back(self) -> None:
        """When Betfair Exchange is absent, exchange_*_prob = market average."""
        bookmakers: dict[str, list[dict[str, Any]]] = {
            "Bet365": [
                {"name": "ML", "odds": [{"home": "2.00", "draw": "3.00", "away": "4.00"}]}
            ],
            "SBOBet": [
                {"name": "ML", "odds": [{"home": "2.10", "draw": "3.10", "away": "3.80"}]}
            ],
        }
        features = build_odds_features(bookmakers)
        assert features["exchange_home_prob"] == pytest.approx(
            features["market_avg_home_prob"], abs=1e-10
        )

    def test_empty_bookmakers(self) -> None:
        features = build_odds_features({})
        assert features["exchange_home_prob"] == 0.0
        assert features["market_avg_home_prob"] == 0.0

    def test_bookmaker_without_ml_skipped(self) -> None:
        bookmakers: dict[str, list[dict[str, Any]]] = {
            "Betfair Exchange": [
                {"name": "Totals", "odds": [{"hdp": 2.5, "over": "1.90", "under": "1.90"}]}
            ],
            "Bet365": [
                {"name": "ML", "odds": [{"home": "2.00", "draw": "3.00", "away": "4.00"}]}
            ],
        }
        features = build_odds_features(bookmakers)
        # Betfair Exchange had no ML, so falls back to Bet365 average
        assert features["exchange_home_prob"] > 0
        assert features["bookmaker_odds_std"] == 0.0  # only 1 bookmaker with ML


# ---------------------------------------------------------------------------
# extract_bet365_implied_probs
# ---------------------------------------------------------------------------


class TestExtractBet365ImpliedProbs:
    def test_bet365_present(
        self, arsenal_bookmakers: dict[str, list[dict[str, Any]]]
    ) -> None:
        probs = extract_bet365_implied_probs(arsenal_bookmakers)
        assert probs is not None
        assert probs["home_win"] > 0
        assert probs["draw"] > 0
        assert probs["away_win"] > 0
        assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-10

    def test_bet365_absent(self) -> None:
        bookmakers: dict[str, list[dict[str, Any]]] = {
            "Betfair Exchange": [
                {"name": "ML", "odds": [{"home": "2.0", "draw": "3.0", "away": "4.0"}]}
            ]
        }
        assert extract_bet365_implied_probs(bookmakers) is None

    def test_empty_bookmakers(self) -> None:
        assert extract_bet365_implied_probs({}) is None


# ---------------------------------------------------------------------------
# _filter_bookmakers
# ---------------------------------------------------------------------------


class TestFilterBookmakers:
    def test_filters_to_selected(self, event_odds_data: dict[str, Any]) -> None:
        filtered = _filter_bookmakers(event_odds_data)
        for name in filtered["bookmakers"]:
            assert name in SELECTED_BOOKMAKERS

    def test_pinnacle_removed(self, event_odds_data: dict[str, Any]) -> None:
        assert "Pinnacle" in event_odds_data["bookmakers"]
        filtered = _filter_bookmakers(event_odds_data)
        assert "Pinnacle" not in filtered["bookmakers"]

    def test_five_bookmakers_kept(self, event_odds_data: dict[str, Any]) -> None:
        filtered = _filter_bookmakers(event_odds_data)
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
        assert delta == pytest.approx(0.10, abs=0.001)
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
        base_url="https://api.odds-api.io/v3",
        transport=httpx.MockTransport(handler),
    )
    return client


class TestOddsApiClientGetEvents:
    async def test_parses_event_list(self) -> None:
        events_list = [
            {
                "id": 100001,
                "home": "Arsenal",
                "away": "Chelsea",
                "date": "2024-03-16T15:00:00Z",
                "status": "pending",
            },
            {
                "id": 100002,
                "home": "Liverpool",
                "away": "Man City",
                "date": "2024-03-16T17:30:00Z",
                "status": "pending",
            },
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=events_list)

        client = _make_odds_client(handler)
        events = await client.get_events(sport="football")
        assert len(events) == 2
        assert events[0]["home"] == "Arsenal"
        await client.close()

    async def test_api_key_in_params(self) -> None:
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.update(dict(request.url.params))
            return httpx.Response(200, json=[])

        client = _make_odds_client(handler)
        await client.get_events(sport="football", league="england-premier-league")
        assert captured_params["apiKey"] == "test_key"
        assert captured_params["sport"] == "football"
        assert captured_params["league"] == "england-premier-league"
        await client.close()


class TestOddsApiClientGetOdds:
    async def test_parses_odds_response(
        self, event_odds_data: dict[str, Any]
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=event_odds_data)

        client = _make_odds_client(handler)
        result = await client.get_odds(event_id=100001, bookmakers="Bet365,Betfair Exchange")
        # Pinnacle filtered out, 5 selected kept
        assert len(result["bookmakers"]) == 5
        assert "Pinnacle" not in result["bookmakers"]
        await client.close()

    async def test_odds_params_sent(self, event_odds_data: dict[str, Any]) -> None:
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.update(dict(request.url.params))
            return httpx.Response(200, json=event_odds_data)

        client = _make_odds_client(handler)
        await client.get_odds(event_id=100001, bookmakers="Bet365")
        assert captured_params["apiKey"] == "test_key"
        assert captured_params["eventId"] == "100001"
        assert captured_params["bookmakers"] == "Bet365"
        await client.close()


class TestOddsApiClientGetOddsMovements:
    async def test_parses_movements(self) -> None:
        movements_data = {
            "eventid": "100001",
            "bookmaker": "Bet365",
            "opening": {"home": 2.0, "draw": 3.2, "away": 3.5, "timestamp": 1734400000},
            "latest": {"home": 2.1, "draw": 3.3, "away": 3.4, "timestamp": 1734450000},
            "movements": [
                {"home": 2.0, "draw": 3.2, "away": 3.5, "timestamp": 1734400000},
                {"home": 2.1, "draw": 3.3, "away": 3.4, "timestamp": 1734450000},
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=movements_data)

        client = _make_odds_client(handler)
        result = await client.get_odds_movements(event_id=100001, bookmaker="Bet365")
        assert result["opening"]["home"] == 2.0
        assert len(result["movements"]) == 2
        await client.close()


# ---------------------------------------------------------------------------
# SELECTED_BOOKMAKERS constant
# ---------------------------------------------------------------------------


def test_selected_bookmakers_has_five() -> None:
    assert len(SELECTED_BOOKMAKERS) == 5
    assert "Betfair Exchange" in SELECTED_BOOKMAKERS
    assert "Bet365" in SELECTED_BOOKMAKERS
    assert "SBOBet" in SELECTED_BOOKMAKERS
    assert "1xBet" in SELECTED_BOOKMAKERS
    assert "DraftKings" in SELECTED_BOOKMAKERS
