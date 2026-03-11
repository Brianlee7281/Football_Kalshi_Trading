"""Match engine runtime configuration, loaded from environment variables.

All configuration is injected by the orchestrator at container launch via
Docker environment variables.  No config files are read at runtime — the
container is stateless and reproducible.

Reference: docs/orchestration.md Container Entry Point
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class MatchEngineConfig:
    """Runtime configuration for a single match container.

    All fields are populated from environment variables by ``from_env()``.

    Attributes:
        match_id: Goalserve match identifier.
        trading_mode: ``"paper"`` or ``"live"``.
        param_version: Phase 1 parameter version pinned at container launch.
        kalshi_tickers: List of Kalshi market tickers for this match.
        db_url: asyncpg-compatible PostgreSQL DSN.
        redis_url: Redis URL (e.g. ``redis://redis:6379``).
        kalshi_api_key: Kalshi REST/WS API key.
        odds_api_key: The Odds API key for bet365 live odds.
        goalserve_api_key: Goalserve API key for live scores.
        fee_rate: Kalshi fee rate (default 0.07).
        z: Conservative-P z-score (default 1.645).
        K_frac: Fractional Kelly coefficient (default 0.25).
    """

    match_id: str
    trading_mode: str
    param_version: int
    kalshi_tickers: list[str]
    db_url: str
    redis_url: str
    kalshi_api_key: str
    odds_api_key: str
    goalserve_api_key: str
    fee_rate: float = 0.07
    z: float = 1.645
    K_frac: float = 0.25

    @classmethod
    def from_env(cls) -> MatchEngineConfig:
        """Build config from environment variables.

        Raises:
            KeyError: If a required environment variable is missing.
            ValueError: If PARAM_VERSION is not a valid integer, or
                        TRADING_MODE is not 'paper' or 'live'.
        """
        trading_mode = os.environ["TRADING_MODE"]
        if trading_mode not in ("paper", "live"):
            raise ValueError(
                f"TRADING_MODE must be 'paper' or 'live', got {trading_mode!r}"
            )

        param_version = int(os.environ["PARAM_VERSION"])

        tickers_raw = os.environ.get("KALSHI_TICKERS", "[]")
        kalshi_tickers: list[str] = json.loads(tickers_raw)

        return cls(
            match_id=os.environ["MATCH_ID"],
            trading_mode=trading_mode,
            param_version=param_version,
            kalshi_tickers=kalshi_tickers,
            db_url=os.environ["DB_URL"],
            redis_url=os.environ["REDIS_URL"],
            kalshi_api_key=os.environ["KALSHI_API_KEY"],
            odds_api_key=os.environ["ODDS_API_KEY"],
            goalserve_api_key=os.environ["GOALSERVE_API_KEY"],
            fee_rate=float(os.environ.get("FEE_RATE", "0.07")),
            z=float(os.environ.get("Z_SCORE", "1.645")),
            K_frac=float(os.environ.get("K_FRAC", "0.25")),
        )
