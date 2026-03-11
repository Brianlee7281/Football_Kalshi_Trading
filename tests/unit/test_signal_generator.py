"""Unit tests for signal_generator.py.

Tests cover:
  - execute_with_reservation: reserve→execute→confirm/release cycle
  - _publish_signal_to_redis: fire-and-forget publish
  - signal_generator loop: HOLD skips, order_allowed=False skips, fill updates bankroll
  - DB helper guard: no db_pool → returns None
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.types import FillResult, PaperFill, Signal
from src.execution.signal_generator import (
    _fill_price,
    _fill_quantity,
    _publish_signal_to_redis,
    execute_with_reservation,
    signal_generator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    direction: str = "BUY_YES",
    EV: float = 0.04,
    P_cons: float = 0.58,
    P_kalshi: float = 0.55,
    market_ticker: str = "SOCC-ABC-YES",
) -> Signal:
    return Signal(
        direction=direction,
        EV=EV,
        P_cons=P_cons,
        P_kalshi=P_kalshi,
        rough_qty=10,
        alignment_status="ALIGNED",
        kelly_multiplier=0.8,
        market_ticker=market_ticker,
    )


def _paper_fill(*, price: float = 0.55, quantity: int = 10) -> PaperFill:
    return PaperFill(
        price=price,
        quantity=quantity,
        timestamp=time.time(),
        is_paper=True,
        slippage=0.01,
        partial=False,
        fill_delay=1.0,
    )


def _fill_result(*, price: float = 0.55, quantity: int = 10) -> FillResult:
    return FillResult(
        success=True,
        fill_price=price,
        fill_quantity=quantity,
        order_id="ord-001",
    )


def _model(
    *,
    bankroll: float = 10_000.0,
    trading_mode: str = "paper",
    engine_phase: str = "FINISHED",
    active_tickers: list[str] | None = None,
    ticker_to_model_key: dict[str, str] | None = None,
    p_true_dict: dict[str, float] | None = None,
    has_db: bool = True,
    has_redis: bool = True,
    has_execution: bool = True,
) -> MagicMock:
    model = MagicMock()
    model.bankroll = bankroll
    model.trading_mode = trading_mode
    model.is_paper = trading_mode == "paper"
    model.engine_phase = engine_phase
    model.match_id = "match-001"
    model.active_tickers = active_tickers or []
    model.ticker_to_model_key = ticker_to_model_key or {}
    model.config.fee_rate = 0.07
    model.config.z = 1.645
    model.config.K_frac = 0.25
    model.bet365_implied = {}
    model.ob_syncs = {}
    model.db_pool = MagicMock() if has_db else None
    model.redis = MagicMock() if has_redis else None
    model.execution = MagicMock() if has_execution else None
    return model


# ---------------------------------------------------------------------------
# _fill_quantity / _fill_price
# ---------------------------------------------------------------------------


def test_fill_quantity_paper_fill() -> None:
    assert _fill_quantity(_paper_fill(quantity=15)) == 15


def test_fill_quantity_fill_result() -> None:
    assert _fill_quantity(_fill_result(quantity=7)) == 7


def test_fill_quantity_fill_result_none_quantity() -> None:
    fr = FillResult(success=False, fill_quantity=None)
    assert _fill_quantity(fr) == 0


def test_fill_price_paper_fill() -> None:
    assert _fill_price(_paper_fill(price=0.56)) == pytest.approx(0.56)


def test_fill_price_fill_result() -> None:
    assert _fill_price(_fill_result(price=0.55)) == pytest.approx(0.55)


def test_fill_price_fill_result_none_price() -> None:
    fr = FillResult(success=False, fill_price=None)
    assert _fill_price(fr) == 0.0


# ---------------------------------------------------------------------------
# execute_with_reservation — no infra
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_reservation_no_db_returns_none() -> None:
    """No db_pool → skip reservation, return None."""
    model = _model(has_db=False, has_redis=True)
    ob = MagicMock()
    result = await execute_with_reservation(_signal(), 100.0, ob, model)
    assert result is None


@pytest.mark.asyncio
async def test_execute_with_reservation_no_redis_returns_none() -> None:
    """No redis → skip reservation, return None."""
    model = _model(has_db=True, has_redis=False)
    ob = MagicMock()
    result = await execute_with_reservation(_signal(), 100.0, ob, model)
    assert result is None


# ---------------------------------------------------------------------------
# execute_with_reservation — reservation denied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_reservation_limit_exceeded_returns_none() -> None:
    """_reserve_exposure returns None (limit exceeded) → returns None."""
    model = _model()
    ob = MagicMock()

    with patch(
        "src.execution.signal_generator._reserve_exposure",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await execute_with_reservation(_signal(), 100.0, ob, model)

    assert result is None
    model.execution.submit_order.assert_not_called()


# ---------------------------------------------------------------------------
# execute_with_reservation — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_reservation_paper_fill_confirmed() -> None:
    """Successful fill → confirm_reservation called with actual amount."""
    fill = _paper_fill(price=0.56, quantity=10)
    model = _model()
    ob = MagicMock()
    model.execution.submit_order = AsyncMock(return_value=fill)

    with (
        patch(
            "src.execution.signal_generator._reserve_exposure",
            new_callable=AsyncMock,
            return_value=42,
        ),
        patch(
            "src.execution.signal_generator._confirm_reservation",
            new_callable=AsyncMock,
        ) as mock_confirm,
        patch(
            "src.execution.signal_generator._release_reservation",
            new_callable=AsyncMock,
        ) as mock_release,
    ):
        result = await execute_with_reservation(_signal(), 100.0, ob, model)

    assert result is fill
    mock_confirm.assert_called_once_with(model.db_pool, 42, pytest.approx(5.6))
    mock_release.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_reservation_no_fill_released() -> None:
    """None fill → release_reservation called, not confirm."""
    model = _model()
    ob = MagicMock()
    model.execution.submit_order = AsyncMock(return_value=None)

    with (
        patch(
            "src.execution.signal_generator._reserve_exposure",
            new_callable=AsyncMock,
            return_value=99,
        ),
        patch(
            "src.execution.signal_generator._confirm_reservation",
            new_callable=AsyncMock,
        ) as mock_confirm,
        patch(
            "src.execution.signal_generator._release_reservation",
            new_callable=AsyncMock,
        ) as mock_release,
    ):
        result = await execute_with_reservation(_signal(), 100.0, ob, model)

    assert result is None
    mock_confirm.assert_not_called()
    mock_release.assert_called_once_with(model.db_pool, 99)


@pytest.mark.asyncio
async def test_execute_with_reservation_exception_releases() -> None:
    """Exception during submit_order → release_reservation called, exception re-raised."""
    model = _model()
    ob = MagicMock()
    model.execution.submit_order = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch(
            "src.execution.signal_generator._reserve_exposure",
            new_callable=AsyncMock,
            return_value=77,
        ),
        patch(
            "src.execution.signal_generator._release_reservation",
            new_callable=AsyncMock,
        ) as mock_release,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await execute_with_reservation(_signal(), 100.0, ob, model)

    mock_release.assert_called_once_with(model.db_pool, 77)


@pytest.mark.asyncio
async def test_execute_with_reservation_no_execution_layer_releases() -> None:
    """model.execution is None → release_reservation, return None."""
    model = _model(has_execution=False)
    ob = MagicMock()

    with (
        patch(
            "src.execution.signal_generator._reserve_exposure",
            new_callable=AsyncMock,
            return_value=55,
        ),
        patch(
            "src.execution.signal_generator._release_reservation",
            new_callable=AsyncMock,
        ) as mock_release,
    ):
        result = await execute_with_reservation(_signal(), 100.0, ob, model)

    assert result is None
    mock_release.assert_called_once_with(model.db_pool, 55)


# ---------------------------------------------------------------------------
# _publish_signal_to_redis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_signal_to_redis_calls_publish_twice() -> None:
    """Publishes to signal:{match_id} and position_update channels."""
    model = _model()
    model.redis.publish = AsyncMock()
    fill = _paper_fill()
    sig = _signal()

    await _publish_signal_to_redis(model, "SOCC-ABC-YES", sig, fill)

    assert model.redis.publish.call_count == 2
    channels = [call.args[0] for call in model.redis.publish.call_args_list]
    assert "signal:match-001" in channels
    assert "position_update" in channels


@pytest.mark.asyncio
async def test_publish_signal_to_redis_no_redis_noop() -> None:
    """model.redis is None → no-op, no exception."""
    model = _model(has_redis=False)
    fill = _paper_fill()
    await _publish_signal_to_redis(model, "SOCC-ABC-YES", _signal(), fill)
    # no exception


@pytest.mark.asyncio
async def test_publish_signal_to_redis_suppresses_exception() -> None:
    """Redis error is suppressed (fire-and-forget)."""
    model = _model()
    model.redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))

    await _publish_signal_to_redis(model, "SOCC-ABC-YES", _signal(), _paper_fill())
    # no exception raised


# ---------------------------------------------------------------------------
# signal_generator loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_generator_exits_when_finished() -> None:
    """engine_phase=FINISHED before any tick → exits immediately."""
    model = _model(engine_phase="FINISHED")
    model.phase4_queue = asyncio.Queue()

    await signal_generator(model)  # should return without blocking


@pytest.mark.asyncio
async def test_signal_generator_skips_when_order_not_allowed() -> None:
    """order_allowed=False tick → no signal generated, loop advances."""
    from src.common.types import TickData

    model = _model(engine_phase="FIRST_HALF")
    model.phase4_queue = asyncio.Queue()

    tick = TickData(
        P_true={"home_win": 0.60},
        sigma_MC={"home_win": 0.002},
        order_allowed=False,
    )
    await model.phase4_queue.put(tick)

    # Set FINISHED then push a dummy tick so the loop wakes up and exits
    async def _set_finished() -> None:
        await asyncio.sleep(0.01)
        model.engine_phase = "FINISHED"
        await model.phase4_queue.put(
            TickData(P_true={}, sigma_MC={}, order_allowed=False)
        )

    await asyncio.gather(signal_generator(model), _set_finished())
    # No exceptions, no fills attempted


@pytest.mark.asyncio
async def test_signal_generator_bankroll_decremented_on_fill() -> None:
    """Successful fill → model.bankroll decremented by fill_cost."""
    from src.common.types import TickData

    ticker = "SOCC-M1-YES"
    model = _model(
        engine_phase="FIRST_HALF",
        bankroll=10_000.0,
        active_tickers=[ticker],
        ticker_to_model_key={ticker: "home_win"},
    )

    ob_mock = MagicMock()
    ob_mock.liquidity_ok.return_value = True
    model.ob_syncs = {ticker: ob_mock}

    tick = TickData(
        P_true={"home_win": 0.65},
        sigma_MC={"home_win": 0.002},
        order_allowed=True,
    )
    model.phase4_queue = asyncio.Queue()
    await model.phase4_queue.put(tick)

    fill = _paper_fill(price=0.56, quantity=10)

    with (
        patch(
            "src.execution.signal_generator.generate_signal",
            return_value=_signal(direction="BUY_YES"),
        ),
        patch(
            "src.execution.signal_generator.compute_kelly",
            return_value=0.01,
        ),
        patch(
            "src.execution.signal_generator.execute_with_reservation",
            new_callable=AsyncMock,
            return_value=fill,
        ),
        patch(
            "src.execution.signal_generator._publish_signal_to_redis",
            new_callable=AsyncMock,
        ),
    ):
        async def _set_finished() -> None:
            from src.common.types import TickData

            await asyncio.sleep(0.05)
            model.engine_phase = "FINISHED"
            await model.phase4_queue.put(
                TickData(P_true={}, sigma_MC={}, order_allowed=False)
            )

        await asyncio.gather(signal_generator(model), _set_finished())

    expected_bankroll = 10_000.0 - (0.56 * 10)
    assert model.bankroll == pytest.approx(expected_bankroll)


@pytest.mark.asyncio
async def test_signal_generator_hold_skips_execution() -> None:
    """HOLD signal → no execute_with_reservation call."""
    from src.common.types import TickData

    ticker = "SOCC-M1-YES"
    model = _model(
        engine_phase="FIRST_HALF",
        active_tickers=[ticker],
        ticker_to_model_key={ticker: "home_win"},
    )

    ob_mock = MagicMock()
    ob_mock.liquidity_ok.return_value = True
    model.ob_syncs = {ticker: ob_mock}

    tick = TickData(
        P_true={"home_win": 0.52},
        sigma_MC={"home_win": 0.002},
        order_allowed=True,
    )
    model.phase4_queue = asyncio.Queue()
    await model.phase4_queue.put(tick)

    with (
        patch(
            "src.execution.signal_generator.generate_signal",
            return_value=_signal(direction="HOLD", EV=0.0),
        ),
        patch(
            "src.execution.signal_generator.execute_with_reservation",
            new_callable=AsyncMock,
        ) as mock_exec,
    ):
        async def _set_finished_hold() -> None:
            from src.common.types import TickData

            await asyncio.sleep(0.05)
            model.engine_phase = "FINISHED"
            await model.phase4_queue.put(
                TickData(P_true={}, sigma_MC={}, order_allowed=False)
            )

        await asyncio.gather(signal_generator(model), _set_finished_hold())

    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_signal_generator_skips_no_ob_sync() -> None:
    """No ob_sync for ticker → skips that ticker."""
    from src.common.types import TickData

    ticker = "SOCC-M1-YES"
    model = _model(
        engine_phase="FIRST_HALF",
        active_tickers=[ticker],
        ticker_to_model_key={ticker: "home_win"},
    )
    model.ob_syncs = {}  # no ob_sync registered

    tick = TickData(
        P_true={"home_win": 0.65},
        sigma_MC={"home_win": 0.002},
        order_allowed=True,
    )
    model.phase4_queue = asyncio.Queue()
    await model.phase4_queue.put(tick)

    with patch(
        "src.execution.signal_generator.execute_with_reservation",
        new_callable=AsyncMock,
    ) as mock_exec:
        async def _set_finished_no_ob() -> None:
            from src.common.types import TickData

            await asyncio.sleep(0.05)
            model.engine_phase = "FINISHED"
            await model.phase4_queue.put(
                TickData(P_true={}, sigma_MC={}, order_allowed=False)
            )

        await asyncio.gather(signal_generator(model), _set_finished_no_ob())

    mock_exec.assert_not_called()
