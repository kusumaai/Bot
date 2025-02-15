#! /usr/bin/env python3
# tests/unit/test_execution_main_loop.py
"""
Module: tests.unit
Provides unit testing functionality for the execution main loop module.
"""
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.queries import DatabaseQueries
from execution.exchange_interface import ExchangeInterface
from execution.main_loop import MainLoop
from risk.manager import RiskManager
from utils.logger import setup_logging


@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    return MagicMock(spec=ExchangeInterface)


@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    return MagicMock(spec=RiskManager)


@pytest.fixture
def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logger.Logger)


@pytest.fixture
def main_loop(mock_exchange_interface, mock_risk_manager, db_queries, logger):
    """Provide a MainLoop instance."""
    return MainLoop(
        exchange_interface=mock_exchange_interface,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger,
    )


@pytest.mark.asyncio
async def test_main_loop_run(main_loop):
    """Test running the main loop."""
    main_loop.execute_cycle = AsyncMock(return_value=True)
    main_loop.shutdown = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = asyncio.Future()
        mock_sleep.return_value.set_result(None)

        # Run a single cycle
        await main_loop.run_cycle()
        main_loop.execute_cycle.assert_awaited_once()
        mock_sleep.assert_awaited_once_with(1)  # Assuming loop interval is 1 second


@pytest.mark.asyncio
async def test_main_loop_execute_cycle_success(main_loop):
    """Test successful execution cycle."""
    main_loop.exchange_interface.fetch_market_data.return_value = {
        "BTC/USDT": {"price": Decimal("50000")}
    }
    main_loop.risk_manager.validate_market_conditions.return_value = True
    main_loop.exchange_interface.place_trade.return_value = {
        "success": True,
        "order_id": "order006",
    }
    main_loop.db_queries.log_trade.return_value = True

    result = await main_loop.execute_cycle()
    assert result is True
    main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
    main_loop.risk_manager.validate_market_conditions.assert_awaited_once_with(
        {"BTC/USDT": {"price": Decimal("50000")}}
    )
    main_loop.exchange_interface.place_trade.assert_awaited_once()
    main_loop.db_queries.log_trade.assert_awaited_once_with(
        "order006",
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": Decimal("1"),
            "price": Decimal("50000"),
        },
    )


@pytest.mark.asyncio
async def test_main_loop_execute_cycle_risk_failure(main_loop):
    """Test execution cycle when risk validation fails."""
    main_loop.exchange_interface.fetch_market_data.return_value = {
        "BTC/USDT": {"price": Decimal("50000")}
    }
    main_loop.risk_manager.validate_market_conditions.return_value = False

    result = await main_loop.execute_cycle()
    assert result is False
    main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
    main_loop.risk_manager.validate_market_conditions.assert_awaited_once_with(
        {"BTC/USDT": {"price": Decimal("50000")}}
    )
    main_loop.exchange_interface.place_trade.assert_not_awaited()
    main_loop.db_queries.log_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_loop_execute_cycle_trade_failure(main_loop):
    """Test execution cycle when trade placement fails."""
    main_loop.exchange_interface.fetch_market_data.return_value = {
        "BTC/USDT": {"price": Decimal("50000")}
    }
    main_loop.risk_manager.validate_market_conditions.return_value = True
    main_loop.exchange_interface.place_trade.return_value = {
        "success": False,
        "error": "Trade Failed",
    }

    result = await main_loop.execute_cycle()
    assert result is False
    main_loop.exchange_interface.fetch_market_data.assert_awaited_once()
    main_loop.risk_manager.validate_market_conditions.assert_awaited_once_with(
        {"BTC/USDT": {"price": Decimal("50000")}}
    )
    main_loop.exchange_interface.place_trade.assert_awaited_once()
    main_loop.db_queries.log_trade.assert_not_awaited()
    main_loop.logger.error.assert_called_with("Trade placement failed: Trade Failed")


@pytest.mark.asyncio
async def test_main_loop_shutdown(main_loop):
    """Test proper shutdown of the main loop."""
    await main_loop.shutdown()
    main_loop.logger.info.assert_called_with("✨ MainLoop has been shutdown.")
