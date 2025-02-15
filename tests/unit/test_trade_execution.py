#! /usr/bin/env python3
# tests/unit/test_trade_execution.py
"""
Module: tests.unit
Provides unit testing functionality for the trade execution module.
"""
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from database.queries import DatabaseQueries
from exchanges.exchange_manager import ExchangeManager
from execution.exchange_interface import ExchangeInterface
from risk.manager import RiskManager
from utils.error_handler import ExchangeError
from utils.exceptions import DatabaseError


# test trade execution functionality
@pytest.fixture
def mock_exchange_manager():
    """Provide a mocked ExchangeManager."""
    manager = ExchangeManager(
        exchange_id="binance",
        api_key="test_key",
        api_secret="test_secret",
        logger=logging.getLogger("TestExchangeManager"),
        sandbox=True,
    )
    manager.exchange = AsyncMock()
    return manager


# test risk manager functionality
@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager."""
    mock_risk = MagicMock(spec=RiskManager)
    mock_risk.validate_trade = AsyncMock(return_value=(True, None))
    return mock_risk


# test database queries functionality
@pytest.fixture
def mock_db_queries():
    """Provide a mocked DatabaseQueries."""
    mock_db = AsyncMock(spec=DatabaseQueries)
    mock_db.log_trade = AsyncMock()
    return mock_db


# test logger functionality
@pytest.fixture
def mock_logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


# test exchange interface functionality
@pytest.fixture
def exchange_interface(
    mock_exchange_manager, mock_risk_manager, mock_db_queries, mock_logger
):
    """Provide an ExchangeInterface instance with mocked dependencies."""
    ctx = MagicMock()
    ctx.logger = mock_logger
    ctx.risk_manager = mock_risk_manager
    ctx.db_queries = mock_db_queries
    ctx.config = {
        "exchange_id": "binance",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "paper_mode": True,
        "database": {"path": "data/test_trading.db"},
        "initial_balance": "10000",
        "rate_limit_per_second": 5,
    }
    # test exchange interface functionality
    exchange_interface = ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=mock_db_queries,
        logger=mock_logger,
    )
    # test exchange interface functionality
    exchange_interface.exchange_manager.exchange.create_order = AsyncMock(
        return_value={
            "id": "order_test",
            "symbol": "BTC/USDT",
            "status": "open",
            "price": "50000",
            "amount": "0.1",
        }
    )
    return exchange_interface


# test execute trade within risk limits functionality
@pytest.mark.asyncio
async def test_execute_trade_within_risk_limits(exchange_interface):
    """Test executing a trade that is within risk limits."""
    mock_risk_manager = MagicMock(spec=RiskManager)
    mock_risk_manager.validate_trade = AsyncMock(return_value=(True, None))
    # test execute trade within risk limits functionality
    result = await exchange_interface.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        amount=Decimal("0.1"),
        order_type="limit",
        price=Decimal("50000"),
    )

    assert result["success"] is True
    assert result["order_id"] == "order_test"
    exchange_interface.exchange_manager.exchange.create_order.assert_awaited_once()
    mock_risk_manager.validate_trade.assert_awaited_once()
    exchange_interface.db_queries.log_trade.assert_awaited_once()


# test execute trade exceeds risk limits functionality
@pytest.mark.asyncio
async def test_execute_trade_exceeds_risk_limits(exchange_interface):
    """Test executing a trade that exceeds risk limits."""
    mock_risk_manager = MagicMock(spec=RiskManager)
    mock_risk_manager.validate_trade = AsyncMock(
        return_value=(False, "Trade validation failed.")
    )

    result = await exchange_interface.execute_trade(
        symbol="ETH/USDT", side="sell", amount=Decimal("10"), order_type="market"
    )

    assert result["success"] is False
    assert result["error"] == "Trade validation failed."
    exchange_interface.exchange_manager.exchange.create_order.assert_not_awaited()
    mock_risk_manager.validate_trade.assert_awaited_once()
    exchange_interface.db_queries.log_trade.assert_not_awaited()


# test execute trade exchange failure functionality to test the exchange error handling in order to handle the exchange error
@pytest.mark.asyncio
async def test_execute_trade_exchange_failure(exchange_interface):
    """Test executing a trade when exchange raises an error."""
    mock_risk_manager = MagicMock(spec=RiskManager)
    mock_risk_manager.validate_trade = AsyncMock(return_value=(True, None))
    mock_exchange_manager = MagicMock()
    mock_exchange_manager.exchange.create_order.side_effect = ExchangeError(
        "Order Failed"
    )

    result = await exchange_interface.execute_trade(
        symbol="ETH/USDT",
        side="buy",
        amount=Decimal("5"),
        order_type="limit",
        price=Decimal("3000"),
    )

    assert result["success"] is False
    assert result["error"] == "Order Failed"


# test execute trade database failure functionality to test the database error handling in order to handle the database error
@pytest.mark.asyncio
async def test_execute_trade_database_failure(exchange_interface):
    """Test executing a trade when database raises an error."""
    mock_risk_manager = MagicMock(spec=RiskManager)
    mock_risk_manager.validate_trade = AsyncMock(return_value=(True, None))
    mock_db_queries = MagicMock()
    mock_db_queries.log_trade.side_effect = DatabaseError("Log trade failed")

    result = await exchange_interface.execute_trade(
        symbol="BTC/USDT",
        side="sell",
        amount=Decimal("2"),
        order_type="market",
        price=Decimal("40000"),
    )
    # test execute trade database failure functionality to test the database error handling in order to handle the database error
    assert result["success"] is False
    assert result["error"] == "Log trade failed"
    # test execute trade database failure functionality to test the database error handling in order to handle the database error
    mock_exchange_manager.exchange.create_order.assert_awaited_once()
    mock_db_queries.log_trade.assert_awaited_once()
    mock_risk_manager.validate_trade.assert_awaited_once()
    exchange_interface.db_queries.log_trade.assert_not_awaited()
