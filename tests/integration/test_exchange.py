#! /usr/bin/env python3
# tests/integration/test_exchange.py
"""
Module: tests.integration
Provides integration testing functionality for the exchange.
"""
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.queries import DatabaseQueries
from src.exchanges.exchange_manager import ExchangeManager, RateLimitConfig, RateLimiter
from src.execution.exchange_interface import ExchangeInterface
from src.risk.limits import RiskLimits
from src.risk.manager import RiskManager
from src.utils.error_handler import ExchangeError


@pytest.fixture
def logger():
    """Provide a logger instance."""
    return logging.getLogger("test")


@pytest.fixture
def risk_manager():
    """Provide a risk manager instance."""
    manager = MagicMock(spec=RiskManager)
    manager.validate_trade = AsyncMock(return_value=(True, None))
    return manager


@pytest.fixture
def db_queries():
    """Provide database queries instance."""
    queries = MagicMock(spec=DatabaseQueries)
    queries.log_trade = AsyncMock()
    return queries


@pytest.fixture
def sample_candles():
    """Provide sample candle data."""
    now = datetime.now()
    return [
        {
            "timestamp": int((now - timedelta(minutes=i * 15)).timestamp() * 1000),
            "open": "50000",
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "100",
        }
        for i in range(20)
    ]


@pytest.fixture
def exchange_credentials():
    """Provide exchange credentials for testing."""
    return {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "exchange_id": "binance",
    }


@pytest.fixture
def mock_exchange_manager(exchange_credentials, logger):
    """Provide a mocked ExchangeManager."""
    manager = ExchangeManager(
        exchange_id=exchange_credentials["exchange_id"],
        api_key=exchange_credentials["api_key"],
        api_secret=exchange_credentials["api_secret"],
        sandbox=True,
        logger=logger,
    )
    manager.exchange = AsyncMock()
    manager.initialize = AsyncMock(return_value=True)
    return manager


@pytest.fixture
async def exchange_interface_fixture(
    mock_exchange_manager, risk_manager, db_queries, logger
):
    """Provide a configured ExchangeInterface fixture."""
    ctx = MagicMock()
    ctx.logger = logger
    ctx.config = {
        "exchange_id": "binance",
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "paper_mode": True,
        "database": {"path": "data/test_trading.db"},
        "initial_balance": "10000",
    }
    ctx.risk_manager = risk_manager
    ctx.db_queries = db_queries
    ctx.exchange_manager = mock_exchange_manager

    exchange_interface = ExchangeInterface(ctx)
    await exchange_interface.initialize()
    return exchange_interface


# test fetch ticker success
@pytest.mark.integration
# test asyncio
@pytest.mark.asyncio
async def test_fetch_ticker_success(exchange_interface_fixture):
    """Test successful fetching of ticker data."""
    mock_ticker = {"symbol": "BTC/USDT", "price": "50000"}
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.return_value = (
        mock_ticker
    )

    ticker = await exchange_interface_fixture.get_ticker("BTC/USDT")
    assert ticker == mock_ticker
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.assert_awaited_once_with(
        "BTC/USDT"
    )


# test fetch ticker exchange error
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_ticker_exchange_error(exchange_interface_fixture):
    """Test fetching ticker data when exchange raises an error."""
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.side_effect = (
        ExchangeError("API Error")
    )

    with pytest.raises(ExchangeError, match="API Error"):
        await exchange_interface_fixture.get_ticker("BTC/USDT")


# test execute trade limit order success
@pytest.mark.integration
# test asyncio
@pytest.mark.asyncio
async def test_execute_trade_limit_order_success(exchange_interface_fixture):
    """Test successful execution of a limit order."""
    mock_order = {
        "id": "order123",
        "symbol": "BTC/USDT",
        "status": "open",
        "price": "50000",
        "amount": "0.1",
    }
    exchange_interface_fixture.exchange_manager.exchange.create_order.return_value = (
        mock_order
    )

    result = await exchange_interface_fixture.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        amount=Decimal("0.1"),
        order_type="limit",
        price=Decimal("50000"),
    )
    assert result["success"] is True
    assert result["order_id"] == "order123"
    exchange_interface_fixture.exchange_manager.exchange.create_order.assert_awaited_once()


# test execute trade market order success
@pytest.mark.integration
# test asyncio
@pytest.mark.asyncio
async def test_execute_trade_market_order_success(exchange_interface_fixture):
    """Test successful execution of a market order."""
    mock_order = {
        "id": "order124",
        "symbol": "ETH/USDT",
        "status": "filled",
        "price": "3000",
        "amount": "10",
    }
    exchange_interface_fixture.exchange_manager.exchange.create_order.return_value = (
        mock_order
    )

    result = await exchange_interface_fixture.execute_trade(
        symbol="ETH/USDT", side="sell", amount=Decimal("10"), order_type="market"
    )
    assert result["success"] is True
    assert result["order_id"] == "order124"
    exchange_interface_fixture.exchange_manager.exchange.create_order.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_order_success(exchange_interface_fixture):
    """Test successful cancellation of an existing order."""
    mock_order = {"id": "order123", "status": "closed"}
    exchange_interface_fixture.exchange_manager.exchange.close_order.return_value = (
        mock_order
    )

    success = await exchange_interface_fixture.cancel_trade("order123")
    assert success is True
    exchange_interface_fixture.exchange_manager.exchange.close_order.assert_awaited_once_with(
        "order123"
    )


# test cancel order not found
@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_order_not_found(exchange_interface_fixture):
    """Test cancellation of a non-existent order."""
    exchange_interface_fixture.exchange_manager.exchange.close_order.return_value = None

    success = await exchange_interface_fixture.cancel_trade("invalid_order")
    assert success is False
    exchange_interface_fixture.exchange_manager.exchange.close_order.assert_awaited_once_with(
        "invalid_order"
    )


# test fetch candles success
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_candles_success(exchange_interface_fixture, sample_candles):
    """Test successful fetching of candle data."""
    formatted_candles = [
        [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
        for c in sample_candles
    ]
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.return_value = (
        formatted_candles
    )

    candles = await exchange_interface_fixture.fetch_candles("BTC/USDT", "15m", 20)
    assert len(candles) == 20
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with(
        "BTC/USDT", "15m", limit=20
    )

    # Verify candle data
    for original, fetched in zip(sample_candles, candles):
        assert fetched["timestamp"] == original["timestamp"]
        assert fetched["open"] == original["open"]
        assert fetched["high"] == original["high"]
        assert fetched["low"] == original["low"]
        assert fetched["close"] == original["close"]
        assert fetched["volume"] == original["volume"]


# test fetch candles exchange error
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_candles_exchange_error(exchange_interface_fixture):
    """Test fetching candle data when exchange raises an error."""
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.side_effect = (
        ExchangeError("API Error")
    )

    candles = await exchange_interface_fixture.fetch_candles("BTC/USDT", "15m", 20)
    assert candles == []
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with(
        "BTC/USDT", "15m", limit=20
    )
