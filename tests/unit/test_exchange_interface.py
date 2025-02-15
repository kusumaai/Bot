#! /usr/bin/env python3
#tests/unit/test_exchange_interface.py
"""
Module: tests.unit
Provides unit testing functionality for the exchange interface module.
""" 
import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from src.exchanges.exchange_manager import ExchangeManager
from src.execution.exchange_interface import ExchangeInterface
from src.utils.error_handler import ExchangeError
from src.risk.manager import RiskManager
from src.database.queries import DatabaseQueries


@pytest.fixture
def mock_exchange_manager():
    """Provide a mocked ExchangeManager"""
    manager = ExchangeManager(
        exchange_id='binance',
        api_key='test_key',
        api_secret='test_secret',
        logger=MagicMock(spec=logging.Logger),
        db_queries=AsyncMock(spec=DatabaseQueries)
    )
    manager.exchange = AsyncMock()
    return manager


@pytest.fixture
def mock_risk_manager():
    """Provide a mocked RiskManager"""
    return AsyncMock(spec=RiskManager)


@pytest.fixture
def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_exchange_interface():
    mock = MagicMock(spec=ExchangeInterface)
    mock.fetch_candles = AsyncMock(return_value=[{"timestamp": 1600000000000, "close": "50000"}])
    return mock


@pytest.fixture
def exchange_interface(mock_exchange_manager, mock_risk_manager, db_queries, logger):
    """Provide an ExchangeInterface instance with mocked dependencies."""
    return ExchangeInterface(
        exchange_manager=mock_exchange_manager,
        risk_manager=mock_risk_manager,
        db_queries=db_queries,
        logger=logger
    )


@pytest.mark.asyncio
async def test_execute_trade_success(exchange_interface):
    """Test successful trade execution."""
    mock_order = {
        'id': 'order124',
        'status': 'open',
        'symbol': 'ETH/USDT',
        'side': 'sell',
        'amount': '10',
        'price': '3000'
    }
    exchange_interface.exchange_manager.exchange.create_order.return_value = mock_order

    result = await exchange_interface.execute_trade(
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('10'),
        order_type='market',
        price=Decimal('3000')
    )
    assert result['success'] is True
    assert result['order_id'] == 'order124'
    exchange_interface.exchange_manager.exchange.create_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_risk_validation_failure(exchange_interface):
    """Test trade execution when risk validation fails."""
    # Mock risk validation to fail
    exchange_interface.risk_manager.validate_trade.return_value = False

    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('1'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result['success'] is False
    assert result['error'] == "Trade validation failed."
    exchange_interface.exchange_manager.exchange.create_order.assert_not_awaited()
    exchange_interface.risk_manager.validate_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_trade_exchange_error(exchange_interface):
    """Test trade execution when exchange raises an error."""
    exchange_interface.exchange_manager.exchange.create_order.side_effect = ExchangeError("Order Failed")

    result = await exchange_interface.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('1'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result['success'] is False
    assert result['error'] == "Order Failed"
    exchange_interface.exchange_manager.exchange.create_order.assert_awaited_once()
    exchange_interface.risk_manager.validate_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_trade_success(exchange_interface):
    """Test successful cancellation of a trade."""
    mock_order = {'id': 'order123', 'status': 'closed'}
    exchange_interface.exchange_manager.exchange.close_order.return_value = mock_order

    success = await exchange_interface.cancel_trade('order123')
    assert success is True
    exchange_interface.exchange_manager.exchange.close_order.assert_awaited_once_with('order123')


@pytest.mark.asyncio
async def test_cancel_trade_failure(exchange_interface):
    """Test cancellation of a non-existent trade."""
    exchange_interface.exchange_manager.exchange.close_order.return_value = None

    success = await exchange_interface.cancel_trade('invalid_order')
    assert success is False
    exchange_interface.exchange_manager.exchange.close_order.assert_awaited_once_with('invalid_order')


@pytest.mark.asyncio
async def test_fetch_candles_success(exchange_interface):
    """Test successful fetching of candle data."""
    mock_candles = [
        [1609459200000, 29000, 29500, 28900, 29400, 100],
        [1609460100000, 29400, 29800, 29300, 29700, 150],
    ]
    exchange_interface.exchange_manager.exchange.fetch_ohlcv.return_value = mock_candles

    candles = await exchange_interface.fetch_candles('BTC/USDT', '15m', 2)
    assert len(candles) == 2
    exchange_interface.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with('BTC/USDT', '15m', limit=2)


@pytest.mark.asyncio
async def test_fetch_candles_exchange_error(exchange_interface):
    """Test fetching candle data when exchange raises an error."""
    exchange_interface.exchange_manager.exchange.fetch_ohlcv.side_effect = ExchangeError("API Error")

    candles = await exchange_interface.fetch_candles('BTC/USDT', '15m', 20)
    assert candles == []
    exchange_interface.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with('BTC/USDT', '15m', limit=20) 