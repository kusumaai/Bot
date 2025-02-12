import pytest
from decimal import Decimal
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from exchanges.exchange_manager import ExchangeManager, RateLimiter
from execution.exchange_interface import ExchangeInterface
from utils.error_handler import ExchangeError
from risk.manager import RiskManager
from risk.limits import RiskLimits
from database.queries import DatabaseQueries


@pytest.fixture
def exchange_credentials():
    """Provide exchange credentials for testing."""
    return {
        'api_key': 'test_api_key',
        'api_secret': 'test_api_secret',
        'exchange_id': 'binance'
    }


@pytest.fixture
async def exchange_manager_fixture(exchange_credentials, logger, db_queries):
    """Provide a configured ExchangeManager fixture."""
    manager = ExchangeManager(
        exchange_id=exchange_credentials['exchange_id'],
        api_key=exchange_credentials['api_key'],
        api_secret=exchange_credentials['api_secret'],
        logger=logger,
        db_queries=db_queries
    )
    manager.exchange = AsyncMock()
    yield manager
    await manager.close()


@pytest.fixture
async def exchange_interface_fixture(exchange_manager_fixture, risk_manager, db_queries, logger):
    """Provide a configured ExchangeInterface fixture."""
    return ExchangeInterface(
        exchange_manager=exchange_manager_fixture,
        risk_manager=risk_manager,
        db_queries=db_queries,
        logger=logger
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_ticker_success(exchange_interface_fixture):
    """Test successful fetching of ticker data."""
    mock_ticker = {'symbol': 'BTC/USDT', 'price': '50000'}
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.return_value = mock_ticker
    
    ticker = await exchange_interface_fixture.get_ticker("BTC/USDT")
    assert ticker == mock_ticker
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.assert_awaited_once_with("BTC/USDT")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_ticker_exchange_error(exchange_interface_fixture):
    """Test fetching ticker data when exchange raises an error."""
    exchange_interface_fixture.exchange_manager.exchange.fetch_ticker.side_effect = ExchangeError("API Error")
    
    with pytest.raises(ExchangeError, match="API Error"):
        await exchange_interface_fixture.get_ticker("BTC/USDT")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_trade_limit_order_success(exchange_interface_fixture):
    """Test successful execution of a limit order."""
    mock_order = {
        'id': 'order123',
        'symbol': 'BTC/USDT',
        'status': 'open',
        'price': '50000',
        'amount': '0.1'
    }
    exchange_interface_fixture.exchange_manager.exchange.create_order.return_value = mock_order
    
    result = await exchange_interface_fixture.execute_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.1'),
        order_type='limit',
        price=Decimal('50000')
    )
    assert result['success'] is True
    assert result['order_id'] == 'order123'
    exchange_interface_fixture.exchange_manager.exchange.create_order.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_trade_market_order_success(exchange_interface_fixture):
    """Test successful execution of a market order."""
    mock_order = {
        'id': 'order124',
        'symbol': 'ETH/USDT',
        'status': 'filled',
        'price': '3000',
        'amount': '10'
    }
    exchange_interface_fixture.exchange_manager.exchange.create_order.return_value = mock_order
    
    result = await exchange_interface_fixture.execute_trade(
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('10'),
        order_type='market'
    )
    assert result['success'] is True
    assert result['order_id'] == 'order124'
    exchange_interface_fixture.exchange_manager.exchange.create_order.assert_awaited_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_order_success(exchange_interface_fixture):
    """Test successful cancellation of an existing order."""
    mock_order = {'id': 'order123', 'status': 'closed'}
    exchange_interface_fixture.exchange_manager.exchange.close_order.return_value = mock_order
    
    success = await exchange_interface_fixture.cancel_trade('order123')
    assert success is True
    exchange_interface_fixture.exchange_manager.exchange.close_order.assert_awaited_once_with('order123')


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_order_not_found(exchange_interface_fixture):
    """Test cancellation of a non-existent order."""
    exchange_interface_fixture.exchange_manager.exchange.close_order.return_value = None
    
    success = await exchange_interface_fixture.cancel_trade('invalid_order')
    assert success is False
    exchange_interface_fixture.exchange_manager.exchange.close_order.assert_awaited_once_with('invalid_order')


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_candles_success(exchange_interface_fixture, sample_candles):
    """Test successful fetching of candle data."""
    formatted_candles = [
        [c['timestamp'], c['open'], c['high'], c['low'], c['close'], c['volume']] for c in sample_candles
    ]
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.return_value = formatted_candles
    
    candles = await exchange_interface_fixture.fetch_candles('BTC/USDT', '15m', 20)
    assert len(candles) == 20
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with('BTC/USDT', '15m', limit=20)
    
    # Verify candle data
    for original, fetched in zip(sample_candles, candles):
        assert fetched['timestamp'] == original['timestamp']
        assert fetched['open'] == original['open']
        assert fetched['high'] == original['high']
        assert fetched['low'] == original['low']
        assert fetched['close'] == original['close']
        assert fetched['volume'] == original['volume']


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_candles_exchange_error(exchange_interface_fixture):
    """Test fetching candle data when exchange raises an error."""
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.side_effect = ExchangeError("API Error")
    
    candles = await exchange_interface_fixture.fetch_candles('BTC/USDT', '15m', 20)
    assert candles == []
    exchange_interface_fixture.exchange_manager.exchange.fetch_ohlcv.assert_awaited_once_with('BTC/USDT', '15m', limit=20) 