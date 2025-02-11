import pytest
from decimal import Decimal
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, Any

from exchanges.exchange_manager import ExchangeManager, RateLimiter
from execution.exchange_interface import ExchangeInterface
from utils.error_handler import ExchangeError
from risk.manager import RiskManager
from risk.limits import RiskLimits

@pytest.fixture
def exchange_credentials():
    """Get exchange credentials from environment"""
    return {
        'api_key': os.getenv('TEST_EXCHANGE_API_KEY'),
        'api_secret': os.getenv('TEST_EXCHANGE_SECRET'),
        'exchange_id': os.getenv('TEST_EXCHANGE_ID', 'binance')
    }

@pytest.fixture
async def exchange_manager(exchange_credentials, logger, db_queries):
    """Provide configured exchange manager"""
    manager = ExchangeManager(
        exchange_id=exchange_credentials['exchange_id'],
        api_key=exchange_credentials['api_key'],
        api_secret=exchange_credentials['api_secret'],
        logger=logger,
        db_queries=db_queries
    )
    yield manager
    await manager.close()

@pytest.fixture
async def exchange_interface(
    exchange_manager,
    risk_manager,
    db_queries,
    logger
) -> ExchangeInterface:
    """Provide configured exchange interface"""
    return ExchangeInterface(
        exchange_manager=exchange_manager,
        risk_manager=risk_manager,
        db_queries=db_queries,
        logger=logger
    )

@pytest.mark.integration
async def test_rate_limiter():
    """Test rate limiter functionality"""
    limiter = RateLimiter({
        'test': RateLimit(max_requests=2, time_window=1)
    })
    
    # First two requests should be immediate
    start_time = datetime.utcnow()
    await limiter.acquire('test')
    await limiter.acquire('test')
    
    # Third request should wait
    await limiter.acquire('test')
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    assert elapsed >= 1.0

@pytest.mark.integration
async def test_market_data_fetching(exchange_manager):
    """Test market data retrieval"""
    # Test markets loading
    markets = await exchange_manager.get_markets()
    assert isinstance(markets, dict)
    assert len(markets) > 0
    
    # Test ticker fetching
    symbol = "BTC/USDT"
    ticker = await exchange_manager.fetch_ticker(symbol)
    assert isinstance(ticker, dict)
    assert 'last' in ticker
    assert 'volume' in ticker
    
    # Test caching
    cached_markets = await exchange_manager.get_markets()
    assert id(cached_markets) == id(markets)  # Should return cached copy

@pytest.mark.integration
async def test_order_validation(exchange_interface):
    """Test order validation and execution"""
    symbol = "BTC/USDT"
    
    # Test invalid order parameters
    result = await exchange_interface.execute_trade(
        symbol=symbol,
        side='invalid',
        amount=Decimal('0.001')
    )
    assert not result.success
    assert result.error is not None
    
    # Test minimum size validation
    result = await exchange_interface.execute_trade(
        symbol=symbol,
        side='buy',
        amount=Decimal('0.000001')
    )
    assert not result.success
    assert 'size' in result.error.lower()

@pytest.mark.integration
async def test_position_management(exchange_interface, db_queries):
    """Test position opening and closing"""
    symbol = "BTC/USDT"
    
    # Open test position
    result = await exchange_interface.execute_trade(
        symbol=symbol,
        side='buy',
        amount=Decimal('0.001'),
        order_type='limit',
        price=Decimal('1000')  # Use unrealistic price to avoid execution
    )
    assert result.success
    assert result.order_id is not None
    
    # Verify position storage
    positions = await db_queries.get_active_positions(symbol)
    assert len(positions) > 0
    
    # Test position closure
    if positions:
        close_result = await exchange_interface.close_position(
            symbol=symbol,
            position_id=positions[0]['id']
        )
        assert close_result.success

@pytest.mark.integration
async def test_error_handling(exchange_manager):
    """Test error handling for various scenarios"""
    
    # Test invalid symbol
    with pytest.raises(ExchangeError):
        await exchange_manager.fetch_ticker("INVALID/PAIR")
    
    # Test authentication error
    bad_manager = ExchangeManager(
        exchange_id='binance',
        api_key='invalid',
        api_secret='invalid'
    )
    with pytest.raises(ExchangeError):
        await bad_manager.fetch_balance()
    await bad_manager.close()

@pytest.mark.integration
async def test_market_data_caching(exchange_interface):
    """Test market data caching mechanism"""
    symbol = "BTC/USDT"
    
    # First fetch
    ticker1 = await exchange_interface.get_ticker(symbol)
    
    # Immediate second fetch should return cached data
    ticker2 = await exchange_interface.get_ticker(symbol)
    assert id(ticker1) == id(ticker2)
    
    # Wait for cache to expire
    await asyncio.sleep(6)
    ticker3 = await exchange_interface.get_ticker(symbol)
    assert id(ticker1) != id(ticker3)

@pytest.mark.integration
async def test_risk_integration(exchange_interface, risk_manager):
    """Test integration between risk management and execution"""
    symbol = "BTC/USDT"
    
    # Test position size calculation
    account_size = Decimal('10000')
    size = await risk_manager.calculate_position_size(
        symbol=symbol,
        account_size=account_size
    )
    
    # Attempt trade with calculated size
    result = await exchange_interface.execute_trade(
        symbol=symbol,
        side='buy',
        amount=size,
        order_type='limit',
        price=Decimal('1000')  # Use unrealistic price to avoid execution
    )
    assert result.success 