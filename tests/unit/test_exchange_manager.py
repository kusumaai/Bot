import pytest
from decimal import Decimal
import logging
from unittest.mock import AsyncMock, MagicMock

from exchanges.exchange_manager import ExchangeManager, RateLimiter, RateLimitConfig
from utils.error_handler import ExchangeError, RateLimitExceeded
from database.queries import DatabaseQueries


@pytest.fixture
def rate_limit_config():
    """Provide a RateLimitConfig instance."""
    return RateLimitConfig(max_requests=5, time_window=60)


@pytest.fixture
def exchange_manager_fixture(logger, mocker):
    """Provide an ExchangeManager instance with mocked dependencies."""
    mock_db_queries = AsyncMock(spec=DatabaseQueries)
    manager = ExchangeManager(
        exchange_id='binance',
        api_key='test_key',
        api_secret='test_secret',
        logger=logger,
        db_queries=mock_db_queries
    )
    manager.exchange = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_exchange_manager_initialization(exchange_manager_fixture):
    """Test initialization of ExchangeManager."""
    await exchange_manager_fixture.initialize()
    exchange_manager_fixture.exchange.load_markets.assert_awaited_once()
    assert exchange_manager_fixture.exchange.loaded is True


@pytest.mark.asyncio
async def test_exchange_manager_initialize_failure(exchange_manager_fixture, mocker):
    """Test ExchangeManager initialization failure."""
    exchange_manager_fixture.exchange.initialize.side_effect = ExchangeError("Initialization Failed")
    
    with pytest.raises(ExchangeError, match="Initialization Failed"):
        await exchange_manager_fixture.initialize()


@pytest.mark.asyncio
async def test_rate_limiter_within_limits(rate_limit_config):
    """Test RateLimiter allows requests within limits."""
    limiter = RateLimiter(rate_limit_config)
    for _ in range(rate_limit_config.max_requests):
        assert await limiter.acquire() is True
    # Next request should wait
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire()


@pytest.mark.asyncio
async def test_exchange_manager_rate_limit(exchange_manager_fixture):
    """Test ExchangeManager rate limiting."""
    rate_limit_config = RateLimitConfig(max_requests=5, time_window=1)  # 5 requests per second
    limiter = RateLimiter(rate_limit_config)
    exchange_manager_fixture.rate_limiter = limiter
    
    # Mock fetch_ticker
    exchange_manager_fixture.exchange.fetch_ticker.return_value = {'symbol': 'BTC/USDT', 'price': '50000'}
    
    # Execute 5 fetch_ticker within rate limit
    for _ in range(5):
        ticker = await exchange_manager_fixture.get_ticker('BTC/USDT')
        assert ticker == {'symbol': 'BTC/USDT', 'price': '50000'}
    
    # 6th request should exceed rate limit
    with pytest.raises(RateLimitExceeded):
        await exchange_manager_fixture.get_ticker('BTC/USDT') 