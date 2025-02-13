from unittest.mock import AsyncMock
import pytest
from decimal import Decimal
from datetime import datetime, timezone
import pandas as pd

from src.execution.market_data import MarketData
from src.database.queries import DatabaseQueries
from src.utils.error_handler import ExchangeError
from src.signals.market_state import prepare_market_state, MarketState


@pytest.fixture
def mock_db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def market_data_fixture(mock_db_queries, logger):
    """Provide a MarketData instance with mocked dependencies."""
    return MarketData(
        db_queries=mock_db_queries,
        logger=logger
    )


@pytest.mark.asyncio
async def test_fetch_market_data_success(market_data_fixture):
    """Test successful fetching of market data."""
    # Mock fetch_ticker for multiple symbols
    market_data_fixture.exchange_interface.get_ticker = AsyncMock(return_value={'symbol': 'BTC/USDT', 'price': '50000'})
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    await market_data_fixture.update_market_data(symbols)
    
    assert market_data_fixture.data['BTC/USDT']['price'] == Decimal('50000')
    assert market_data_fixture.last_update['BTC/USDT'] == pytest.approx(datetime.now(tz=timezone.utc).timestamp(), rel=1e-3)
    market_data_fixture.exchange_interface.get_ticker.assert_any_await('BTC/USDT')
    market_data_fixture.exchange_interface.get_ticker.assert_any_await('ETH/USDT')


@pytest.mark.asyncio
async def test_fetch_market_data_exchange_error(market_data_fixture):
    """Test fetching market data when exchange raises an error."""
    market_data_fixture.exchange_interface.get_ticker = AsyncMock(side_effect=ExchangeError("API Failure"))
    
    symbols = ['BTC/USDT']
    await market_data_fixture.update_market_data(symbols)
    
    assert 'BTC/USDT' not in market_data_fixture.data
    market_data_fixture.logger.error.assert_called_with("Failed to fetch market data for BTC/USDT: API Failure")


@pytest.mark.asyncio
async def test_get_market_price(market_data_fixture):
    """Test retrieving market price from MarketData."""
    market_data_fixture.data['BTC/USDT'] = {'price': Decimal('50000')}
    
    price = market_data_fixture.get_price('BTC/USDT')
    assert price == Decimal('50000')
    
    # Test retrieving price for non-existent symbol
    price_none = market_data_fixture.get_price('SOL/USDT')
    assert price_none is None


@pytest.mark.asyncio
async def test_market_data_validation(market_data_fixture):
    """Test validation of fetched market data."""
    market_data_fixture.data['BTC/USDT'] = {'price': Decimal('50000'), 'volume': Decimal('10000')}
    
    is_valid = market_data_fixture.validate_market_data('BTC/USDT')
    assert is_valid is True
    
    # Invalidate market data
    market_data_fixture.data['BTC/USDT'] = {'price': Decimal('-50000'), 'volume': Decimal('10000')}
    is_valid = market_data_fixture.validate_market_data('BTC/USDT')
    assert is_valid is False
    market_data_fixture.logger.warning.assert_called_with("Invalid price for BTC/USDT: -50000")


def test_market_state_preparation():
    # Create test data
    data = pd.DataFrame({
        'high': [100, 101, 102],
        'low': [98, 97, 99],
        'close': [99, 100, 101],
        'volume': [1000, 1100, 900]
    })
    
    market_state = prepare_market_state(data)
    
    assert isinstance(market_state, MarketState)
    assert market_state.trend in ['bullish', 'bearish']
    assert isinstance(market_state.volatility, float)
    assert isinstance(market_state.volume, float) 