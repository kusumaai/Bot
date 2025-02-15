#! /usr/bin/env python3
# tests/unit/test_market_data.py
"""
Module: tests.unit
Provides unit testing functionality for the market data module.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from database.queries import DatabaseQueries
from execution.market_data import MarketData
from signals.market_state import MarketState, prepare_market_state
from utils.error_handler import ExchangeError, MarketDataError


@pytest.fixture
def mock_db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def market_data_fixture(mock_db_queries, logger):
    """Provide a MarketData instance with mocked dependencies."""
    return MarketData(db_queries=mock_db_queries, logger=logger)


@pytest.mark.asyncio
async def test_fetch_market_data_success(market_data_fixture):
    """Test successful fetching of market data."""
    # Mock fetch_ticker for multiple symbols
    market_data_fixture.exchange_interface.get_ticker = AsyncMock(
        return_value={"symbol": "BTC/USDT", "price": "50000"}
    )

    symbols = ["BTC/USDT", "ETH/USDT"]
    await market_data_fixture.update_market_data(symbols)

    assert market_data_fixture.data["BTC/USDT"]["price"] == Decimal("50000")
    assert market_data_fixture.last_update["BTC/USDT"] == pytest.approx(
        datetime.now(tz=timezone.utc).timestamp(), rel=1e-3
    )
    market_data_fixture.exchange_interface.get_ticker.assert_any_await("BTC/USDT")
    market_data_fixture.exchange_interface.get_ticker.assert_any_await("ETH/USDT")


@pytest.mark.asyncio
async def test_fetch_market_data_exchange_error(market_data_fixture):
    """Test fetching market data when exchange raises an error."""
    market_data_fixture.exchange_interface.get_ticker = AsyncMock(
        side_effect=ExchangeError("API Failure")
    )

    symbols = ["BTC/USDT"]
    await market_data_fixture.update_market_data(symbols)

    assert "BTC/USDT" not in market_data_fixture.data
    market_data_fixture.logger.error.assert_called_with(
        "Failed to fetch market data for BTC/USDT: API Failure"
    )


@pytest.mark.asyncio
async def test_get_market_price(market_data_fixture):
    """Test retrieving market price from MarketData."""
    market_data_fixture.data["BTC/USDT"] = {"price": Decimal("50000")}

    price = market_data_fixture.get_price("BTC/USDT")
    assert price == Decimal("50000")

    # Test retrieving price for non-existent symbol
    price_none = market_data_fixture.get_price("SOL/USDT")
    assert price_none is None


@pytest.mark.asyncio
async def test_market_data_validation(market_data_fixture):
    """Test validation of fetched market data."""
    market_data_fixture.data["BTC/USDT"] = {
        "price": Decimal("50000"),
        "volume": Decimal("10000"),
    }

    is_valid = market_data_fixture.validate_market_data("BTC/USDT")
    assert is_valid is True

    # Invalidate market data
    market_data_fixture.data["BTC/USDT"] = {
        "price": Decimal("-50000"),
        "volume": Decimal("10000"),
    }
    is_valid = market_data_fixture.validate_market_data("BTC/USDT")
    assert is_valid is False
    market_data_fixture.logger.warning.assert_called_with(
        "Invalid price for BTC/USDT: -50000"
    )


def test_market_state_preparation():
    # Create test data
    data = pd.DataFrame(
        {
            "high": [100, 101, 102],
            "low": [98, 97, 99],
            "close": [99, 100, 101],
            "volume": [1000, 1100, 900],
        }
    )

    market_state = prepare_market_state(data)

    assert isinstance(market_state, MarketState)
    assert market_state.trend in ["bullish", "bearish"]
    assert isinstance(market_state.volatility, float)
    assert isinstance(market_state.volume, float)


@pytest.mark.asyncio
class TestMarketData:
    """Test suite for market data operations."""

    @pytest.fixture
    async def market_data(self, trading_context):
        """Provide configured market data manager."""
        from execution.market_data import MarketData

        manager = MarketData(trading_context)
        await manager.initialize()
        return manager

    @pytest.fixture
    def sample_candles(self):
        """Provide sample candle data."""
        base_time = int(datetime.now().timestamp())
        return [
            {
                "timestamp": base_time - i * 300,  # 5-minute intervals
                "open": Decimal("50000"),
                "high": Decimal("50100"),
                "low": Decimal("49900"),
                "close": Decimal("50050"),
                "volume": Decimal("10.5"),
            }
            for i in range(100)
        ]

    @pytest.mark.asyncio
    async def test_market_data_retrieval(self, market_data):
        """Test market data retrieval functionality."""
        # Setup mock responses
        market_data.exchange.fetch_ticker = AsyncMock(
            return_value={"last": Decimal("50000")}
        )

        # Test ticker retrieval
        ticker = await market_data.get_current_price("BTC/USDT")
        assert ticker == Decimal("50000")

        # Test error handling
        market_data.exchange.fetch_ticker.side_effect = Exception("API Error")
        with pytest.raises(MarketDataError):
            await market_data.get_current_price("BTC/USDT")

    @pytest.mark.asyncio
    async def test_candle_operations(self, market_data, sample_candles):
        """Test candle data operations."""
        symbol = "BTC/USDT"
        timeframe = "5m"

        # Mock candle retrieval
        market_data.exchange.fetch_ohlcv = AsyncMock(return_value=sample_candles)

        # Test candle retrieval
        candles = await market_data.get_candles(symbol, timeframe, limit=100)
        assert len(candles) == 100
        assert all(isinstance(c["close"], Decimal) for c in candles)

        # Test candle validation
        invalid_candle = {
            "timestamp": 1234567890,
            "open": "invalid",
            "high": None,
            "low": -1,
            "close": "50000",
            "volume": 0,
        }
        with pytest.raises(MarketDataError):
            await market_data.validate_candle(invalid_candle)

    @pytest.mark.asyncio
    async def test_market_data_validation(self, market_data):
        """Test market data validation."""
        # Test price validation
        valid_price = Decimal("50000")
        assert await market_data.validate_price(valid_price)

        with pytest.raises(MarketDataError):
            await market_data.validate_price(Decimal("-1"))

        # Test volume validation
        valid_volume = Decimal("10.5")
        assert await market_data.validate_volume(valid_volume)

        with pytest.raises(MarketDataError):
            await market_data.validate_volume(Decimal("0"))

    @pytest.mark.asyncio
    async def test_symbol_handling(self, market_data):
        """Test symbol handling and validation."""
        # Test symbol normalization
        assert market_data.normalize_symbol("btc/usdt") == "BTC/USDT"
        assert market_data.normalize_symbol("ETH-USDT") == "ETH/USDT"

        # Test symbol validation
        assert await market_data.validate_symbol("BTC/USDT")

        with pytest.raises(MarketDataError):
            await market_data.validate_symbol("INVALID")

        # Test symbol info retrieval
        market_data.exchange.load_markets = AsyncMock()
        market_data.exchange.markets = {
            "BTC/USDT": {
                "precision": {"price": 2, "amount": 6},
                "limits": {"amount": {"min": 0.001}},
            }
        }

        info = await market_data.get_symbol_info("BTC/USDT")
        assert info["precision"]["price"] == 2
        assert info["limits"]["amount"]["min"] == 0.001

    @pytest.mark.asyncio
    async def test_market_analysis(self, market_data, sample_candles):
        """Test market analysis functionality."""
        # Setup test data
        market_data.get_candles = AsyncMock(return_value=sample_candles)

        # Test volatility calculation
        volatility = await market_data.calculate_volatility("BTC/USDT", "5m")
        assert isinstance(volatility, Decimal)
        assert volatility >= 0

        # Test volume analysis
        volume = await market_data.analyze_volume("BTC/USDT", "5m")
        assert isinstance(volume, dict)
        assert "average" in volume
        assert "spike_detected" in volume

        # Test price trend analysis
        trend = await market_data.analyze_trend("BTC/USDT", "5m")
        assert trend in ["bullish", "bearish", "sideways"]

    @pytest.mark.asyncio
    async def test_data_freshness(self, market_data):
        """Test market data freshness checks."""
        # Test fresh data
        market_data.last_update = datetime.now()
        assert await market_data.is_data_fresh()

        # Test stale data
        market_data.last_update = datetime.now() - timedelta(minutes=10)
        assert not await market_data.is_data_fresh()

        # Test update timestamp
        await market_data.update_last_fetch()
        assert (datetime.now() - market_data.last_update).seconds < 1

    @pytest.mark.asyncio
    async def test_market_data_caching(self, market_data):
        """Test market data caching mechanism."""
        symbol = "BTC/USDT"

        # Test cache miss
        assert await market_data.get_from_cache(symbol) is None

        # Test cache storage
        data = {"price": Decimal("50000"), "timestamp": datetime.now()}
        await market_data.store_in_cache(symbol, data)

        # Test cache hit
        cached = await market_data.get_from_cache(symbol)
        assert cached is not None
        assert cached["price"] == data["price"]

        # Test cache expiry
        old_data = {
            "price": Decimal("49000"),
            "timestamp": datetime.now() - timedelta(minutes=5),
        }
        await market_data.store_in_cache("ETH/USDT", old_data)
        assert await market_data.get_from_cache("ETH/USDT") is None

    @pytest.mark.asyncio
    async def test_error_handling(self, market_data):
        """Test error handling in market data operations."""
        # Test network errors
        market_data.exchange.fetch_ticker.side_effect = Exception("Network Error")
        with pytest.raises(MarketDataError) as exc_info:
            await market_data.get_current_price("BTC/USDT")
        assert "Network Error" in str(exc_info.value)

        # Test rate limiting
        market_data.exchange.fetch_ticker.side_effect = Exception("Rate limit exceeded")
        with pytest.raises(MarketDataError) as exc_info:
            await market_data.get_current_price("BTC/USDT")
        assert "Rate limit" in str(exc_info.value)

        # Test invalid response format
        market_data.exchange.fetch_ticker = AsyncMock(return_value={})
        with pytest.raises(MarketDataError):
            await market_data.get_current_price("BTC/USDT")
