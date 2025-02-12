import logging
import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from data.candles import CandleManager, calculate_atr, CandleProcessor
from utils.error_handler import ValidationError
from database.database import DatabaseConnection
from utils.logger import setup_logging


@pytest.fixture
def db_connection():
    """Provide a mocked DatabaseConnection."""
    connection = AsyncMock(spec=DatabaseConnection)
    return connection


@pytest.fixture
def candle_manager(db_connection, logger):
    """Provide a CandleManager instance."""
    return CandleManager(
        db_connection=db_connection,
        logger=logger
    )


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


def test_calculate_atr():
    """Test ATR calculation with valid candles."""
    candles = [
        {'high': Decimal('50'), 'low': Decimal('30'), 'close': Decimal('40')},
        {'high': Decimal('55'), 'low': Decimal('35'), 'close': Decimal('45')},
        {'high': Decimal('60'), 'low': Decimal('40'), 'close': Decimal('50')},
    ]
    atr = calculate_atr(candles)
    expected_tr = [
        Decimal('50') - Decimal('30'),  # 20
        Decimal('55') - Decimal('35'),  # 20
        Decimal('60') - Decimal('40'),  # 20
    ]
    expected_atr = sum(expected_tr) / len(expected_tr)
    assert atr == expected_atr


def test_calculate_atr_insufficient_data():
    """Test ATR calculation with insufficient candle data."""
    candles = [{'high': Decimal('50'), 'low': Decimal('30'), 'close': Decimal('40')}]
    with pytest.raises(ValidationError, match="Insufficient candle data to calculate ATR"):
        calculate_atr(candles)


@pytest.mark.asyncio
async def test_insert_candles_success(candle_manager, logger):
    """Test successful insertion of candle data into the database."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    candles = [
        {'timestamp': 1609459200, 'open': Decimal('29000'), 'high': Decimal('29500'),
         'low': Decimal('28900'), 'close': Decimal('29400'), 'volume': Decimal('100')},
        {'timestamp': 1609460100, 'open': Decimal('29400'), 'high': Decimal('29800'),
         'low': Decimal('29300'), 'close': Decimal('29700'), 'volume': Decimal('150')},
    ]

    with patch.object(candle_manager.db_connection, 'insert_candles', new_callable=AsyncMock) as mock_insert:
        result = await candle_manager.insert_candles(symbol, timeframe, candles)
        assert result is True
        mock_insert.assert_awaited_once_with(symbol, timeframe, candles)


@pytest.mark.asyncio
async def test_insert_candles_database_error(candle_manager, logger):
    """Test insertion of candle data when database raises an error."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    candles = [
        {'timestamp': 1609459200, 'open': Decimal('29000'), 'high': Decimal('29500'),
         'low': Decimal('28900'), 'close': Decimal('29400'), 'volume': Decimal('100')},
    ]

    with patch.object(candle_manager.db_connection, 'insert_candles', new_callable=AsyncMock) as mock_insert:
        mock_insert.side_effect = Exception("Database Insert Error")
        result = await candle_manager.insert_candles(symbol, timeframe, candles)
        assert result is False
        mock_insert.assert_awaited_once_with(symbol, timeframe, candles)
        logger.error.assert_called_with(
            f"Failed to insert candles for {symbol} - {timeframe}: Database Insert Error"
        )


@pytest.mark.asyncio
async def test_fetch_and_store_candles(candle_manager, logger):
    """Test fetching and storing candle data."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    limit = 2
    mock_candles = [
        {'timestamp': 1609459200, 'open': Decimal('29000'), 'high': Decimal('29500'),
         'low': Decimal('28900'), 'close': Decimal('29400'), 'volume': Decimal('100')},
        {'timestamp': 1609460100, 'open': Decimal('29400'), 'high': Decimal('29800'),
         'low': Decimal('29300'), 'close': Decimal('29700'), 'volume': Decimal('150')},
    ]

    with patch.object(candle_manager.db_connection, 'insert_candles', new_callable=AsyncMock) as mock_insert, \
         patch('ccxt.async_support.binance.fetch_ohlcv', new_callable=AsyncMock) as mock_fetch:

        mock_fetch.return_value = [
            [1609459200000, 29000, 29500, 28900, 29400, 100],
            [1609460100000, 29400, 29800, 29300, 29700, 150],
        ]

        result = await candle_manager.fetch_and_store_candles(symbol, timeframe, limit)
        assert result is True
        mock_fetch.assert_awaited_once_with(symbol, timeframe, limit=limit)
        mock_insert.assert_awaited_once_with(symbol, timeframe, mock_candles)


@pytest.mark.asyncio
async def test_fetch_and_store_candles_exchange_error(candle_manager, logger):
    """Test fetching candle data when exchange raises an error."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    limit = 2

    with patch('ccxt.async_support.binance.fetch_ohlcv', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = Exception("Exchange API Error")

        result = await candle_manager.fetch_and_store_candles(symbol, timeframe, limit)
        assert result is False
        mock_fetch.assert_awaited_once_with(symbol, timeframe, limit=limit)
        logger.error.assert_called_with(
            f"Failed to fetch candles for {symbol} - {timeframe}: Exchange API Error"
        ) 