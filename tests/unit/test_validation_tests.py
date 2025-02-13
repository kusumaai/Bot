import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from tests.validation_tests import (
    test_store_trade_invalid_data,
  #  there are specific functions or classes to import
)
from src.utils.error_handler import DatabaseError
from src.database.queries import DatabaseQueries


@pytest.mark.asyncio
async def test_store_trade_invalid_data(db_queries):
    """Test storing trade with invalid data."""
    with pytest.raises(DatabaseError):
        invalid_trade = {
            'id': 'trade_invalid',
            'symbol': 'BTC/USDT',
            # Missing required fields
        }
        await db_queries.store_trade(invalid_trade)


@pytest.mark.asyncio
async def test_insert_candle_data_invalid(db_queries):
    """Test inserting invalid candle data."""
    invalid_candles = [{'invalid_field': 'invalid_value'}]
    with pytest.raises(DatabaseError):
        await db_queries.insert_candle_data("INVALID/PAIR", "invalid_timeframe", invalid_candles)


@pytest.mark.asyncio
async def test_store_trade_missing_fields(db_queries):
    """Test storing a trade with missing fields."""
    incomplete_trade = {
        'id': 'trade_missing',
        'symbol': 'ETH/USDT',
        # Missing 'amount' and 'price'
    }
    with pytest.raises(DatabaseError):
        await db_queries.store_trade(incomplete_trade) 