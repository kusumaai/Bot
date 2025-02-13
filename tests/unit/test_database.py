import pytest
from decimal import Decimal
import logging
from datetime import datetime, timezone

from src.database.connection import DatabaseConnection
from src.database.queries import DatabaseQueries
from src.utils.error_handler import DatabaseError


@pytest.mark.asyncio
async def test_database_connection(db_connection):
    """Test that the database connection can be established and closed."""
    assert db_connection is not None
    async with db_connection.get_connection() as conn:
        assert conn is not None
        version = await conn.execute_fetchone("SELECT sqlite_version();")
        assert version is not None


@pytest.mark.asyncio
async def test_insert_candles(db_queries, sample_candles):
    """Test inserting candle data into the database."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    
    # Insert candles
    success = await db_queries.insert_candle_data(symbol, timeframe, sample_candles)
    assert success is True
    
    # Retrieve candles
    candles = await db_queries.get_recent_candles(symbol, timeframe, limit=20)
    assert len(candles) == 20
    
    # Verify data integrity
    for original, stored in zip(sorted(sample_candles, key=lambda x: x['timestamp']),
                               sorted(candles, key=lambda x: x['timestamp'])):
        assert stored['timestamp'] == original['timestamp']
        assert stored['open'] == original['open']
        assert stored['high'] == original['high']
        assert stored['low'] == original['low']
        assert stored['close'] == original['close']
        assert stored['volume'] == original['volume']


@pytest.mark.asyncio
async def test_store_trade_signal(db_queries):
    """Test storing and retrieving trade signals."""
    symbol = "ETH/USDT"
    signal_type = "GA"
    direction = "long"
    metadata = {"confidence": 0.85, "indicators": {"rsi": 30}}
    
    # Store signal
    success = await db_queries.store_trade_signal(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        metadata=metadata
    )
    assert success is True
    
    # Retrieve the latest signal
    signal = await db_queries.get_latest_trade_signal(symbol)
    assert signal is not None
    assert signal['symbol'] == symbol
    assert signal['signal_type'] == signal_type
    assert signal['direction'] == direction
    assert signal['metadata'] == metadata


@pytest.mark.asyncio
async def test_position_management(db_queries):
    """Test position creation, updating, and retrieval."""
    # Create a new position
    position_data = {
        'symbol': "BTC/USDT",
        'direction': "long",
        'entry_price': 35000.0,
        'size': 0.1,
        'status': "active",
        'timestamp': int(datetime.now(tz=timezone.utc).timestamp())
    }
    success = await db_queries.create_position(position_data)
    assert success is True
    
    # Retrieve active positions
    positions = await db_queries.get_active_positions("BTC/USDT")
    assert len(positions) == 1
    position = positions[0]
    assert position['symbol'] == position_data['symbol']
    assert position['direction'] == position_data['direction']
    assert position['entry_price'] == position_data['entry_price']
    assert position['size'] == position_data['size']
    assert position['status'] == position_data['status']
    
    # Update position status
    new_metadata = {"exit_price": 36000.0, "pnl": 1000.0}
    update_success = await db_queries.update_position_status(
        position_id=position['id'],
        status="closed",
        metadata=new_metadata
    )
    assert update_success is True
    
    # Verify update
    active_positions = await db_queries.get_active_positions("BTC/USDT")
    assert len(active_positions) == 0
    
    # Retrieve closed position
    closed_position = await db_queries.get_position_by_id(position['id'])
    assert closed_position is not None
    assert closed_position['status'] == "closed"
    assert closed_position['metadata'] == new_metadata


@pytest.mark.asyncio
async def test_error_handling(db_queries):
    """Test error handling for invalid operations."""
    with pytest.raises(DatabaseError):
        # Attempt to insert invalid candle data
        invalid_candles = [{'invalid_field': 'invalid_value'}]
        await db_queries.insert_candle_data("INVALID/PAIR", "invalid_timeframe", invalid_candles)
    
    with pytest.raises(DatabaseError):
        # Attempt to store a trade with missing fields
        incomplete_trade = {
            'id': 'trade_invalid',
            'symbol': 'BTC/USDT'
            # Missing other required fields
        }
        await db_queries.store_trade(incomplete_trade) 