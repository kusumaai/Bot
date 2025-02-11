import pytest
from decimal import Decimal
import logging
from datetime import datetime, timezone

from database.connection import DatabaseConnection
from database.queries import DatabaseQueries
from utils.error_handler import DatabaseError

async def test_database_connection(db_connection):
    """Test database connection is working"""
    assert db_connection is not None
    async with db_connection.get_connection() as conn:
        assert conn is not None

async def test_insert_candles(db_queries, sample_candles):
    """Test inserting candle data"""
    symbol = "BTC/USDT"
    timeframe = "15m"
    
    # Insert candles
    await db_queries.insert_candle_data(symbol, timeframe, sample_candles)
    
    # Verify insertion
    candles = await db_queries.get_recent_candles(symbol, timeframe)
    assert len(candles) == len(sample_candles)
    
    # Verify data integrity
    for original, stored in zip(
        sorted(sample_candles, key=lambda x: x['timestamp']),
        sorted(candles, key=lambda x: x['timestamp'])
    ):
        assert stored['timestamp'] == original['timestamp']
        assert stored['open'] == original['open']
        assert stored['high'] == original['high']
        assert stored['low'] == original['low']
        assert stored['close'] == original['close']
        assert stored['volume'] == original['volume']

async def test_store_trade_signal(db_queries):
    """Test storing and retrieving trade signals"""
    symbol = "ETH/USDT"
    signal_type = "GA"
    direction = "long"
    metadata = {"confidence": 0.85, "indicators": {"rsi": 30}}
    
    # Store signal
    await db_queries.store_trade_signal(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        metadata=metadata
    )
    
    # Verify storage
    query = "SELECT * FROM trade_signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1"
    result = await db_queries.connection.execute(query, [symbol], fetch=True)
    assert len(result) == 1
    signal = result[0]
    
    assert signal['symbol'] == symbol
    assert signal['signal_type'] == signal_type
    assert signal['direction'] == direction

async def test_position_management(db_queries):
    """Test position creation and updates"""
    # Create position
    query = """
        INSERT INTO positions (
            symbol, direction, entry_price, size, status, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    params = ["BTC/USDT", "long", 35000.0, 0.1, "active", int(datetime.now().timestamp())]
    await db_queries.connection.execute(query, params)
    
    # Test get_active_positions
    positions = await db_queries.get_active_positions("BTC/USDT")
    assert len(positions) == 1
    position = positions[0]
    assert position['symbol'] == "BTC/USDT"
    assert position['direction'] == "long"
    
    # Test update_position_status
    new_metadata = {"exit_price": 36000.0, "pnl": 1000.0}
    await db_queries.update_position_status(
        position_id=position['id'],
        status="closed",
        metadata=new_metadata
    )
    
    # Verify update
    active_positions = await db_queries.get_active_positions("BTC/USDT")
    assert len(active_positions) == 0

async def test_error_handling(db_queries):
    """Test error handling for invalid operations"""
    with pytest.raises(DatabaseError):
        await db_queries.insert_candle_data(
            symbol="INVALID/PAIR",
            timeframe="invalid",
            candles=[{'invalid': 'data'}]
        ) 