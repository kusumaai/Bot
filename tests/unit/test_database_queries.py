import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from database.queries import DatabaseQueries
from database.connection import DatabaseConnection
from utils.error_handler import DatabaseError


@pytest.fixture
def db_connection():
    """Provide a mocked DatabaseConnection."""
    connection = DatabaseConnection(db_path='test.db', logger=MagicMock())
    connection.execute = AsyncMock()
    connection.fetch_one = AsyncMock()
    connection.fetch_all = AsyncMock()
    return connection


@pytest.fixture
def database_queries(db_connection):
    """Provide a DatabaseQueries instance."""
    return DatabaseQueries(connection=db_connection)


@pytest.mark.asyncio
async def test_store_trade_success(database_queries):
    """Test successful storage of a trade."""
    trade = {
        'id': 'trade125',
        'symbol': 'BTC/USDT',
        'entry_price': Decimal('50000'),
        'size': Decimal('0.1'),
        'side': 'buy',
        'strategy': 'TestStrategy',
        'metadata': {}
    }
    database_queries.connection.execute.return_value = None

    result = await database_queries.store_trade(trade)
    assert result is True
    database_queries.connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_trade_failure(database_queries):
    """Test failure in storing a trade due to database error."""
    trade = {
        'id': 'trade126',
        'symbol': 'ETH/USDT',
        'entry_price': Decimal('3000'),
        'size': Decimal('1'),
        'side': 'sell',
        'strategy': 'TestStrategy',
        'metadata': {}
    }
    database_queries.connection.execute.side_effect = Exception("DB Insert Error")

    with pytest.raises(DatabaseError):
        await database_queries.store_trade(trade)
    database_queries.connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_candle_data_success(database_queries):
    """Test successful insertion of candle data."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    candles = [
        {'timestamp': 1609459200, 'open': Decimal('29000'), 'high': Decimal('29500'),
         'low': Decimal('28900'), 'close': Decimal('29400'), 'volume': Decimal('100')},
        {'timestamp': 1609460100, 'open': Decimal('29400'), 'high': Decimal('29800'),
         'low': Decimal('29300'), 'close': Decimal('29700'), 'volume': Decimal('150')},
    ]
    database_queries.connection.execute.return_value = None

    result = await database_queries.insert_candle_data(symbol, timeframe, candles)
    assert result is True
    assert database_queries.connection.execute.call_count == len(candles)


@pytest.mark.asyncio
async def test_insert_candle_data_failure(database_queries):
    """Test insertion of candle data when database raises an error."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    candles = [{'invalid_field': 'invalid_value'}]
    database_queries.connection.execute.side_effect = Exception("DB Insert Error")

    with pytest.raises(DatabaseError):
        await database_queries.insert_candle_data(symbol, timeframe, candles)
    assert database_queries.connection.execute.call_count == len(candles)


@pytest.mark.asyncio
async def test_store_order_success(database_queries):
    """Test successful storage of an order."""
    order = {
        'id': 'order125',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': Decimal('0.1'),
        'price': Decimal('50000'),
        'status': 'open'
    }
    database_queries.connection.execute.return_value = None

    result = await database_queries.store_order(order)
    assert result is True
    database_queries.connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_order_failure(database_queries):
    """Test failure in storing an order due to database error."""
    order = {
        'id': 'order126',
        'symbol': 'ETH/USDT',
        'side': 'sell',
        'amount': Decimal('1'),
        'price': Decimal('3000'),
        'status': 'open'
    }
    database_queries.connection.execute.side_effect = Exception("DB Insert Error")

    with pytest.raises(DatabaseError):
        await database_queries.store_order(order)
    database_queries.connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_and_retrieve_trade_signal(db_queries):
    """Test inserting and retrieving trade signals."""
    symbol = "ETH/USDT"
    signal_type = "GA"
    direction = "long"
    metadata = {"confidence": 0.85, "indicators": {"rsi": 30}}
    
    # Insert trade signal
    success = await db_queries.store_trade_signal(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        metadata=metadata
    )
    assert success is True
    
    # Retrieve latest trade signal
    signal = await db_queries.get_latest_trade_signal(symbol)
    assert signal is not None
    assert signal['symbol'] == symbol
    assert signal['signal_type'] == signal_type
    assert signal['direction'] == direction
    assert signal['metadata'] == metadata


@pytest.mark.asyncio
async def test_insert_and_retrieve_candles(db_queries, sample_candles):
    """Test inserting and retrieving candle data."""
    symbol = "BTC/USDT"
    timeframe = "15m"
    
    # Insert candles
    success = await db_queries.insert_candle_data(symbol, timeframe, sample_candles)
    assert success is True
    
    # Retrieve candles
    candles = await db_queries.get_recent_candles(symbol, timeframe, limit=20)
    assert len(candles) == 20
    
    # Validate retrieved data
    for orig, retrieved in zip(sorted(sample_candles, key=lambda x: x['timestamp']),
                               sorted(candles, key=lambda x: x['timestamp'])):
        assert retrieved['symbol'] == symbol
        assert retrieved['timeframe'] == timeframe
        assert retrieved['timestamp'] == orig['timestamp']
        assert retrieved['open'] == orig['open']
        assert retrieved['high'] == orig['high']
        assert retrieved['low'] == orig['low']
        assert retrieved['close'] == orig['close']
        assert retrieved['volume'] == orig['volume']


@pytest.mark.asyncio
async def test_store_trade(db_queries, sample_trades):
    """Test storing and retrieving trades."""
    trade = sample_trades[0]
    
    # Store trade
    success = await db_queries.store_trade(trade)
    assert success is True
    
    # Retrieve trade by ID
    retrieved_trade = await db_queries.get_trade_by_id(trade['id'])
    assert retrieved_trade is not None
    assert retrieved_trade['id'] == trade['id']
    assert retrieved_trade['symbol'] == trade['symbol']
    assert retrieved_trade['entry_price'] == Decimal(trade['entry_price'])
    assert retrieved_trade['size'] == Decimal(trade['size'])
    assert retrieved_trade['side'] == trade['side']
    assert retrieved_trade['strategy'] == trade['strategy']
    assert retrieved_trade['metadata'] == trade['metadata']
    
    # Attempt to retrieve non-existent trade
    non_existent = await db_queries.get_trade_by_id("nonexistent")
    assert non_existent is None


@pytest.mark.asyncio
async def test_create_and_update_position(db_queries):
    """Test creating and updating a position."""
    position_data = {
        'symbol': "BTC/USDT",
        'direction': "long",
        'entry_price': 50000.0,
        'size': 1.0,
        'status': "active",
        'timestamp': int(datetime.now(tz=timezone.utc).timestamp())
    }
    
    # Create position
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
    
    # Update position status and metadata
    new_metadata = {"exit_price": 51000.0, "pnl": 1000.0}
    update_success = await db_queries.update_position_status(
        position_id=position['id'],
        status="closed",
        metadata=new_metadata
    )
    assert update_success is True
    
    # Verify update
    updated_position = await db_queries.get_position_by_id(position['id'])
    assert updated_position['status'] == "closed"
    assert updated_position['metadata'] == new_metadata 