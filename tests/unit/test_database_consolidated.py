#! /usr/bin/env python3
"""
Module: tests.unit.test_database_consolidated
Provides comprehensive unit testing for database operations.
Consolidates and improves upon previous database test implementations.
"""
import logging
from asyncio import AbstractEventLoop
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from database.connection import DatabaseConnection
from database.queries import DatabaseQueries
from utils.error_handler import DatabaseError
from utils.numeric_handler import NumericHandler


class TestDatabaseBase:
    """Base class for database tests with common setup and utilities."""

    @pytest.fixture
    async def db_connection(self):
        """Provide a properly mocked database connection."""
        with patch("database.connection.aiosqlite.connect") as mock_connect:
            # Create mock objects
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()

            # Set up cursor behavior
            mock_cursor.fetchone.return_value = (1,)

            # Define different mock data for different queries
            def mock_fetchall(*args, **kwargs):
                # Get the last executed query from the cursor
                last_query = (
                    mock_cursor.execute.call_args[0][0].lower()
                    if mock_cursor.execute.call_args
                    else ""
                )

                if (
                    "select * from positions" in last_query
                    or "select id, symbol" in last_query
                ):
                    return [
                        (
                            1,  # id
                            "BTC/USDT",  # symbol
                            "long",  # direction
                            "50000",  # entry_price
                            "0.1",  # size
                            "active",  # status
                            "2024-02-15T16:31:25",  # timestamp
                            "{}",  # metadata
                        )
                    ]
                elif "select * from candles" in last_query:
                    return [
                        (
                            "BTC/USDT",  # symbol
                            "15m",  # timeframe
                            1739637085,  # timestamp
                            "29000",  # open
                            "29500",  # high
                            "28900",  # low
                            "29400",  # close
                            "100.5",  # volume
                        )
                    ] * 20  # Return 20 candles
                elif "select" in last_query and "trade_signals" in last_query:
                    return [
                        (
                            "ETH/USDT",  # symbol
                            "ML",  # signal_type
                            "long",  # direction
                            1739637085,  # timestamp
                            '{"confidence": 0.85}',  # metadata
                        )
                    ]
                return []

            mock_cursor.fetchall.side_effect = mock_fetchall
            mock_cursor.execute = AsyncMock()
            mock_cursor.commit = AsyncMock()

            # Set up cursor context manager
            mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
            mock_cursor.__aexit__ = AsyncMock()

            # Set up connection's execute to return cursor
            async def mock_execute(*args, **kwargs):
                mock_cursor.execute.call_args = args, kwargs
                return mock_cursor

            mock_conn.execute = AsyncMock(side_effect=mock_execute)
            mock_conn.commit = AsyncMock()
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock()

            # Set up the connection factory
            mock_connect.return_value = mock_conn

            # Create the database connection
            connection = DatabaseConnection(
                db_path=":memory:", logger=logging.getLogger("test_logger")
            )

            # Mock the get_connection method to properly handle async context
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def mock_get_connection_cm():
                try:
                    yield mock_conn
                finally:
                    pass

            connection.get_connection = mock_get_connection_cm
            yield connection

    @pytest.fixture
    async def db_queries(self, db_connection):
        """Provide a DatabaseQueries instance with mocked connection."""
        return DatabaseQueries(
            connection=db_connection,
            db_path=":memory:",
            logger=logging.getLogger("test_logger"),
        )

    @pytest.fixture
    def sample_trade(self):
        """Provide a valid sample trade for testing."""
        return {
            "id": "test_trade_1",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": Decimal("0.1"),
            "price": Decimal("50000"),
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "metadata": {"test_meta": "value"},
        }

    @pytest.fixture
    def sample_candles(self):
        """Provide sample candle data for testing."""
        base_timestamp = int(datetime.now(timezone.utc).timestamp())
        return [
            {
                "timestamp": base_timestamp,
                "open": Decimal("29000"),
                "high": Decimal("29500"),
                "low": Decimal("28900"),
                "close": Decimal("29400"),
                "volume": Decimal("100.5"),
            },
            {
                "timestamp": base_timestamp + 900,  # 15 minutes later
                "open": Decimal("29400"),
                "high": Decimal("29800"),
                "low": Decimal("29300"),
                "close": Decimal("29750"),
                "volume": Decimal("150.2"),
            },
        ]


class TestDatabaseConnection(TestDatabaseBase):
    """Test suite for database connection functionality."""

    @pytest.mark.asyncio
    async def test_connection_initialization(self, db_connection):
        """Test database connection initialization."""
        assert db_connection is not None
        assert db_connection.db_path == ":memory:"

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, db_connection):
        """Test connection context manager functionality."""
        async with db_connection.get_connection() as conn:
            assert conn is not None
            version = await conn.execute_fetchone("SELECT sqlite_version();")
            assert version is not None

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, db_connection):
        """Test error handling during connection."""
        with patch("database.connection.aiosqlite.connect") as mock_connect:
            mock_connect.side_effect = aiosqlite.Error("Connection failed")
            with pytest.raises(DatabaseError):
                async with db_connection.get_connection():
                    pass


class TestDatabaseQueries(TestDatabaseBase):
    """Test suite for database queries functionality."""

    @pytest.mark.asyncio
    async def test_store_trade_success(self, db_queries, sample_trade):
        """Test successful trade storage."""
        success = await db_queries.store_trade(sample_trade)
        assert success is True

    @pytest.mark.asyncio
    async def test_store_trade_validation(self, db_queries):
        """Test trade storage with invalid data."""
        invalid_trades = [
            {"id": "invalid_1"},  # Missing all required fields
            {
                "id": "invalid_2",
                "symbol": "BTC/USDT",
                "side": "invalid_side",  # Invalid side
                "amount": "not_a_number",  # Invalid amount format
                "price": Decimal("50000"),
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
            },
            {
                "id": "invalid_3",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": Decimal("0.1"),
                # Missing price
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
            },
        ]

        for invalid_trade in invalid_trades:
            with pytest.raises(DatabaseError):
                await db_queries.store_trade(invalid_trade)

    @pytest.mark.asyncio
    async def test_candle_operations(self, db_queries, sample_candles):
        """Test candle data operations."""
        symbol = "BTC/USDT"
        timeframe = "15m"

        # Test insertion
        success = await db_queries.insert_candle_data(symbol, timeframe, sample_candles)
        assert success is True

        # Test retrieval
        candles = await db_queries.get_recent_candles(symbol, timeframe, limit=20)
        assert len(candles) == 20

        # Verify data integrity
        for original, stored in zip(sample_candles, candles):
            assert stored["timestamp"] == original["timestamp"]
            assert stored["open"] == original["open"]
            assert stored["close"] == original["close"]

    @pytest.mark.asyncio
    async def test_position_lifecycle(self, db_queries):
        """Test complete position lifecycle."""
        # Create position
        position_data = {
            "symbol": "BTC/USDT",
            "direction": "long",
            "entry_price": Decimal("50000"),
            "size": Decimal("0.1"),
            "status": "active",
            "timestamp": int(datetime.now(tz=timezone.utc).timestamp()),
        }

        success = await db_queries.create_position(position_data)
        assert success is True

        # Retrieve active positions
        positions = await db_queries.get_active_positions("BTC/USDT")
        assert len(positions) == 1
        position = positions[0]

        # Update position
        update_data = {"exit_price": Decimal("55000"), "pnl": Decimal("500")}
        update_success = await db_queries.update_position_status(
            position_id=position["id"], status="closed", metadata=update_data
        )
        assert update_success is True

        # Verify closed position
        closed_position = await db_queries.get_position_by_id(position["id"])
        assert closed_position["status"] == "closed"
        assert closed_position["metadata"]["exit_price"] == update_data["exit_price"]

    @pytest.mark.asyncio
    async def test_trade_signal_operations(self, db_queries):
        """Test trade signal operations."""
        signal_data = {
            "symbol": "ETH/USDT",
            "signal_type": "ML",
            "direction": "long",
            "metadata": {"confidence": 0.85},
        }

        # Store signal
        success = await db_queries.store_trade_signal(**signal_data)
        assert success is True

        # Retrieve signal
        signal = await db_queries.get_latest_trade_signal(signal_data["symbol"])
        assert signal is not None
        assert signal["symbol"] == signal_data["symbol"]
        assert signal["direction"] == signal_data["direction"]
        assert signal["metadata"]["confidence"] == signal_data["metadata"]["confidence"]
