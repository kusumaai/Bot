#! /usr/bin/env python3
"""
Module: tests.unit.conftest
Provides shared fixtures for database testing.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from database.connection import DatabaseConnection
from database.queries import DatabaseQueries
from utils.numeric_handler import NumericHandler


@pytest.fixture
def logger():
    """Provide a test logger instance."""
    return logging.getLogger("test_logger")


@pytest.fixture
async def mock_connection():
    """Provide a mock database connection."""
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = []
    return mock_conn


@pytest.fixture
async def mock_pool(mock_connection):
    """Provide a mock connection pool."""
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = mock_connection
    return mock_pool


@pytest.fixture
async def base_db_connection(logger):
    """Provide a base database connection for testing."""
    with patch("database.connection.aiosqlite.connect") as mock_connect:
        connection = DatabaseConnection(db_path=":memory:", logger=logger)
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = []

        yield connection

        if hasattr(connection, "_pool"):
            await connection._pool.close()


@pytest.fixture
async def base_db_queries(base_db_connection):
    """Provide a base DatabaseQueries instance."""
    return DatabaseQueries(
        connection=base_db_connection, numeric_handler=NumericHandler()
    )


@pytest.fixture
def sample_trade_data():
    """Provide sample trade data for testing."""
    return {
        "id": "test_trade_1",
        "symbol": "BTC/USDT",
        "entry_price": Decimal("50000"),
        "size": Decimal("0.1"),
        "side": "buy",
        "strategy": "TestStrategy",
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp()),
        "metadata": {"test_meta": "value"},
    }


@pytest.fixture
def sample_candle_data():
    """Provide sample candle data for testing."""
    return [
        {
            "timestamp": int(datetime.now(tz=timezone.utc).timestamp()) - i * 900,
            "open": Decimal("29000"),
            "high": Decimal("29500"),
            "low": Decimal("28900"),
            "close": Decimal("29400"),
            "volume": Decimal("100"),
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_position_data():
    """Provide sample position data for testing."""
    return {
        "symbol": "BTC/USDT",
        "direction": "long",
        "entry_price": Decimal("50000"),
        "size": Decimal("0.1"),
        "status": "active",
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp()),
        "metadata": {},
    }


@pytest.fixture
def sample_trade_signal_data():
    """Provide sample trade signal data for testing."""
    return {
        "symbol": "ETH/USDT",
        "signal_type": "ML",
        "direction": "long",
        "metadata": {"confidence": 0.85, "indicators": {"rsi": 30, "macd": "bullish"}},
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp()),
    }


@pytest.fixture
async def mock_db_error_connection(logger):
    """Provide a database connection that simulates errors."""
    with patch("database.connection.aiosqlite.connect") as mock_connect:
        connection = DatabaseConnection(db_path=":memory:", logger=logger)
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = aiosqlite.Error("Database error")

        yield connection

        if hasattr(connection, "_pool"):
            await connection._pool.close()


@pytest.fixture
def mock_db_error_queries(mock_db_error_connection):
    """Provide a DatabaseQueries instance configured to simulate errors."""
    return DatabaseQueries(
        connection=mock_db_error_connection, numeric_handler=NumericHandler()
    )
