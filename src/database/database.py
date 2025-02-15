#! /usr/bin/env python3
# src/database/database.py
"""
Module: database/database.py
Context manager for SQLite connections and helper functions to execute SQL queries with automatic commits.  
"""

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
from aiosqlite import connect

from utils.error_handler import DatabaseError, handle_error_async
from utils.exceptions import DatabaseError
from utils.numeric_handler import NumericHandler

logger = logging.getLogger("TradingBot")


class QueryBuilder:
    """Placeholder for QueryBuilder class to safely build SQL queries"""

    def build_insert_trade(self, trade: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
        """Build SQL insert statement for a trade"""
        query = """
            INSERT INTO trades (id, symbol, entry_price, size, side, strategy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trade.get("id"),
            trade.get("symbol"),
            trade.get("entry_price"),
            trade.get("size"),
            trade.get("side"),
            trade.get("strategy"),
            json.dumps(trade.get("metadata")) if trade.get("metadata") else None,
        )
        return query, params


class DatabaseConnection:
    """Synchronous context manager for SQLite database connections."""

    def __init__(self, db_path: str, logger=None, **kwargs):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.conn = None

    def __enter__(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            return self.conn
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            self.conn.close()

    async def initialize(self):
        try:
            directory = os.path.dirname(self.db_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self.conn = await connect(self.db_path)
            await self.create_tables()
            return True
        except Exception as e:
            import traceback

            self.logger.error(f"Failed to initialize database: {e}")
            self.logger.error(traceback.format_exc())
            return False

    async def create_tables(self):
        # Implementation to create necessary tables
        pass


@asynccontextmanager
async def async_db_connection(db_path: str):
    """Async context manager for database connections using aiosqlite."""
    try:
        async with aiosqlite.connect(db_path) as conn:
            yield conn
    except Exception as e:
        logger.error(f"Async DB connection error: {e}")
        raise


async def execute_sql(query: str, params: Tuple[Any, ...], db_path: str) -> bool:
    try:
        async with database.get_db_connection(db_path) as db:
            await db.execute(query, params)
            await db.commit()
        return True
    except aiosqlite.Error as e:
        handle_error_async(e, "execute_sql", logger)
        return False
    except Exception as e:
        handle_error_async(e, "execute_sql", logger)
        return False


class DatabaseQueries:
    """Safe database query implementations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[asyncio.Pool] = None
        self.logger = logger
        self.nh = NumericHandler()
        self.db_connection = DatabaseConnection(config["database"]["dbname"])
        self._lock = asyncio.Lock()
        self.query_builder = QueryBuilder()

    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        try:
            self.pool = await asyncio.create_pool(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                user=self.config["database"]["user"],
                password=self.config["database"]["password"],
                database=self.config["database"]["dbname"],
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            return False

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()

    async def execute(
        self, query: str, params: Union[List[Any], Tuple[Any, ...]] = ()
    ) -> Any:
        """Execute a SQL query with parameters"""
        try:
            async with self.db_connection.get_connection() as conn:
                async with conn.execute(query, params) as cursor:
                    await conn.commit()
                    result = await cursor.fetchall()
                    return result
        except aiosqlite.Error as e:
            self.logger.error(f"Database execution error: {e}")
            raise DatabaseError(f"Database execution error: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error in execute: {e}")
            raise DatabaseError(f"Unexpected error in execute: {e}") from e

    async def store_trade(self, trade: Dict[str, Any]) -> bool:
        """Store a trade in the database"""
        query, params = self.query_builder.build_insert_trade(trade)
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to store trade: {e}")
            return False

    async def store_ticker(self, symbol: str, ticker: Dict[str, Any]) -> bool:
        """Store ticker data in the database"""
        query = """
            INSERT INTO tickers (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
        """
        params = (
            symbol,
            ticker.get("timestamp"),
            ticker.get("open"),
            ticker.get("high"),
            ticker.get("low"),
            ticker.get("last"),
            ticker.get("baseVolume"),
        )
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to store ticker for {symbol}: {e}")
            return False

    async def update_position_status(
        self, position_id: str, status: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update the status of a position"""
        query = """
            UPDATE positions
            SET status = ?, metadata = ?
            WHERE id = ?
        """
        params = (status, json.dumps(metadata) if metadata else None, position_id)
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to update position status: {e}")
            return False

    async def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Retrieve all trades for a given symbol"""
        query = "SELECT * FROM trades WHERE symbol = ?"
        try:
            rows = await self.execute(query, (symbol,))
            return [dict(row) for row in rows]
        except DatabaseError as e:
            self.logger.error(f"Failed to get trades for {symbol}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when getting trades for {symbol}: {e}")
            return []

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        query = "SELECT * FROM positions WHERE status = 'OPEN'"
        try:
            rows = await self.execute(query, ())
            return [
                {
                    "symbol": pos["symbol"],
                    "size": self.nh.to_decimal(pos["size"]),
                    "entry_price": self.nh.to_decimal(pos["entry_price"]),
                    "current_price": self.nh.to_decimal(pos["current_price"]),
                    "unrealized_pnl": self.nh.to_decimal(pos["unrealized_pnl"]),
                    "last_update": pos["last_update"],
                }
                for pos in rows
            ]
        except DatabaseError as e:
            self.logger.error(f"Failed to retrieve open positions: {e}")
            return []
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Invalid data when retrieving open positions: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when retrieving open positions: {e}")
            return []

    async def ping(self) -> bool:
        """Check database connection"""
        try:
            async with aiosqlite.connect(self.config["database"]["dbname"]) as db:
                await db.execute("SELECT 1")
                return True
        except Exception:
            return False

    async def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by its ID"""
        query = "DELETE FROM trades WHERE id = ?"
        params = (trade_id,)
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to delete trade {trade_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error when deleting trade {trade_id}: {e}")
            return False

    async def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update trade details"""
        if not updates:
            self.logger.warning("No updates provided for trade.")
            return False

        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        query = f"UPDATE trades SET {set_clause} WHERE id = ?"
        params = list(updates.values()) + [trade_id]
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to update trade {trade_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error when updating trade {trade_id}: {e}")
            return False

    async def get_all_trades(self) -> List[Dict[str, Any]]:
        """Retrieve all trades"""
        query = "SELECT * FROM trades"
        try:
            rows = await self.execute(query, ())
            return [dict(row) for row in rows]
        except DatabaseError as e:
            self.logger.error(f"Failed to retrieve all trades: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when retrieving all trades: {e}")
            return []
