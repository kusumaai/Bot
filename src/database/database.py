#! /usr/bin/env python3
# src/database/database.py
"""
Module: src.database.database
Provides a production-grade database connection manager using SQLAlchemy.
This module implements the DBConnection context manager for robust connection pooling, pre-ping, and error handling.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.utils.error_handler import DatabaseError, handle_error_async
from src.utils.exceptions import DatabaseError
from src.utils.numeric_handler import NumericHandler

logger = logging.getLogger("TradingBot")


def _get_connect():
    import sys

    return sys.modules["aiosqlite"].connect


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


class DBConnection:
    """DEPRECATED: Use DatabaseConnection instead. This class will be removed in future versions."""

    def __init__(self, db_url: str):
        raise DeprecationWarning(
            "DBConnection is deprecated. Please use DatabaseConnection with async/await pattern instead."
        )


class DatabaseConnection:
    """Asynchronous database connection manager with proper resource handling."""

    def __init__(self, db_path: str, numeric_handler=None):
        self.db_path = db_path
        self.nh = numeric_handler or NumericHandler()
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._active_transactions: Dict[int, int] = (
            {}
        )  # transaction_id -> nesting level
        self._transaction_timeouts: Dict[int, float] = {}  # transaction_id -> deadline
        self._max_nesting_depth = 3  # Limit nesting to prevent stack overflow
        self._transaction_timeout = 30.0  # 30 second timeout
        self._deadlock_detection_interval = 5.0  # Check every 5 seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    @asynccontextmanager
    async def transaction(self):
        """
        Thread-safe transaction context manager with deadlock prevention.

        Features:
        - Limited nesting depth
        - Transaction timeouts
        - Deadlock detection
        - Automatic cleanup of stale transactions
        """
        transaction_id = id(asyncio.current_task())

        try:
            async with self._lock:
                # Check nesting depth
                current_depth = self._active_transactions.get(transaction_id, 0)
                if current_depth >= self._max_nesting_depth:
                    raise DatabaseError(
                        f"Maximum transaction nesting depth ({self._max_nesting_depth}) exceeded"
                    )

                # Set transaction deadline
                deadline = time.time() + self._transaction_timeout
                self._transaction_timeouts[transaction_id] = deadline

                # Increment nesting level
                self._active_transactions[transaction_id] = current_depth + 1

            # Start deadlock detection if this is the first transaction
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._monitor_transactions())

            async with aiosqlite.connect(self.db_path) as conn:
                try:
                    if current_depth == 0:
                        await conn.execute("BEGIN")
                    else:
                        # Create numbered savepoint for nested transaction
                        await conn.execute(f"SAVEPOINT sp_{current_depth}")

                    yield conn

                    # Commit only if this is the outermost transaction
                    if current_depth == 0:
                        await conn.commit()
                    else:
                        await conn.execute(f"RELEASE SAVEPOINT sp_{current_depth}")

                except Exception as e:
                    # Rollback to appropriate point
                    if current_depth == 0:
                        await conn.rollback()
                    else:
                        await conn.execute(f"ROLLBACK TO SAVEPOINT sp_{current_depth}")
                    raise DatabaseError(f"Transaction failed: {str(e)}")

        finally:
            async with self._lock:
                # Decrement nesting level
                current_depth = self._active_transactions.get(transaction_id, 0)
                if current_depth <= 1:
                    # Clean up if this is the outermost transaction
                    self._active_transactions.pop(transaction_id, None)
                    self._transaction_timeouts.pop(transaction_id, None)
                else:
                    self._active_transactions[transaction_id] = current_depth - 1

    async def _monitor_transactions(self):
        """Monitor for deadlocks and stale transactions."""
        try:
            while self._active_transactions:
                current_time = time.time()

                async with self._lock:
                    # Check for timed out transactions
                    timed_out = [
                        tid
                        for tid, deadline in self._transaction_timeouts.items()
                        if current_time > deadline
                    ]

                    # Clean up timed out transactions
                    for tid in timed_out:
                        self.logger.error(
                            f"Transaction {tid} timed out after {self._transaction_timeout} seconds"
                        )
                        self._active_transactions.pop(tid, None)
                        self._transaction_timeouts.pop(tid, None)

                        # Force rollback the connection if possible
                        try:
                            async with aiosqlite.connect(self.db_path) as conn:
                                await conn.rollback()
                        except Exception as e:
                            self.logger.error(
                                f"Error rolling back timed out transaction: {e}"
                            )

                await asyncio.sleep(self._deadlock_detection_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in transaction monitor: {e}")

    async def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query with proper error handling and timeouts."""
        try:
            async with self.transaction() as conn:
                async with conn.execute(query, params or {}) as cursor:
                    # Convert results to dictionaries
                    columns = (
                        [col[0] for col in cursor.description]
                        if cursor.description
                        else []
                    )
                    rows = await cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]

        except aiosqlite.Error as e:
            raise DatabaseError(f"Database error executing query: {str(e)}")
        except asyncio.TimeoutError:
            raise DatabaseError("Query execution timed out")
        except Exception as e:
            raise DatabaseError(f"Unexpected error executing query: {str(e)}")

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> None:
        """Execute multiple queries in a single transaction with proper error handling."""
        if not params_list:
            return

        try:
            async with self.transaction() as conn:
                async with conn.executemany(query, params_list) as cursor:
                    await cursor.fetchall()  # Ensure all results are consumed

        except aiosqlite.Error as e:
            raise DatabaseError(f"Database error executing multiple queries: {str(e)}")
        except asyncio.TimeoutError:
            raise DatabaseError("Query execution timed out")
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error executing multiple queries: {str(e)}"
            )

    async def close(self) -> None:
        """Safely close the database connection."""
        try:
            # Cancel transaction monitor
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Force cleanup any remaining transactions
            async with self._lock:
                self._active_transactions.clear()
                self._transaction_timeouts.clear()

        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")


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
    """Execute a SQL query with parameters and return success status.

    Args:
        query: SQL query string
        params: Query parameters
        db_path: Path to the database file

    Returns:
        bool: True if query executed successfully, False otherwise
    """
    try:
        async with async_db_connection(db_path) as db:
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
    """Safe database query implementations with connection pooling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[asyncio.Pool] = None
        self.logger = logger
        self.nh = NumericHandler()
        self.db_connection = DatabaseConnection(config["database"]["dbname"])
        self._lock = asyncio.Lock()
        self.query_builder = QueryBuilder()
        self.initialized = False
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        async with self._lock:
            try:
                if self.initialized:
                    return True

                # Initialize the database connection
                if not await self.db_connection.initialize():
                    self.logger.error("Failed to initialize database connection")
                    return False

                # Create connection pool
                self.pool = await asyncio.create_pool(
                    host=self.config["database"]["host"],
                    port=self.config["database"]["port"],
                    user=self.config["database"]["user"],
                    password=self.config["database"]["password"],
                    database=self.config["database"]["dbname"],
                    minsize=5,
                    maxsize=20,
                    timeout=60.0,
                    echo=True,
                )

                self.initialized = True
                self.logger.info("Database queries initialized successfully")
                return True

            except Exception as e:
                self.logger.error(f"Failed to initialize database pool: {e}")
                return False

    async def close(self):
        """Close database connection pool"""
        async with self._lock:
            try:
                if self.pool:
                    self.pool.close()
                    await self.pool.wait_closed()
                if self.db_connection:
                    await self.db_connection.close()
                self.initialized = False
            except Exception as e:
                self.logger.error(f"Error closing database connections: {e}")
                raise DatabaseError(f"Failed to close database connections: {e}") from e

    async def execute(
        self,
        query: str,
        params: Union[List[Any], Tuple[Any, ...]] = (),
        retries: int = 0,
    ) -> Any:
        """Execute a SQL query with parameters and retry logic"""
        if not self.initialized and not await self.initialize():
            raise DatabaseError("Database not initialized")

        try:
            async with self.db_connection.transaction() as session:
                result = await session.execute(text(query), params)
                rows = await result.fetchall()
                return [dict(zip(row.keys(), row)) for row in rows]

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}") from e

    async def execute_many(
        self, query: str, params_list: List[Union[List[Any], Tuple[Any, ...]]]
    ) -> bool:
        """Execute multiple SQL queries in a transaction"""
        if not self.initialized and not await self.initialize():
            raise DatabaseError("Database not initialized")

        try:
            async with self.db_connection.transaction() as session:
                for params in params_list:
                    await session.execute(text(query), params)
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute multiple queries: {e}")
            raise DatabaseError(f"Failed to execute multiple queries: {e}")

    async def execute_transaction(self, queries: List[Tuple[str, Any]]) -> bool:
        """Execute multiple queries in a single transaction with proper timeout handling.

        Args:
            queries: List of (query, params) tuples to execute

        Returns:
            bool: True if transaction succeeded

        Raises:
            DatabaseError: If transaction fails or times out
            TimeoutError: If transaction exceeds timeout
        """
        if not self.initialized and not await self.initialize():
            raise DatabaseError("Database not initialized")

        # Validate queries
        if not queries:
            self.logger.warning("Empty transaction queries list")
            return True

        # Configuration
        TRANSACTION_TIMEOUT = 30  # 30 second timeout
        MAX_RETRIES = 3
        BASE_RETRY_DELAY = 0.1

        retry_count = 0
        start_time = time.time()

        while retry_count < MAX_RETRIES:
            try:
                async with asyncio.timeout(TRANSACTION_TIMEOUT):
                    async with self.db_connection.transaction() as session:
                        # Set transaction isolation level
                        await session.execute(
                            "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
                        )

                        # Set statement timeout
                        await session.execute(
                            f"SET LOCAL statement_timeout = {TRANSACTION_TIMEOUT * 1000}"
                        )  # milliseconds

                        # Track query execution time
                        for query, params in queries:
                            query_start = time.time()
                            try:
                                await session.execute(text(query), params)
                                query_duration = time.time() - query_start

                                # Log slow queries
                                if query_duration > 1.0:  # 1 second threshold
                                    self.logger.warning(
                                        f"Slow query detected: {query_duration:.2f}s\nQuery: {query}"
                                    )

                            except Exception as e:
                                self.logger.error(
                                    f"Query failed: {query}\nError: {str(e)}"
                                )
                                raise

                        # Transaction succeeded
                        transaction_duration = time.time() - start_time
                        if transaction_duration > 5.0:  # 5 second threshold
                            self.logger.warning(
                                f"Long transaction detected: {transaction_duration:.2f}s\nQueries: {len(queries)}"
                            )

                        return True

            except asyncio.TimeoutError:
                self.logger.error(
                    f"Transaction timed out after {TRANSACTION_TIMEOUT}s\n"
                    f"Queries: {len(queries)}"
                )
                raise TimeoutError(
                    f"Transaction timed out after {TRANSACTION_TIMEOUT}s"
                )

            except Exception as e:
                retry_count += 1

                # Check if error is retryable
                if self._is_retryable_error(e):
                    if retry_count < MAX_RETRIES:
                        # Exponential backoff
                        delay = BASE_RETRY_DELAY * (2 ** (retry_count - 1))
                        self.logger.warning(
                            f"Retrying transaction after error: {str(e)}\n"
                            f"Attempt {retry_count} of {MAX_RETRIES}, "
                            f"waiting {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                self.logger.error(
                    f"Transaction failed: {str(e)}\n"
                    f"Queries: {len(queries)}\n"
                    f"Duration: {time.time() - start_time:.2f}s"
                )
                raise DatabaseError(f"Transaction failed: {str(e)}")

        raise DatabaseError(f"Transaction failed after {MAX_RETRIES} retries")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()

        # Common retryable errors
        retryable_patterns = [
            "deadlock detected",
            "could not serialize access",
            "concurrent update",
            "lock timeout",
            "connection reset",
            "operation timed out",
            "connection refused",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self.initialized and not await self.initialize():
            raise DatabaseError("Database not initialized")

        try:
            return await self.db_connection.transaction()
        except Exception as e:
            self.logger.error(f"Failed to get database connection: {e}")
            raise DatabaseError(f"Failed to get database connection: {e}")

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions"""
        if not self.initialized and not await self.initialize():
            raise DatabaseError("Database not initialized")

        conn = None
        try:
            conn = await self.get_connection()
            await conn.execute("BEGIN")
            yield conn
            await conn.commit()
        except Exception as e:
            if conn:
                await conn.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise DatabaseError(f"Transaction failed: {e}")
        finally:
            if conn:
                await conn.close()

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self.initialized and not await self.initialize():
                return False

            async with self.db_connection.transaction() as session:
                result = await session.execute(text("SELECT 1"))
                return result is not None and result[0] == 1

        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False

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


async def execute_query(
    db_url: str, query: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute a query using the async DatabaseConnection."""
    db = DatabaseConnection(db_url)
    try:
        await db.initialize()
        return await db.execute(query, params)
    finally:
        await db.close()
