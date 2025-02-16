#! /usr/bin/env python3
# src/database/queries.py
"""
Module: database/queries.py
Provides safe database query implementations with proper concurrency handling.
"""
import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite
from aiosqlite import Error as SQLiteError

from database.database import execute_sql
from utils.error_handler import DatabaseError, handle_error_async
from utils.exceptions import DatabaseError
from utils.numeric_handler import NumericHandler

from .connection import DatabaseConnection

logger = logging.getLogger("TradingBot")

# Constants for SQLite retry handling
MAX_RETRIES = 3
RETRY_DELAY = 0.1  # seconds
BUSY_TIMEOUT = 5000  # milliseconds


class QueryBuilder:
    """Helper class to build safe SQL queries"""

    @staticmethod
    def build_select(
        table: str,
        fields: List[str],
        conditions: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> tuple[str, List[Any]]:
        """
        Build a safe SELECT query with parameters

        Args:
            table: Table name
            fields: List of fields to select
            conditions: Optional WHERE conditions
            order_by: Optional ORDER BY clause
            limit: Optional LIMIT value
        """
        query = f"SELECT {', '.join(fields)} FROM {table}"
        params: List[Any] = []

        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                if value is None:
                    where_clauses.append(f"{key} IS NULL")
                else:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        return query, params

    def build_insert_trade(self, trade: Dict[str, Any]) -> Tuple[str, tuple]:
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
            None if not trade.get("metadata") else trade.get("metadata"),
        )
        return query, params


class DatabaseQueries:
    """Database queries with proper concurrency handling"""

    def __init__(self, connection=None, db_path=None, logger=None):
        self.db_connection = connection
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.nh = NumericHandler()
        self._lock = asyncio.Lock()
        self.query_builder = QueryBuilder()
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize database connection with proper concurrency settings"""
        try:
            if self.initialized:
                return True

            if not self.db_connection and self.db_path:
                self.db_connection = DatabaseConnection(self.db_path)

            if not self.db_connection:
                self.logger.error("No database connection available")
                return False

            # Set busy timeout and journal mode
            async with self.get_connection() as conn:
                await conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT}")
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA synchronous = NORMAL")

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False

    async def execute_with_retry(
        self,
        query: str,
        params: Union[List[Any], Tuple[Any, ...]] = (),
        retries: int = MAX_RETRIES,
    ) -> Any:
        """Execute a query with retry logic for handling SQLite concurrency"""
        last_error = None

        for attempt in range(retries):
            try:
                async with self.get_connection() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(query, params)
                        await conn.commit()
                        return await cursor.fetchall()

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    last_error = e
                    if attempt < retries - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                raise DatabaseError(f"Database locked after {retries} retries: {e}")

            except SQLiteError as e:
                raise DatabaseError(f"SQLite error: {e}")

            except Exception as e:
                raise DatabaseError(f"Unexpected error: {e}")

        raise DatabaseError(
            f"Failed after {retries} attempts. Last error: {last_error}"
        )

    async def execute_transaction(
        self, queries: List[Tuple[str, Union[List[Any], Tuple[Any, ...]]]]
    ) -> bool:
        """Execute multiple queries in a single transaction with proper locking"""
        async with self._lock:  # Ensure only one transaction at a time
            try:
                async with self.get_connection() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("BEGIN TRANSACTION")

                        for query, params in queries:
                            await cursor.execute(query, params)

                        await conn.commit()
                        return True

            except Exception as e:
                self.logger.error(f"Transaction failed: {e}")
                try:
                    await conn.rollback()
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
                raise DatabaseError(f"Transaction failed: {e}")

    async def store_trade(self, trade: Dict[str, Any]) -> bool:
        """Store trade data with proper concurrency handling and validation"""
        try:
            # Validate required fields
            required_fields = ["symbol", "side", "amount", "price", "timestamp"]
            missing_fields = [field for field in required_fields if field not in trade]
            if missing_fields:
                raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

            # Validate side
            if trade["side"] not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {trade['side']}")

            # Convert and validate numeric values
            amount_decimal = self.nh.to_decimal(trade["amount"])
            price_decimal = self.nh.to_decimal(trade["price"])
            if amount_decimal <= 0 or price_decimal <= 0:
                raise ValueError("Amount and price must be positive")

            timestamp_iso = datetime.fromtimestamp(
                trade["timestamp"], tz=timezone.utc
            ).isoformat()

            # Prepare query and parameters
            query = """
                INSERT INTO trades (
                    symbol, side, amount, price, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """
            params = (
                trade["symbol"],
                trade["side"],
                str(amount_decimal),
                str(price_decimal),
                timestamp_iso,
            )

            # Execute with retry logic
            await self.execute_with_retry(query, params)
            return True

        except (InvalidOperation, KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Invalid trade data: {e}")
            return False
        except DatabaseError as e:
            self.logger.error(f"Database error storing trade: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error storing trade: {e}")
            return False

    async def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get trades with proper concurrency handling"""
        query = "SELECT * FROM trades WHERE symbol = ?"
        try:
            rows = await self.execute_with_retry(query, (symbol,))
            return [dict(row) for row in rows]
        except DatabaseError as e:
            self.logger.error(f"Failed to get trades for {symbol}: {e}")
            return []

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection with proper timeout settings"""
        conn = None
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=BUSY_TIMEOUT / 1000.0,  # Convert to seconds
                isolation_level=None,  # Enable autocommit mode
            )
            yield conn
        finally:
            if conn:
                await conn.close()

    async def insert_candle_data(
        self, symbol: str, timeframe: str, candles: List[Dict[str, Any]]
    ) -> bool:
        """
        Safely insert OHLCV candle data

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            candles: List of candle data dictionaries
        Returns:
            bool: True if successful, raises DatabaseError otherwise
        """
        query = """
            INSERT OR REPLACE INTO candles (
                symbol, timeframe, timestamp,
                open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            for candle in candles:
                params = [
                    symbol,
                    timeframe,
                    candle["timestamp"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                ]
                await self.execute_with_retry(query, params)
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to insert candle data for {symbol}: {str(e)}")

    async def get_recent_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent candles for a symbol

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to fetch
        """
        query, params = self.query_builder.build_select(
            table="candles",
            fields=["timestamp", "open", "high", "low", "close", "volume"],
            conditions={"symbol": symbol, "timeframe": timeframe},
            order_by="timestamp DESC",
            limit=limit,
        )

        try:
            results = await self.execute_with_retry(query, params)
            return results or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to fetch recent candles for {symbol}: {str(e)}"
            )

    async def store_trade_signal(
        self, symbol: str, signal_type: str, direction: str, metadata: Dict[str, Any]
    ) -> bool:
        """
        Store a trading signal with metadata

        Args:
            symbol: Trading pair symbol
            signal_type: Type of signal (e.g., 'GA', 'ML')
            direction: Trade direction ('long' or 'short')
            metadata: Additional signal information
        Returns:
            bool: True if successful, raises DatabaseError otherwise
        """
        query = """
            INSERT INTO trade_signals (
                symbol, signal_type, direction,
                timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?)
        """

        try:
            params = [
                symbol,
                signal_type,
                direction,
                datetime.now(timezone.utc).timestamp(),
                json.dumps(metadata),
            ]
            await self.execute_with_retry(query, params)
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to store trade signal for {symbol}: {str(e)}")

    async def get_active_positions(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch active trading positions

        Args:
            symbol: Optional symbol filter
        """
        conditions = {"status": "active"}
        if symbol:
            conditions["symbol"] = symbol

        query, params = self.query_builder.build_select(
            table="positions",
            fields=[
                "id",
                "symbol",
                "direction",
                "entry_price",
                "size",
                "timestamp",
                "metadata",
            ],
            conditions=conditions,
        )

        try:
            results = await self.execute_with_retry(query, params)
            return results or []
        except Exception as e:
            raise DatabaseError(f"Failed to fetch active positions: {str(e)}")

    async def update_position_status(
        self, position_id: int, status: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update position status and metadata

        Args:
            position_id: Position ID
            status: New status
            metadata: Optional updated metadata
        """
        query = """
            UPDATE positions 
            SET status = ?, 
                metadata = CASE 
                    WHEN ? THEN ?
                    ELSE metadata 
                END,
                updated_at = ?
            WHERE id = ?
        """

        try:
            metadata_flag = metadata is not None
            metadata_json = json.dumps(metadata) if metadata else None
            updated_at = datetime.utcnow().isoformat()
            params = [status, metadata_flag, metadata_json, updated_at, position_id]
            await self.execute_with_retry(query, params)

        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Invalid data when updating position {position_id}: {e}")
            raise DatabaseError(f"Invalid data: {e}")
        except DatabaseError as e:
            self.logger.error(f"Failed to update position {position_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error when updating position {position_id}: {e}"
            )
            raise DatabaseError(f"Unexpected error: {e}")

    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            async with await self.get_connection() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM positions WHERE status = 'OPEN'"
                ) as cursor:
                    positions = await cursor.fetchall()
                    return [
                        {
                            "symbol": pos["symbol"],
                            "size": self.nh.to_decimal(pos["size"]),
                            "entry_price": self.nh.to_decimal(pos["entry_price"]),
                            "current_price": self.nh.to_decimal(pos["current_price"]),
                            "unrealized_pnl": self.nh.to_decimal(pos["unrealized_pnl"]),
                            "last_update": pos["last_update"],
                        }
                        for pos in positions
                    ]
        except Exception as e:
            self.logger.error(f"Error retrieving open positions: {e}")
            return []

    async def insert_trade(self, trade: Dict[str, Any]) -> bool:
        query, params = self.query_builder.build_insert_trade(trade)
        success = await execute_sql(query, params, self.db_path)
        if not success:
            raise DatabaseError("Failed to insert trade into database.")
        return True

    async def get_account_balance(self) -> Decimal:
        """Get current account balance from database"""
        try:
            async with self.get_connection() as conn:
                query = "SELECT balance FROM account_balance ORDER BY timestamp DESC LIMIT 1"
                result = await conn.execute(query)
                row = await result.fetchone()
                if row:
                    return Decimal(str(row["balance"]))
                return Decimal("0")
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return Decimal("0")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions from database"""
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM positions WHERE status = 'open'"
                result = await conn.execute(query)
                rows = await result.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []

    async def create_position(self, position_data: Dict[str, Any]) -> bool:
        """Create a new trading position."""
        query = """
            INSERT INTO positions (
                symbol, direction, entry_price, size,
                status, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            params = [
                position_data["symbol"],
                position_data["direction"],
                str(self.nh.to_decimal(position_data["entry_price"])),
                str(self.nh.to_decimal(position_data["size"])),
                position_data.get("status", "active"),
                position_data["timestamp"],
                json.dumps(position_data.get("metadata", {})),
            ]
            await self.execute_with_retry(query, params)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create position: {e}")
            raise DatabaseError(f"Failed to create position: {e}")

    async def get_latest_trade_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest trade signal for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Optional[Dict[str, Any]]: Latest trade signal data or None if not found
        """
        query = """
            SELECT symbol, signal_type, direction, timestamp, metadata
            FROM trade_signals
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            result = await self.execute_with_retry(query, [symbol])
            if not result:
                return None

            row = result[0]
            return {
                "symbol": row[0],
                "signal_type": row[1],
                "direction": row[2],
                "timestamp": row[3],
                "metadata": json.loads(row[4]),
            }
        except Exception as e:
            self.logger.error(f"Failed to get latest trade signal for {symbol}: {e}")
            raise DatabaseError(f"Failed to get latest trade signal: {e}")

    # Add other database query methods as needed
