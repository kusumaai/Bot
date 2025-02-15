#! /usr/bin/env python3
# src/database/queries.py
"""
Module: database/queries.py
Provides safe database query implementations.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite

from database.database import execute_sql
from utils.error_handler import DatabaseError, handle_error_async
from utils.exceptions import DatabaseError
from utils.numeric_handler import NumericHandler

from .connection import DatabaseConnection

logger = logging.getLogger("TradingBot")


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
    """Safe database query implementations"""

    def __init__(self, connection=None, db_path=None, logger=None):
        self.connection = connection
        self.db_path = db_path
        self.logger = logger
        self.nh = NumericHandler()
        self.db_connection = DatabaseConnection(db_path)
        self._lock = asyncio.Lock()
        self.query_builder = QueryBuilder()
        # Alias so that store_order is available if needed by tests.
        self.store_order = self.store_trade

    async def initialize(self) -> bool:
        """Initialize database connection"""
        try:
            # Initialize the underlying connection first
            if not await self.db_connection.initialize():
                return False

            # Test the connection
            async with self._lock:
                query = "SELECT 1"
                await self.db_connection.execute_sql(query)
                self.logger.info("Database connection initialized successfully")
                return True

        except DatabaseError as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during database initialization: {e}")
            return False

    async def execute(
        self, query: str, params: Union[List[Any], Tuple[Any, ...]] = ()
    ) -> Any:
        """Execute a SQL query with parameters"""
        try:
            return await self.db_connection.execute_sql(query, list(params))
        except DatabaseError as e:
            self.logger.error(f"Database execution error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in execute: {e}")
            raise DatabaseError(f"Unexpected error in execute: {e}")

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
                await self.execute(query, params)
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
            results = await self.execute(query, params)
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
            await self.execute(query, params)
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
            results = await self.execute(query, params)
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
            await self.execute(query, params)

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

    async def store_trade(self, trade: Dict[str, Any]) -> bool:
        """Store trade data with validation"""
        query = """
            INSERT INTO trades (
                symbol, side, amount, price, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """
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

            params = (
                trade["symbol"],
                trade["side"],
                str(amount_decimal),
                str(price_decimal),
                timestamp_iso,
            )
            await self.execute(query, params)
            return True

        except (InvalidOperation, KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Invalid trade data: {e}")
            raise DatabaseError(f"Invalid trade data: {e}")
        except Exception as e:
            self.logger.error(f"Failed to store trade: {e}")
            raise DatabaseError(f"Failed to store trade: {e}")

    async def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
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

    async def get_connection(self):
        """Asynchronously get a database connection"""
        try:
            return await aiosqlite.connect(self.db_path)
        except Exception as e:
            self.logger.error(f"Error getting connection: {e}")
            raise e

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
            async with self.db_connection.get_connection() as conn:
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
            async with self.db_connection.get_connection() as conn:
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
            await self.execute(query, params)
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
            result = await self.execute(query, [symbol])
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
