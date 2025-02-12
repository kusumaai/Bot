#!/usr/bin/env python3
"""
Module: database/database.py

Provides a production-ready context manager for SQLite connections and
helper functions to execute SQL queries with automatic commits.
"""

import sqlite3
import logging
from typing import Any, List, Dict, Optional, Union, Tuple
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
import asyncio
import aiosqlite
import json
from datetime import datetime
from utils.numeric_handler import NumericHandler
from utils.error_handler import handle_error, handle_error_async
from trading.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    pass

class DatabaseQueries:
    """Safe database query implementations"""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.nh = NumericHandler()

    async def execute(self, query: str, params: Union[List[Any], Tuple[Any, ...]] = ()) -> Any:
        """Execute a SQL query with parameters"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    await db.commit()
                    return await cursor.fetchall()
        except aiosqlite.Error as e:
            self.logger.error(f"Database execution error: {e}")
            raise DatabaseError(f"Database execution error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in execute: {e}")
            raise DatabaseError(f"Unexpected error in execute: {e}")

    async def store_trade(self, trade: Dict[str, Any]) -> bool:
        """Store a trade in the database"""
        query = """
            INSERT INTO trades (id, symbol, entry_price, size, side, strategy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trade.get('id'),
            trade.get('symbol'),
            trade.get('entry_price'),
            trade.get('size'),
            trade.get('side'),
            trade.get('strategy'),
            json.dumps(trade.get('metadata')) if trade.get('metadata') else None
        )
        try:
            await self.execute(query, params)
            return True
        except DatabaseError as e:
            self.logger.error(f"Failed to store trade: {e}")
            return False

    async def update_position_status(
        self,
        position_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update the status of a position"""
        query = """
            UPDATE positions
            SET status = ?, metadata = ?
            WHERE id = ?
        """
        params = (
            status,
            json.dumps(metadata) if metadata else None,
            position_id
        )
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
            return [{
                'symbol': pos['symbol'],
                'size': self.nh.to_decimal(pos['size']),
                'entry_price': self.nh.to_decimal(pos['entry_price']),
                'current_price': self.nh.to_decimal(pos['current_price']),
                'unrealized_pnl': self.nh.to_decimal(pos['unrealized_pnl']),
                'last_update': pos['last_update']
            } for pos in rows]
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
            async with aiosqlite.connect(self.db_path) as db:
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

        set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
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
