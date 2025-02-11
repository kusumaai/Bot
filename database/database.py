#!/usr/bin/env python3
"""
Module: database/database.py

Provides a production-ready context manager for SQLite connections and
helper functions to execute SQL queries with automatic commits.
"""

import sqlite3
import logging
from typing import Any, List, Dict, Optional, Union
from contextlib import contextmanager
from decimal import Decimal
import asyncio
import aiosqlite
import json
from datetime import datetime
from utils.numeric import NumericHandler
from utils.error_handler import handle_error, handle_error_async

logger = logging.getLogger(__name__)

class DBConnection:
    """Context manager for DB connection with proper error handling."""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self._init_pool()
        self.nh = NumericHandler()

    async def _init_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool.maxsize):
            conn = await aiosqlite.connect(self.db_path)
            await self.pool.put(conn)

    async def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> Any:
        """Execute SQL with connection pooling and proper error handling"""
        conn = await self.pool.get()
        try:
            async with conn.cursor() as cursor:
                if params:
                    await cursor.execute(query, params)
                else:
                    await cursor.execute(query)
                await conn.commit()
                return await cursor.fetchall()
        except Exception as e:
            await handle_error_async(e, "DBConnection.execute_sql", self.logger)
            raise
        finally:
            await self.pool.put(conn)

    async def initialize(self):
        """Initialize database with required tables"""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price TEXT NOT NULL,
                        exit_price TEXT,
                        size TEXT NOT NULL,
                        side TEXT NOT NULL,
                        status TEXT NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        pnl TEXT,
                        strategy TEXT NOT NULL,
                        metadata TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        size TEXT NOT NULL,
                        entry_price TEXT NOT NULL,
                        current_price TEXT NOT NULL,
                        unrealized_pnl TEXT NOT NULL,
                        last_update TIMESTAMP NOT NULL
                    );
                    
                    CREATE TABLE IF NOT EXISTS daily_performance (
                        date DATE PRIMARY KEY,
                        starting_balance TEXT NOT NULL,
                        ending_balance TEXT NOT NULL,
                        pnl TEXT NOT NULL,
                        trade_count INTEGER NOT NULL,
                        win_rate TEXT,
                        metadata TEXT
                    );
                """)
                
    async def record_trade(self, trade: Dict) -> bool:
        """Record trade with proper decimal handling"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO trades (
                        id, symbol, entry_price, size, side, status,
                        entry_time, strategy, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade['id'],
                    trade['symbol'],
                    str(self.nh.to_decimal(trade['entry_price'])),
                    str(self.nh.to_decimal(trade['size'])),
                    trade['side'],
                    'OPEN',
                    datetime.utcnow(),
                    trade['strategy'],
                    json.dumps(trade.get('metadata', {}))
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return False
            
    async def update_trade(self, 
                          trade_id: str, 
                          exit_price: Decimal,
                          pnl: Decimal) -> bool:
        """Update trade with exit information"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE trades 
                    SET exit_price = ?, exit_time = ?, pnl = ?, status = ?
                    WHERE id = ?
                """, (
                    str(exit_price),
                    datetime.utcnow(),
                    str(pnl),
                    'CLOSED',
                    trade_id
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
            return False
            
    async def record_daily_performance(self, 
                                     performance: Dict) -> bool:
        """Record daily performance metrics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, starting_balance, ending_balance,
                        pnl, trade_count, win_rate, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance['date'],
                    str(self.nh.to_decimal(performance['starting_balance'])),
                    str(self.nh.to_decimal(performance['ending_balance'])),
                    str(self.nh.to_decimal(performance['pnl'])),
                    performance['trade_count'],
                    str(self.nh.to_decimal(performance['win_rate'])),
                    json.dumps(performance.get('metadata', {}))
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to record performance: {e}")
            return False
            
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM positions") as cursor:
                    positions = await cursor.fetchall()
                    return [{
                        'symbol': pos['symbol'],
                        'size': self.nh.to_decimal(pos['size']),
                        'entry_price': self.nh.to_decimal(pos['entry_price']),
                        'current_price': self.nh.to_decimal(pos['current_price']),
                        'unrealized_pnl': self.nh.to_decimal(pos['unrealized_pnl']),
                        'last_update': pos['last_update']
                    } for pos in positions]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
            
    async def ping(self) -> bool:
        """Check database connection"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
                return True
        except Exception:
            return False

def execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    params: Optional[Union[List[Any], Dict[str, Any]]] = None
) -> List[sqlite3.Row]:
    """Execute SQL with proper error handling and parameter validation."""
    if params is None:
        params = []
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}\nQuery: {sql}\nParams: {params}")
        conn.rollback()
        raise

def execute_sql_one(
    conn: sqlite3.Connection,
    sql: str,
    params: Optional[Union[List[Any], Dict[str, Any]]] = None
) -> Optional[sqlite3.Row]:
    """Execute SQL and return first row with proper error handling."""
    try:
        rows = execute_sql(conn, sql, params)
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}\nQuery: {sql}\nParams: {params}")
        raise

@contextmanager
def transaction(conn: sqlite3.Connection):
    """Transaction context manager with proper rollback."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
