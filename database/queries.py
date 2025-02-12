from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import json
import asyncio

from .connection import DatabaseConnection
from utils.error_handler import DatabaseError

class QueryBuilder:
    """Helper class to build safe SQL queries"""
    
    @staticmethod
    def build_select(
        table: str,
        fields: List[str],
        conditions: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
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

class DatabaseQueries:
    """Safe database query implementations"""
    
    def __init__(self, ctx: Any, logger: logging.Logger):
        self.ctx = ctx
        self.logger = logger
        self._lock = asyncio.Lock()
        self.db_pool = None
        self.query_builder = QueryBuilder()
    
    async def execute(self, query: str, params: tuple = ()) -> List[Any]:
        """Execute SQL with injection protection"""
        async with self._lock:
            try:
                async with DatabaseConnection(self.db_pool) as conn:
                    return await conn.execute_sql(query, params)
            except Exception as e:
                self.logger.error(f"Database error: {e}")
                raise DatabaseError(f"Query failed: {str(e)}")

    async def insert_candle_data(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]]
    ) -> None:
        """
        Safely insert OHLCV candle data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            candles: List of candle data dictionaries
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
                    candle['timestamp'],
                    candle['open'],
                    candle['high'],
                    candle['low'],
                    candle['close'],
                    candle['volume']
                ]
                await self.execute(query, params)
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to insert candle data for {symbol}: {str(e)}"
            )
    
    async def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
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
            limit=limit
        )
        
        try:
            results = await self.execute(query, params)
            return results or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to fetch recent candles for {symbol}: {str(e)}"
            )
    
    async def store_trade_signal(
        self,
        symbol: str,
        signal_type: str,
        direction: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store a trading signal with metadata
        
        Args:
            symbol: Trading pair symbol
            signal_type: Type of signal (e.g., 'GA', 'ML')
            direction: Trade direction ('long' or 'short')
            metadata: Additional signal information
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
                datetime.utcnow().timestamp(),
                json.dumps(metadata)
            ]
            await self.execute(query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to store trade signal for {symbol}: {str(e)}"
            )
    
    async def get_active_positions(
        self,
        symbol: Optional[str] = None
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
                "id", "symbol", "direction", "entry_price",
                "size", "timestamp", "metadata"
            ],
            conditions=conditions
        )
        
        try:
            results = await self.execute(query, params)
            return results or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to fetch active positions: {str(e)}"
            )
    
    async def update_position_status(
        self,
        position_id: int,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
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
                    WHEN ? IS NOT NULL THEN ?
                    ELSE metadata 
                END,
                updated_at = ?
            WHERE id = ?
        """
        
        try:
            params = [
                status,
                metadata is not None,
                json.dumps(metadata) if metadata else None,
                datetime.utcnow().timestamp(),
                position_id
            ]
            await self.execute(query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to update position {position_id}: {str(e)}"
            )

    async def store_trade(self, trade: Dict[str, Any]) -> None:
        query = """
            INSERT INTO trades (
                symbol, side, amount, price, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """
        params = (
            trade['symbol'],
            trade['side'], 
            str(trade['amount']),
            str(trade['price']),
            trade['timestamp']
        )
        await self.execute(query, params)

    async def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        query = "SELECT * FROM trades WHERE symbol = ?"
        return await self.execute(query, (symbol,)) 