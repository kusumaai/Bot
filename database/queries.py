from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json
import asyncio

from .connection import DatabaseConnection
from utils.error_handler import DatabaseError
from utils.numeric_handler import NumericHandler

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
    
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.nh = NumericHandler()
        self.db_connection = DBConnection(db_path)
        self._lock = asyncio.Lock()
        self.db_pool = None
        self.query_builder = QueryBuilder()
    
    async def execute(self, query: str, params: Union[List[Any], Tuple[Any, ...]] = ()) -> Any:
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
            params = [
                status,
                metadata_flag,
                metadata_json,
                updated_at,
                position_id
            ]
            await self.execute(query, params)
            
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Invalid data when updating position {position_id}: {e}")
            raise DatabaseError(f"Invalid data: {e}")
        except DatabaseError as e:
            self.logger.error(f"Failed to update position {position_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error when updating position {position_id}: {e}")
            raise DatabaseError(f"Unexpected error: {e}")

    async def store_trade(self, trade: Dict[str, Any]) -> None:
        query = """
            INSERT INTO trades (
                symbol, side, amount, price, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """
        try:
            amount_decimal = self.nh.to_decimal(trade['amount'])
            price_decimal = self.nh.to_decimal(trade['price'])
            timestamp_iso = datetime.utcfromtimestamp(trade['timestamp']).isoformat()
            params = (
                trade['symbol'],
                trade['side'], 
                str(amount_decimal),
                str(price_decimal),
                timestamp_iso
            )
            await self.execute(query, params)
        except (InvalidOperation, KeyError, TypeError) as e:
            self.logger.error(f"Invalid trade data: {e}")
            raise DatabaseError(f"Invalid trade data: {e}")
        except DatabaseError as e:
            self.logger.error(f"Failed to store trade: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error when storing trade: {e}")
            raise DatabaseError(f"Unexpected error: {e}")

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

    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            async with self.get_connection() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM positions WHERE status = 'OPEN'") as cursor:
                    positions = await cursor.fetchall()
                    return [{
                        'symbol': pos['symbol'],
                        'size': self.nh.to_decimal(pos['size']),
                        'entry_price': self.nh.to_decimal(pos['entry_price']),
                        'current_price': self.nh.to_decimal(pos['current_price']),
                        'unrealized_pnl': self.nh.to_decimal(pos['unrealized_pnl']),
                        'last_update': pos['last_update']
                    } for pos in positions]
        except aiosqlite.Error as e:
            self.logger.error(f"Database error when retrieving positions: {e}")
            return []
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Invalid data when retrieving positions: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when retrieving positions: {e}")
            return [] 