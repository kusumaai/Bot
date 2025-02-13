from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Any, Dict
import aiosqlite
import logging
import asyncio
from datetime import datetime
from pathlib import Path

from utils.error_handler import DatabaseError, ErrorHandler

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages database connections with proper pooling and error handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = None
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """Initialize the database connection pool"""
        try:
            import aiosqlite
            self.pool = await aiosqlite.connect(self.db_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            return False

    async def execute_sql(self, query: str, params: List[Any] = None) -> Any:
        """Execute a SQL query with proper error handling"""
        if not self.pool:
            raise DatabaseError("Database connection not initialized")
            
        async with self._lock:
            try:
                async with self.pool.execute(query, params or []) as cursor:
                    result = await cursor.fetchall()
                    await self.pool.commit()
                    return result
            except Exception as e:
                await self.pool.rollback()
                raise DatabaseError(f"Query execution failed: {str(e)}")

    async def close(self):
        """Close the database connection"""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def execute(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a query with proper error handling"""
        async with self.get_connection() as conn:
            try:
                cursor = await conn.execute(query, params or [])
                if fetch:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                await conn.commit()
                return None
                
            except Exception as e:
                await conn.rollback()
                await self.error_handler.handle_error(
                    e, "DatabaseConnection.execute",
                    metadata={
                        "query": query,
                        "params": params,
                        "fetch": fetch
                    }
                )
                raise DatabaseError("Query execution failed") from e 

class DBConnection:
    def __init__(self, pool: Any):
        self.pool = pool
        self.conn = None
        self.tx = None
        
    async def __aenter__(self):
        self.conn = await self.pool.acquire()
        self.tx = self.conn.transaction()
        await self.tx.start()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                await self.tx.commit()
            else:
                await self.tx.rollback()
        finally:
            await self.pool.release(self.conn)
            
    async def execute_sql(self, query: str, params: tuple) -> List[Any]:
        """Execute SQL with proper parameter binding"""
        return await self.conn.fetch(query, *params) 