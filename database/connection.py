from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Any, Dict
import aiosqlite
import logging
import asyncio
from datetime import datetime
from pathlib import Path

from utils.error_handler import DatabaseError, ErrorHandler

class DatabaseConnection:
    """Manages database connections with proper pooling and error handling"""
    
    def __init__(
        self,
        db_path: Path,
        logger: logging.Logger,
        pool_size: int = 5,
        max_connections: int = 20,
        timeout: float = 30.0
    ):
        self.db_path = db_path
        self.logger = logger
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.timeout = timeout
        self.error_handler = ErrorHandler(logger)
        
        # Connection management
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._active_connections: int = 0
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the connection pool"""
        try:
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)
        except Exception as e:
            await self.error_handler.handle_error(
                e, "DatabaseConnection.initialize",
                metadata={"db_path": str(self.db_path)}
            )
            raise DatabaseError("Failed to initialize database pool") from e
            
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        try:
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except Exception as e:
            raise DatabaseError(f"Failed to create database connection: {e}")
            
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a database connection from the pool"""
        conn = None
        try:
            async with self._lock:
                if self._active_connections >= self.max_connections:
                    raise DatabaseError("Maximum connection limit reached")
                self._active_connections += 1
            
            conn = await asyncio.wait_for(self._pool.get(), self.timeout)
            yield conn
            await self._pool.put(conn)
            
        except asyncio.TimeoutError:
            raise DatabaseError("Timeout waiting for database connection")
        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            await self.error_handler.handle_error(
                e, "DatabaseConnection.get_connection",
                metadata={"db_path": str(self.db_path)}
            )
            raise DatabaseError("Database connection error") from e
        finally:
            if conn:
                async with self._lock:
                    self._active_connections -= 1
                    
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