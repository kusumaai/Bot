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
    
    def __init__(
        self,
        db_path: str,
        logger: Optional[logging.Logger] = None,
        pool_size: int = 5,
        max_connections: int = 20,
        timeout: float = 30.0
    ):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.timeout = timeout
        self.error_handler = ErrorHandler(self.logger)
        
        # Connection management
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._active_connections: int = 0
        self._lock = asyncio.Lock()
        self._initialize_pool_task = asyncio.create_task(self.initialize_pool())

    async def initialize_pool(self) -> None:
        """Initialize the connection pool"""
        try:
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)
        except Exception as e:
            await self.error_handler.handle_error(
                e, "DatabaseConnection.initialize_pool",
                metadata={"db_path": str(self.db_path)}
            )
            raise DatabaseError("Failed to initialize database pool") from e

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        try:
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except aiosqlite.Error as e:
            await self.error_handler.handle_error(
                e, "DatabaseConnection._create_connection",
                metadata={"db_path": str(self.db_path)}
            )
            raise DatabaseError(f"Failed to create database connection: {e}") from e

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Provide a connection from the pool"""
        conn = None
        try:
            conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
            yield conn
        except asyncio.TimeoutError:
            await self.error_handler.handle_error(
                TimeoutError("Database connection pool timeout."),
                "DatabaseConnection.get_connection",
                metadata={"db_path": self.db_path}
            )
            raise DatabaseError("Database connection pool timeout.") from None
        except Exception as e:
            await self.error_handler.handle_error(
                e, "DatabaseConnection.get_connection",
                metadata={"db_path": self.db_path}
            )
            raise DatabaseError(f"Failed to get database connection: {e}") from e
        finally:
            if conn:
                await self._pool.put(conn)

    async def close_all(self) -> None:
        """Close all database connections"""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()

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