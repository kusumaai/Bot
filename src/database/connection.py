#! /usr/bin/env python3
#src/database/connection.py
"""
Module: database/connection.py
Provides database connection management.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Any, Dict
import aiosqlite
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3

from utils.error_handler import DatabaseError, ErrorHandler

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages database connections with proper pooling and error handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = None
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        self.connection = None
        
    async def initialize(self) -> bool:
        """Initialize the database connection"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA foreign_keys = ON;")
                # Test connection
                await conn.execute("SELECT 1")
                self.logger.info("Database connection initialized successfully")
                return True
        except aiosqlite.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during database initialization: {e}")
            return False

    @asynccontextmanager
    async def get_connection(self):
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA foreign_keys = ON;")
                yield conn
        except aiosqlite.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database connection error: {e}") from e

    async def execute_sql(self, query: str, params: List[Any] = None) -> Any:
        """Execute a SQL query with proper error handling"""
        try:
            async with self.get_connection() as conn:
                async with conn.execute(query, params or []) as cursor:
                    result = await cursor.fetchall()
                    await conn.commit()
                    return result
        except Exception as e:
            self.logger.error(f"Failed to execute SQL: {str(e)}")
            raise DatabaseError(f"Database query failed: {str(e)}") from e

    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        with self.connection:
            cursor = self.connection.cursor()
            
            # Add your table creation SQL statements here
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    status TEXT NOT NULL
                )
            ''')
            # Add more table creation statements as needed

    async def close(self):
        """Close the database connection"""
        if self.connection:
            await self.connection.close()
            self.connection = None

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