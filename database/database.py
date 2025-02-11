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

logger = logging.getLogger(__name__)

class DBConnection:
    """Context manager for DB connection with proper error handling."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self) -> sqlite3.Connection:
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            # Enable WAL and foreign_keys for better concurrency 
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA busy_timeout = 5000;")  # 5 second timeout
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to database {self.db_path}: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                try:
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Failed to commit transaction: {str(e)}")
                    self.conn.rollback()
            else:
                self.conn.rollback()
            self.conn.close()

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
