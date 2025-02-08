#!/usr/bin/env python3
"""
Module: database/database.py

Provides a production-ready context manager for SQLite connections and
helper functions to execute SQL queries with automatic commits.
"""

import sqlite3

class DBConnection:
    """
    Context manager for DB connection.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable WAL and foreign_keys for better concurrency & integrity
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()


def execute_sql(conn, sql: str, params=None):
    if params is None:
        params = []
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur.fetchall()


def execute_sql_one(conn, sql: str, params=None):
    rows = execute_sql(conn, sql, params)
    return rows[0] if rows else None
