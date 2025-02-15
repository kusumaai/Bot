#! /usr/bin/env python3
# src/utils/error_handler.py
"""
Module: src.utils
Provides error handling functionality.
"""
import asyncio
import json
import logging
import sqlite3
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from utils.exceptions import (
    CircuitBreakerError,
    DatabaseError,
    ExchangeAPIError,
    ExchangeError,
    InvalidOrderError,
    MarketDataValidationError,
    MathError,
    PortfolioError,
    PositionError,
    RatchetError,
    RateLimitExceeded,
    TradingBotError,
)

# Global state
_db_pool: Optional[str] = None
_error_counts: Dict[str, int] = defaultdict(int)
_last_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_error_thresholds: Dict[str, int] = {
    "RateLimitExceeded": 5,
    "ExchangeAPIError": 3,
    "MarketDataValidationError": 10,
    "PositionError": 5,
}
MAX_ERROR_HISTORY = 100
ERROR_EXPIRY = timedelta(hours=1)


@asynccontextmanager
async def db_connection():
    """Async context manager for database connections"""
    if not _db_pool:
        raise DatabaseError("Database not initialized")

    conn = None
    try:
        conn = sqlite3.connect(_db_pool)
        yield conn
    finally:
        if conn:
            conn.close()


def init_error_handler(db_path: str) -> None:
    """Initialize error handler with database connection and tables"""
    global _db_pool

    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _db_pool = db_path

        # Initialize database tables
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    location TEXT NOT NULL,
                    message TEXT NOT NULL,
                    traceback TEXT,
                    metadata TEXT,
                    severity INTEGER DEFAULT 1
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS error_aggregates (
                    error_type TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    severity_sum INTEGER DEFAULT 0
                )
            """
            )
            conn.commit()

    except Exception as e:
        logging.error(f"Failed to initialize error handler: {str(e)}")
        raise


def handle_error(exception, context, logger, **kwargs):
    extra = kwargs.get("extra") or kwargs.get("metadata", {})
    logger.error(
        f"Error in {context}: {exception}",
        exc_info=kwargs.get("exc_info", False),
        extra=extra,
    )


async def handle_error_async(exception, context, logger, **kwargs):
    extra = kwargs.get("extra") or kwargs.get("metadata", {})
    logger.error(
        f"Error in {context}: {exception}",
        exc_info=kwargs.get("exc_info", False),
        extra=extra,
    )
    await asyncio.sleep(0)  # ensure coroutine is never None


async def handle_error_threshold_exceeded(
    error_type: str, location: str, logger: Optional[logging.Logger] = None
) -> None:
    """Handle cases where error thresholds are exceeded"""
    msg = f"Error threshold exceeded for {error_type} in {location}"
    if logger:
        logger.critical(msg)
    else:
        logging.critical(msg)

    # Implement circuit breaker or emergency shutdown logic here
    # This could raise a fatal error or trigger system shutdown


async def get_error_stats(
    error_type: Optional[str] = None, time_window: Optional[timedelta] = None
) -> Dict[str, Any]:
    """Get error statistics for analysis"""
    try:
        if not _db_pool:
            return {}

        async with db_connection() as conn:
            cursor = conn.cursor()

            if error_type:
                # Get stats for specific error type
                cursor.execute(
                    """
                    SELECT * FROM error_aggregates WHERE error_type = ?
                """,
                    (error_type,),
                )
            else:
                # Get all error stats
                cursor.execute("SELECT * FROM error_aggregates")

            return {
                row[0]: {
                    "count": row[1],
                    "first_seen": row[2],
                    "last_seen": row[3],
                    "avg_severity": row[4] / row[1] if row[1] > 0 else 0,
                }
                for row in cursor.fetchall()
            }

    except Exception as e:
        logging.error(f"Failed to get error stats: {str(e)}")
        return {}


class ApplicationError(Exception):
    """Base exception class for application-specific errors"""

    pass


class ExchangeError(ApplicationError):
    """Exchange interaction errors"""

    pass


class DatabaseError(Exception):
    """Custom exception for database errors."""

    pass


class RiskError(Exception):
    """Custom exception for risk errors."""

    pass


class ModelError(ApplicationError):
    """ML model-related errors"""

    pass


class CircuitBreakerError(ApplicationError):
    """Circuit breaker errors"""

    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class OrderError(Exception):
    pass


class ExecutionError(Exception):
    """Custom exception for execution-related errors."""

    pass


class MarketDataError(Exception):
    """Custom exception for market data-related errors."""

    pass


class HealthCheckError(Exception):
    """Custom exception for health check-related errors."""

    pass


class ErrorHandler:
    """Handles and logs errors consistently across the application."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def handle_error(
        self,
        exception: Exception,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle exceptions by logging them with context and metadata."""
        self.logger.error(
            f"Error in {context}: {str(exception)}", exc_info=True, extra=metadata or {}
        )

    def handle_error_sync(
        self,
        exception: Exception,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle synchronous exceptions."""
        self.logger.error(
            f"Error in {context}: {str(exception)}", exc_info=True, extra=metadata or {}
        )


def handle_error_async(exception, context, logger, **kwargs):
    metadata = kwargs.get("metadata", {})
    logger.error(f"Async Error in {context}: {exception}. Metadata: {metadata}")


def init_error_handler():
    """Initialize global error handlers if necessary."""
    pass  # Implement global error handling setup if required
    pass  # Implement global error handling setup if required
