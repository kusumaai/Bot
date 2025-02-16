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
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import aiosqlite

from src.utils.exceptions import (
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
_db_pool: Optional[aiosqlite.Connection] = None
_connection_pool: List[aiosqlite.Connection] = []
_max_pool_size = 10
_pool_lock = asyncio.Lock()
_error_counts: Dict[str, int] = defaultdict(int)
_last_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_error_thresholds: Dict[str, Dict[str, Any]] = {
    "RateLimitExceeded": {
        "base": 5,
        "window": timedelta(minutes=5),
        "scale_factor": 1.0,  # Will be adjusted based on volume
    },
    "ExchangeAPIError": {
        "base": 3,
        "window": timedelta(minutes=1),
        "scale_factor": 1.0,
    },
    "MarketDataValidationError": {
        "base": 10,
        "window": timedelta(minutes=15),
        "scale_factor": 1.0,
    },
    "PositionError": {"base": 5, "window": timedelta(minutes=5), "scale_factor": 1.0},
}
_volume_metrics = {
    "trades_per_minute": 0,
    "last_update": datetime.now(),
    "total_trades": 0,
}
MAX_ERROR_HISTORY = 100
ERROR_EXPIRY = timedelta(hours=1)


@asynccontextmanager
async def get_db_connection():
    """Get a connection from the pool"""
    conn = None
    try:
        async with _pool_lock:
            while not _connection_pool:
                await asyncio.sleep(0.1)  # Wait for a connection to become available
            conn = _connection_pool.pop()
        yield conn
    finally:
        if conn:
            async with _pool_lock:
                _connection_pool.append(conn)


async def init_error_handler(db_path: str, pool_size: int = 10) -> None:
    """Initialize error handler with connection pool and tables"""
    global _db_pool, _max_pool_size

    try:
        _max_pool_size = pool_size
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize main connection for schema setup
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
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

            await conn.execute(
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
            await conn.commit()

        # Initialize connection pool
        for _ in range(_max_pool_size):
            conn = await aiosqlite.connect(db_path)
            _connection_pool.append(conn)

    except Exception as e:
        logging.error(f"Failed to initialize error handler: {str(e)}")
        raise


def handle_error(exception, context, logger, **kwargs):
    extra = kwargs.get("extra") or kwargs.get("metadata", {})
    logger.error(
        f"Error in {context}: {exception}",
        exc_info=kwargs.get("exc_info", True),
        extra=extra,
    )


async def handle_error_async(exception, context, logger, **kwargs):
    """Enhanced async error handling with dynamic thresholds"""
    error_type = exception.__class__.__name__
    current_threshold = await get_current_threshold(error_type)

    try:
        async with get_db_connection() as conn:
            # Log error
            await conn.execute(
                "INSERT INTO error_log (timestamp, error_type, location, message, traceback, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    error_type,
                    context,
                    str(exception),
                    "".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                    json.dumps(kwargs.get("metadata", {})),
                ),
            )
            await conn.commit()

        # Update error counts
        _error_counts[error_type] += 1

        # Check if threshold exceeded
        if _error_counts[error_type] >= current_threshold:
            await handle_error_threshold_exceeded(error_type, context, logger)

    except Exception as e:
        logger.error(f"Error in error handler: {str(e)}")

    # Log the error
    extra = kwargs.get("extra") or kwargs.get("metadata", {})
    logger.error(
        f"Error in {context}: {exception}",
        exc_info=kwargs.get("exc_info", True),
        extra=extra,
    )
    await asyncio.sleep(0)


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

        async with get_db_connection() as conn:
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


class TradingError(Exception):
    """Base class for all trading related errors"""

    pass


class ValidationError(TradingError):
    """Validation related errors"""

    pass


class BacktestError(TradingError):
    """Backtesting related errors"""

    pass


class PositionError(TradingError):
    """Position management errors"""

    pass


class MarketDataError(TradingError):
    """Market data related errors"""

    pass


class ExchangeError(TradingError):
    """Exchange interaction errors"""

    pass


class HealthCheckError(Exception):
    """Custom exception for health check-related errors."""

    pass


class LoggingError(Exception):
    """Exception raised for logging-related errors."""

    pass


class MonitoringError(Exception):
    """Exception raised for monitoring-related errors."""

    pass


class InitializationError(Exception):
    """Exception raised for initialization-related errors."""

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


def init_error_handler():
    """Initialize global error handlers if necessary."""
    pass  # Implement global error handling setup if required
    pass  # Implement global error handling setup if required


async def update_volume_metrics(trades_count: int):
    """Update trading volume metrics for dynamic thresholds"""
    global _volume_metrics
    current_time = datetime.now()
    time_diff = (current_time - _volume_metrics["last_update"]).total_seconds() / 60

    if time_diff > 0:
        _volume_metrics["trades_per_minute"] = trades_count / time_diff
        _volume_metrics["last_update"] = current_time
        _volume_metrics["total_trades"] += trades_count

        # Update error thresholds based on volume
        await _adjust_error_thresholds()


async def _adjust_error_thresholds():
    """Adjust error thresholds based on trading volume"""
    base_volume = 100  # baseline trades per minute
    current_volume = _volume_metrics["trades_per_minute"]

    if current_volume > 0:
        volume_factor = (
            current_volume / base_volume
        ) ** 0.5  # Square root to dampen effect

        for error_type in _error_thresholds:
            _error_thresholds[error_type]["scale_factor"] = max(1.0, volume_factor)


async def get_current_threshold(error_type: str) -> int:
    """Get current threshold for error type, adjusted for volume"""
    if error_type not in _error_thresholds:
        return 5  # Default threshold

    threshold_config = _error_thresholds[error_type]
    return int(threshold_config["base"] * threshold_config["scale_factor"])


async def handle_error_async(
    error: Exception, context: str, logger: Optional[logging.Logger] = None
) -> None:
    """Unified async error handler"""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.error(f"Error in {context}: {str(error)}", exc_info=True)

    if isinstance(error, ValidationError):
        # Handle validation errors
        logger.warning(f"Validation failed in {context}: {str(error)}")
    elif isinstance(error, ExchangeError):
        # Handle exchange errors
        logger.error(f"Exchange error in {context}: {str(error)}")
    else:
        # Handle unexpected errors
        logger.critical(f"Unexpected error in {context}: {str(error)}")


def handle_error(
    error: Exception, context: str, logger: Optional[logging.Logger] = None
) -> None:
    """Unified synchronous error handler"""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
