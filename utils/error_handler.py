#!/usr/bin/env python3
"""
Module: utils/error_handler.py
Centralized error handling utility with database persistence and error aggregation
"""

import logging
import traceback
from typing import Optional, Dict, Any, Type, List, Union
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import time
from collections import defaultdict
import sqlite3
from contextlib import asynccontextmanager

from utils.exceptions import (
    RatchetError, RateLimitExceeded, PositionError,
    InvalidOrderError, ExchangeError, ExchangeAPIError,
    PortfolioError, MathError, CircuitBreakerError,
    MarketDataValidationError, TradingBotError,
    DatabaseError
)

# Global state
_db_pool: Optional[str] = None
_error_counts: Dict[str, int] = defaultdict(int)
_last_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_error_thresholds: Dict[str, int] = {
    "RateLimitExceeded": 5,
    "ExchangeAPIError": 3,
    "MarketDataValidationError": 10,
    "PositionError": 5
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
            conn.execute("""
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
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_aggregates (
                    error_type TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    severity_sum INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            
    except Exception as e:
        logging.error(f"Failed to initialize error handler: {str(e)}")
        raise

async def handle_error_async(error: Exception, context: str, logger: logging.Logger) -> None:
    """Asynchronous error handler"""
    error_msg = f"Error in {context}: {str(error)}"
    if logger:
        logger.error(error_msg, exc_info=True)
    else:
        print(error_msg)  # Fallback if logger isn't available

async def handle_error_threshold_exceeded(
    error_type: str,
    location: str,
    logger: Optional[logging.Logger] = None
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
    error_type: Optional[str] = None,
    time_window: Optional[timedelta] = None
) -> Dict[str, Any]:
    """Get error statistics for analysis"""
    try:
        if not _db_pool:
            return {}
            
        async with db_connection() as conn:
            cursor = conn.cursor()
            
            if error_type:
                # Get stats for specific error type
                cursor.execute("""
                    SELECT * FROM error_aggregates WHERE error_type = ?
                """, (error_type,))
            else:
                # Get all error stats
                cursor.execute("SELECT * FROM error_aggregates")
                
            return {row[0]: {
                "count": row[1],
                "first_seen": row[2],
                "last_seen": row[3],
                "avg_severity": row[4] / row[1] if row[1] > 0 else 0
            } for row in cursor.fetchall()}
            
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
class ExchangeError(Exception):
    pass

class OrderError(Exception):
    pass
class ErrorHandler:
    """Handles and logs errors consistently across the application."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def handle_error(
        self, 
        exception: Exception, 
        context: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle exceptions by logging them with context and metadata."""
        self.logger.error(
            f"Error in {context}: {str(exception)}",
            exc_info=True,
            extra=metadata or {}
        )

    def handle_error_sync(
        self, 
        exception: Exception, 
        context: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle synchronous exceptions."""
        self.logger.error(
            f"Error in {context}: {str(exception)}",
            exc_info=True,
            extra=metadata or {}
        )

def handle_error(e: Exception, context: str, logger: Optional[logging.Logger] = None):
    """Handle errors synchronously by logging."""
    logger = logger or logging.getLogger(__name__)
    logger.error(f"Error in {context}: {e}")
    # Implement additional synchronous error handling actions if necessary

class ErrorTracker:
    def __init__(self, logger: logging.Logger):
        self._lock = asyncio.Lock()
        self.logger = logger
        self.errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.MAX_ERRORS = 100
        self.CLEANUP_INTERVAL = 3600  # 1 hour

    async def track_error(
        self,
        error: Exception,
        context: str,
        severity: str = 'ERROR'
    ) -> None:
        async with self._lock:
            try:
                error_entry = {
                    'timestamp': int(time.time()),
                    'type': type(error).__name__,
                    'message': str(error),
                    'context': context,
                    'severity': severity
                }
                
                self.errors[context].append(error_entry)
                await self._cleanup_old_errors()
                
                if severity in ['ERROR', 'CRITICAL']:
                    self.logger.error(
                        f"{severity} in {context}: {str(error)}",
                        exc_info=True
                    )
                    
            except Exception as e:
                self.logger.error(f"Error tracking failed: {e}")

    async def _cleanup_old_errors(self) -> None:
        now = int(time.time())
        for context in list(self.errors.keys()):
            self.errors[context] = [
                e for e in self.errors[context]
                if now - e['timestamp'] < self.CLEANUP_INTERVAL
            ][:self.MAX_ERRORS] 

def handle_error(e: Exception, context: str, logger: logging.Logger) -> None:
    logger.error(f"Error in {context}: {str(e)}", exc_info=True)

def handle_error_async(e: Exception, context: str, logger: logging.Logger) -> None:
    logger.error(f"Async error in {context}: {str(e)}", exc_info=True) 

def init_error_handler():
    """Initialize global error handlers if necessary."""
    pass  # Implement global error handling setup if required 