#!/usr/bin/env python3
"""
Module: utils/error_handler.py
Centralized error handling utility
"""

import logging
import traceback
from typing import Optional, Dict, Any, Type, TYPE_CHECKING, List
from datetime import datetime
import json
import asyncio
from pathlib import Path
import time
from collections import defaultdict
from utils.exceptions import (
    RatchetError,
    RateLimitExceeded,
    PositionError,
    InvalidOrderError,
    ExchangeError,
    ExchangeAPIError,
    PortfolioError,
    MathError,
    CircuitBreakerError,
    MarketDataValidationError,
    TradingBotError
)

if TYPE_CHECKING:
    from database.database import DBConnection


# Global connection pool
_db_pool: Optional[str] = None

def init_error_handler(db_path: str) -> None:
    """Initialize error handler with database connection"""
    global _db_pool
    _db_pool = db_path

class ApplicationError(Exception):
    """Base exception class for application-specific errors"""
    pass

class ExchangeError(ApplicationError):
    """Exchange interaction errors"""
    pass

class DatabaseError(Exception):
    """Custom exception for database errors."""
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

async def handle_error_async(e: Exception, context: str, logger: Optional[logging.Logger] = None):
    """Asynchronously handle errors by logging and performing necessary actions."""
    logger = logger or logging.getLogger(__name__)
    logger.error(f"Error in {context}: {e}")
    # Implement additional asynchronous error handling actions if necessary

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