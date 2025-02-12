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

class DatabaseError(ApplicationError):
    """Database-related errors"""
    pass

class ExchangeError(ApplicationError):
    """Exchange interaction errors"""
    pass
class ModelError(ApplicationError):
    """ML model-related errors"""
    pass

class CircuitBreakerError(ApplicationError):
    """Circuit breaker errors"""
    pass
class ValidationError(ApplicationError):
    """Data validation errors"""
    pass

class ErrorHandler:
    """Centralized error handling with logging and tracking"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.critical_error_threshold = 5

    def handle_error(self, error: Exception, context: str, logger: Optional[logging.Logger] = None) -> None:
        """Handle non-critical errors"""
        log = logger or self.logger
        log.error(f"Error in {context}: {error}")
        key = f"{context}:{type(error).__name__}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    async def handle_error_async(self, error: Exception, context: str, logger: Optional[logging.Logger] = None) -> None:
        """Handle non-critical errors asynchronously"""
        await asyncio.to_thread(self.handle_error, error, context, logger)

    def handle_critical_error(self, error_detail: Dict[str, Any]) -> None:
        """Handle critical errors with additional actions"""
        try:
            # Notify administrators
            self._send_notification(error_detail)
            
            # Check if circuit breaker should be triggered
            if self._should_trigger_circuit_breaker(error_detail):
                self._trigger_circuit_breaker()
                
        except Exception as e:
            self.logger.critical(f"Critical error handler failed: {e}")

    def _send_notification(self, error_detail: Dict[str, Any]) -> None:
        """Send notification for critical errors"""
        # Implementation depends on notification system
        # Example: Send an email or Slack message
        message = f"Critical Error: {error_detail}"
        self.logger.critical(f"Sending notification: {message}")
        # Placeholder for actual notification logic

    def _should_trigger_circuit_breaker(self, error_detail: Dict[str, Any]) -> bool:
        """Determine if circuit breaker should be triggered"""
        critical_sources = {'OrderManager', 'RiskManager', 'ExchangeInterface'}
        source_type = error_detail.get('source', '')
        error_type = error_detail.get('type', '')
        key = f"{source_type}:{error_type}"
        count = self.error_counts.get(key, 0)
        return key in critical_sources and count >= self.critical_error_threshold

    def _trigger_circuit_breaker(self) -> None:
        """Trigger the circuit breaker mechanism"""
        # Implementation depends on system architecture
        # Example: Update a shared state or notify relevant components
        self.logger.critical("Circuit breaker triggered due to critical errors.")
        # Placeholder for actual circuit breaker logic

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error counts and patterns"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': dict(self.error_counts),
            'last_update': datetime.utcnow().isoformat()
        }

    async def _store_error(self, error_info: Dict[str, Any]) -> None:
        """Store error information in database"""
        query = """
            INSERT INTO error_log (
                timestamp, context, error_type, 
                error_message, traceback, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        params = [
            error_info['timestamp'],
            error_info['context'],
            error_info['type'],
            error_info['message'],
            error_info['traceback'],
            json.dumps(error_info['metadata'])
        ]
        
        async with self.db_connection.get_connection() as conn:
            await conn.execute(query, params)
            
    def get_error_counts(self) -> Dict[str, int]:
        """Get current error counts by type"""
        return self.error_counts.copy()

async def handle_error_async(exception: Exception, location: str, logger: logging.Logger):
    logger.error(f"Error at {location}: {exception}")

def handle_error(exception: Exception, location: str, logger: Optional[logging.Logger] = None):
    logger = logger or logging.getLogger(__name__)
    logger.error(f"Error at {location}: {exception}")

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