#!/usr/bin/env python3
"""
Module: utils/error_handler.py
Centralized error handling utility
"""

import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
import json
import asyncio

from database.database import DBConnection

# Global connection pool
_db_pool: Optional[str] = None

def init_error_handler(db_path: str) -> None:
    """Initialize error handler with database connection"""
    global _db_pool
    _db_pool = db_path

class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._error_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        
    async def handle_error(self, 
                          error: Exception, 
                          source: str, 
                          context: Optional[Dict] = None) -> None:
        """Handle error with proper logging and tracking"""
        async with self._lock:
            try:
                error_type = type(error).__name__
                error_key = f"{source}:{error_type}"
                
                # Update error counts
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
                
                # Prepare error details
                error_details = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': source,
                    'type': error_type,
                    'message': str(error),
                    'traceback': traceback.format_exc(),
                    'context': context or {},
                    'count': self._error_counts[error_key]
                }
                
                # Log error with appropriate severity
                if self._is_critical_error(error_type, self._error_counts[error_key]):
                    self.logger.critical(json.dumps(error_details, indent=2))
                    await self._handle_critical_error(error_details)
                else:
                    self.logger.error(json.dumps(error_details, indent=2))
                    
            except Exception as e:
                # Fallback error handling
                self.logger.critical(f"Error handler failed: {e}")
                
    def _is_critical_error(self, error_type: str, count: int) -> bool:
        """Determine if error is critical based on type and frequency"""
        critical_types = {
            'DatabaseError': 1,
            'ExchangeError': 3,
            'OrderError': 5,
            'ValidationError': 10
        }
        return count >= critical_types.get(error_type, 20)
        
    async def _handle_critical_error(self, error_details: Dict) -> None:
        """Handle critical errors with additional actions"""
        try:
            # Notify administrators
            await self._send_notification(error_details)
            
            # Check if circuit breaker should be triggered
            if self._should_trigger_circuit_breaker(error_details):
                await self._trigger_circuit_breaker()
                
        except Exception as e:
            self.logger.critical(f"Critical error handler failed: {e}")
            
    async def _send_notification(self, error_details: Dict) -> None:
        """Send notification for critical errors"""
        # Implementation depends on notification system
        # Could be email, Slack, etc.
        pass
        
    def _should_trigger_circuit_breaker(self, error_details: Dict) -> bool:
        """Determine if circuit breaker should be triggered"""
        critical_sources = {'OrderManager', 'RiskManager', 'ExchangeInterface'}
        return (
            error_details['source'] in critical_sources and
            self._error_counts.get(f"{error_details['source']}:{error_details['type']}", 0) >= 5
        )
        
    async def _trigger_circuit_breaker(self) -> None:
        """Trigger circuit breaker in critical situations"""
        try:
            self.logger.critical("Triggering circuit breaker due to critical errors")
            # Implementation depends on circuit breaker interface
            pass
        except Exception as e:
            self.logger.critical(f"Failed to trigger circuit breaker: {e}")
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error counts and patterns"""
        return {
            'total_errors': sum(self._error_counts.values()),
            'error_counts': dict(self._error_counts),
            'last_update': datetime.utcnow().isoformat()
        }

async def handle_error_async(
    e: Exception,
    context: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Centralized async error handling"""
    error_info = {
        'error': str(e),
        'type': type(e).__name__,
        'context': context,
        'timestamp': datetime.utcnow().isoformat(),
        'traceback': traceback.format_exc()
    }
    
    if logger:
        logger.error(f"Error in {context}: {json.dumps(error_info)}")
    
    # Store error in database for monitoring
    if _db_pool:
        try:
            async with DBConnection(_db_pool) as conn:
                await conn.execute_sql(
                    """INSERT INTO error_log 
                       (timestamp, context, error_type, error_message, traceback)
                       VALUES (?, ?, ?, ?, ?)""",
                    [error_info['timestamp'], context, error_info['type'],
                     str(e), error_info['traceback']]
                )
        except Exception as db_error:
            if logger:
                logger.critical(f"Failed to log error to database: {db_error}")

def handle_error(
    e: Exception,
    context: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """Synchronous version of error handler"""
    error_info = {
        'error': str(e),
        'type': type(e).__name__,
        'context': context,
        'timestamp': datetime.utcnow().isoformat(),
        'traceback': traceback.format_exc()
    }
    
    if logger:
        logger.error(f"Error in {context}: {json.dumps(error_info)}")
    
    # For sync version, we'll just log but not store in DB to avoid sync/async conflicts 