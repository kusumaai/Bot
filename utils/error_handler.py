#!/usr/bin/env python3
"""
Module: utils/error_handler.py
Production-grade error handling and monitoring system
"""

import logging
import traceback
import time
import json
import asyncio
from typing import Any, Optional, Dict, List, Callable
from functools import wraps
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager

class ErrorSeverity:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ErrorTracker:
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, List[Dict[str, Any]]] = {}
        self.error_thresholds = {
            ErrorSeverity.WARNING: 10,
            ErrorSeverity.ERROR: 5,
            ErrorSeverity.CRITICAL: 1
        }
        self.window_minutes = 5
        self.last_reset = time.time()

    def record_error(self, context: str, severity: int, error: Exception) -> None:
        """Record error occurrence"""
        now = time.time()
        
        # Reset counters if window expired
        if now - self.last_reset > self.window_minutes * 60:
            self.error_counts = {}
            self.last_errors = {}
            self.last_reset = now
            
        # Update counts
        key = f"{context}:{severity}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Store error details
        if context not in self.last_errors:
            self.last_errors[context] = []
        self.last_errors[context].append({
            'time': datetime.now().isoformat(),
            'severity': severity,
            'error': str(error),
            'traceback': traceback.format_exc()
        })
        
        # Trim error history
        self.last_errors[context] = self.last_errors[context][-10:]

    def should_alert(self, context: str, severity: int) -> bool:
        """Check if error threshold exceeded"""
        key = f"{context}:{severity}"
        count = self.error_counts.get(key, 0)
        threshold = self.error_thresholds.get(severity, float('inf'))
        return count >= threshold

class SystemHealth:
    def __init__(self):
        self.start_time = time.time()
        self.last_heartbeat = self.start_time
        self.healthy = True
        self.status_checks: Dict[str, bool] = {
            'database': True,
            'exchange': True,
            'memory': True,
            'disk': True
        }
        self.component_latencies: Dict[str, float] = {}

    def update_heartbeat(self) -> None:
        """Update system heartbeat"""
        self.last_heartbeat = time.time()

    def update_component_status(self, component: str, status: bool) -> None:
        """Update component health status"""
        self.status_checks[component] = status
        self.healthy = all(self.status_checks.values())

    def record_latency(self, component: str, latency: float) -> None:
        """Record component operation latency"""
        self.component_latencies[component] = latency

    def get_health_report(self) -> Dict[str, Any]:
        """Generate system health report"""
        return {
            'uptime': time.time() - self.start_time,
            'last_heartbeat': time.time() - self.last_heartbeat,
            'healthy': self.healthy,
            'component_status': self.status_checks,
            'latencies': self.component_latencies
        }

class ErrorHandler:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.error_tracker = ErrorTracker()
        self.health_monitor = SystemHealth()
        self.emergency_shutdown_triggered = False
        
        # Configure logging
        self.setup_logging()
        
        # Start monitoring task
        asyncio.create_task(self.monitor_system_health())

    def setup_logging(self) -> None:
        """Setup enhanced logging"""
        log_config = self.ctx.config.get('log_settings', {})
        
        # File handler
        file_handler = logging.FileHandler(log_config.get('file', 'trading_bot.log'))
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: int = ErrorSeverity.ERROR
    ) -> None:
        """Handle error with proper logging and tracking"""
        # Record error
        self.error_tracker.record_error(context, severity, error)
        
        # Log error
        error_msg = f"Error in {context}: {str(error)}"
        if severity >= ErrorSeverity.ERROR:
            self.ctx.logger.error(error_msg)
            self.ctx.logger.debug(traceback.format_exc())
        else:
            self.ctx.logger.warning(error_msg)
            
        # Check for alerts
        if self.error_tracker.should_alert(context, severity):
            self.trigger_alert(context, severity)
            
        # Check for emergency shutdown
        if severity == ErrorSeverity.CRITICAL:
            self.evaluate_emergency_shutdown()

    def trigger_alert(self, context: str, severity: int) -> None:
        """Trigger error alert"""
        alert_msg = {
            'context': context,
            'severity': severity,
            'error_count': self.error_tracker.error_counts.get(f"{context}:{severity}", 0),
            'recent_errors': self.error_tracker.last_errors.get(context, []),
            'health_status': self.health_monitor.get_health_report()
        }
        
        # Log alert
        self.ctx.logger.critical(f"Alert triggered: {json.dumps(alert_msg, indent=2)}")
        
        # TODO: Add external alert notifications (email, Slack, etc)

    def evaluate_emergency_shutdown(self) -> None:
        """Evaluate if emergency shutdown needed"""
        if self.emergency_shutdown_triggered:
            return
            
        critical_errors = sum(
            count for key, count in self.error_tracker.error_counts.items()
            if key.endswith(f":{ErrorSeverity.CRITICAL}")
        )
        
        if critical_errors >= 3 or not self.health_monitor.healthy:
            self.emergency_shutdown_triggered = True
            self.ctx.logger.critical("Emergency shutdown triggered")
            asyncio.create_task(self.execute_emergency_shutdown())

    async def execute_emergency_shutdown(self) -> None:
        """Execute emergency shutdown procedure"""
        try:
            # Close all positions
            if hasattr(self.ctx, 'exchange_interface'):
                await self.ctx.exchange_interface.emergency_shutdown()
                
            # Cancel all orders
            if hasattr(self.ctx, 'order_manager'):
                await self.ctx.order_manager.cancel_all_orders()
                
            # Log final status
            self.ctx.logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            self.ctx.logger.critical(f"Error during emergency shutdown: {str(e)}")
            
        finally:
            # Stop the event loop
            loop = asyncio.get_event_loop()
            loop.stop()

    async def monitor_system_health(self) -> None:
        """Monitor system health in background"""
        while not self.emergency_shutdown_triggered:
            try:
                # Update heartbeat
                self.health_monitor.update_heartbeat()
                
                # Check database
                db_start = time.time()
                try:
                    with self.ctx.db_pool.connection() as conn:
                        conn.execute("SELECT 1")
                    self.health_monitor.update_component_status('database', True)
                    self.health_monitor.record_latency(
                        'database',
                        time.time() - db_start
                    )
                except Exception:
                    self.health_monitor.update_component_status('database', False)
                    
                # Check exchange connection
                if hasattr(self.ctx, 'exchange_interface'):
                    ex_start = time.time()
                    try:
                        await self.ctx.exchange_interface.ping()
                        self.health_monitor.update_component_status('exchange', True)
                        self.health_monitor.record_latency(
                            'exchange',
                            time.time() - ex_start
                        )
                    except Exception:
                        self.health_monitor.update_component_status('exchange', False)
                        
                # Check system resources
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    self.health_monitor.update_component_status(
                        'memory',
                        memory.percent < 90
                    )
                    self.health_monitor.update_component_status(
                        'disk',
                        disk.percent < 90
                    )
                except ImportError:
                    pass
                    
                # Log health status if not healthy
                if not self.health_monitor.healthy:
                    self.ctx.logger.warning(
                        f"System health check failed: "
                        f"{json.dumps(self.health_monitor.get_health_report(), indent=2)}"
                    )
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.ctx.logger.error(f"Error in health monitor: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

def retry_with_logging(retries: int = 3, delay: int = 1):
    """Decorator for retry with logging"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = args[0] if args else None
            last_error = None
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if ctx and hasattr(ctx, 'logger'):
                        ctx.logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {str(e)}"
                        )
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
                        
            if ctx and hasattr(ctx, 'error_handler'):
                ctx.error_handler.handle_error(
                    last_error,
                    f"retry_wrapper.{func.__name__}",
                    ErrorSeverity.ERROR
                )
            raise last_error
            
        return wrapper
    return decorator

# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler(ctx: Any) -> ErrorHandler:
    """Get or create global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(ctx)
    return _error_handler

def handle_error(error: Exception, context: str, logger: Optional[logging.Logger] = None) -> None:
    """Global error handling function"""
    if _error_handler is not None:
        _error_handler.handle_error(error, context)
    elif logger is not None:
        logger.error(f"Error in {context}: {str(error)}")
        logger.debug(traceback.format_exc())