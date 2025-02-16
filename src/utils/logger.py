#! /usr/bin/env python3
# src/utils/logger.py
"""
Enhanced logging system with structured logging, rotation, and error aggregation.
"""
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.error_handler import LoggingError


class StructuredFormatter(logging.Formatter):
    """Format logs in a structured JSON format."""

    def __init__(self, *args, **kwargs):
        self.include_extra_fields = kwargs.pop("include_extra_fields", True)
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        message = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add error information if present
        if record.exc_info:
            message["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        if self.include_extra_fields and hasattr(record, "extra_fields"):
            message.update(record.extra_fields)

        return json.dumps(message)


class ErrorAggregator:
    """Aggregate and track error occurrences with proper memory management."""

    def __init__(self, max_history: int = 1000, cleanup_interval: int = 3600):
        self.max_history = max_history
        self.cleanup_interval = cleanup_interval  # Cleanup every hour by default
        self.errors = deque(maxlen=max_history)  # Use deque with max length
        self.error_counts = {}  # Dict[str, Dict[str, Any]]
        self._lock = threading.Lock()
        self._last_cleanup = datetime.now()
        self.error_expiry = timedelta(hours=24)  # Expire error counts after 24 hours

    def add_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Add an error to the aggregator with timestamp tracking."""
        with self._lock:
            # Create error entry
            error_entry = {
                "type": error_type,
                "message": error_message,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }

            # Add to circular buffer
            self.errors.append(error_entry)

            # Update error counts with timestamp
            error_key = f"{error_type}:{error_message}"
            if error_key not in self.error_counts:
                self.error_counts[error_key] = {
                    "count": 0,
                    "first_seen": datetime.now(),
                    "last_seen": datetime.now(),
                }

            self.error_counts[error_key]["count"] += 1
            self.error_counts[error_key]["last_seen"] = datetime.now()

            # Perform cleanup if needed
            self._maybe_cleanup()

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked errors."""
        with self._lock:
            # Clean up expired errors before generating summary
            self._maybe_cleanup()

            active_errors = {
                k: v
                for k, v in self.error_counts.items()
                if datetime.now() - v["last_seen"] < self.error_expiry
            }

            return {
                "total_errors": len(self.errors),
                "unique_errors": len(active_errors),
                "error_counts": dict(
                    sorted(
                        {k: v["count"] for k, v in active_errors.items()}.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ),
                "recent_errors": list(self.errors)[-10:],  # Last 10 errors
            }

    def clear(self):
        """Clear error history and counts."""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()
            self._last_cleanup = datetime.now()

    def _maybe_cleanup(self):
        """Perform cleanup if cleanup_interval has elapsed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() >= self.cleanup_interval:
            # Remove expired error counts
            expired_keys = [
                k
                for k, v in self.error_counts.items()
                if now - v["last_seen"] >= self.error_expiry
            ]
            for k in expired_keys:
                del self.error_counts[k]

            self._last_cleanup = now


class StructuredLogger:
    """Enhanced logger with structured logging and error tracking."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_level: str = "INFO",
        include_console: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.error_aggregator = ErrorAggregator()

        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Set up file handlers with rotation
        handlers = []

        # Main log file
        main_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        main_handler.setFormatter(StructuredFormatter())
        handlers.append(main_handler)

        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

        # Console handler
        if include_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
            handlers.append(console_handler)

        # Add all handlers
        for handler in handlers:
            self.logger.addHandler(handler)

    def _log(
        self,
        level: int,
        msg: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        exc_info: Optional[BaseException] = None,
    ):
        """Internal method to handle logging with extra fields."""
        if extra_fields:
            extra = {"extra_fields": extra_fields}
        else:
            extra = None

        self.logger.log(level, msg, extra=extra, exc_info=exc_info)

        # Track errors
        if level >= logging.ERROR and exc_info:
            self.error_aggregator.add_error(
                error_type=exc_info.__class__.__name__,
                error_message=str(exc_info),
                context={"message": msg, "extra_fields": extra_fields or {}},
            )

    def debug(self, msg: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log(logging.DEBUG, msg, extra_fields)

    def info(self, msg: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log(logging.INFO, msg, extra_fields)

    def warning(self, msg: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log(logging.WARNING, msg, extra_fields)

    def error(
        self,
        msg: str,
        exc_info: Optional[BaseException] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """Log an error message."""
        self._log(logging.ERROR, msg, extra_fields, exc_info)

    def critical(
        self,
        msg: str,
        exc_info: Optional[BaseException] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, extra_fields, exc_info)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked errors."""
        return self.error_aggregator.get_error_summary()

    def clear_error_history(self):
        """Clear error tracking history."""
        self.error_aggregator.clear()


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(name, log_level=log_level)


def setup_logging(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logging with enhanced features.

    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: None)
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Convert string level to logging level
    try:
        numeric_level = getattr(logging, log_level.upper())
    except (AttributeError, TypeError):
        raise LoggingError(f"Invalid log level: {log_level}")

    logger.setLevel(numeric_level)

    # Create formatter
    formatter = StructuredFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_dir is specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log", maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: StructuredLogger, level: Union[int, str]):
        self.logger = logger
        self.level = (
            level if isinstance(level, int) else getattr(logging, level.upper())
        )
        self.previous_level = logger.logger.level

    def __enter__(self):
        self.logger.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.logger.setLevel(self.previous_level)
