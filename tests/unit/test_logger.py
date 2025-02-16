"""
Test suite for the enhanced logging system.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.error_handler import LoggingError
from utils.logger import (
    ErrorAggregator,
    LoggerContext,
    StructuredFormatter,
    StructuredLogger,
    setup_logging,
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def structured_logger(temp_log_dir):
    """Create a structured logger instance."""
    return StructuredLogger(
        name="test_logger",
        log_dir=temp_log_dir,
        log_level="DEBUG",
        include_console=False,
    )


def test_structured_formatter():
    """Test structured formatter output."""
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert "timestamp" in data
    assert data["level"] == "INFO"
    assert data["logger"] == "test"
    assert data["message"] == "Test message"
    assert data["module"] == "test"
    assert data["line"] == 1

    # Test with extra fields
    record.extra_fields = {"user": "test_user", "action": "login"}
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["user"] == "test_user"
    assert data["action"] == "login"

    # Test with error information
    try:
        raise ValueError("Test error")
    except ValueError:
        record.exc_info = sys.exc_info()
        formatted = formatter.format(record)
        data = json.loads(formatted)
        assert "error" in data
        assert data["error"]["type"] == "ValueError"
        assert data["error"]["message"] == "Test error"
        assert "traceback" in data["error"]


def test_error_aggregator():
    """Test error aggregator functionality."""
    aggregator = ErrorAggregator(max_history=2)

    # Add errors
    aggregator.add_error(
        "ValueError", "Invalid input", {"user": "test_user", "input": "invalid"}
    )
    aggregator.add_error(
        "TypeError", "Invalid type", {"user": "test_user", "type": "str"}
    )

    # Test error history limit
    aggregator.add_error("RuntimeError", "Test error", {"user": "test_user"})
    assert len(aggregator.errors) == 2  # Max history is 2

    # Test error summary
    summary = aggregator.get_error_summary()
    assert summary["total_errors"] == 2
    assert summary["unique_errors"] == 2
    assert len(summary["recent_errors"]) == 2
    assert "TypeError:Invalid type" in summary["error_counts"]

    # Test clear
    aggregator.clear()
    assert len(aggregator.errors) == 0
    assert len(aggregator.error_counts) == 0


def test_structured_logger_initialization(temp_log_dir):
    """Test structured logger initialization."""
    logger = StructuredLogger(
        name="test",
        log_dir=temp_log_dir,
        log_level="INFO",
    )

    # Check log files creation
    log_path = Path(temp_log_dir)
    assert (log_path / "test.log").exists()
    assert (log_path / "test_error.log").exists()

    # Check handlers
    assert len(logger.logger.handlers) == 3  # Main log, error log, console
    assert any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        for h in logger.logger.handlers
    )
    assert any(isinstance(h, logging.StreamHandler) for h in logger.logger.handlers)


def test_structured_logger_logging(structured_logger, temp_log_dir):
    """Test logging functionality."""
    # Test different log levels
    structured_logger.debug("Debug message", {"extra": "debug"})
    structured_logger.info("Info message", {"extra": "info"})
    structured_logger.warning("Warning message", {"extra": "warning"})
    structured_logger.error("Error message", {"extra": "error"})

    # Read log file
    log_path = Path(temp_log_dir) / "test_logger.log"
    with open(log_path) as f:
        logs = f.readlines()

    # Verify log contents
    assert len(logs) == 4
    for log in logs:
        data = json.loads(log)
        assert "timestamp" in data
        assert "level" in data
        assert "message" in data
        assert "extra" in data

    # Verify error log
    error_log_path = Path(temp_log_dir) / "test_logger_error.log"
    with open(error_log_path) as f:
        error_logs = f.readlines()

    assert len(error_logs) == 1  # Only error message
    error_data = json.loads(error_logs[0])
    assert error_data["level"] == "ERROR"
    assert error_data["extra"] == "error"


def test_error_tracking(structured_logger):
    """Test error tracking functionality."""
    # Generate some errors
    try:
        raise ValueError("Test error 1")
    except ValueError as e:
        structured_logger.error("Error occurred", exc_info=e)

    try:
        raise TypeError("Test error 2")
    except TypeError as e:
        structured_logger.error("Another error", exc_info=e)

    # Check error summary
    summary = structured_logger.get_error_summary()
    assert summary["total_errors"] == 2
    assert summary["unique_errors"] == 2
    assert len(summary["recent_errors"]) == 2

    # Clear errors
    structured_logger.clear_error_history()
    summary = structured_logger.get_error_summary()
    assert summary["total_errors"] == 0
    assert len(summary["recent_errors"]) == 0


def test_logger_context(structured_logger):
    """Test logger context manager."""
    original_level = structured_logger.logger.level

    # Change log level temporarily
    with LoggerContext(structured_logger, "DEBUG"):
        assert structured_logger.logger.level == logging.DEBUG
        structured_logger.debug("Debug message")

    # Verify level is restored
    assert structured_logger.logger.level == original_level


def test_setup_logging(temp_log_dir):
    """Test logging setup function."""
    # Test successful setup
    logger = setup_logging(name="test_setup", log_dir=temp_log_dir, log_level="INFO")
    assert isinstance(logger, StructuredLogger)
    assert logger.logger.level == logging.INFO

    # Test invalid log level
    with pytest.raises(LoggingError):
        setup_logging(name="test_setup", log_dir=temp_log_dir, log_level="INVALID")

    # Test invalid directory
    with pytest.raises(LoggingError):
        setup_logging(name="test_setup", log_dir="/invalid/directory", log_level="INFO")


def test_log_rotation(temp_log_dir):
    """Test log file rotation."""
    logger = StructuredLogger(
        name="rotation_test",
        log_dir=temp_log_dir,
        max_bytes=100,  # Small size to trigger rotation
        backup_count=2,
    )

    # Generate enough logs to trigger rotation
    for i in range(100):
        logger.info(f"Test message {i}" * 10)  # Long message to fill up log file

    log_files = list(Path(temp_log_dir).glob("rotation_test.log*"))
    assert len(log_files) == 3  # Original + 2 backups


def test_concurrent_logging(temp_log_dir):
    """Test concurrent logging operations."""
    import threading
    import time

    logger = StructuredLogger(
        name="concurrent_test",
        log_dir=temp_log_dir,
    )

    def log_messages():
        for i in range(100):
            logger.info(f"Thread message {i}")
            time.sleep(0.001)

    # Create multiple threads
    threads = [threading.Thread(target=log_messages) for _ in range(5)]

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify logs
    log_path = Path(temp_log_dir) / "concurrent_test.log"
    with open(log_path) as f:
        logs = f.readlines()

    assert len(logs) == 500  # 5 threads * 100 messages
    for log in logs:
        assert json.loads(log)["message"].startswith("Thread message")
