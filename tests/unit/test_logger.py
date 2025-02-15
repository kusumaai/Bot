#! /usr/bin/env python3
#tests/unit/test_logger.py
"""
Module: tests.unit
Provides unit testing functionality for the logger module.
"""
import pytest
import logging
import os
from unittest.mock import patch, MagicMock

from src.utils.logger import setup_logging


def test_setup_logging_console_only():
    """Test logger setup with only console handler."""
    handler = setup_logging(console_only=True)
    assert len(handler.handlers) == 1  # Assuming only one handler is added for console


def test_setup_logging_file_output():
    """Test logger setup with file output specified via environment variable."""
    with patch.dict(os.environ, {"LOG_OUTPUT": "test_logs/trading_bot_test.log"}):
        with patch("os.makedirs") as mocked_makedirs, \
             patch("logging.FileHandler") as mock_file_handler:
            mock_file = MagicMock()
            mock_file_handler.return_value = mock_file
            logger = setup_logging(name="FileLogger", level="INFO")
            mock_file_handler.assert_called_once_with("test_logs/trading_bot_test.log")
            mocked_makedirs.assert_called_once_with("test_logs", exist_ok=True)
            assert len(logger.handlers) == 2
            assert logger.handlers[1] == mock_file 