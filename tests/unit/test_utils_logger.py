#! /usr/bin/env python3
#tests/unit/test_utils_logger.py
"""
Module: tests.unit
Provides unit testing functionality for the logger module.
"""
import pytest
import logging
import os
from unittest.mock import patch, MagicMock

from src.utils.logger import setup_logging

def test_setup_logging_default():
    """Test logger setup with default settings."""
    with patch("logging.FileHandler") as mock_file_handler, \
         patch("os.makedirs") as mock_makedirs:
        
        mock_file = MagicMock()
        mock_file_handler.return_value = mock_file
        
        logger = setup_logging(name="DefaultLogger", level="INFO")
        
        assert logger.name == "DefaultLogger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # StreamHandler and FileHandler
        mock_file_handler.assert_called_once_with("logs/KillaBot.log")
        mock_makedirs.assert_called_once_with("logs", exist_ok=True)


def test_setup_logging_custom_log_output():
    """Test logger setup with custom log output via environment variable."""
    with patch.dict(os.environ, {"LOG_OUTPUT": "custom_logs/trading_bot.log"}), \
         patch("os.makedirs") as mock_makedirs, \
         patch("logging.FileHandler") as mock_file_handler:
        
        mock_file = MagicMock()
        mock_file_handler.return_value = mock_file
        logger = setup_logging(name="CustomLogger", level="DEBUG")
        
        mock_makedirs.assert_called_once_with("custom_logs", exist_ok=True)
        mock_file_handler.assert_called_once_with("custom_logs/trading_bot.log")
        assert len(logger.handlers) == 2
        assert logger.handlers[1] == mock_file


def test_setup_logging_invalid_level():
    """Test logger setup with invalid log level."""
    with pytest.raises(ValueError):
        setup_logging(name="InvalidLevelLogger", level="INVALID")


def test_setup_logging_console_only():
    """Test logger setup with console only."""
    with patch("logging.StreamHandler") as mock_stream_handler, \
         patch("os.makedirs", return_value=None) as mock_makedirs:
        
        mock_stream = MagicMock()
        mock_stream_handler.return_value = mock_stream
        
        logger = setup_logging(name="ConsoleOnlyLogger", level="DEBUG", log_file=None)
        
        assert logger.name == "ConsoleOnlyLogger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1  # Only StreamHandler
        mock_stream_handler.assert_called_once()
        mock_makedirs.assert_not_called()  # Since log_file is None 