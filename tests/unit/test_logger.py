import pytest
import logging
import os
from unittest.mock import patch, MagicMock

from utils.logger import setup_logging


def test_setup_logging_console_only():
    """Test logger setup with only console handler."""
    logger = setup_logging(name="TestLogger", level="DEBUG")
    assert logger.name == "TestLogger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # Console and File handlers
    
    # Check if handlers are correctly set
    console_handler = logger.handlers[0]
    file_handler = logger.handlers[1]
    
    assert isinstance(console_handler, logging.StreamHandler)
    assert isinstance(file_handler, logging.FileHandler)
    
    # Check formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    assert console_handler.formatter == formatter
    assert file_handler.formatter == formatter


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