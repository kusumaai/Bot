import pytest
import logging
import os
from unittest.mock import patch, MagicMock

from utils.logger import setup_logging


def test_setup_logging_default():
    """Test default logger setup."""
    with patch("logging.FileHandler") as mock_file_handler:
        mock_file = MagicMock()
        mock_file_handler.return_value = mock_file
        logger = setup_logging(name="DefaultLogger", level="INFO")
        
        assert logger.name == "DefaultLogger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # StreamHandler and FileHandler
        mock_file_handler.assert_called_once_with("logs/KillaBot.log")
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.handlers[1] == mock_file


def test_setup_logging_custom_log_output():
    """Test logger setup with custom log output."""
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