#! /usr/bin/env python3
# tests/unit/test_logger_extras.py
"""
Module: tests.unit
Provides unit testing functionality for the logger extras module.
"""
import logging
import os

import pytest

from utils.logger import setup_logging


def test_logger_format():
    logger = setup_logging(name="TestFormatLogger", level="INFO")
    # Check that at least one handler (e.g., a StreamHandler) has the correct formatter
    stream_handler = next(
        (h for h in logger.handlers if isinstance(h, logging.StreamHandler)), None
    )
    assert stream_handler is not None
    # Expect the formatter pattern to match the standard format
    expected_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert stream_handler.formatter._fmt == expected_fmt


def test_logger_invalid_directory(monkeypatch):
    # Set a log output to an invalid/unwritable location
    monkeypatch.setenv("LOG_OUTPUT", "/invalid_directory/trading_bot.log")
    with pytest.raises(Exception):
        setup_logging(name="TestInvalidLogger", level="INFO")
