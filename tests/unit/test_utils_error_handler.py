#! /usr/bin/env python3
#tests/unit/test_utils_error_handler.py
"""
Module: tests.unit
Provides unit testing functionality for the error handler module.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.utils.error_handler import (
    DatabaseError,
    ExchangeError,
    handle_error,
    handle_error_async
)
import logging

@pytest.fixture
def mock_logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


def test_handle_error_sync(mock_logger):
    """Test synchronous error handling."""
    exception = Exception("Test Exception")
    context = "TestContext"

    handle_error(exception, context, mock_logger, metadata={"key": "value"})

    mock_logger.error.assert_called_with(
        "Error in TestContext: Test Exception",
        exc_info=True,
        extra={"key": "value"}
    )


@pytest.mark.asyncio
async def test_handle_error_async_sync(mock_logger):
    """Test asynchronous error handling."""
    exception = Exception("Async Test Exception")
    context = "AsyncTestContext"

    await handle_error_async(exception, context, mock_logger, metadata={"async_key": "async_value"})

    mock_logger.error.assert_called_with(
        "Error in AsyncTestContext: Async Test Exception",
        exc_info=True,
        extra={"async_key": "async_value"}
    ) 