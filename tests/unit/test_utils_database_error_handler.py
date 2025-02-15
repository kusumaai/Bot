#! /usr/bin/env python3
# tests/unit/test_utils_database_error_handler.py
"""
Module: tests.unit
Provides unit testing functionality for the database error handler module.
"""
import logging
from unittest.mock import MagicMock

import pytest

from utils.error_handler import DatabaseError, handle_error, handle_error_async


@pytest.fixture
def mock_logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


def test_handle_error_database_error_sync(mock_logger):
    """Test handling of a synchronous DatabaseError."""
    exception = DatabaseError("Database Insert Failed")
    context = "TestContext"

    handle_error(exception, context, mock_logger, metadata={"operation": "insert"})

    mock_logger.error.assert_called_with(
        "Error in TestContext: Database Insert Failed",
        exc_info=True,
        extra={"operation": "insert"},
    )


@pytest.mark.asyncio
async def test_handle_error_database_error_async(mock_logger):
    """Test handling of an asynchronous DatabaseError."""
    exception = DatabaseError("Database Query Failed")
    context = "AsyncTestContext"

    await handle_error_async(
        exception, context, mock_logger, metadata={"operation": "query"}
    )

    mock_logger.error.assert_called_with(
        "Error in AsyncTestContext: Database Query Failed",
        exc_info=True,
        extra={"operation": "query"},
    )
