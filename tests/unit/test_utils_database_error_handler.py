import logging
import pytest
from unittest.mock import MagicMock

from utils.error_handler import DatabaseError, handle_error
from utils.error_handler import handle_error_async


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
        extra={"operation": "insert"}
    )


@pytest.mark.asyncio
async def test_handle_error_database_error_async(mock_logger):
    """Test handling of an asynchronous DatabaseError."""
    exception = DatabaseError("Database Query Failed")
    context = "AsyncTestContext"

    await handle_error_async(exception, context, mock_logger, metadata={"operation": "query"})
    
    mock_logger.error.assert_called_with(
        "Error in AsyncTestContext: Database Query Failed",
        exc_info=True,
        extra={"operation": "query"}
    ) 