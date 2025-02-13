import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.startup.system_init import SystemInitializer
from src.database.connection import DatabaseConnection
from src.exchanges.exchange_manager import ExchangeManager
from src.utils.error_handler import handle_error, handle_error_async


@pytest.fixture
def mock_db_connection():
    """Provide a mocked DatabaseConnection."""
    connection = MagicMock(spec=DatabaseConnection)
    connection.verify_connection = AsyncMock(return_value=True)
    return connection


@pytest.fixture
def mock_exchange_manager():
    """Provide a mocked ExchangeManager."""
    manager = MagicMock(spec=ExchangeManager)
    manager.verify_exchange_connection = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def system_initializer(mock_db_connection, mock_exchange_manager, logger):
    """Provide a SystemInitializer instance."""
    return SystemInitializer(
        db_connection=mock_db_connection,
        exchange_manager=mock_exchange_manager,
        logger=logger
    )


@pytest.mark.asyncio
async def test_initialize_success(system_initializer):
    """Test successful system initialization."""
    result = await system_initializer.initialize_system()
    assert result is True
    system_initializer.db_connection.verify_connection.assert_awaited_once()
    system_initializer.exchange_manager.verify_exchange_connection.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_database_failure(system_initializer):
    """Test system initialization failure due to database connection error."""
    system_initializer.db_connection.verify_connection.side_effect = Exception("DB Connection Failed")

    result = await system_initializer.initialize_system()
    assert result is False
    system_initializer.db_connection.verify_connection.assert_awaited_once()
    system_initializer.exchange_manager.verify_exchange_connection.assert_not_awaited()
    system_initializer.logger.error.assert_called_with(
        "Failed to initialize system: DB Connection Failed"
    )


@pytest.mark.asyncio
async def test_initialize_exchange_failure(system_initializer):
    """Test system initialization failure due to exchange connection error."""
    system_initializer.exchange_manager.verify_exchange_connection.side_effect = Exception("Exchange Unreachable")

    result = await system_initializer.initialize_system()
    assert result is False
    system_initializer.db_connection.verify_connection.assert_awaited_once()
    system_initializer.exchange_manager.verify_exchange_connection.assert_awaited_once()
    system_initializer.logger.error.assert_called_with(
        "Failed to initialize system: Exchange Unreachable"
    ) 