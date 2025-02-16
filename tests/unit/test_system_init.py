#! /usr/bin/env python3
# tests/unit/test_system_init.py
"""
Module: tests.unit
Provides unit testing functionality for the system initialization module.
"""
import asyncio
import gc
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.connection import DatabaseConnection
from exchanges.exchange_manager import ExchangeManager
from startup.system_init import SystemInitializer
from utils.error_handler import handle_error, handle_error_async
from utils.logger import StructuredLogger


@pytest.fixture
def mock_db_connection():
    """Provide a mocked DatabaseConnection."""
    connection = AsyncMock(spec=DatabaseConnection)
    connection.close = AsyncMock()
    connection.verify_connection = AsyncMock(return_value=True)
    return connection


@pytest.fixture
def mock_exchange_manager():
    """Provide a mocked ExchangeManager."""
    manager = AsyncMock(spec=ExchangeManager)
    manager.close = AsyncMock()
    manager.verify_exchange_connection = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_file_handle():
    """Provide a mocked file handle."""
    handle = MagicMock()
    handle.closed = False
    handle.close = MagicMock()
    return handle


@pytest.fixture
def mock_task():
    """Provide a mocked asyncio task."""
    task = AsyncMock()
    task.done = MagicMock(return_value=False)
    task.cancel = MagicMock()
    return task


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def config():
    return {
        "database": {"path": ":memory:"},
        "exchange": {"type": "paper"},
        "risk": {"max_drawdown": 0.1},
    }


@pytest.fixture
async def system_init(config, logger):
    system = SystemInitializer(config, logger)
    yield system
    await system.shutdown()


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
    system_initializer.db_connection.verify_connection.side_effect = Exception(
        "DB Connection Failed"
    )

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
    system_initializer.exchange_manager.verify_exchange_connection.side_effect = (
        Exception("Exchange Unreachable")
    )

    result = await system_initializer.initialize_system()
    assert result is False
    system_initializer.db_connection.verify_connection.assert_awaited_once()
    system_initializer.exchange_manager.verify_exchange_connection.assert_awaited_once()
    system_initializer.logger.error.assert_called_with(
        "Failed to initialize system: Exchange Unreachable"
    )


@pytest.mark.asyncio
async def test_concurrent_initialization(system_init):
    """Test concurrent initialization of components."""

    # Create slow mock initializations
    async def slow_init():
        await asyncio.sleep(0.1)
        return True

    with (
        patch.object(
            system_init, "_init_database", new_callable=AsyncMock, return_value=True
        ),
        patch.object(
            system_init, "_init_exchange", new_callable=AsyncMock, side_effect=slow_init
        ),
        patch.object(
            system_init,
            "_init_market_data",
            new_callable=AsyncMock,
            side_effect=slow_init,
        ),
    ):
        # Try to initialize the same component multiple times
        tasks = [
            asyncio.create_task(system_init._initialize_component("database"))
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks)

        # Only one initialization should succeed
        assert sum(results) == 1
        assert system_init._components_status.get("database", False)
        assert "database" not in system_init._state_transitions


@pytest.mark.asyncio
async def test_dependency_initialization_order(system_init):
    """Test that components wait for their dependencies."""
    init_order = []

    async def mock_init(component):
        await asyncio.sleep(0.1)
        init_order.append(component)
        return True

    with (
        patch.object(
            system_init,
            "_init_database",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init("database"),
        ),
        patch.object(
            system_init,
            "_init_exchange",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init("exchange"),
        ),
        patch.object(
            system_init,
            "_init_market_data",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init("market_data"),
        ),
    ):
        success = await system_init.initialize_system()
        assert success

        # Verify initialization order respects dependencies
        for i, component in enumerate(init_order):
            deps = system_init._dependencies.get(component, set())
            for dep in deps:
                assert dep in init_order[:i]


@pytest.mark.asyncio
async def test_component_failure_cleanup(system_init):
    """Test cleanup of dependent components on failure."""
    initialized_components = set()

    async def mock_init_success(component):
        initialized_components.add(component)
        return True

    async def mock_init_failure(component):
        raise Exception(f"Failed to initialize {component}")

    with (
        patch.object(
            system_init,
            "_init_database",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init_success("database"),
        ),
        patch.object(
            system_init,
            "_init_exchange",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init_success("exchange"),
        ),
        patch.object(
            system_init,
            "_init_market_data",
            new_callable=AsyncMock,
            side_effect=lambda: mock_init_failure("market_data"),
        ),
    ):
        success = await system_init.initialize_system()
        assert not success

        # Verify dependent components are cleaned up
        assert "database" not in system_init._components_status
        assert "exchange" not in system_init._components_status
        assert not system_init._state_transitions
        assert not system_init.initialized


@pytest.mark.asyncio
async def test_shutdown_race_conditions(system_init):
    """Test concurrent shutdown handling."""
    # Initialize system first
    with (
        patch.object(system_init, "_init_database", return_value=True),
        patch.object(system_init, "_init_exchange", return_value=True),
    ):
        await system_init.initialize_system()

    # Try to shutdown multiple times concurrently
    shutdown_tasks = [asyncio.create_task(system_init.shutdown()) for _ in range(3)]
    await asyncio.gather(*shutdown_tasks)

    # Verify clean shutdown
    assert not system_init.initialized
    assert not system_init._components_status
    assert not system_init._state_transitions
    assert system_init._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_component_timeout(system_init):
    """Test handling of component initialization timeout."""

    async def slow_init():
        await asyncio.sleep(31)  # Longer than the timeout
        return True

    with patch.object(
        system_init, "_init_database", new_callable=AsyncMock, side_effect=slow_init
    ):
        success = await system_init._initialize_component("database")
        assert not success
        assert "database" not in system_init._components_status
        assert "database" not in system_init._state_transitions


@pytest.mark.asyncio
async def test_cleanup_during_initialization(system_init):
    """Test cleanup while initialization is in progress."""
    init_started = asyncio.Event()

    async def slow_init():
        init_started.set()
        await asyncio.sleep(0.5)
        return True

    with patch.object(
        system_init, "_init_database", new_callable=AsyncMock, side_effect=slow_init
    ):
        # Start initialization
        init_task = asyncio.create_task(system_init.initialize_system())

        # Wait for initialization to start
        await init_started.wait()

        # Trigger shutdown while initializing
        await system_init.shutdown()

        # Wait for initialization to complete
        await init_task

        # Verify clean state
        assert not system_init.initialized
        assert not system_init._components_status
        assert not system_init._state_transitions


@pytest.mark.asyncio
async def test_resource_tracking(
    system_init, mock_db_connection, mock_exchange_manager, mock_file_handle
):
    """Test resource tracking functionality."""
    # Track resources
    await system_init._track_resource("db_connections", mock_db_connection)
    await system_init._track_resource("exchange_connections", mock_exchange_manager)
    await system_init._track_resource("file_handles", mock_file_handle)

    # Verify resources are tracked
    assert mock_db_connection in system_init._resource_tracker["db_connections"]
    assert (
        mock_exchange_manager in system_init._resource_tracker["exchange_connections"]
    )
    assert mock_file_handle in system_init._resource_tracker["file_handles"]

    # Untrack resources
    await system_init._untrack_resource("db_connections", mock_db_connection)
    assert mock_db_connection not in system_init._resource_tracker["db_connections"]


@pytest.mark.asyncio
async def test_database_cleanup(system_init, mock_db_connection):
    """Test database connection cleanup."""
    await system_init._track_resource("db_connections", mock_db_connection)
    await system_init._cleanup_resources("db_connections")

    mock_db_connection.close.assert_awaited_once()
    assert not system_init._resource_tracker["db_connections"]


@pytest.mark.asyncio
async def test_exchange_cleanup(system_init, mock_exchange_manager):
    """Test exchange connection cleanup."""
    await system_init._track_resource("exchange_connections", mock_exchange_manager)
    await system_init._cleanup_resources("exchange_connections")

    mock_exchange_manager.close.assert_awaited_once()
    assert not system_init._resource_tracker["exchange_connections"]


@pytest.mark.asyncio
async def test_file_handle_cleanup(system_init, mock_file_handle):
    """Test file handle cleanup."""
    await system_init._track_resource("file_handles", mock_file_handle)
    await system_init._cleanup_resources("file_handles")

    mock_file_handle.close.assert_called_once()
    assert not system_init._resource_tracker["file_handles"]


@pytest.mark.asyncio
async def test_task_cleanup(system_init, mock_task):
    """Test task cleanup."""
    await system_init._track_resource("tasks", mock_task)
    await system_init._cleanup_resources("tasks")

    mock_task.cancel.assert_called_once()
    assert not system_init._resource_tracker["tasks"]


@pytest.mark.asyncio
async def test_cleanup_timeout_handling(system_init, mock_db_connection):
    """Test cleanup timeout handling."""
    # Mock a slow closing connection
    mock_db_connection.close = AsyncMock(side_effect=asyncio.sleep(2))
    await system_init._track_resource("db_connections", mock_db_connection)

    # Set short timeout
    system_init._cleanup_timeout = 0.1

    # Cleanup should not hang
    await system_init._cleanup_resources("db_connections")
    system_init.logger.error.assert_called_with("Timeout cleaning up db_connections")


@pytest.mark.asyncio
async def test_cleanup_error_handling(system_init, mock_db_connection):
    """Test cleanup error handling."""
    # Mock a failing close
    mock_db_connection.close = AsyncMock(side_effect=Exception("Test error"))
    await system_init._track_resource("db_connections", mock_db_connection)

    await system_init._cleanup_resources("db_connections")
    system_init.logger.error.assert_called()
    assert not system_init._resource_tracker["db_connections"]


@pytest.mark.asyncio
async def test_comprehensive_shutdown(
    system_init, mock_db_connection, mock_exchange_manager, mock_task
):
    """Test comprehensive shutdown with multiple resources."""
    # Track multiple resources
    await system_init._track_resource("db_connections", mock_db_connection)
    await system_init._track_resource("exchange_connections", mock_exchange_manager)
    await system_init._track_resource("tasks", mock_task)

    system_init.initialized = True
    await system_init.shutdown()

    # Verify all resources were cleaned up
    mock_db_connection.close.assert_awaited_once()
    mock_exchange_manager.close.assert_awaited_once()
    mock_task.cancel.assert_called_once()

    # Verify all trackers are empty
    assert all(not resources for resources in system_init._resource_tracker.values())
    assert not system_init.initialized


@pytest.mark.asyncio
async def test_gc_after_cleanup(system_init):
    """Test garbage collection after cleanup."""
    with patch("gc.collect") as mock_gc:
        await system_init.shutdown()
        mock_gc.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_resource_cleanup(
    system_init, mock_db_connection, mock_exchange_manager
):
    """Test concurrent cleanup of multiple resources."""
    # Create multiple connections
    connections = [AsyncMock(spec=DatabaseConnection) for _ in range(5)]
    for conn in connections:
        conn.close = AsyncMock()
        await system_init._track_resource("db_connections", conn)

    # Cleanup should handle all connections concurrently
    await system_init._cleanup_resources("db_connections")

    # Verify all connections were closed
    for conn in connections:
        conn.close.assert_awaited_once()
    assert not system_init._resource_tracker["db_connections"]
