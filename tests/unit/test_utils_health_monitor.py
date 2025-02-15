#! /usr/bin/env python3
# tests/unit/test_utils_health_monitor.py
"""
Module: tests.unit
Provides unit testing functionality for the health monitor module.
"""
import asyncio
import logging
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from database.queries import DatabaseQueries
from utils.error_handler import handle_error_async
from utils.health_monitor import HealthMonitor


# fixture for database queries
@pytest.fixture
def mock_db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


# fixture for exchange interface
@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    mock_interface = MagicMock()
    mock_interface.exchange = MagicMock()
    mock_interface.exchange.ping = AsyncMock(return_value=True)
    return mock_interface


# fixture for logger
@pytest.fixture
def mock_logger():
    mock = MagicMock()
    return mock


# fixture for context
@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.logger = MagicMock()
    ctx.db_connection = MagicMock()
    ctx.exchange_interface = MagicMock()
    ctx.market_data = MagicMock()
    return ctx


# fixture for health monitor
@pytest.fixture
def health_monitor(mock_logger, mock_db_queries):
    """Provide a HealthMonitor instance."""
    return HealthMonitor(db_queries=mock_db_queries, logger=mock_logger)


# test successful database connectivity check
@pytest.mark.asyncio
async def test_check_database_success(health_monitor):
    """Test successful database connectivity check."""
    health_monitor.db_queries.ping_database.return_value = True

    success, response_time, error = await health_monitor.check_database()
    assert success is True
    assert response_time >= 0
    assert error is None
    health_monitor.db_queries.ping_database.assert_awaited_once()


# test database connectivity failure
@pytest.mark.asyncio
async def test_check_database_failure(health_monitor):
    """Test database connectivity failure."""
    health_monitor.db_queries.ping_database.side_effect = Exception(
        "DB Connection Failed"
    )

    success, response_time, error = await health_monitor.check_database()
    assert success is False
    assert response_time >= 0
    assert error == "DB Connection Failed"
    health_monitor.db_queries.ping_database.assert_awaited_once()


# test successful exchange connectivity check
@pytest.mark.asyncio
async def test_check_exchange_success(health_monitor):
    """Test successful exchange connectivity check."""
    health_monitor.exchange_interface.exchange.ping.return_value = True

    success, response_time, error = await health_monitor.check_exchange()
    assert success is True
    assert response_time >= 0
    assert error is None
    health_monitor.exchange_interface.exchange.ping.assert_awaited_once()


# test exchange connectivity failure
@pytest.mark.asyncio
async def test_check_exchange_failure(health_monitor):
    """Test exchange connectivity failure."""
    health_monitor.exchange_interface.exchange.ping.side_effect = Exception(
        "Exchange Unreachable"
    )

    success, response_time, error = await health_monitor.check_exchange()
    assert success is False
    assert response_time >= 0
    assert error == "Exchange Unreachable"
    health_monitor.exchange_interface.exchange.ping.assert_awaited_once()


# test overall system health when all components are healthy
@pytest.mark.asyncio
async def test_is_system_healthy_all_components_healthy(health_monitor):
    """Test overall system health when all components are healthy."""
    health_monitor.check_database = AsyncMock(return_value=(True, 0.1, None))
    health_monitor.check_exchange = AsyncMock(return_value=(True, 0.2, None))
    health_monitor.check_market_data = AsyncMock(return_value=True)

    is_healthy = await health_monitor.is_system_healthy()
    assert is_healthy is True
    health_monitor.check_database.assert_awaited_once()
    health_monitor.check_exchange.assert_awaited_once()
    health_monitor.check_market_data.assert_awaited_once()


# test overall system health when one component is unhealthy
@pytest.mark.asyncio
async def test_is_system_healthy_one_component_unhealthy(health_monitor):
    """Test overall system health when one component is unhealthy."""
    health_monitor.check_database = AsyncMock(return_value=(False, 0.1, "DB Error"))
    health_monitor.check_exchange = AsyncMock(return_value=(True, 0.2, None))
    health_monitor.check_market_data = AsyncMock(return_value=True)

    is_healthy = await health_monitor.is_system_healthy()
    assert is_healthy is False
    health_monitor.check_database.assert_awaited_once()
    health_monitor.check_exchange.assert_awaited_once()
    health_monitor.check_market_data.assert_awaited_once()


# test successful collection of system metrics
@pytest.mark.asyncio
async def test_collect_system_metrics_success(health_monitor):
    """Test successful collection of system metrics."""
    with (
        patch("psutil.cpu_percent", return_value=50.0),
        patch("psutil.virtual_memory", return_value=MagicMock(percent=75.0)),
        patch("psutil.disk_usage", return_value=MagicMock(percent=60.0)),
    ):

        metrics = await health_monitor.collect_system_metrics()
        assert metrics["cpu_usage"] == 50.0
        assert metrics["memory_usage"] == 75.0
        assert metrics["disk_usage"] == 60.0


@pytest.mark.asyncio
async def test_collect_system_metrics_failure(health_monitor, mock_logger):
    """Test failure in collecting system metrics."""
    with (
        patch("psutil.cpu_percent", side_effect=Exception("CPU fetch error")),
        patch("psutil.virtual_memory", return_value=MagicMock(percent=75.0)),
        patch("psutil.disk_usage", return_value=MagicMock(percent=60.0)),
    ):

        metrics = await health_monitor.collect_system_metrics()
        assert "cpu_usage" not in metrics  # Example assumption
        mock_logger.error.assert_called_with(
            "Failed to collect CPU usage: CPU fetch error"
        )


# test successful system resource check
@pytest.mark.asyncio
async def test_check_system_resources_success(health_monitor):
    """Test successful system resource check."""
    with (
        patch("psutil.virtual_memory", return_value=MagicMock(percent=50.0)),
        patch("psutil.disk_usage", return_value=MagicMock(percent=60.0)),
        patch("psutil.cpu_percent", return_value=70.0),
    ):

        metrics = health_monitor.check_system_resources()
        assert metrics["healthy"] is True
        assert metrics["memory_used_pct"] == 50.0
        assert metrics["disk_used_pct"] == 60.0
        assert metrics["cpu_used_pct"] == 70.0
        assert metrics["memory_healthy"] is True
        assert metrics["disk_healthy"] is True
        assert metrics["cpu_healthy"] is True


@pytest.mark.asyncio
async def test_check_system_resources_critical(health_monitor):
    """Test system resources in critical state."""
    with (
        patch("psutil.virtual_memory", return_value=MagicMock(percent=95.0)),
        patch("psutil.disk_usage", return_value=MagicMock(percent=92.0)),
        patch("psutil.cpu_percent", return_value=98.0),
    ):

        metrics = health_monitor.check_system_resources()
        assert metrics["healthy"] is False
        assert metrics["memory_healthy"] is False
        assert metrics["disk_healthy"] is False
        assert metrics["cpu_healthy"] is False
        assert (
            "System resources critical" in health_monitor.components["system"].message
        )


# test successful market data check
@pytest.mark.asyncio
async def test_check_market_data_success(health_monitor):
    """Test successful market data check."""
    health_monitor.ctx.market_data.get_latest_data = AsyncMock(
        return_value={"BTC/USDT": {"price": "50000"}}
    )

    success, response_time, error = await health_monitor.check_market_data()
    assert success is True
    assert response_time >= 0
    assert error is None
    assert health_monitor.components["market_data"].status is True


@pytest.mark.asyncio
async def test_check_market_data_failure(health_monitor):
    """Test market data check failure."""
    health_monitor.ctx.market_data.get_latest_data = AsyncMock(return_value=None)

    success, response_time, error = await health_monitor.check_market_data()
    assert success is False
    assert response_time >= 0
    assert "No market data available" in error
    assert health_monitor.components["market_data"].status is False


# test component metric updates
@pytest.mark.asyncio
async def test_update_component_metrics(health_monitor):
    """Test component metric updates."""
    component = "exchange"
    await health_monitor.update_component_metrics(component, 0.1)

    # Test with multiple updates to check statistics
    for _ in range(10):
        await health_monitor.update_component_metrics(component, 0.1)

    assert len(health_monitor.latency_history[component]) == 11
    assert health_monitor.components[component].response_time == 0.1


@pytest.mark.asyncio
async def test_should_emergency_shutdown(health_monitor):
    """Test emergency shutdown conditions."""
    # Simulate critical component failures
    health_monitor.components["database"].status = False
    health_monitor.components["exchange"].status = False
    health_monitor.components["database"].error_count = 6

    assert health_monitor.should_emergency_shutdown() is True


# test system metrics collection failure
@pytest.mark.asyncio
async def test_get_system_metrics_failure(health_monitor):
    """Test system metrics collection failure."""
    with patch("psutil.cpu_percent", side_effect=Exception("CPU Error")):
        metrics = health_monitor.get_system_metrics()
        assert metrics["cpu_used_pct"] == 100.0  # Fail-safe value
        assert metrics["memory_used_pct"] == 100.0
        assert metrics["disk_used_pct"] == 100.0


# test system readiness check with fresh system (no data)
@pytest.mark.asyncio
async def test_check_system_readiness_fresh_system(health_monitor):
    """Test system readiness check with fresh system (no data)."""
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.side_effect = [
        (0,),  # No active models
        (0,),  # No GA data
        (0,),  # No market data
    ]

    mock_conn = AsyncMock()
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

    health_monitor.ctx.db_connection.pool.acquire.return_value = mock_conn

    is_ready, readiness = await health_monitor.check_system_readiness()

    assert is_ready is False
    assert readiness == {
        "database": True,
        "models": False,
        "ga_data": False,
        "market_data": False,
        "overall": False,
    }


# test system readiness check with partial data
@pytest.mark.asyncio
async def test_check_system_readiness_partially_ready(health_monitor):
    """Test system readiness check with partial data."""
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.side_effect = [
        (1,),  # Has active models
        (0,),  # No GA data
        (100,),  # Has market data
    ]

    mock_conn = AsyncMock()
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

    health_monitor.ctx.db_connection.pool.acquire.return_value = mock_conn

    is_ready, readiness = await health_monitor.check_system_readiness()

    assert is_ready is False
    assert readiness == {
        "database": True,
        "models": True,
        "ga_data": False,
        "market_data": True,
        "overall": False,
    }


@pytest.mark.asyncio
async def test_check_system_readiness_fully_ready(health_monitor):
    """Test system readiness check with all required data."""
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.side_effect = [
        (2,),  # Has active models
        (5,),  # Has GA data
        (1000,),  # Has market data
    ]

    mock_conn = AsyncMock()
    mock_conn.__aenter__.return_value = mock_conn
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

    health_monitor.ctx.db_connection.pool.acquire.return_value = mock_conn

    is_ready, readiness = await health_monitor.check_system_readiness()

    assert is_ready is True
    assert readiness == {
        "database": True,
        "models": True,
        "ga_data": True,
        "market_data": True,
        "overall": True,
    }


# test system readiness check with database error
@pytest.mark.asyncio
async def test_check_system_readiness_db_error(health_monitor):
    """Test system readiness check with database error."""
    health_monitor.ctx.db_connection.pool.acquire.side_effect = Exception(
        "DB Connection Error"
    )

    is_ready, readiness = await health_monitor.check_system_readiness()

    assert is_ready is False
    assert readiness == {
        "database": False,
        "models": False,
        "ga_data": False,
        "market_data": False,
        "overall": False,
    }


# test system readiness check with no database connection
@pytest.mark.asyncio
async def test_check_system_readiness_no_db_connection(health_monitor):
    """Test system readiness check with no database connection."""
    health_monitor.ctx.db_connection = None

    is_ready, readiness = await health_monitor.check_system_readiness()

    assert is_ready is False
    assert readiness == {
        "database": False,
        "models": False,
        "ga_data": False,
        "market_data": False,
        "overall": False,
    }
