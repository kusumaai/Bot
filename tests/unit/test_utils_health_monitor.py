import logging
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from utils.health_monitor import HealthMonitor
from utils.error_handler import handle_error_async
from database.queries import DatabaseQueries


@pytest.fixture
def mock_db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return AsyncMock(spec=DatabaseQueries)


@pytest.fixture
def mock_exchange_interface():
    """Provide a mocked ExchangeInterface."""
    mock_interface = MagicMock()
    mock_interface.exchange = MagicMock()
    mock_interface.exchange.ping = AsyncMock(return_value=True)
    return mock_interface


@pytest.fixture
def logger():
    """Provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def health_monitor(mock_db_queries, mock_exchange_interface, logger):
    """Provide a HealthMonitor instance."""
    return HealthMonitor(
        db_queries=mock_db_queries,
        exchange_interface=mock_exchange_interface,
        logger=logger
    )


@pytest.mark.asyncio
async def test_check_database_success(health_monitor):
    """Test successful database connectivity check."""
    health_monitor.db_queries.ping_database.return_value = True

    success, response_time, error = await health_monitor.check_database()
    assert success is True
    assert response_time >= 0
    assert error is None
    health_monitor.db_queries.ping_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_database_failure(health_monitor):
    """Test database connectivity failure."""
    health_monitor.db_queries.ping_database.side_effect = Exception("DB Connection Failed")

    success, response_time, error = await health_monitor.check_database()
    assert success is False
    assert response_time >= 0
    assert error == "DB Connection Failed"
    health_monitor.db_queries.ping_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_exchange_success(health_monitor):
    """Test successful exchange connectivity check."""
    health_monitor.exchange_interface.exchange.ping.return_value = True

    success, response_time, error = await health_monitor.check_exchange()
    assert success is True
    assert response_time >= 0
    assert error is None
    health_monitor.exchange_interface.exchange.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_exchange_failure(health_monitor):
    """Test exchange connectivity failure."""
    health_monitor.exchange_interface.exchange.ping.side_effect = Exception("Exchange Unreachable")

    success, response_time, error = await health_monitor.check_exchange()
    assert success is False
    assert response_time >= 0
    assert error == "Exchange Unreachable"
    health_monitor.exchange_interface.exchange.ping.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_collect_system_metrics_success(health_monitor):
    """Test successful collection of system metrics."""
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=75.0)), \
         patch('psutil.disk_usage', return_value=MagicMock(percent=60.0)):
        
        metrics = await health_monitor.collect_system_metrics()
        assert metrics['cpu_usage'] == 50.0
        assert metrics['memory_usage'] == 75.0
        assert metrics['disk_usage'] == 60.0


@pytest.mark.asyncio
async def test_collect_system_metrics_failure(health_monitor, logger):
    """Test failure in collecting system metrics."""
    with patch('psutil.cpu_percent', side_effect=Exception("CPU fetch error")), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=75.0)), \
         patch('psutil.disk_usage', return_value=MagicMock(percent=60.0)):
        
        metrics = await health_monitor.collect_system_metrics()
        assert 'cpu_usage' not in metrics  # Assuming implementation skips failed metrics
        logger.error.assert_called_with("Failed to collect CPU usage: CPU fetch error") 