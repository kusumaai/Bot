import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.utils.health_monitor import HealthMonitor
from src.database.queries import DatabaseQueries
from src.utils.error_handler import handle_error_async


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
def health_monitor_fixture(mock_db_queries, mock_exchange_interface, logger):
    """Provide a HealthMonitor instance with mocked dependencies."""
    hm = HealthMonitor(
        db_queries=mock_db_queries,
        exchange_interface=mock_exchange_interface,
        logger=logger
    )
    return hm


@pytest.mark.asyncio
async def test_collect_system_metrics_success(health_monitor_fixture):
    """Test successful collection of system metrics."""
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=75.0)), \
         patch('psutil.disk_usage', return_value=MagicMock(percent=60.0)):
        
        metrics = await health_monitor_fixture.collect_system_metrics()
        assert metrics['cpu_usage'] == 50.0
        assert metrics['memory_usage'] == 75.0
        assert metrics['disk_usage'] == 60.0


@pytest.mark.asyncio
async def test_collect_system_metrics_failure(health_monitor_fixture, logger):
    """Test failure in collecting system metrics."""
    with patch('psutil.cpu_percent', side_effect=Exception("CPU fetch error")), \
         patch('psutil.virtual_memory', return_value=MagicMock(percent=75.0)), \
         patch('psutil.disk_usage', return_value=MagicMock(percent=60.0)):
        
        metrics = await health_monitor_fixture.collect_system_metrics()
        # CPU usage failed, should handle gracefully or set to None
        # Depending on implementation, adjust assertions accordingly
        assert 'cpu_usage' not in metrics  # Example assumption
        logger.error.assert_called_with("Failed to collect CPU usage: CPU fetch error") 