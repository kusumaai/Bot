#! /usr/bin/env python3
# tests/unit/test_health_monitor.py
"""
Module: tests.unit
Provides unit testing functionality for the health monitor module.
"""
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.queries import DatabaseQueries
from src.utils.error_handler import handle_error_async
from src.utils.health_monitor import HealthMonitor


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
        logger=logger,
    )
    return hm


@pytest.mark.asyncio
async def test_health_monitor_initialization(health_monitor_fixture):
    """Test initialization of HealthMonitor."""
    assert health_monitor_fixture.db_queries is not None
    assert health_monitor_fixture.exchange_interface is not None
    assert health_monitor_fixture.logger is not None


@pytest.mark.asyncio
async def test_health_monitor_database_check_success(
    health_monitor_fixture, mock_db_queries
):
    """Test database connectivity check passing."""
    mock_db_queries.connection.execute.return_value = AsyncMock()

    success, response_time, error = await health_monitor_fixture.check_database()
    assert success is True
    assert response_time >= 0
    assert error is None
    mock_db_queries.connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_monitor_database_check_failure(
    health_monitor_fixture, mock_db_queries
):
    """Test database connectivity check failing."""
    mock_db_queries.connection.execute.side_effect = Exception("DB Down")

    success, response_time, error = await health_monitor_fixture.check_database()
    assert success is False
    assert response_time >= 0
    assert error == "DB Down"


@pytest.mark.asyncio
async def test_health_monitor_exchange_check_success(health_monitor_fixture):
    """Test exchange connectivity check passing."""
    success, response_time, error = await health_monitor_fixture.check_exchange()
    assert success is True
    assert response_time >= 0
    assert error is None
    health_monitor_fixture.exchange_interface.exchange.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_monitor_exchange_check_failure(health_monitor_fixture):
    """Test exchange connectivity check failing."""
    health_monitor_fixture.exchange_interface.exchange.ping.side_effect = Exception(
        "Exchange Unreachable"
    )

    success, response_time, error = await health_monitor_fixture.check_exchange()
    assert success is False
    assert response_time >= 0
    assert error == "Exchange Unreachable"


@pytest.mark.asyncio
async def test_health_monitor_market_data_freshness(health_monitor_fixture, logger):
    """Test market data freshness evaluation."""
    health_monitor_fixture.ctx.config = {"market_list": ["BTC/USDT", "ETH/USDT"]}
    health_monitor_fixture.ctx.market_data = MagicMock()

    # Fresh data
    health_monitor_fixture.ctx.market_data.last_update = {
        "BTC/USDT": 1600000000,
        "ETH/USDT": 1600000300,
    }
    with patch("time.time", return_value=1600000300):
        is_fresh = await health_monitor_fixture.check_market_data()
        assert is_fresh is True

    # Stale data
    with patch("time.time", return_value=1600000600):
        is_fresh = await health_monitor_fixture.check_market_data()
        assert is_fresh is False
        logger.warning.assert_called_with("Stale market data for BTC/USDT")
