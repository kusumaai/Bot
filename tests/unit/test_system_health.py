#! /usr/bin/env python3
"""
Module: tests.unit.test_system_health
Comprehensive testing of system health monitoring including component health,
system metrics, and emergency protocols.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.error_handler import HealthCheckError
from utils.health_monitor import HealthMonitor


class TestSystemHealth:
    """Test suite for system health monitoring."""

    @pytest.fixture
    async def health_monitor(self, trading_context):
        """Provide configured health monitor."""
        monitor = HealthMonitor(trading_context)
        monitor.CRITICAL_CPU_THRESHOLD = 90
        monitor.CRITICAL_MEMORY_THRESHOLD = 90
        monitor.MARKET_DATA_MAX_AGE = 60
        await monitor.initialize()
        return monitor

    @pytest.mark.asyncio
    async def test_component_health_checks(self, health_monitor):
        """Test individual component health checks."""
        # Database health
        health_monitor.ctx.db_connection.execute = AsyncMock(return_value=True)
        assert await health_monitor.check_database_health()

        # Exchange health
        health_monitor.ctx.exchange_interface.ping = AsyncMock(return_value=True)
        assert await health_monitor.check_exchange_health()

        # Market data health
        health_monitor.ctx.market_data.get_last_update = AsyncMock(
            return_value=datetime.now()
        )
        assert await health_monitor.check_market_data_health()

    @pytest.mark.asyncio
    async def test_system_metrics(self, health_monitor):
        """Test system metrics monitoring."""
        # Mock system metrics
        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
        ):

            metrics = await health_monitor.get_system_metrics()
            assert metrics["cpu_usage"] <= health_monitor.CRITICAL_CPU_THRESHOLD
            assert metrics["memory_usage"] <= health_monitor.CRITICAL_MEMORY_THRESHOLD

            # Test high resource usage warning
            with patch("psutil.cpu_percent", return_value=95.0):
                metrics = await health_monitor.get_system_metrics()
                assert metrics["cpu_usage"] > health_monitor.CRITICAL_CPU_THRESHOLD
                with pytest.raises(HealthCheckError):
                    await health_monitor.validate_system_metrics(metrics)

    @pytest.mark.asyncio
    async def test_market_data_freshness(self, health_monitor):
        """Test market data freshness checks."""
        # Fresh data
        recent_time = datetime.now()
        health_monitor.ctx.market_data.get_last_update = AsyncMock(
            return_value=recent_time
        )
        assert await health_monitor.check_market_data_freshness()

        # Stale data
        stale_time = datetime.now() - timedelta(seconds=120)
        health_monitor.ctx.market_data.get_last_update = AsyncMock(
            return_value=stale_time
        )
        with pytest.raises(HealthCheckError):
            await health_monitor.check_market_data_freshness()

    @pytest.mark.asyncio
    async def test_system_readiness(self, health_monitor):
        """Test overall system readiness check."""
        # All components healthy
        health_monitor.check_database_health = AsyncMock(return_value=True)
        health_monitor.check_exchange_health = AsyncMock(return_value=True)
        health_monitor.check_market_data_health = AsyncMock(return_value=True)
        health_monitor.get_system_metrics = AsyncMock(
            return_value={"cpu_usage": 50.0, "memory_usage": 60.0}
        )

        is_ready = await health_monitor.check_system_readiness()
        assert is_ready is True

        # Component failure
        health_monitor.check_database_health = AsyncMock(return_value=False)
        is_ready = await health_monitor.check_system_readiness()
        assert is_ready is False

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, health_monitor):
        """Test emergency shutdown protocol."""
        # Trigger emergency shutdown
        with patch("psutil.cpu_percent", return_value=95.0):
            await health_monitor.monitor_system_health()
            assert health_monitor.emergency_shutdown_triggered

        # Verify trading is blocked
        assert await health_monitor.is_trading_allowed() is False

    @pytest.mark.asyncio
    async def test_recovery_procedures(self, health_monitor):
        """Test system recovery procedures."""
        # Trigger shutdown
        health_monitor.emergency_shutdown_triggered = True

        # System returns to normal
        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
        ):

            await health_monitor.check_recovery_conditions()
            assert health_monitor.emergency_shutdown_triggered is False
            assert await health_monitor.is_trading_allowed()

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, health_monitor):
        """Test continuous monitoring loop."""
        # Setup monitoring conditions
        health_checks = []

        async def mock_monitor():
            health_checks.append(await health_monitor.check_system_readiness())
            if len(health_checks) >= 3:  # Run 3 cycles
                health_monitor.stop_monitoring()

        # Run monitoring loop
        health_monitor.monitor_system_health = mock_monitor
        await health_monitor.start_monitoring()

        assert len(health_checks) == 3  # Verify monitoring cycles completed
        assert all(health_checks)  # Verify all checks passed
