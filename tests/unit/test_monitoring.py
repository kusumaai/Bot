"""
Test suite for the monitoring system with enhanced functionality tests.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from monitoring.metrics import (
    MetricsCollector,
    PerformanceMetrics,
    ResourceMetrics,
    SystemHealth,
)
from utils.error_handler import MonitoringError


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_logger")


@pytest.fixture
def metrics_collector(logger):
    """Create a metrics collector instance."""
    return MetricsCollector(logger)


@pytest.mark.asyncio
async def test_enhanced_performance_metrics_update():
    """Test updating performance metrics with enhanced tracking."""
    metrics = PerformanceMetrics()

    # Test successful operation with operation type
    metrics.update(True, 0.5, operation_type="query", error_type=None)
    assert metrics.total_operations == 1
    assert metrics.successful_operations == 1
    assert metrics.failed_operations == 0
    assert metrics.consecutive_failures == 0
    assert metrics.operation_counts["query"] == 1
    assert not metrics.error_types

    # Test failed operation with error type
    metrics.update(False, 0.3, operation_type="query", error_type="timeout")
    assert metrics.total_operations == 2
    assert metrics.consecutive_failures == 1
    assert metrics.error_types["timeout"] == 1
    assert metrics.operation_counts["query"] == 2

    # Test peak latency tracking
    metrics.update(True, 1.0, operation_type="query")
    assert metrics.peak_latency == 1.0


@pytest.mark.asyncio
async def test_enhanced_system_health_update():
    """Test enhanced system health status tracking."""
    health = SystemHealth()

    # Test warning severity
    health.update("component1", False, "Warning message", severity="warning")
    assert health.warning_count == 1
    assert health.error_count == 0
    assert "component1" in health.degraded_components
    assert health.recovery_attempts["component1"] == 1

    # Test error severity
    health.update("component2", False, "Error message", severity="error")
    assert health.error_count == 1
    assert len(health.degraded_components) == 2
    assert isinstance(health.last_errors[0], dict)
    assert health.last_errors[0]["severity"] == "warning"

    # Test recovery
    health.update("component1", True)
    assert "component1" not in health.degraded_components
    assert "component1" not in health.recovery_attempts
    assert len(health.health_check_history) > 0


@pytest.mark.asyncio
async def test_resource_metrics_collection(metrics_collector):
    """Test enhanced resource metrics collection."""
    with (
        patch("psutil.cpu_percent", return_value=50.0),
        patch("psutil.virtual_memory", return_value=MagicMock(percent=75.0)),
        patch("psutil.disk_usage", return_value=MagicMock(percent=80.0)),
        patch("psutil.net_io_counters", return_value=MagicMock(errin=5, errout=3)),
    ):
        await metrics_collector._update_resource_metrics()

        assert metrics_collector.resource_metrics.cpu_usage == 50.0
        assert metrics_collector.resource_metrics.memory_usage == 0.75
        assert metrics_collector.resource_metrics.disk_usage == 0.8
        assert metrics_collector.resource_metrics.network_errors == 8
        assert metrics_collector.resource_metrics.last_updated is not None


@pytest.mark.asyncio
async def test_alert_rate_limiting(metrics_collector):
    """Test alert rate limiting functionality."""
    component = "test_component"
    alert_type = "performance"
    message = "Test alert"

    # First alert should be raised
    await metrics_collector._raise_alert(message, component, alert_type, datetime.now())
    assert len(metrics_collector._alert_history) == 1

    # Second alert within cooldown period should be suppressed
    await metrics_collector._raise_alert(message, component, alert_type, datetime.now())
    assert len(metrics_collector._alert_history) == 1

    # Alert after cooldown should be raised
    future_time = datetime.now() + timedelta(seconds=301)
    await metrics_collector._raise_alert(message, component, alert_type, future_time)
    assert len(metrics_collector._alert_history) == 2


@pytest.mark.asyncio
async def test_metric_history_storage(metrics_collector):
    """Test metric history storage and cleanup."""
    # Add some test metrics
    component = "test_component"
    metrics = PerformanceMetrics()
    metrics.update(True, 0.5, "operation1")
    metrics_collector.performance_metrics[component] = metrics

    # Store metrics
    await metrics_collector._store_metric_history()
    assert component in metrics_collector._metric_history
    assert len(metrics_collector._metric_history[component]) == 1
    assert "system_resources" in metrics_collector._metric_history

    # Test cleanup
    old_time = datetime.now() - timedelta(days=8)
    metrics_collector._metric_history[component][0]["timestamp"] = old_time
    await metrics_collector._cleanup_old_metrics()
    assert len(metrics_collector._metric_history[component]) == 0


@pytest.mark.asyncio
async def test_comprehensive_system_status(metrics_collector):
    """Test comprehensive system status reporting."""
    # Setup test data
    component = "test_component"
    metrics = PerformanceMetrics()
    metrics.update(True, 0.5, "operation1")
    metrics.update(False, 1.0, "operation2", "error1")
    metrics_collector.performance_metrics[component] = metrics

    # Add some alerts
    await metrics_collector._raise_alert(
        "Test alert", component, "performance", datetime.now()
    )

    # Get status
    status = metrics_collector.get_system_status()

    # Verify comprehensive status
    assert "health" in status
    assert "performance" in status
    assert "resources" in status
    assert "alerts" in status

    # Verify performance metrics
    perf = status["performance"][component]
    assert "peak_latency" in perf
    assert "operation_types" in perf
    assert "error_types" in perf
    assert perf["consecutive_failures"] == 1

    # Verify alerts
    assert len(status["alerts"]["recent"]) == 1
    assert "thresholds" in status["alerts"]


@pytest.mark.asyncio
async def test_component_alert_conditions(metrics_collector):
    """Test component-specific alert conditions."""
    component = "test_component"
    current_time = datetime.now()

    # Setup metrics with various alert conditions
    metrics = PerformanceMetrics()
    metrics.error_rate = 0.2  # Above threshold
    metrics.avg_latency = 2.0  # Above threshold
    metrics.consecutive_failures = 4  # Above threshold
    metrics.operation_counts["critical_op"] = 1500  # Above threshold

    await metrics_collector._check_component_alerts(component, metrics, current_time)

    # Verify alerts were raised
    alerts = [alert["message"] for alert in metrics_collector._alert_history]
    assert any("error rate" in alert.lower() for alert in alerts)
    assert any("latency" in alert.lower() for alert in alerts)
    assert any("consecutive failures" in alert.lower() for alert in alerts)
    assert any("operation count" in alert.lower() for alert in alerts)


@pytest.mark.asyncio
async def test_resource_alert_conditions(metrics_collector):
    """Test resource-specific alert conditions."""
    # Setup resource metrics above thresholds
    metrics_collector.resource_metrics.memory_usage = 0.95
    metrics_collector.resource_metrics.cpu_usage = 0.85
    metrics_collector.resource_metrics.disk_usage = 0.95
    metrics_collector.resource_metrics.io_wait = 0.4
    metrics_collector.resource_metrics.network_errors = 15
    metrics_collector.resource_metrics.thread_count = 150

    await metrics_collector._check_resource_alerts(datetime.now())

    # Verify alerts were raised
    alerts = [alert["message"] for alert in metrics_collector._alert_history]
    assert any("memory usage" in alert.lower() for alert in alerts)
    assert any("cpu usage" in alert.lower() for alert in alerts)
    assert any("disk usage" in alert.lower() for alert in alerts)
    assert any("io wait" in alert.lower() for alert in alerts)
    assert any("network errors" in alert.lower() for alert in alerts)
    assert any("thread count" in alert.lower() for alert in alerts)


@pytest.mark.asyncio
async def test_health_alert_conditions(metrics_collector):
    """Test health-specific alert conditions."""
    current_time = datetime.now()

    # Setup unhealthy state for over an hour
    metrics_collector.system_health.is_healthy = False
    metrics_collector.system_health.last_healthy_state = current_time - timedelta(
        hours=2
    )

    # Setup multiple recovery attempts
    metrics_collector.system_health.recovery_attempts = {"component1": 4}

    await metrics_collector._check_health_alerts(current_time)

    # Verify alerts were raised
    alerts = [alert["message"] for alert in metrics_collector._alert_history]
    assert any("system unhealthy for" in alert.lower() for alert in alerts)
    assert any("multiple recovery attempts" in alert.lower() for alert in alerts)


@pytest.mark.asyncio
async def test_concurrent_metric_updates(metrics_collector):
    """Test concurrent metric updates."""
    component = "test_component"
    update_count = 100

    async def update_metrics():
        for _ in range(update_count):
            metrics = metrics_collector.performance_metrics.get(
                component, PerformanceMetrics()
            )
            metrics.update(True, 0.1, "test_operation")
            await asyncio.sleep(0.01)

    # Run multiple concurrent updates
    tasks = [update_metrics() for _ in range(5)]
    await asyncio.gather(*tasks)

    # Verify metrics were updated correctly
    metrics = metrics_collector.performance_metrics.get(component)
    assert metrics is not None
    assert metrics.operation_counts["test_operation"] == update_count * 5
