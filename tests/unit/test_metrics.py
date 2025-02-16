"""
Test suite for enhanced monitoring metrics functionality.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.monitoring.metrics import (
    MetricsCollector,
    PerformanceMetrics,
    ResourceMetrics,
    SystemHealth,
)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_logger")


@pytest.fixture
def metrics_collector(logger):
    """Create a metrics collector instance."""
    return MetricsCollector(logger)


@pytest.fixture
def performance_metrics():
    """Create a performance metrics instance."""
    return PerformanceMetrics()


@pytest.fixture
def resource_metrics():
    """Create a resource metrics instance."""
    return ResourceMetrics()


@pytest.fixture
def system_health():
    """Create a system health instance."""
    return SystemHealth()


def test_performance_metrics_concurrent_operations(performance_metrics):
    """Test tracking of concurrent operations."""
    # Simulate multiple concurrent operations
    performance_metrics.update(True, 0.1, "op1")
    performance_metrics.update(True, 0.2, "op2")
    performance_metrics.update(True, 0.15, "op3")

    assert performance_metrics.max_concurrent_operations >= 1
    assert (
        performance_metrics.concurrent_operations == 0
    )  # Should reset after operations complete


def test_performance_metrics_memory_tracking(performance_metrics):
    """Test memory usage tracking per operation."""
    performance_metrics.update(True, 0.1, memory_usage=100.0)
    performance_metrics.update(True, 0.2, memory_usage=150.0)

    assert performance_metrics.peak_memory_per_op == 150.0


def test_performance_metrics_dependency_failures(performance_metrics):
    """Test tracking of dependency failures."""
    dependencies = ["db", "cache"]
    performance_metrics.update(
        False, 1.0, dependencies=dependencies, error_type="connection_error"
    )

    assert "db" in performance_metrics.dependency_failures
    assert "cache" in performance_metrics.dependency_failures
    assert performance_metrics.dependency_failures["db"] == 1


def test_performance_metrics_timeout_tracking(performance_metrics):
    """Test tracking of operation timeouts."""
    performance_metrics.update(False, 1.0, error_type="timeout_error")
    performance_metrics.update(False, 1.0, error_type="connection_timeout")

    assert performance_metrics.operation_timeouts == 2


def test_resource_metrics_enhanced_tracking(resource_metrics):
    """Test enhanced resource metrics tracking."""
    resource_metrics.gc_stats = {"collections": 10, "collected": 1000}
    resource_metrics.disk_io_stats = {"read_bytes": 1024, "write_bytes": 2048}
    resource_metrics.network_bandwidth = {"rx_bytes": 5000, "tx_bytes": 3000}
    resource_metrics.memory_fragmentation = 0.15

    assert resource_metrics.gc_stats["collections"] == 10
    assert resource_metrics.disk_io_stats["read_bytes"] == 1024
    assert resource_metrics.network_bandwidth["rx_bytes"] == 5000
    assert resource_metrics.memory_fragmentation == 0.15


def test_resource_metrics_leak_detection(resource_metrics):
    """Test resource leak detection."""
    resource_metrics.resource_leaks = {"file_handles": 5, "db_connections": 2}
    resource_metrics.connection_pool_usage = {"main_db": 0.8, "replica_db": 0.6}

    assert resource_metrics.resource_leaks["file_handles"] == 5
    assert resource_metrics.connection_pool_usage["main_db"] == 0.8


def test_system_health_cascading_failures(system_health):
    """Test detection and tracking of cascading failures."""
    # Set up component dependencies
    dependencies1: Set[str] = {"cache", "db"}
    dependencies2: Set[str] = {"api"}

    # Update components with dependencies
    system_health.update("web", False, "Web service error", dependencies=dependencies1)
    system_health.update("cache", False, "Cache error", dependencies=dependencies2)

    assert len(system_health.cascading_failures) > 0
    assert "web" in system_health.degraded_components
    assert "cache" in system_health.degraded_components


def test_system_health_partial_degradation(system_health):
    """Test tracking of partial system degradation."""
    # Update component with latency
    system_health.update("api", False, latency=2.5)

    assert "api" in system_health.partial_degradation
    assert 0 <= system_health.partial_degradation["api"] <= 1


def test_system_health_recovery_tracking(system_health):
    """Test tracking of component recovery."""
    # Simulate component failure and recovery
    system_health.update("db", False, "Database error")
    assert "db" in system_health.recovery_attempts

    # Simulate recovery
    system_health.update("db", True)
    assert "db" in system_health.recovery_success_rate
    assert 0 <= system_health.recovery_success_rate["db"] <= 1


def test_system_health_component_latencies(system_health):
    """Test tracking of component latencies."""
    system_health.update("api", True, latency=1.5)
    system_health.update("db", True, latency=0.5)

    assert "api" in system_health.component_latencies
    assert "db" in system_health.component_latencies
    assert system_health.component_latencies["api"] == 1.5


def test_system_health_history_management(system_health):
    """Test management of health check history."""
    # Generate multiple health updates
    for i in range(system_health.max_error_history + 5):
        system_health.update(f"component_{i}", i % 2 == 0)

    assert len(system_health.health_check_history) <= system_health.max_error_history
    assert len(system_health.last_errors) <= system_health.max_error_history


def test_performance_metrics_recovery_times(performance_metrics):
    """Test tracking of recovery times between failures."""
    # Simulate a failure followed by success
    performance_metrics.update(False, 1.0)
    time.sleep(0.1)  # Simulate time passing
    performance_metrics.update(True, 0.5)

    assert len(performance_metrics.recovery_times) > 0
    assert all(t > 0 for t in performance_metrics.recovery_times)


@pytest.mark.asyncio
async def test_metrics_collection_lifecycle(metrics_collector):
    """Test metrics collection start and stop."""
    await metrics_collector.start_collection()
    assert metrics_collector._is_collecting
    assert len(metrics_collector._collection_tasks) == 4  # All collection tasks started

    await metrics_collector.stop_collection()
    assert not metrics_collector._is_collecting
    assert not metrics_collector._collection_tasks  # All tasks cleaned up


@pytest.mark.asyncio
async def test_metric_aggregation(metrics_collector):
    """Test metric aggregation and trend detection."""
    # Add some test metrics
    component = "test_component"
    metrics_collector._metric_history[component] = [
        {"timestamp": datetime.now(), "error_rate": 0.2, "avg_latency": 1.5},
        {"timestamp": datetime.now(), "error_rate": 0.3, "avg_latency": 1.8},
    ]

    # Run aggregation
    await metrics_collector._aggregate_metrics()

    # Verify alerts were raised for high error rate and latency
    alerts = [a for a in metrics_collector._alert_history if a["type"] == "trend"]
    assert len(alerts) > 0
    assert any("error rate" in a["message"].lower() for a in alerts)
    assert any("latency" in a["message"].lower() for a in alerts)


@pytest.mark.asyncio
async def test_gc_monitoring(metrics_collector):
    """Test garbage collection monitoring."""
    with (
        patch("gc.get_count", return_value=(100, 10, 1)),
        patch("gc.get_objects", return_value=[object() for _ in range(1000)]),
        patch("gc.garbage", [object() for _ in range(150)]),
    ):  # Above threshold

        await metrics_collector._monitor_gc()

        # Verify GC stats were collected
        assert "collections" in metrics_collector.resource_metrics.gc_stats
        assert "objects" in metrics_collector.resource_metrics.gc_stats
        assert "garbage" in metrics_collector.resource_metrics.gc_stats

        # Verify alert was raised for high garbage count
        alerts = [a for a in metrics_collector._alert_history if a["type"] == "memory"]
        assert len(alerts) > 0
        assert any("garbage objects" in a["message"].lower() for a in alerts)


@pytest.mark.asyncio
async def test_alert_severity_handling(metrics_collector):
    """Test alert handling with different severity levels."""
    current_time = datetime.now()

    # Test different severity levels
    await metrics_collector._raise_alert(
        "Info message", "test", "info", current_time, severity="info"
    )
    await metrics_collector._raise_alert(
        "Warning message", "test", "warning", current_time, severity="warning"
    )
    await metrics_collector._raise_alert(
        "Error message", "test", "error", current_time, severity="error"
    )
    await metrics_collector._raise_alert(
        "Critical message", "test", "critical", current_time, severity="critical"
    )

    alerts = metrics_collector._alert_history
    assert len(alerts) == 4

    severity_levels = [a["severity_level"] for a in alerts]
    assert severity_levels == [0, 1, 2, 3]  # Increasing severity levels


@pytest.mark.asyncio
async def test_alert_pattern_detection(metrics_collector):
    """Test detection of alert patterns."""
    current_time = datetime.now()
    component = "test_component"

    # Generate multiple alerts for the same component
    for i in range(6):
        await metrics_collector._raise_alert(
            f"Test alert {i}", component, "test", current_time
        )

    # Verify pattern alert was raised
    pattern_alerts = [
        a
        for a in metrics_collector._alert_history
        if a["type"] == "pattern" and component in a["message"]
    ]
    assert len(pattern_alerts) > 0


def test_alert_pattern_analysis(metrics_collector):
    """Test alert pattern analysis functionality."""
    # Generate test alerts
    current_time = datetime.now()
    metrics_collector._alert_history = [
        {
            "timestamp": current_time,
            "message": f"Test alert {i}",
            "component": "component1" if i < 4 else "component2",
            "type": "error" if i % 2 == 0 else "warning",
            "severity": "error" if i % 3 == 0 else "warning",
        }
        for i in range(6)
    ]

    patterns = metrics_collector.get_alert_patterns()

    assert "by_component" in patterns
    assert "by_type" in patterns
    assert "by_severity" in patterns
    assert "trending" in patterns

    assert patterns["by_component"]["component1"] == 4
    assert "component1" in [t["component"] for t in patterns["trending"]]


@pytest.mark.asyncio
async def test_alert_cooldown(metrics_collector):
    """Test alert rate limiting with cooldown period."""
    current_time = datetime.now()

    # First alert should be raised
    await metrics_collector._raise_alert("Test alert", "test", "test", current_time)
    assert len(metrics_collector._alert_history) == 1

    # Second alert within cooldown should be suppressed
    await metrics_collector._raise_alert("Test alert", "test", "test", current_time)
    assert len(metrics_collector._alert_history) == 1

    # Alert after cooldown should be raised
    future_time = current_time + timedelta(
        seconds=metrics_collector._alert_cooldown + 1
    )
    await metrics_collector._raise_alert("Test alert", "test", "test", future_time)
    assert len(metrics_collector._alert_history) == 2


@pytest.mark.asyncio
async def test_collection_task_error_handling(metrics_collector):
    """Test error handling in collection tasks."""
    with patch.object(
        metrics_collector,
        "_update_resource_metrics",
        side_effect=Exception("Test error"),
    ):

        # Start collection
        await metrics_collector.start_collection()

        # Wait for error handling
        await asyncio.sleep(0.1)

        # Verify collection continues despite error
        assert metrics_collector._is_collecting

        # Cleanup
        await metrics_collector.stop_collection()
