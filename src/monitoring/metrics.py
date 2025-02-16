#! /usr/bin/env python3
# src/monitoring/metrics.py
"""
Module: monitoring/metrics.py
Provides comprehensive system-wide monitoring and metrics collection.
"""
import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import psutil

from utils.error_handler import MonitoringError


@dataclass
class PerformanceMetrics:
    """Track performance metrics for a component."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_latency: float = 0.0
    last_operation_time: Optional[datetime] = None
    error_rate: float = 0.0
    avg_latency: float = 0.0
    peak_latency: float = 0.0
    operation_counts: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    recovery_times: List[float] = field(default_factory=list)
    operation_timeouts: int = 0
    peak_memory_per_op: float = 0.0
    dependency_failures: Dict[str, int] = field(default_factory=dict)
    concurrent_operations: int = 0
    max_concurrent_operations: int = 0

    def update(
        self,
        success: bool,
        latency: float,
        operation_type: str = "default",
        error_type: Optional[str] = None,
        memory_usage: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """Update metrics with new operation result."""
        self.total_operations += 1
        self.concurrent_operations += 1
        self.max_concurrent_operations = max(
            self.max_concurrent_operations, self.concurrent_operations
        )

        if success:
            self.successful_operations += 1
            self.consecutive_failures = 0
            if self.recovery_times and self.last_operation_time:
                recovery_time = (
                    datetime.now() - self.last_operation_time
                ).total_seconds()
                self.recovery_times.append(recovery_time)
        else:
            self.failed_operations += 1
            self.consecutive_failures += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                if "timeout" in error_type.lower():
                    self.operation_timeouts += 1

        if dependencies:
            for dep in dependencies:
                if not success:
                    self.dependency_failures[dep] = (
                        self.dependency_failures.get(dep, 0) + 1
                    )

        self.total_latency += latency
        self.peak_latency = max(self.peak_latency, latency)
        self.last_operation_time = datetime.now()
        self.error_rate = (
            self.failed_operations / self.total_operations
            if self.total_operations > 0
            else 0
        )
        self.avg_latency = (
            self.total_latency / self.total_operations
            if self.total_operations > 0
            else 0
        )
        self.operation_counts[operation_type] = (
            self.operation_counts.get(operation_type, 0) + 1
        )

        if memory_usage is not None:
            self.peak_memory_per_op = max(self.peak_memory_per_op, memory_usage)

        self.concurrent_operations -= 1


@dataclass
class ResourceMetrics:
    """Track resource usage metrics."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    db_connections: int = 0
    last_updated: Optional[datetime] = None
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    io_wait: float = 0.0
    network_errors: int = 0
    open_file_handles: int = 0
    thread_count: int = 0
    process_uptime: float = 0.0
    memory_fragmentation: float = 0.0
    gc_stats: Dict[str, int] = field(default_factory=dict)
    disk_io_stats: Dict[str, float] = field(default_factory=dict)
    network_bandwidth: Dict[str, float] = field(default_factory=dict)
    connection_pool_usage: Dict[str, float] = field(default_factory=dict)
    resource_leaks: Dict[str, int] = field(default_factory=dict)
    system_load: List[float] = field(default_factory=list)


@dataclass
class SystemHealth:
    """Track overall system health status."""

    is_healthy: bool = True
    last_check: Optional[datetime] = None
    error_count: int = 0
    warning_count: int = 0
    components_status: Dict[str, bool] = field(default_factory=dict)
    last_errors: List[str] = field(default_factory=list)
    max_error_history: int = 100
    degraded_components: Set[str] = field(default_factory=set)
    recovery_attempts: Dict[str, int] = field(default_factory=dict)
    last_healthy_state: Optional[datetime] = None
    health_check_history: List[Dict] = field(default_factory=list)
    component_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    cascading_failures: List[Dict] = field(default_factory=list)
    partial_degradation: Dict[str, float] = field(default_factory=dict)
    recovery_success_rate: Dict[str, float] = field(default_factory=dict)
    component_latencies: Dict[str, float] = field(default_factory=dict)

    def update(
        self,
        component: str,
        is_healthy: bool,
        error_msg: Optional[str] = None,
        severity: str = "error",
        dependencies: Optional[Set[str]] = None,
        latency: Optional[float] = None,
    ):
        """Update system health status with enhanced tracking."""
        self.last_check = datetime.now()
        self.components_status[component] = is_healthy

        if not is_healthy:
            if severity == "error":
                self.error_count += 1
            elif severity == "warning":
                self.warning_count += 1

            self.degraded_components.add(component)
            self.recovery_attempts[component] = (
                self.recovery_attempts.get(component, 0) + 1
            )

            if error_msg:
                timestamp = datetime.now()
                error_entry = {
                    "timestamp": timestamp,
                    "component": component,
                    "message": error_msg,
                    "severity": severity,
                }
                self.last_errors.append(error_entry)
                if len(self.last_errors) > self.max_error_history:
                    self.last_errors.pop(0)

            # Track cascading failures
            if dependencies:
                self.component_dependencies[component] = dependencies
                affected_components = self._check_cascading_failures(component)
                if affected_components:
                    self.cascading_failures.append(
                        {
                            "timestamp": datetime.now(),
                            "source": component,
                            "affected": list(affected_components),
                        }
                    )
        else:
            self.degraded_components.discard(component)
            if component in self.recovery_attempts:
                attempts = self.recovery_attempts[component]
                successes = self.recovery_success_rate.get(component, 0)
                self.recovery_success_rate[component] = (successes * attempts + 1) / (
                    attempts + 1
                )
                del self.recovery_attempts[component]

        if latency is not None:
            self.component_latencies[component] = latency
            if not is_healthy and latency > 0:
                self.partial_degradation[component] = min(
                    1.0, latency / 5.0
                )  # Normalize to [0,1]

        # Update overall health status
        previous_health = self.is_healthy
        self.is_healthy = all(self.components_status.values())

        # Track health state changes
        if self.is_healthy != previous_health:
            if self.is_healthy:
                self.last_healthy_state = datetime.now()
            self.health_check_history.append(
                {
                    "timestamp": datetime.now(),
                    "state": "healthy" if self.is_healthy else "unhealthy",
                    "trigger_component": component,
                }
            )

        # Maintain history size
        if len(self.health_check_history) > self.max_error_history:
            self.health_check_history.pop(0)

    def _check_cascading_failures(self, component: str) -> Set[str]:
        """Identify components affected by a failure."""
        affected = set()
        for dep_component, deps in self.component_dependencies.items():
            if component in deps and self.components_status.get(dep_component, True):
                affected.add(dep_component)
                affected.update(self._check_cascading_failures(dep_component))
        return affected


class MetricsCollector:
    """Collects and manages system metrics with proper resource cleanup"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._state_transitions = {}
        self._cleanup_interval = 300  # 5 minutes
        self._max_history = 1000  # Maximum events to keep
        self._last_cleanup = time.time()

        # Resource tracking with bounds
        self._resource_limits = {
            "db_connections": 100,
            "exchange_connections": 50,
            "file_handles": 1000,
            "tasks": 500,
            "events": 1000,
        }

        # Resource tracking with timestamps
        self._resource_tracker = {
            "db_connections": ResourceSet(
                max_size=self._resource_limits["db_connections"]
            ),
            "exchange_connections": ResourceSet(
                max_size=self._resource_limits["exchange_connections"]
            ),
            "file_handles": ResourceSet(max_size=self._resource_limits["file_handles"]),
            "tasks": ResourceSet(max_size=self._resource_limits["tasks"]),
            "events": ResourceSet(max_size=self._resource_limits["events"]),
        }

        # Component events with cleanup
        self._component_events = {
            component: ManagedEvent() for component in self._dependencies.keys()
        }

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def track_resource(
        self, resource_type: str, resource: Any, metadata: Dict = None
    ) -> None:
        """Track a resource with proper bounds checking and cleanup"""
        try:
            if resource_type not in self._resource_tracker:
                self.logger.error(f"Unknown resource type: {resource_type}")
                return

            resource_set = self._resource_tracker[resource_type]
            await resource_set.add(resource, metadata=metadata, timestamp=time.time())

            # Check if cleanup needed
            if resource_set.size >= self._resource_limits[resource_type] * 0.8:
                await self._cleanup_resources(resource_type)

        except Exception as e:
            self.logger.error(f"Failed to track resource: {e}")

    async def _cleanup_resources(self, resource_type: str) -> None:
        """Clean up resources of a specific type"""
        try:
            resource_set = self._resource_tracker[resource_type]
            stale_resources = await resource_set.get_stale(
                older_than=self._cleanup_interval
            )

            for resource in stale_resources:
                try:
                    await self._safely_cleanup_resource(resource_type, resource)
                    await resource_set.remove(resource)
                except Exception as e:
                    self.logger.error(f"Failed to cleanup {resource_type}: {e}")

        except Exception as e:
            self.logger.error(f"Resource cleanup failed for {resource_type}: {e}")

    async def _safely_cleanup_resource(self, resource_type: str, resource: Any) -> None:
        """Safely cleanup a specific resource"""
        try:
            if resource_type == "tasks":
                if not resource.done():
                    resource.cancel()
                    try:
                        await resource
                    except (asyncio.CancelledError, Exception):
                        pass

            elif resource_type == "db_connections":
                await resource.close()

            elif resource_type == "exchange_connections":
                await resource.close()

            elif resource_type == "file_handles":
                resource.close()

            elif resource_type == "events":
                resource.clear()

        except Exception as e:
            self.logger.error(f"Failed to safely cleanup {resource_type}: {e}")

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up resources"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                # Clean up all resource types
                for resource_type in self._resource_tracker:
                    await self._cleanup_resources(resource_type)

                # Clean up state transitions
                self._cleanup_state_history()

                # Update last cleanup time
                self._last_cleanup = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def _cleanup_state_history(self) -> None:
        """Clean up old state transition history"""
        try:
            current_time = time.time()
            for component in list(self._state_transitions.keys()):
                transitions = self._state_transitions[component]
                # Keep only recent transitions
                self._state_transitions[component] = [
                    t
                    for t in transitions
                    if current_time - t["timestamp"] < self._cleanup_interval
                ]
        except Exception as e:
            self.logger.error(f"State history cleanup failed: {e}")

    async def cleanup(self) -> None:
        """Clean up all resources when shutting down"""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Clean up all resource types
            for resource_type in self._resource_tracker:
                await self._cleanup_resources(resource_type)

            # Clear all events
            for event in self._component_events.values():
                event.clear()

            # Clear state history
            self._state_transitions.clear()

        except Exception as e:
            self.logger.error(f"Final cleanup failed: {e}")


class ResourceSet:
    """Thread-safe resource set with size limits"""

    def __init__(self, max_size: int):
        self._resources = {}  # resource -> (timestamp, metadata)
        self._max_size = max_size
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._resources)

    async def add(self, resource: Any, timestamp: float, metadata: Dict = None) -> None:
        async with self._lock:
            if len(self._resources) >= self._max_size:
                # Remove oldest if at capacity
                oldest = min(self._resources.items(), key=lambda x: x[1][0])
                del self._resources[oldest[0]]
            self._resources[resource] = (timestamp, metadata)

    async def remove(self, resource: Any) -> None:
        async with self._lock:
            self._resources.pop(resource, None)

    async def get_stale(self, older_than: float) -> List[Any]:
        current_time = time.time()
        async with self._lock:
            return [
                resource
                for resource, (timestamp, _) in self._resources.items()
                if current_time - timestamp > older_than
            ]


class ManagedEvent:
    """Event with cleanup capabilities"""

    def __init__(self):
        self._event = asyncio.Event()
        self._set_time = None

    def set(self) -> None:
        self._event.set()
        self._set_time = time.time()

    def clear(self) -> None:
        self._event.clear()
        self._set_time = None

    def is_stale(self, threshold: float) -> bool:
        return self._set_time is not None and time.time() - self._set_time > threshold
