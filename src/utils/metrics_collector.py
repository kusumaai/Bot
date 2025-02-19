import asyncio
import gc
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logger
        self._dependencies: Set[str] = set()
        self._metrics_history: List[Dict] = []
        self._alert_history: List[Dict] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)
        self._component_lock = asyncio.Lock()

        # Performance metrics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_latency = 0.0
        self.last_operation_time = None
        self.operation_timeouts = 0
        self.peak_memory_per_op = 0.0
        self.dependency_failures = {}
        self.concurrent_operations = 0
        self.max_concurrent_operations = 1
        self.recovery_times = []

    def add_dependency(self, name: str):
        """Add a dependency to track."""
        self._dependencies.add(name)
        if name not in self.dependency_failures:
            self.dependency_failures[name] = 0

    def record_operation(self, success: bool, latency: float):
        """Record operation metrics."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.total_latency += latency
        self.last_operation_time = datetime.now()

        # Update memory metrics
        gc.collect()
        current_memory = gc.get_stats()[0]["size"]
        self.peak_memory_per_op = max(self.peak_memory_per_op, current_memory)

    def record_dependency_failure(self, dependency: str):
        """Record dependency failure."""
        if dependency in self.dependency_failures:
            self.dependency_failures[dependency] += 1

    def record_timeout(self):
        """Record operation timeout."""
        self.operation_timeouts += 1

    def record_concurrent_operation(self, delta: int = 1):
        """Record concurrent operation count change."""
        self.concurrent_operations += delta
        self.max_concurrent_operations = max(
            self.max_concurrent_operations, self.concurrent_operations
        )

    def record_recovery_time(self, seconds: float):
        """Record time taken to recover from failure."""
        self.recovery_times.append(seconds)

    def get_success_rate(self) -> float:
        """Get operation success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    def get_average_latency(self) -> float:
        """Get average operation latency."""
        if self.total_operations == 0:
            return 0.0
        return self.total_latency / self.total_operations

    def get_failure_rate(self) -> float:
        """Get operation failure rate."""
        if self.total_operations == 0:
            return 0.0
        return self.failed_operations / self.total_operations

    def get_timeout_rate(self) -> float:
        """Get operation timeout rate."""
        if self.total_operations == 0:
            return 0.0
        return self.operation_timeouts / self.total_operations

    def get_dependency_health(self) -> Dict[str, float]:
        """Get dependency health scores."""
        health_scores = {}
        for dep in self._dependencies:
            failures = self.dependency_failures.get(dep, 0)
            if failures == 0:
                health_scores[dep] = 1.0
            else:
                health_scores[dep] = 1.0 / (1.0 + failures)
        return health_scores

    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "total_operations": self.total_operations,
            "success_rate": self.get_success_rate(),
            "average_latency": self.get_average_latency(),
            "failure_rate": self.get_failure_rate(),
            "timeout_rate": self.get_timeout_rate(),
            "peak_memory_per_op": self.peak_memory_per_op,
            "concurrent_operations": self.concurrent_operations,
            "max_concurrent_operations": self.max_concurrent_operations,
            "dependency_health": self.get_dependency_health(),
        }

    async def collect_metrics(self):
        """Collect current metrics and store in history."""
        async with self._component_lock:
            metrics = self.get_metrics_summary()
            metrics["timestamp"] = datetime.now()
            self._metrics_history.append(metrics)

            # Trim history to last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            self._metrics_history = [
                m for m in self._metrics_history if m["timestamp"] > cutoff
            ]

    def should_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent based on cooldown."""
        now = datetime.now()
        if alert_type in self._last_alert_time:
            last_alert = self._last_alert_time[alert_type]
            if now - last_alert < self._alert_cooldown:
                return False
        self._last_alert_time[alert_type] = now
        return True

    async def process_alert(self, alert_type: str, message: str, severity: str):
        """Process and store alert."""
        if not self.should_alert(alert_type):
            return

        async with self._component_lock:
            alert = {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(),
            }
            self._alert_history.append(alert)

            # Trim alert history to last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            self._alert_history = [
                a for a in self._alert_history if a["timestamp"] > cutoff
            ]
