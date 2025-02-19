#! /usr/bin/env python3
# src/utils/health_monitor.py
"""
Module: src.utils
Provides health monitoring functionality.
"""
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import psutil

from src.utils.exceptions import HealthCheckError
from utils.error_handler import handle_error, handle_error_async
from utils.numeric_handler import NumericHandler


# dataclass for component health
@dataclass
class ComponentHealth:
    """Component health status with enhanced tracking"""

    name: str
    status: bool
    message: str = ""
    last_checked: float = field(default_factory=time.time)
    error_count: int = 0
    response_time: float = 0.0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    degradation_level: float = 0.0  # 0.0 to 1.0
    recovery_attempts: int = 0

    @property
    def is_healthy(self) -> bool:
        return (
            self.status
            and self.error_count < 3
            and self.consecutive_failures < 2
            and self.degradation_level < 0.8
        )


# dataclass for health status
@dataclass
class HealthStatus:
    """Health status"""

    timestamp: float
    is_healthy: bool
    warnings: List[str]
    errors: List[str]
    components: Dict[str, ComponentHealth]
    system_metrics: Dict[str, Any]


# health monitor class
class HealthMonitor:
    """Enhanced health monitoring system"""

    def __init__(self, ctx):
        """Initialize HealthMonitor with context."""
        self.ctx = ctx
        self.logger = ctx.logger
        self.db_queries = getattr(ctx, "db_queries", None)
        self.nh = NumericHandler()
        self._component_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._metric_lock = asyncio.Lock()
        self.initialized = False

        # Use deque for fixed-size history
        self.latency_history = defaultdict(lambda: deque(maxlen=1000))
        self.max_history = 1000

        # Initialize other attributes
        self.last_check = 0
        self.error_thresholds = {
            "max_errors": 5,
            "max_consecutive_failures": 2,
            "memory": 0.85,  # 85% memory usage
            "disk": 0.90,  # 90% disk usage
            "cpu": 0.95,  # 95% CPU usage
            "critical_component_failures": 1,  # Number of critical component failures allowed
            "degradation_threshold": 0.8,  # 80% degradation threshold
            "max_recovery_attempts": 3,
        }
        self.error_counts = defaultdict(int)
        self.check_interval = 30  # seconds

        # Initialize status and components
        self._init_status_and_components()

        # Add the following new attribute
        self._monitoring = True
        self.emergency_shutdown_triggered = False
        self.last_health_check = time.time()
        self._shutdown_in_progress = False
        self._shutdown_lock = asyncio.Lock()

        # Critical components and their weights
        self.critical_components = {
            "database": 1.0,
            "exchange": 1.0,
            "position_manager": 0.8,
            "risk_manager": 0.9,
            "circuit_breaker": 1.0,
        }

        self.CRITICAL_CPU_THRESHOLD = 90.0
        self.CRITICAL_MEMORY_THRESHOLD = 90.0
        self.MAX_CONSECUTIVE_FAILURES = 3
        self.consecutive_failures = 0

    def _init_status_and_components(self):
        # Initialize status
        self.status = HealthStatus(
            timestamp=time.time(),
            is_healthy=True,
            warnings=[],
            errors=[],
            components={},
            system_metrics={},
        )

        # Component tracking
        self.components: Dict[str, ComponentHealth] = {
            "database": ComponentHealth(name="database", status=True),
            "exchange": ComponentHealth(name="exchange", status=True),
            "order_manager": ComponentHealth(name="order_manager", status=True),
            "position_manager": ComponentHealth(name="position_manager", status=True),
            "risk_manager": ComponentHealth(name="risk_manager", status=True),
            "system": ComponentHealth(name="system", status=True),
        }

        # Performance tracking
        self.error_window = timedelta(minutes=5)
        self.degradation_threshold = Decimal("2.0")

        self.exchange_interface = getattr(self.ctx, "exchange_interface", None)
        self.market_data = getattr(self.ctx, "market_data", None)

        # initialize health monitor

    async def initialize(self) -> bool:
        # If operating in paper mode, bypass full initialization
        if self.ctx.config.get("paper_mode", False):
            self.logger.info(
                "HealthMonitor: Paper mode detected, setting up dummy services."
            )
            # Attach dummy market_data if missing
            if not hasattr(self.ctx, "market_data"):

                class DummyMarketData:
                    async def get_last_update(self, symbol):
                        return datetime.now()

                    async def get_latest_data(self):
                        return {}

                self.ctx.market_data = DummyMarketData()
            self.initialized = True
            return True
        try:
            if self.initialized:
                return True
            self._monitor_task = asyncio.create_task(self.monitor_loop())
            self.initialized = True
            self.logger.info("HealthMonitor initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"HealthMonitor initialization failed: {e}")
            return False

    async def start_monitoring(self):
        """Start the health monitoring loop with proper error handling"""
        while self._monitoring:
            try:
                await self.monitor_system_health()
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(self.check_interval)

    async def monitor_system_health(self):
        async with self._component_lock:
            health_status = await self.check_system_health()
            if not health_status.get("healthy", True):
                self.logger.warning("System health check failed")
            metrics = await self.get_system_metrics()
            if self.should_emergency_shutdown():
                self.emergency_shutdown_triggered = True
            else:
                self.emergency_shutdown_triggered = False

    async def check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check database
            db_healthy, _, _ = await self.check_database()
            # Check exchange
            exchange_healthy, _, _ = await self.check_exchange()
            # Check system resources
            sys_metrics = await self.check_system_resources()
            # Check market data
            market_healthy, _, _ = await self.check_market_data()

            return all(
                [
                    db_healthy,
                    exchange_healthy,
                    market_healthy,
                    sys_metrics["memory_used_pct"] < self.error_thresholds["memory"],
                    sys_metrics["disk_used_pct"] < self.error_thresholds["disk"],
                    sys_metrics["cpu_used_pct"] < self.error_thresholds["cpu"],
                ]
            )

        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return False

    # check memory
    async def _check_memory(self) -> bool:
        """Check memory usage"""
        try:
            usage = psutil.Process().memory_percent()
            return usage < self.error_thresholds["memory"]
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return False

    # check api health
    async def _check_api_health(self) -> bool:
        """Check API health"""
        try:
            # Implementation of _check_api_health method
            # For example, send a lightweight request to verify API functionality
            # Here, it's a placeholder returning True
            return True
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            return False

    # check database
    async def _check_database(self) -> bool:
        """Verify database connection and performance"""
        try:
            start_time = time.time()
            if self.ctx.config.get("paper_mode", False):
                self.components["database"].status = True
                return True

            if not hasattr(self.ctx, "db_connection") or not self.ctx.db_connection:
                error_msg = "Database connection not initialized"
                self.logger.warning(error_msg)
                self.components["database"].status = True
                self.components["database"].message = "Fresh database initialization"
                return True

            # Try to get a connection from the pool
            try:
                if (
                    hasattr(self.ctx.db_connection, "pool")
                    and self.ctx.db_connection.pool
                ):
                    async with self.ctx.db_connection.pool.acquire() as conn:
                        async with conn.cursor() as cursor:
                            await cursor.execute("SELECT 1")
                            await cursor.fetchone()
                elif (
                    hasattr(self.ctx.db_connection, "conn")
                    and self.ctx.db_connection.conn is not None
                ):
                    await self.ctx.db_connection.conn.execute("SELECT 1")
                else:
                    raise Exception("No valid database connection found.")
            except Exception as db_error:
                if self.components["database"].last_checked > 0:
                    raise Exception(f"Database query failed: {str(db_error)}")
                else:
                    self.logger.info("Database being initialized for first time")
                    return True
            # response time
            response_time = time.time() - start_time

            # Update component status
            self.components["database"].status = True
            self.components["database"].last_checked = time.time()
            self.components["database"].response_time = response_time
            self.components["database"].message = ""
            self.components["database"].error_count = 0

            return True

        except Exception as e:
            error_msg = f"Database health check failed: {str(e)}"
            response_time = time.time() - start_time

            # Update component status
            self.components["database"].status = False
            self.components["database"].last_checked = time.time()
            self.components["database"].response_time = response_time
            self.components["database"].message = error_msg
            self.components["database"].error_count += 1
            # log the error for the database health check
            self.logger.warning(error_msg)
            return False

    # check exchange
    async def _check_exchange(self) -> bool:
        """Verify exchange connectivity"""
        try:
            # Check exchange connectivity if available
            if self.exchange_interface and hasattr(
                self.exchange_interface.exchange, "ping"
            ):
                try:
                    return self.exchange_interface.exchange.ping()
                except Exception:
                    return False
            return False

        except Exception as e:
            self.logger.error(f"Exchange check failed: {e}")
            return False

    # check market data
    async def _check_market_data(self) -> bool:
        """Verify market data freshness"""
        try:
            for symbol in self.ctx.config["market_list"]:
                last_update = self.ctx.market_data.last_update.get(symbol)
                if not last_update or time.time() - last_update > 300:  # 5 minutes
                    self.logger.warning(f"Stale market data for {symbol}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Market data check failed: {e}")
            return False

    # check system resources
    async def check_system_resources(self) -> Dict[str, float]:
        """Enhanced system resource checking with trending"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_pct = psutil.cpu_percent(interval=1)

            metrics = {
                "memory_used_pct": memory.percent / 100,
                "disk_used_pct": disk.percent / 100,
                "cpu_used_pct": cpu_pct / 100,
                "timestamp": time.time(),
            }

            # Store metrics for trending
            await self._store_resource_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {
                "memory_used_pct": 1.0,
                "disk_used_pct": 1.0,
                "cpu_used_pct": 1.0,
                "timestamp": time.time(),
            }

    async def _store_resource_metrics(self, metrics: Dict[str, float]):
        """Store resource metrics for trend analysis"""
        if not hasattr(self, "_resource_history"):
            self._resource_history = []

        self._resource_history.append(metrics)

        # Keep last 10 minutes of metrics
        cutoff_time = time.time() - 600
        self._resource_history = [
            m for m in self._resource_history if m["timestamp"] > cutoff_time
        ]

    async def get_resource_trends(self) -> Dict[str, float]:
        """Calculate resource usage trends"""
        if not hasattr(self, "_resource_history") or len(self._resource_history) < 2:
            return {"memory_trend": 0, "disk_trend": 0, "cpu_trend": 0}

        recent = self._resource_history[-1]
        past = self._resource_history[0]
        time_diff = recent["timestamp"] - past["timestamp"]

        if time_diff == 0:
            return {"memory_trend": 0, "disk_trend": 0, "cpu_trend": 0}

        return {
            "memory_trend": (recent["memory_used_pct"] - past["memory_used_pct"])
            / time_diff,
            "disk_trend": (recent["disk_used_pct"] - past["disk_used_pct"]) / time_diff,
            "cpu_trend": (recent["cpu_used_pct"] - past["cpu_used_pct"]) / time_diff,
        }

    # check market data
    async def check_market_data(self) -> Tuple[bool, float, Optional[str]]:
        """Check market data service health"""
        start_time = time.time()
        try:
            if not self.ctx.market_data:
                return False, 0.0, "Market data service not initialized"

            # Check if we can fetch recent market data
            data = await self.ctx.market_data.get_latest_data()
            if not data:
                raise ValueError("No market data available")

            response_time = time.time() - start_time

            # Update component status
            self.components["market_data"].status = True
            self.components["market_data"].last_checked = time.time()
            self.components["market_data"].response_time = response_time
            self.components["market_data"].message = ""

            return True, response_time, None

        except Exception as e:
            error_msg = f"Market data health check failed: {str(e)}"
            response_time = time.time() - start_time

            # Update component status
            self.components["market_data"].status = False
            self.components["market_data"].last_checked = time.time()
            self.components["market_data"].response_time = response_time
            self.components["market_data"].message = error_msg
            self.components["market_data"].error_count += 1

            return False, response_time, error_msg

    # get system metrics
    async def get_system_metrics(self) -> dict:
        try:
            metrics = await self.check_system_resources()
            return {
                "memory_usage": float(metrics.get("memory_used_pct", 0)),
                "disk_usage": float(metrics.get("disk_used_pct", 0)),
                "cpu_usage": float(metrics.get("cpu_used_pct", 0)),
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {
                "memory_usage": 100.0,
                "disk_usage": 100.0,
                "cpu_usage": 100.0,
            }

    # update component
    def update_component(
        self, component: str, healthy: bool, latency: float, error: Optional[str] = None
    ):
        """Update component health status"""
        if component not in self.components:
            self.logger.warning(f"Attempted to update unknown component: {component}")
            return

        comp = self.components[component]
        comp.status = healthy
        comp.last_checked = time.time()
        comp.response_time = latency

        if error:
            comp.message = error
            comp.error_count += 1
            comp.last_error = error
        else:
            comp.message = ""

    # check emergency shutdown
    async def should_emergency_shutdown(self) -> bool:
        """Enhanced emergency shutdown decision with better failure detection"""
        try:
            async with self._shutdown_lock:
                if self._shutdown_in_progress:
                    return True

                # Check critical component failures with weighted impact
                critical_failure_impact = 0.0
                for component, weight in self.critical_components.items():
                    if component in self.components:
                        health = self.components[component]
                        if not health.is_healthy:
                            critical_failure_impact += weight
                            if (
                                health.consecutive_failures
                                >= self.error_thresholds["max_consecutive_failures"]
                            ):
                                critical_failure_impact += weight * 0.5

                if (
                    critical_failure_impact
                    >= self.error_thresholds["critical_component_failures"]
                ):
                    self.logger.critical(
                        f"Critical component failure impact: {critical_failure_impact}"
                    )
                    return True

                # Check system resources with trending
                sys_metrics = await self.check_system_resources()
                resource_trends = await self.get_resource_trends()

                # Immediate shutdown conditions
                if any(
                    [
                        sys_metrics["memory_used_pct"]
                        > self.error_thresholds["memory"],
                        sys_metrics["disk_used_pct"] > self.error_thresholds["disk"],
                        sys_metrics["cpu_used_pct"] > self.error_thresholds["cpu"],
                    ]
                ):
                    self.logger.critical("Resource thresholds exceeded")
                    return True

                # Trending shutdown conditions
                if any(
                    [
                        resource_trends["memory_trend"] > 0.1
                        and sys_metrics["memory_used_pct"]
                        > self.error_thresholds["memory"] * 0.9,
                        resource_trends["disk_trend"] > 0.1
                        and sys_metrics["disk_used_pct"]
                        > self.error_thresholds["disk"] * 0.9,
                        resource_trends["cpu_trend"] > 0.2
                        and sys_metrics["cpu_used_pct"]
                        > self.error_thresholds["cpu"] * 0.9,
                    ]
                ):
                    self.logger.critical(
                        "Resource trends indicate imminent threshold breach"
                    )
                    return True

                # Check overall system degradation
                degraded_components = [
                    c
                    for c in self.components.values()
                    if c.degradation_level
                    > self.error_thresholds["degradation_threshold"]
                ]
                if len(degraded_components) >= 2:
                    self.logger.critical(
                        f"Multiple components severely degraded: {len(degraded_components)}"
                    )
                    return True

                return False

        except Exception as e:
            self.logger.critical(f"Error in should_emergency_shutdown: {e}")
            return True  # Fail safe

    # get health status
    async def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        try:
            # Check database
            db_healthy, _, _ = await self.check_database()
            # Check exchange
            exchange_healthy, _, _ = await self.check_exchange()
            # Update system component status
            self.components["system"].status = all(
                [
                    db_healthy,
                    exchange_healthy,
                    (await self.check_system_resources())["healthy"],
                ]
            )

            return HealthStatus(
                healthy=all(c.is_healthy for c in self.components.values()),
                components={
                    name: comp.status for name, comp in self.components.items()
                },
                messages={name: comp.message for name, comp in self.components.items()},
                error_counts={
                    name: comp.error_count for name, comp in self.components.items()
                },
                last_checked=time.time(),
            )

        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return HealthStatus(
                healthy=False,
                components={},
                messages={"error": str(e)},
                error_counts={},
                last_checked=time.time(),
            )

    # get health report
    async def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            now = time.time()
            status = await self.get_health_status()

            return {
                "timestamp": datetime.fromtimestamp(now).isoformat(),
                "healthy": status.is_healthy,
                "components": {
                    name: {
                        "healthy": c.healthy,
                        "last_check_age": now - c.last_checked,
                        "response_time": c.response_time,
                        "error_count": c.error_count,
                        "last_error": c.last_error,
                    }
                    for name, c in status.components.items()
                },
                "system_metrics": status.system_metrics,
                "warnings": status.warnings,
                "errors": status.errors,
            }

        except Exception as e:
            handle_error(e, "HealthMonitor.get_health_report", logger=self.logger)
            return {
                "timestamp": datetime.now().isoformat(),
                "healthy": False,
                "error": str(e),
            }

    # update component metrics
    async def update_component_metrics(self, component: str, latency: float):
        """Thread-safe component metric updates"""
        async with self._metric_lock:
            try:
                if component not in self.latency_history:
                    self.latency_history[component] = deque(maxlen=self.max_history)

                self.latency_history[component].append(latency)

                # Calculate rolling statistics if we have enough data
                if len(self.latency_history[component]) >= 10:
                    median = float(np.median(self.latency_history[component]))
                    p95 = float(np.percentile(self.latency_history[component], 95))

                    comp = self.components.get(component)
                    if comp:
                        comp.response_time = latency
                        # Mark as degraded if current latency is significantly higher than median
                        if latency > median * float(self.degradation_threshold):
                            self.logger.warning(
                                f"{component} showing performance degradation: "
                                f"current={latency:.2f}ms, median={median:.2f}ms"
                            )
                            if not comp.message:
                                comp.message = "Performance degradation detected"

            except Exception as e:
                await handle_error_async(
                    e, "HealthMonitor.update_component_metrics", self.logger
                )

    # monitor loop
    async def monitor_loop(self):
        """Main monitoring loop"""
        try:
            while True:
                await self.start_monitoring()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            self.logger.error(f"Monitor loop failed: {e}")
            # Don't restart automatically - let the circuit breaker handle it
            raise

    # monitor component health
    async def monitor_component_health(self, component: str):
        """Monitor individual component health"""
        async with self._component_lock:
            try:
                start_time = time.time()
                healthy = False
                error_msg = None

                if component == "database":
                    healthy, response_time, error_msg = await self.check_database()
                elif component == "exchange":
                    healthy, response_time, error_msg = await self.check_exchange()
                elif component == "system":
                    metrics = await self.check_system_resources()
                    healthy = all(
                        metrics[k]
                        for k in ["memory_used_pct", "disk_used_pct", "cpu_used_pct"]
                    )
                    response_time = 0.0
                else:
                    # Generic component check
                    start = time.time()
                    check_method = getattr(self.ctx, f"check_{component}", None)
                    if check_method:
                        try:
                            healthy = await check_method()
                            response_time = time.time() - start
                        except Exception as e:
                            error_msg = str(e)
                            response_time = time.time() - start
                    else:
                        self.logger.warning(
                            f"No check method found for component: {component}"
                        )
                        return

                # Update component status
                self.update_component(
                    component=component,
                    healthy=healthy,
                    latency=response_time,
                    error=error_msg,
                )

                # Update metrics
                await self.update_component_metrics(component, response_time)

            except Exception as e:
                await handle_error_async(
                    e, "HealthMonitor.monitor_component_health", self.logger
                )
                self.update_component(
                    component=component,
                    healthy=False,
                    latency=time.time() - start_time,
                    error=str(e),
                )

    async def is_system_healthy(self) -> bool:
        """Check overall system health"""
        try:
            db_healthy, _, _ = await self.check_database()
            exchange_healthy, _, _ = await self.check_exchange()

            # Update system component status
            self.components["system"].status = all(
                [
                    db_healthy,
                    exchange_healthy,
                    await self.check_system_resources()["healthy"],
                ]
            )

            return self.components["system"].status

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return False

    # check system readiness
    async def check_system_readiness(self) -> bool:
        if self.ctx.config.get("paper_mode", False):
            return True
        try:
            readiness = {
                "database": False,
                "models": False,
                "ga_data": False,
                "market_data": False,
            }
            if self.ctx.db_connection:
                async with self.ctx.db_connection.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(
                            "SELECT COUNT(*) FROM models WHERE status = 'active'"
                        )
                        model_count = await cursor.fetchone()
                        readiness["models"] = (
                            (model_count[0] > 0) if model_count else False
                        )

                        await cursor.execute(
                            "SELECT COUNT(*) FROM genetic_algorithms WHERE status = 'complete'"
                        )
                        ga_count = await cursor.fetchone()
                        readiness["ga_data"] = (ga_count[0] > 0) if ga_count else False

                        await cursor.execute("SELECT COUNT(*) FROM market_data")
                        data_count = await cursor.fetchone()
                        readiness["market_data"] = (
                            (data_count[0] > 0) if data_count else False
                        )

                        readiness["database"] = True

            overall = all(value for value in readiness.values())
            if not overall:
                missing = [k for k, v in readiness.items() if not v]
                self.logger.info(
                    f"System not ready for trading. Missing: {', '.join(missing)}"
                )
            return overall
        except Exception as e:
            self.logger.warning(f"Could not check system readiness: {e}")
            return False

    # collect system metrics
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        metrics = {}
        try:
            cpu = psutil.cpu_percent(interval=1)
            metrics["cpu_usage"] = cpu
        except Exception as e:
            self.logger.error(f"Failed to collect CPU usage: {e}")

        try:
            virtual_mem = psutil.virtual_memory()
            metrics["memory_usage"] = virtual_mem.percent
        except Exception as e:
            self.logger.error(f"Failed to collect memory usage: {e}")

        try:
            disk = psutil.disk_usage("/")
            metrics["disk_usage"] = disk.percent
        except Exception as e:
            self.logger.error(f"Failed to collect disk usage: {e}")

        return metrics

    def stop_monitoring(self):
        self._monitoring = False

    async def shutdown(self):
        self.stop_monitoring()
        if hasattr(self, "_monitor_task"):
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                self.logger.info("Monitor loop cancelled successfully.")

    async def check_database_health(self) -> bool:
        healthy, _, _ = await self.check_database()
        return healthy

    async def check_exchange_health(self) -> bool:
        healthy, _, _ = await self.check_exchange()
        return healthy

    async def check_market_data_health(self) -> bool:
        healthy, _, _ = await self.check_market_data()
        return healthy

    async def check_market_data_freshness(self) -> bool:
        try:
            last_update = await self.ctx.market_data.get_last_update()
            if not last_update:
                raise HealthCheckError("No market data updates found")

            staleness = datetime.now() - last_update
            if staleness > timedelta(minutes=5):
                raise HealthCheckError(f"Market data is stale: {staleness}")

            return True

        except Exception as e:
            self.logger.error(f"Market data freshness check failed: {e}")
            raise HealthCheckError(f"Market data check failed: {e}")

    async def validate_system_metrics(self, metrics: dict) -> None:
        from utils.error_handler import HealthCheckError

        if (
            metrics.get("cpu_usage", 0) > self.CRITICAL_CPU_THRESHOLD
            or metrics.get("memory_usage", 0) > self.CRITICAL_MEMORY_THRESHOLD
        ):
            raise HealthCheckError("System metrics out of range.")
        return

    async def check_recovery_conditions(self):
        if not self.should_emergency_shutdown():
            self.emergency_shutdown_triggered = False
        return

    async def is_trading_allowed(self) -> bool:
        """Check if trading should be allowed based on system health"""
        if self.emergency_shutdown_triggered or self._shutdown_in_progress:
            return False

        # Check critical components
        for component in ["exchange", "risk_manager", "circuit_breaker"]:
            if (
                component in self.components
                and not self.components[component].is_healthy
            ):
                return False

        return True

    def update_thresholds(self, thresholds: Dict[str, float]):
        """Update monitoring thresholds"""
        self.error_thresholds.update(thresholds)

    async def check_component_health(self, component: str) -> Tuple[bool, float, str]:
        """Check health of a specific component"""
        try:
            if component == "database":
                healthy, response_time, error_msg = await self.check_database()
            elif component == "exchange":
                healthy, response_time, error_msg = await self.check_exchange()
            elif component == "system":
                metrics = await self.check_system_resources()
                healthy = all(
                    metrics[k]
                    for k in ["memory_used_pct", "disk_used_pct", "cpu_used_pct"]
                )
                response_time = 0.0
                error_msg = ""
            else:
                healthy = False
                response_time = 0.0
                error_msg = f"Unknown component: {component}"

            return healthy, response_time, error_msg

        except Exception as e:
            self.logger.error(f"Error checking component {component}: {e}")
            return False, 0.0, str(e)

    async def check_database(self) -> Tuple[bool, str, Optional[Exception]]:
        """
        Check database health.

        Returns:
            Tuple of (is_healthy, status_message, exception if any)
        """
        try:
            if not hasattr(self, "ctx") or not self.ctx.db_connection:
                return False, "Database connection not initialized", None

            await self.ctx.db_connection.execute("SELECT 1")
            return True, "Database connection healthy", None

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False, f"Database error: {str(e)}", e

    async def get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system metrics.

        Returns:
            Dict with cpu_usage and memory_usage
        """
        try:
            cpu_usage = psutil.cpu_percent() / 100.0
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0

            return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}

        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {"cpu_usage": 0.0, "memory_usage": 0.0}

    async def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.

        Returns:
            Dict with health status and details
        """
        try:
            metrics = await self.get_system_metrics()
            db_healthy = await self.check_database_health()
            market_data_healthy = await self.check_market_data_freshness()

            is_healthy = (
                db_healthy
                and market_data_healthy
                and metrics["cpu_usage"] < self.CRITICAL_CPU_THRESHOLD / 100.0
                and metrics["memory_usage"] < self.CRITICAL_MEMORY_THRESHOLD / 100.0
            )

            if not is_healthy:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0

            return {
                "healthy": is_healthy,
                "database": db_healthy,
                "market_data": market_data_healthy,
                "metrics": metrics,
                "consecutive_failures": self.consecutive_failures,
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def check_system_readiness(self) -> bool:
        """
        Check if system is ready for operation.

        Returns:
            bool: True if system is ready
        """
        health_status = await self.check_system_health()
        return health_status.get("healthy", False)

    async def monitor_system_health(self):
        """Monitor system health continuously."""
        async with self._component_lock:
            health_status = await self.check_system_health()
            if not health_status["healthy"]:
                if (
                    health_status["consecutive_failures"]
                    >= self.MAX_CONSECUTIVE_FAILURES
                ):
                    await self.trigger_emergency_shutdown()
