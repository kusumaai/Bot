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

from utils.error_handler import handle_error, handle_error_async
from utils.numeric import NumericHandler


# dataclass for component health
@dataclass
class ComponentHealth:
    """Component health status"""

    name: str
    status: bool
    message: str = ""
    last_checked: float = field(default_factory=time.time)
    error_count: int = 0
    response_time: float = 0.0
    last_error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.status and self.error_count < 3


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
    """Health monitoring system"""

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
            "api": 3,
            "database": 2,
            "memory": 90,  # percent
            "max_errors": 5,
            "disk": 90,
            "cpu": 85,
        }
        self.error_counts = defaultdict(int)
        self.check_interval = 30  # seconds

        # Initialize status and components
        self._init_status_and_components()

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
                "HealthMonitor: Paper mode detected, skipping full initialization."
            )
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
        while True:
            try:
                async with self._component_lock:
                    health_status = await self.check_system_health()
                    if not health_status["healthy"]:
                        self.logger.warning("System health check failed")

                    # Update system metrics
                    metrics = await self.collect_system_metrics()
                    self.status.system_metrics.update(metrics)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                self.logger.info("Health monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def check_system_health(self) -> Dict[str, bool]:
        """Check system health"""
        try:
            now = time.time()
            if now - self.last_check < self.check_interval:
                return {"healthy": True}

            status = {
                "database": await self._check_database(),
                "exchange": await self._check_exchange(),
                "memory": await self._check_memory(),
                "market_data": await self._check_market_data(),
            }

            self.last_check = now
            self.status.is_healthy = all(status.values())
            self.status.timestamp = now
            self.status.warnings = []
            self.status.errors = []

            for comp, healthy in status.items():
                if not healthy:
                    self.status.errors.append(f"{comp} is unhealthy.")
                else:
                    self.status.components[comp].is_healthy = True

            return status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.status.is_healthy = False
            self.status.errors.append(str(e))
            return {"healthy": False, "error": str(e)}

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
            # Check connectivity via db_queries if available
            if self.db_queries and hasattr(self.db_queries, "ping_database"):
                return await self.db_queries.ping_database()
            return False

        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
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

    # check database
    async def check_database(self) -> Tuple[bool, float, Optional[str]]:
        """Check database connectivity and health"""
        start_time = time.time()
        try:
            if not hasattr(self.ctx, "db_connection") or not self.ctx.db_connection:
                error_msg = "Database connection not initialized"
                self.logger.warning(error_msg)
                self.components["database"].status = True
                self.components["database"].message = "Fresh database initialization"
                return True, 0.0, None

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
                    return True, 0.0, None
            # response time
            response_time = time.time() - start_time

            # Update component status
            self.components["database"].status = True
            self.components["database"].last_checked = time.time()
            self.components["database"].response_time = response_time
            self.components["database"].message = ""
            self.components["database"].error_count = 0

            return True, response_time, None

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
            return False, response_time, error_msg

    # check exchange
    async def check_exchange(self) -> Tuple[bool, float, Optional[str]]:
        """Check exchange connectivity and health"""
        start_time = time.time()
        try:
            if not self.ctx.exchange_interface:
                return False, 0.0, "Exchange interface not initialized"

            await self.ctx.exchange_interface.exchange.ping()
            response_time = time.time() - start_time

            # Update component status
            self.components["exchange"].status = True
            self.components["exchange"].last_checked = time.time()
            self.components["exchange"].response_time = response_time
            self.components["exchange"].message = ""

            return True, response_time, None

        except Exception as e:
            error_msg = f"Exchange health check failed: {str(e)}"
            response_time = time.time() - start_time

            # Update component status
            self.components["exchange"].status = False
            self.components["exchange"].last_checked = time.time()
            self.components["exchange"].response_time = response_time
            self.components["exchange"].message = error_msg
            self.components["exchange"].error_count += 1

            return False, response_time, error_msg

    # check system resources
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu = psutil.cpu_percent(interval=0.1)

            metrics = {
                "memory_used_pct": memory.percent,
                "disk_used_pct": disk.percent,
                "cpu_used_pct": cpu,
                "memory_healthy": memory.percent < self.error_thresholds["memory"],
                "disk_healthy": disk.percent < self.error_thresholds["disk"],
                "cpu_healthy": cpu < self.error_thresholds["cpu"],
                "healthy": True,
            }

            # Update system component status
            self.components["system"].status = all(
                [
                    metrics["memory_healthy"],
                    metrics["disk_healthy"],
                    metrics["cpu_healthy"],
                ]
            )
            self.components["system"].last_checked = time.time()
            self.components["system"].message = ""

            if not self.components["system"].status:
                self.components["system"].message = (
                    f"System resources critical: Memory={memory.percent}%, "
                    f"Disk={disk.percent}%, CPU={cpu}%"
                )

            metrics["healthy"] = self.components["system"].status
            return metrics

        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            self.components["system"].status = False
            self.components["system"].message = f"Resource check error: {str(e)}"
            return {
                "memory_used_pct": 0,
                "disk_used_pct": 0,
                "cpu_used_pct": 0,
                "memory_healthy": False,
                "disk_healthy": False,
                "cpu_healthy": False,
                "healthy": False,
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
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            metrics = self.check_system_resources()
            return {
                "memory_used_pct": float(metrics["memory_used_pct"]),
                "disk_used_pct": float(metrics["disk_used_pct"]),
                "cpu_used_pct": float(metrics["cpu_used_pct"]),
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {
                "memory_used_pct": 100.0,  # Fail safe - assume worst
                "disk_used_pct": 100.0,
                "cpu_used_pct": 100.0,
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
    def should_emergency_shutdown(self) -> bool:
        """Determine if emergency shutdown needed"""
        try:
            # Critical components must be healthy
            critical_components = ["database", "exchange", "position_manager"]
            critical_failures = sum(
                1
                for c in critical_components
                if c in self.components and not self.components[c].is_healthy
            )

            if critical_failures >= 2:
                return True

            # Check error rates
            high_error_components = sum(
                1
                for c in self.components.values()
                if c.error_count >= self.error_thresholds["max_errors"]
            )

            if high_error_components >= 2:
                return True

            # Check system resources
            sys_metrics = self.check_system_resources()
            if any(
                [
                    sys_metrics["memory_used_pct"] > self.error_thresholds["memory"],
                    sys_metrics["disk_used_pct"] > self.error_thresholds["disk"],
                    sys_metrics["cpu_used_pct"] > self.error_thresholds["cpu"],
                ]
            ):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in should_emergency_shutdown: {e}")
            return True  # Fail safe

    # get health status
    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        try:
            now = time.time()
            warnings = []
            errors = []

            # Check component health
            for name, comp in self.components.items():
                if not comp.healthy:
                    errors.append(f"{name} unhealthy: {comp.last_error}")
                elif comp.response_time > float(self.degradation_threshold) * np.median(
                    self.latency_history[name]
                ):
                    warnings.append(f"{name} performance degraded")

            # Check system resources
            sys_metrics = self.check_system_resources()

            return HealthStatus(
                timestamp=now,
                is_healthy=len(errors) == 0,
                warnings=warnings,
                errors=errors,
                components=self.components.copy(),
                system_metrics=sys_metrics,
            )

        except Exception as e:
            handle_error(e, "HealthMonitor.get_health_status", logger=self.logger)
            return HealthStatus(
                timestamp=time.time(),
                is_healthy=False,
                warnings=[],
                errors=[str(e)],
                components={},
                system_metrics={},
            )

    # get health report
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            now = time.time()
            status = self.get_health_status()

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
                    metrics = self.check_system_resources()
                    healthy = all(
                        metrics[k]
                        for k in ["memory_healthy", "disk_healthy", "cpu_healthy"]
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
                [db_healthy, exchange_healthy, self.check_system_resources()["healthy"]]
            )

            return self.components["system"].status

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return False

    # check system readiness
    async def check_system_readiness(self) -> Tuple[bool, Dict[str, bool]]:
        """Check if system components are ready for trading (beyond just health)"""
        readiness = {
            "database": False,
            "models": False,
            "ga_data": False,
            "market_data": False,
            "overall": False,
        }

        try:
            # Check if database has necessary tables and initial data
            if self.ctx.db_connection:
                async with self.ctx.db_connection.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        # Check for trained models
                        await cursor.execute(
                            "SELECT COUNT(*) FROM models WHERE status = 'active'"
                        )
                        model_count = await cursor.fetchone()
                        readiness["models"] = (
                            model_count[0] > 0 if model_count else False
                        )

                        # Check for GA data
                        await cursor.execute(
                            "SELECT COUNT(*) FROM genetic_algorithms WHERE status = 'complete'"
                        )
                        ga_count = await cursor.fetchone()
                        readiness["ga_data"] = ga_count[0] > 0 if ga_count else False

                        # Check for historical data
                        await cursor.execute("SELECT COUNT(*) FROM market_data")
                        data_count = await cursor.fetchone()
                        readiness["market_data"] = (
                            data_count[0] > 0 if data_count else False
                        )

                        readiness["database"] = (
                            True  # Database exists and can be queried
                        )

            # Overall readiness requires all components
            readiness["overall"] = all(
                ready
                for component, ready in readiness.items()
                if component != "overall"
            )
            # if not ready, log missing components
            if not readiness["overall"]:
                missing = [
                    comp
                    for comp, ready in readiness.items()
                    if not ready and comp != "overall"
                ]
                self.logger.info(
                    f"System not ready for trading. Missing: {', '.join(missing)}"
                )
            # return overall readiness and readiness components
            return readiness["overall"], readiness

        except Exception as e:
            self.logger.warning(f"Could not check system readiness: {e}")
            return False, readiness

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

    # check system health
    async def check_system_health(self) -> Dict[str, bool]:
        """Check system health"""
        try:
            # Check database health
            db_healthy, _, _ = await self.check_database()
            # Check exchange health
            exchange_healthy, _, _ = await self.check_exchange()
            # Check system resources
            sys_metrics = self.check_system_resources()
            # Check market data
            market_healthy, _, _ = await self.check_market_data()
            # return the system health
            return {
                "database": db_healthy,
                "exchange": exchange_healthy,
                "system": sys_metrics["healthy"],
                "market": market_healthy,
            }
        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
            return {
                "database": False,
                "exchange": False,
                "system": False,
                "market": False,
            }
