#!/usr/bin/env python3
# src/startup/system_init.py
"""
Module: src.startup
Provides system initialization and component setup with enhanced resource management.
"""
import asyncio
import gc
import logging
import os
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# import required modules
from config.risk_config import RiskConfig
from database.database import DatabaseConnection
from database.queries import DatabaseQueries
from exchanges.exchange_manager import ExchangeManager
from execution.exchange_interface import ExchangeInterface
from execution.market_data import MarketData
from execution.portfolio_manager import PortfolioManager
from monitoring.metrics import MetricsCollector
from risk.manager import RiskManager
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from utils.error_handler import InitializationError, init_error_handler
from utils.health_monitor import HealthMonitor
from utils.logger import StructuredLogger, setup_logging


# system initializer class that initializes the system by running the system initialization and startup checks using the run_system_init and run_startup_checks methods
class SystemInitializer:
    """Manage system initialization and component dependencies with enhanced resource management."""

    def __init__(self, config: Dict, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self._lock = asyncio.Lock()
        self.initialized = False
        self.components: Dict[str, bool] = {}
        self._dependencies = {
            "database": set(),
            "exchange": {"database"},
            "market_data": {"exchange"},
            "risk_manager": {"database", "market_data"},
            "portfolio_manager": {"database", "risk_manager", "exchange"},
            "circuit_breaker": {"portfolio_manager", "risk_manager"},
        }
        self._init_order = self._calculate_init_order()
        self._components_status = {}
        self.metrics = MetricsCollector(logger)
        self._component_locks = {
            component: asyncio.Lock() for component in self._dependencies.keys()
        }
        self._state_transitions = {}
        self._shutdown_event = asyncio.Event()
        self._component_events = {
            component: asyncio.Event() for component in self._dependencies.keys()
        }
        self._resource_tracker = {
            "db_connections": set(),
            "exchange_connections": set(),
            "file_handles": set(),
            "tasks": set(),
        }
        self._cleanup_timeout = 30.0  # seconds

    def _calculate_init_order(self) -> List[str]:
        """Calculate initialization order based on dependencies."""
        init_order = []
        visited = set()

        def visit(component: str):
            if component in visited:
                return
            visited.add(component)
            for dep in self._dependencies.get(component, set()):
                visit(dep)
            init_order.append(component)

        for component in self._dependencies:
            visit(component)
        return init_order

    # run system initialization checks
    async def run_startup_checks(self) -> bool:
        """Run system startup checks"""
        try:
            if not self.initialized:
                self.logger.error(
                    "System initialization failed, system not initialized"
                )
                return False

            # Check database connection
            if not await self.db.test_connection():
                self.logger.error("Database connection failed")
                return False

            # Check disk space
            if not self._check_disk_space():
                self.logger.error("Insufficient disk space")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Startup checks failed: {str(e)}")
            return False

    # check  if sufficient disk space is available
    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space is available"""
        try:
            min_space_mb = 1000  # 1GB minimum
            db_path = Path(self.db.db_path)
            usage = shutil.disk_usage(str(db_path.parent))
            free_space_mb = usage.free / (1024 * 1024)
            return free_space_mb >= min_space_mb
        except Exception as e:
            self.logger.error(f"Failed to check disk space: {str(e)}")
            return False

    # initialize the system by running the system initialization and startup checks using the run_system_init and run_startup_checks methods
    async def initialize_system(self) -> bool:
        """Initialize all system components in correct order."""
        if self.initialized:
            return True

        async with self._lock:
            try:
                self.logger.info("Starting system initialization")
                await self.metrics.start_collection()

                # Initialize components in dependency order
                for component in self._init_order:
                    start_time = asyncio.get_event_loop().time()
                    success = await self._initialize_component(component)
                    end_time = asyncio.get_loop().time()

                    if not success:
                        self.logger.error(f"Failed to initialize {component}")
                        await self._cleanup_failed_init(component)
                        return False

                    # Record metrics
                    await self.metrics.record_operation(
                        component=f"init_{component}",
                        success=True,
                        latency=end_time - start_time,
                    )

                self.initialized = True
                self.logger.info("System initialization completed successfully")
                return True

            except Exception as e:
                self.logger.error(f"System initialization failed: {e}")
                await self._cleanup_failed_init()
                return False

    async def _initialize_component(self, component: str) -> bool:
        """Initialize a single component with atomic state transitions."""
        async with self._component_locks[component]:
            try:
                if component in self._state_transitions:
                    self.logger.warning(
                        f"Component {component} is already in transition state: {self._state_transitions[component]}"
                    )
                    return False

                self._state_transitions[component] = "initializing"

                # Check dependencies
                for dep in self._dependencies.get(component, set()):
                    if not self._components_status.get(dep, False):
                        self.logger.error(
                            f"Cannot initialize {component}: dependency {dep} not initialized"
                        )
                        self._state_transitions.pop(component)
                        return False

                    # Wait for dependency to be fully initialized
                    if not self._component_events[dep].is_set():
                        try:
                            await asyncio.wait_for(
                                self._component_events[dep].wait(), timeout=30.0
                            )
                        except asyncio.TimeoutError:
                            self.logger.error(
                                f"Timeout waiting for dependency {dep} to initialize"
                            )
                            self._state_transitions.pop(component)
                            return False

                self.logger.info(f"Initializing {component}")

                # Initialize component
                success = False
                try:
                    if component == "database":
                        success = await self._init_database()
                    elif component == "exchange":
                        success = await self._init_exchange()
                    elif component == "market_data":
                        success = await self._init_market_data()
                    elif component == "risk_manager":
                        success = await self._init_risk_manager()
                    elif component == "portfolio_manager":
                        success = await self._init_portfolio_manager()
                    elif component == "circuit_breaker":
                        success = await self._init_circuit_breaker()
                    else:
                        self.logger.error(f"Unknown component: {component}")
                        success = False
                finally:
                    if success:
                        self._components_status[component] = True
                        self._component_events[component].set()
                    else:
                        await self._handle_component_failure(component)
                    self._state_transitions.pop(component)

                return success

            except Exception as e:
                self.logger.error(f"Error initializing {component}: {e}")
                await self._handle_component_failure(component)
                return False

    async def _track_resource(self, resource_type: str, resource: Any):
        """Track a resource for cleanup."""
        self._resource_tracker[resource_type].add(resource)

    async def _untrack_resource(self, resource_type: str, resource: Any):
        """Remove a resource from tracking."""
        self._resource_tracker[resource_type].discard(resource)

    async def _cleanup_resources(self, resource_type: str):
        """Clean up tracked resources of a specific type."""
        resources = self._resource_tracker[resource_type].copy()
        cleanup_tasks = []

        for resource in resources:
            if resource_type == "db_connections":
                cleanup_tasks.append(self._close_db_connection(resource))
            elif resource_type == "exchange_connections":
                cleanup_tasks.append(self._close_exchange_connection(resource))
            elif resource_type == "file_handles":
                cleanup_tasks.append(self._close_file_handle(resource))
            elif resource_type == "tasks":
                cleanup_tasks.append(self._cancel_task(resource))

        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=self._cleanup_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout cleaning up {resource_type}")
            except Exception as e:
                self.logger.error(f"Error cleaning up {resource_type}: {e}")

        self._resource_tracker[resource_type].clear()

    async def _close_db_connection(self, connection):
        """Safely close a database connection."""
        try:
            if hasattr(connection, "close"):
                await connection.close()
            await self._untrack_resource("db_connections", connection)
        except Exception as e:
            self.logger.error(f"Error closing DB connection: {e}")

    async def _close_exchange_connection(self, connection):
        """Safely close an exchange connection."""
        try:
            if hasattr(connection, "close"):
                await connection.close()
            await self._untrack_resource("exchange_connections", connection)
        except Exception as e:
            self.logger.error(f"Error closing exchange connection: {e}")

    async def _close_file_handle(self, handle):
        """Safely close a file handle."""
        try:
            if not handle.closed:
                handle.close()
            await self._untrack_resource("file_handles", handle)
        except Exception as e:
            self.logger.error(f"Error closing file handle: {e}")

    async def _cancel_task(self, task):
        """Safely cancel an asyncio task."""
        try:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await self._untrack_resource("tasks", task)
        except Exception as e:
            self.logger.error(f"Error cancelling task: {e}")

    async def _init_database(self) -> bool:
        """Initialize database connection with resource tracking."""
        try:
            self.db = DatabaseConnection(self.config["database"]["path"])
            await self._track_resource("db_connections", self.db)
            success = await self.db.initialize()
            if not success:
                await self._cleanup_resources("db_connections")
            return success
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            await self._cleanup_resources("db_connections")
            return False

    async def _init_exchange(self) -> bool:
        """Initialize exchange interface with resource tracking."""
        try:
            self.exchange = ExchangeInterface(
                self.config["exchange"], self.db, self.logger
            )
            await self._track_resource("exchange_connections", self.exchange)
            success = await self.exchange.initialize()
            if not success:
                await self._cleanup_resources("exchange_connections")
            return success
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {e}")
            await self._cleanup_resources("exchange_connections")
            return False

    async def _init_market_data(self) -> bool:
        """Initialize market data service."""
        try:
            self.market_data = MarketData(
                self.exchange,
                self.db,
                self.logger,
            )
            return await self.market_data.initialize()
        except Exception as e:
            self.logger.error(f"Market data initialization failed: {e}")
            return False

    async def _init_risk_manager(self) -> bool:
        """Initialize risk manager."""
        try:
            self.risk_manager = RiskManager(
                self.config["risk"],
                self.market_data,
                self.db,
                self.logger,
            )
            return await self.risk_manager.initialize()
        except Exception as e:
            self.logger.error(f"Risk manager initialization failed: {e}")
            return False

    async def _init_portfolio_manager(self) -> bool:
        """Initialize portfolio manager."""
        try:
            self.portfolio_manager = PortfolioManager(
                self.exchange,
                self.risk_manager,
                self.db,
                self.logger,
            )
            return await self.portfolio_manager.initialize()
        except Exception as e:
            self.logger.error(f"Portfolio manager initialization failed: {e}")
            return False

    async def _init_circuit_breaker(self) -> bool:
        """Initialize circuit breaker."""
        try:
            self.circuit_breaker = CircuitBreaker(
                self.portfolio_manager,
                self.risk_manager,
                self.logger,
            )
            return await self.circuit_breaker.initialize()
        except Exception as e:
            self.logger.error(f"Circuit breaker initialization failed: {e}")
            return False

    async def _handle_component_failure(self, component: str) -> None:
        """Handle component initialization failure with proper cleanup."""
        async with self._component_locks[component]:
            try:
                self._components_status[component] = False
                self._component_events[component].clear()

                # Clean up dependent components
                dependents = [
                    c for c, deps in self._dependencies.items() if component in deps
                ]

                for dependent in dependents:
                    if self._components_status.get(dependent, False):
                        self.logger.info(f"Cleaning up dependent component {dependent}")
                        await self._cleanup_component(dependent)

            except Exception as e:
                self.logger.error(
                    f"Error handling component failure for {component}: {e}"
                )

    async def _cleanup_component(self, component: str) -> None:
        """Clean up a component with proper state management."""
        async with self._component_locks[component]:
            try:
                if hasattr(self, component):
                    component_instance = getattr(self, component)
                    if hasattr(component_instance, "close"):
                        await component_instance.close()
                self._components_status[component] = False
                self._component_events[component].clear()
            except Exception as e:
                self.logger.error(f"Error cleaning up component {component}: {e}")

    async def _cleanup_failed_init(
        self, failed_component: Optional[str] = None
    ) -> None:
        """Enhanced cleanup after failed initialization."""
        self.logger.info("Cleaning up after failed initialization")

        cleanup_tasks = []
        # Clean up components in reverse initialization order
        for component in reversed(self._init_order):
            if component == failed_component:
                break
            if self._components_status.get(component, False):
                cleanup_tasks.append(self._cleanup_component(component))

        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=self._cleanup_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error("Timeout during component cleanup")
            except Exception as e:
                self.logger.error(f"Error during component cleanup: {e}")

        # Clean up all tracked resources
        for resource_type in self._resource_tracker:
            await self._cleanup_resources(resource_type)

        self._components_status.clear()
        self.initialized = False

        # Force garbage collection
        gc.collect()

    async def shutdown(self) -> None:
        """Enhanced shutdown with comprehensive resource cleanup."""
        if not self.initialized:
            return

        self.logger.info("Starting system shutdown")
        self._shutdown_event.set()
        await self.metrics.stop_collection()

        shutdown_tasks = []
        # Shutdown components in reverse initialization order
        for component in reversed(self._init_order):
            async with self._component_locks[component]:
                if self._components_status.get(component, False):
                    shutdown_tasks.append(self._cleanup_component(component))

        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=self._cleanup_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error("Timeout during shutdown")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

        # Clean up all tracked resources
        for resource_type in self._resource_tracker:
            await self._cleanup_resources(resource_type)

        self._components_status.clear()
        self._state_transitions.clear()
        self.initialized = False

        # Force garbage collection
        gc.collect()

        self.logger.info("System shutdown completed")
