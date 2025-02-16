#! /usr/bin/env python3
# src/trading/circuit_breaker.py
"""
Module: src.trading
Provides circuit breaker implementation for risk management.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional, Set

from bot_types.base_types import RiskLimits
from utils.error_handler import handle_error_async
from utils.exceptions import CircuitBreakerError
from utils.numeric_handler import NumericHandler


# circuit breaker class that implements the circuit breaker pattern for risk management and trading
class CircuitBreaker:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self._state_lock = asyncio.Lock()  # Main state lock
        self._component_locks: Dict[str, asyncio.Lock] = {}  # Per-component locks
        self._shutdown_in_progress = False
        self._last_check = time.time()
        self._monitor_task: Optional[asyncio.Task] = None
        self.nh = NumericHandler()
        self.emergency_triggered = False
        self.risk_limits = None
        self._state = "NORMAL"  # NORMAL, WARNING, EMERGENCY
        self._affected_components: Set[str] = set()
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._last_error: Optional[Exception] = None
        self._shutdown_complete = asyncio.Event()
        self._component_states: Dict[str, str] = {}
        self._component_health: Dict[str, float] = {}  # Atomic health scores
        self._critical_components: Dict[str, float] = {
            "portfolio": 0.4,
            "orders": 0.3,
            "risk": 0.3,
        }
        self._health_threshold = 0.7  # 70% health threshold

    # initialize the circuit breaker components
    async def initialize(self) -> bool:
        """Initialize circuit breaker components with enhanced error handling."""
        try:
            if self.initialized:
                return True

            async with self._state_lock:
                # Ensure portfolio manager is initialized
                if (
                    not hasattr(self.ctx, "portfolio_manager")
                    or not self.ctx.portfolio_manager.initialized
                ):
                    self.logger.error("Portfolio manager must be initialized first")
                    return False

                # Get risk limits
                self.risk_limits = getattr(
                    self.ctx.portfolio_manager, "risk_limits", None
                )
                if not self.risk_limits:
                    self.logger.error("Risk limits not found")
                    return False

                # Initialize component states
                self._component_states = {
                    "portfolio": "NORMAL",
                    "orders": "NORMAL",
                    "risk": "NORMAL",
                }

                # Start monitoring task with error recovery
                try:
                    self._monitor_task = asyncio.create_task(self.monitor_loop())
                    self.initialized = True
                    self._state = "NORMAL"
                    self.logger.info("CircuitBreaker initialized successfully")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to start monitor task: {e}")
                    await self._cleanup()
                    return False

        except Exception as e:
            await handle_error_async(e, "CircuitBreaker.initialize", self.logger)
            return False

    # continuously monitor the portfolio and enforce circuit breakers
    async def monitor_loop(self):
        """Enhanced monitoring loop with error recovery."""
        consecutive_errors = 0
        while not self.triggered and getattr(self.ctx, "running", False):
            try:
                if not self.initialized or not self.risk_limits:
                    await asyncio.sleep(60)
                    continue

                await self.check_conditions()
                consecutive_errors = 0  # Reset on successful check
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                await handle_error_async(e, "CircuitBreaker.monitor_loop", self.logger)

                if consecutive_errors >= 3:
                    self.logger.error(
                        "Multiple consecutive monitoring errors, triggering emergency stop"
                    )
                    await self.trigger_emergency_stop("Multiple monitoring failures")
                    break

                await asyncio.sleep(60)  # Wait before retrying

    # check various risk conditions and enforce circuit breakers
    async def check_conditions(self) -> None:
        """Enhanced condition checking with proper state synchronization."""
        try:
            if not self.initialized or not self.risk_limits:
                return

            async with self._state_lock:
                # Take atomic snapshots of all state we need to check
                current_drawdown = await self._calculate_drawdown()
                daily_loss = await self._calculate_daily_loss()
                emergency_stop_pct = self.risk_limits.emergency_stop_pct
                max_daily_loss = self.risk_limits.max_daily_loss
                warning_threshold = emergency_stop_pct * Decimal("0.8")
                current_state = self._state

                # Determine new state based on conditions
                new_state = current_state
                if (
                    current_drawdown >= emergency_stop_pct
                    or daily_loss >= max_daily_loss
                    or await self.should_emergency_shutdown()
                ):
                    new_state = "EMERGENCY"
                elif current_drawdown >= warning_threshold:
                    new_state = "WARNING"
                else:
                    new_state = "NORMAL"

                # Only transition if state actually changed
                if new_state != current_state:
                    await self._transition_state(new_state)

        except Exception as e:
            self._last_error = e
            await handle_error_async(e, "CircuitBreaker.check_conditions", self.logger)
            if self._recovery_attempts >= self._max_recovery_attempts:
                async with self._state_lock:
                    await self.trigger_emergency_stop("Max recovery attempts exceeded")

    async def trigger_emergency_stop(self, reason: str) -> None:
        """Thread-safe emergency stop with proper state synchronization."""
        async with self._state_lock:
            if self._shutdown_in_progress:
                self.logger.warning("Emergency stop already in progress")
                return

            self._shutdown_in_progress = True
            self.emergency_triggered = True
            self._state = "EMERGENCY"
            self.logger.error(f"Circuit breaker triggered: {reason}")

            # Create shutdown tasks outside the lock
            shutdown_tasks = []
            SHUTDOWN_TIMEOUT = 30

            if hasattr(self.ctx, "portfolio_manager"):
                shutdown_tasks.append(
                    self._safe_component_shutdown_with_timeout(
                        "portfolio",
                        self.ctx.portfolio_manager.close_all_positions,
                        SHUTDOWN_TIMEOUT,
                        "Emergency stop triggered",
                    )
                )

            if hasattr(self.ctx, "order_manager"):
                shutdown_tasks.append(
                    self._safe_component_shutdown_with_timeout(
                        "orders",
                        self.ctx.order_manager.cancel_all_orders,
                        SHUTDOWN_TIMEOUT,
                        "Emergency stop triggered",
                    )
                )

        # Execute shutdown tasks with timeout outside the lock
        try:
            await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=SHUTDOWN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self.logger.error("Emergency stop timed out")
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {str(e)}")
        finally:
            self._shutdown_complete.set()

    async def _safe_component_shutdown_with_timeout(
        self, component: str, shutdown_func: Any, timeout: int, *args
    ) -> None:
        """Safely shutdown a component with timeout."""
        try:
            self._component_states[component] = "SHUTTING_DOWN"
            await asyncio.wait_for(shutdown_func(*args), timeout=timeout)
            self._component_states[component] = "SHUTDOWN"
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout shutting down {component}")
            self._component_states[component] = "TIMEOUT"
            self._affected_components.add(component)
        except Exception as e:
            self.logger.error(f"Error shutting down {component}: {e}")
            self._component_states[component] = "ERROR"
            self._affected_components.add(component)

    async def _safe_component_shutdown(self, component: str, shutdown_func: Any, *args):
        """Safely shutdown a component with state tracking."""
        try:
            self._component_states[component] = "SHUTTING_DOWN"
            await shutdown_func(*args)
            self._component_states[component] = "SHUTDOWN"
        except Exception as e:
            self.logger.error(f"Error shutting down {component}: {e}")
            self._component_states[component] = "ERROR"
            self._affected_components.add(component)

    async def _transition_state(self, new_state: str) -> None:
        """Handle state transitions with proper cleanup."""
        if new_state == self._state:
            return

        self.logger.info(f"Transitioning from {self._state} to {new_state}")
        old_state = self._state
        self._state = new_state

        if new_state == "EMERGENCY":
            await self.trigger_emergency_stop(f"State transition from {old_state}")
        elif new_state == "WARNING":
            # Implement warning state actions (e.g., reduce position sizes)
            pass
        elif new_state == "NORMAL" and old_state in ("WARNING", "EMERGENCY"):
            await self._attempt_recovery()

    async def _attempt_recovery(self) -> None:
        """Attempt to recover from warning/emergency state."""
        if self._recovery_attempts >= self._max_recovery_attempts:
            self.logger.error("Max recovery attempts exceeded")
            return

        self._recovery_attempts += 1
        try:
            # Verify system state
            conditions_ok = await self.check_conditions()
            if not conditions_ok:
                return

            # Reset affected components
            for component in self._affected_components.copy():
                if await self._reset_component(component):
                    self._affected_components.remove(component)

            if not self._affected_components:
                self._recovery_attempts = 0
                self._last_error = None
                self.logger.info("System recovered successfully")

        except Exception as e:
            self._last_error = e
            self.logger.error(f"Recovery attempt failed: {e}")

    async def _reset_component(self, component: str) -> bool:
        """Reset a component to normal state."""
        try:
            if component == "portfolio":
                return await self.ctx.portfolio_manager.reset()
            elif component == "orders":
                return await self.ctx.order_manager.reset()
            return False
        except Exception as e:
            self.logger.error(f"Failed to reset {component}: {e}")
            return False

    async def _cleanup(self) -> None:
        """Cleanup resources and state."""
        try:
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

            self._affected_components.clear()
            self._component_states.clear()
            self._recovery_attempts = 0
            self._last_error = None
            self.initialized = False
            self._state = "NORMAL"
            self._shutdown_complete.clear()

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Thread-safe status retrieval."""
        async with self._state_lock:
            return {
                "state": self._state,
                "triggered": self._shutdown_in_progress,
                "emergency_triggered": self.emergency_triggered,
                "affected_components": list(self._affected_components),
                "component_states": self._component_states.copy(),
                "component_health": self._component_health.copy(),
                "recovery_attempts": self._recovery_attempts,
                "last_error": str(self._last_error) if self._last_error else None,
                "last_check": self._last_check,
            }

    # calculate the current drawdown of the portfolio
    async def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown"""
        try:
            if hasattr(self.ctx, "portfolio_manager"):
                return await self.ctx.portfolio_manager.get_current_drawdown()
            return Decimal("0")
        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker._calculate_drawdown", self.logger
            )
            return Decimal("0")

    # calculate the daily loss of the portfolio
    async def _calculate_daily_loss(self) -> Decimal:
        """Calculate current daily loss"""
        try:
            if hasattr(self.ctx, "portfolio_manager"):
                return await self.ctx.portfolio_manager.get_daily_loss()
            return Decimal("0")
        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker._calculate_daily_loss", self.logger
            )
            return Decimal("0")

    # check if emergency stop conditions are met
    async def check_emergency_stop(self) -> bool:
        """Check if emergency stop conditions are met."""
        async with self._state_lock:
            try:
                if not self.initialized or not self.risk_limits:
                    return False

                current_drawdown = await self._calculate_drawdown()
                if current_drawdown >= self.risk_limits.emergency_stop_pct:
                    await self.trigger_emergency_stop(
                        f"Emergency stop check triggered: drawdown {current_drawdown} >= {self.risk_limits.emergency_stop_pct}"
                    )
                    return True
                return False

            except Exception as e:
                await handle_error_async(
                    e, "CircuitBreaker.check_emergency_stop", self.logger
                )
                return False

    # verify the exchange connectivity
    async def _check_exchange(self) -> bool:
        """Verify exchange connectivity"""
        try:
            if hasattr(self.ctx, "exchange_interface"):
                start_time = time.time()
                await self.ctx.exchange_interface.exchange.ping()
                return True
            return False
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker._check_exchange", self.logger)
            return False

    async def should_emergency_shutdown(self) -> bool:
        """
        Thread-safe check for emergency shutdown conditions.
        Returns True if shutdown should be triggered.
        """
        async with self._state_lock:
            if self._shutdown_in_progress:
                return True

            # Take atomic snapshot of component health
            health_snapshot = self._component_health.copy()
            critical_failure_impact = 0.0

            # Calculate weighted health impact
            for component, weight in self._critical_components.items():
                health = health_snapshot.get(component, 0.0)
                critical_failure_impact += (1 - health) * weight

            # Check if cumulative impact exceeds threshold
            if critical_failure_impact > (1 - self._health_threshold):
                self._shutdown_in_progress = True
                return True

            return False

    async def update_component_health(
        self, component: str, health_score: float
    ) -> None:
        """
        Thread-safe update of component health score.

        Args:
            component: Component name
            health_score: Health score between 0.0 and 1.0
        """
        if component not in self._component_locks:
            self._component_locks[component] = asyncio.Lock()

        async with self._component_locks[component]:
            self._component_health[component] = max(0.0, min(1.0, health_score))

            # Update component state based on health
            if health_score < 0.3:
                new_state = "ERROR"
            elif health_score < 0.7:
                new_state = "WARNING"
            else:
                new_state = "NORMAL"

            await self._update_component_state(component, new_state)

    async def _update_component_state(self, component: str, new_state: str) -> None:
        """Thread-safe update of component state."""
        async with self._state_lock:
            old_state = self._component_states.get(component)
            if old_state != new_state:
                self._component_states[component] = new_state
                if new_state == "ERROR":
                    self._affected_components.add(component)
                elif new_state == "NORMAL":
                    self._affected_components.discard(component)
