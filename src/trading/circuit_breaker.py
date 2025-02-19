#!/usr/bin/env python3
"""Highly optimized circuit breaker with caching, batching and backoff."""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional, Set

import numpy as np

from bot_types.base_types import RiskLimits
from utils.error_handler import handle_error_async
from utils.exceptions import CircuitBreakerError
from utils.numeric_handler import NumericHandler

# Core constants
_HEALTH_THRESHOLD = 0.7
_MAX_RECOVERY_ATTEMPTS = 3
_MONITOR_INTERVAL = 60
_CONSECUTIVE_ERROR_THRESHOLD = 3
_COMPONENT_HEALTH_LOCK_TIMEOUT = 0.1
_STATE_LOCK_TIMEOUT = 0.2
_WARNING_DELAY = 300
_SHUTDOWN_TIMEOUT = 30
_STATE_CACHE_TTL = 0.1  # 100ms cache lifetime
_BACKOFF_FACTOR = 1.5  # Exponential backoff multiplier

# Component weights as numpy array for faster computation
_CRITICAL_COMPONENTS = {
    "portfolio": 0.4,
    "orders": 0.3,
    "risk": 0.3,
}
_HEALTH_WEIGHTS = np.array([w for w in _CRITICAL_COMPONENTS.values()])


class CircuitBreaker:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self._state_lock = asyncio.Lock()
        self._component_locks: Dict[str, asyncio.Lock] = {}
        self._shutdown_in_progress = False
        self._last_check = time.time()
        self._monitor_task: Optional[asyncio.Task] = None
        self.nh = NumericHandler()
        self.emergency_triggered = False
        self.risk_limits = None

        # State management
        self._state = "NORMAL"
        self._affected_components: Set[str] = set()
        self._recovery_attempts = 0
        self._last_error: Optional[str] = None
        self._shutdown_complete = asyncio.Event()

        # Component health tracking with numpy optimization
        self._component_states: Dict[str, str] = {}
        self._component_health: Dict[str, float] = {}
        self._component_health_array = np.zeros(len(_CRITICAL_COMPONENTS))

        # Warning state debouncing
        self._warning_start_time: Optional[float] = None
        self._warning_triggered = False

        # Status caching
        self._last_state_cache: Optional[Dict[str, Any]] = None
        self._last_cache_time = 0.0

        # Health update batching
        self._pending_health_updates: Dict[str, float] = {}
        self._health_update_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        try:
            if self.initialized:
                return True

            async with asyncio.timeout(_STATE_LOCK_TIMEOUT):
                async with self._state_lock:
                    if not self._verify_dependencies():
                        return False

                    self._initialize_component_states()

                    try:
                        self._monitor_task = asyncio.create_task(self._monitor_loop())
                        self.initialized = True
                        self._state = "NORMAL"
                        self.logger.info("CircuitBreaker initialized successfully")
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to start monitor task: {e}")
                        await self._cleanup()
                        return False

        except asyncio.TimeoutError:
            self.logger.error("Timeout initializing circuit breaker")
            return False
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker.initialize", self.logger)
            return False

    def _verify_dependencies(self) -> bool:
        if (
            not hasattr(self.ctx, "portfolio_manager")
            or not self.ctx.portfolio_manager.initialized
        ):
            self.logger.error("Portfolio manager must be initialized first")
            return False

        self.risk_limits = getattr(self.ctx.portfolio_manager, "risk_limits", None)
        if not self.risk_limits:
            self.logger.error("Risk limits not found")
            return False

        return True

    def _initialize_component_states(self):
        self._component_states = {comp: "NORMAL" for comp in _CRITICAL_COMPONENTS}
        self._component_health = {comp: 1.0 for comp in _CRITICAL_COMPONENTS}

    async def _monitor_loop(self):
        consecutive_errors = 0
        while not self.emergency_triggered and getattr(self.ctx, "running", False):
            try:
                if not self.initialized or not self.risk_limits:
                    await asyncio.sleep(_MONITOR_INTERVAL)
                    continue

                await self._check_conditions()
                consecutive_errors = 0
                await asyncio.sleep(_MONITOR_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                await handle_error_async(e, "CircuitBreaker._monitor_loop", self.logger)

                if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
                    self.logger.critical("Multiple consecutive monitoring errors!")
                    await self.trigger_emergency_stop("Multiple monitoring failures")
                    break

                await asyncio.sleep(_MONITOR_INTERVAL)


async def _check_conditions(self) -> None:
    try:
        async with asyncio.timeout(_STATE_LOCK_TIMEOUT):
            async with self._state_lock:
                current_drawdown = await self._calculate_drawdown()
                daily_loss = await self._calculate_daily_loss()
                emergency_stop_pct = self.risk_limits.emergency_stop_pct
                max_daily_loss = self.risk_limits.max_daily_loss
                warning_threshold = emergency_stop_pct * Decimal("0.8")

                # EMERGENCY conditions
                if (
                    current_drawdown >= emergency_stop_pct
                    or daily_loss >= max_daily_loss
                    or await self._calculate_system_health() > (1 - _HEALTH_THRESHOLD)
                    or self._warning_triggered
                ):
                    await self._transition_state("EMERGENCY")
                    return

                # WARNING conditions with debouncing
                if current_drawdown >= warning_threshold:
                    current_time = time.time()
                    if self._warning_start_time is None:
                        self._warning_start_time = current_time
                    elif current_time - self._warning_start_time > _WARNING_DELAY:
                        self._warning_triggered = True
                        await self._transition_state("EMERGENCY")
                        return
                    else:
                        await self._transition_state("WARNING")
                else:
                    self._warning_start_time = None
                    self._warning_triggered = False
                    await self._transition_state("NORMAL")

    except asyncio.TimeoutError:
        self.logger.error("Timeout checking conditions")
        await self.update_component_health("circuit_breaker", 0.2)
    except Exception as e:
        await handle_error_async(e, "CircuitBreaker._check_conditions", self.logger)
        if self._recovery_attempts >= _MAX_RECOVERY_ATTEMPTS:
            await self.trigger_emergency_stop("Max recovery attempts exceeded")


async def trigger_emergency_stop(self, reason: str) -> None:
    async with self._state_lock:
        if self._shutdown_in_progress:
            self.logger.warning("Emergency stop already in progress")
            return

        self.logger.critical(f"Emergency stop triggered: {reason}")
        self._shutdown_in_progress = True
        self.emergency_triggered = True
        self._state = "EMERGENCY"

        shutdown_tasks = []

        # Portfolio shutdown
        if hasattr(self.ctx, "portfolio_manager"):
            shutdown_tasks.append(
                self._safe_component_shutdown(
                    "portfolio",
                    self.ctx.portfolio_manager.close_all_positions,
                    "Emergency stop triggered",
                )
            )

        # Order shutdown
        if hasattr(self.ctx, "order_manager"):
            shutdown_tasks.append(
                self._safe_component_shutdown(
                    "orders",
                    self.ctx.order_manager.cancel_all_orders,
                    "Emergency stop triggered",
                )
            )

        # Exchange shutdown
        if hasattr(self.ctx, "exchange_interface"):
            shutdown_tasks.append(
                self._safe_component_shutdown(
                    "exchange",
                    self._graceful_exchange_disconnect,
                    "Emergency stop triggered",
                )
            )

    try:
        await asyncio.wait_for(
            asyncio.gather(*shutdown_tasks, return_exceptions=True),
            timeout=_SHUTDOWN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        self.logger.error("Emergency stop timed out")
    except Exception as e:
        self.logger.error(f"Error during emergency stop: {str(e)}")
    finally:
        self._shutdown_complete.set()


async def _graceful_exchange_disconnect(self, *args):
    """Gracefully disconnect from exchange with connection drain."""
    try:
        if hasattr(self.ctx, "exchange_interface"):
            try:
                # Cancel any pending requests
                await self.ctx.exchange_interface.cancel_pending_requests()
                # Drain existing connections
                await self.ctx.exchange_interface.drain_connections()
                # Finally disconnect
                await self.ctx.exchange_interface.disconnect()
            except Exception as e:
                self.logger.error(f"Error during exchange disconnect step: {e}")
    except Exception as e:
        self.logger.error(f"Error during exchange disconnect: {e}")


# Rest of the implementation remains the same...
async def _safe_component_shutdown(
    self, component: str, shutdown_func: Any, *args
) -> None:
    try:
        self._component_states[component] = "SHUTTING_DOWN"
        await asyncio.wait_for(shutdown_func(*args), timeout=_SHUTDOWN_TIMEOUT)
        self._component_states[component] = "SHUTDOWN"
    except asyncio.TimeoutError:
        self.logger.error(f"Timeout shutting down {component}")
        self._component_states[component] = "TIMEOUT"
        self._affected_components.add(component)
    except Exception as e:
        self.logger.error(f"Error shutting down {component}: {e}")
        self._component_states[component] = "ERROR"
        self._affected_components.add(component)


async def _calculate_system_health(self) -> float:
    """Optimized system health calculation using numpy."""
    self._component_health_array[:] = [
        self._component_health.get(c, 0.0) for c in _CRITICAL_COMPONENTS
    ]
    return float(np.dot(1 - self._component_health_array, _HEALTH_WEIGHTS))


async def _transition_state(self, new_state: str) -> None:
    if new_state == self._state:
        return

    self.logger.info(f"Transitioning from {self._state} to {new_state}")
    old_state = self._state
    self._state = new_state

    if new_state == "EMERGENCY":
        await self.trigger_emergency_stop(f"State transition from {old_state}")
    elif new_state == "NORMAL" and old_state in ("WARNING", "EMERGENCY"):
        await self._attempt_recovery()


async def _attempt_recovery(self) -> None:
    """Recovery with exponential backoff."""
    if self._recovery_attempts >= _MAX_RECOVERY_ATTEMPTS:
        self.logger.error("Max recovery attempts exceeded")
        return

    # Calculate backoff delay
    recovery_delay = _BACKOFF_FACTOR**self._recovery_attempts
    await asyncio.sleep(recovery_delay)

    self._recovery_attempts += 1
    try:
        components_to_reset = list(self._affected_components)
        reset_results = await asyncio.gather(
            *[self._reset_component(c) for c in components_to_reset],
            return_exceptions=True,
        )

        # Process results
        for component, result in zip(components_to_reset, reset_results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to reset {component}: {result}")
            elif result:
                self._affected_components.discard(component)

        if not self._affected_components:
            self._recovery_attempts = 0
            self._last_error = None
            self.logger.info("System recovered successfully")

    except Exception as e:
        self._last_error = str(e)
        self.logger.error(f"Recovery attempt failed: {e}")


async def _batch_health_update(self):
    """Process batched health updates efficiently."""
    async with self._health_update_lock:
        if not self._pending_health_updates:
            return

        async with self._state_lock:
            for component, score in self._pending_health_updates.items():
                self._component_health[component] = max(0.0, min(1.0, score))

                if score < 0.3:
                    new_state = "ERROR"
                elif score < 0.7:
                    new_state = "WARNING"
                else:
                    new_state = "NORMAL"

                old_state = self._component_states.get(component)
                if old_state != new_state:
                    self._component_states[component] = new_state
                    if new_state == "ERROR":
                        self._affected_components.add(component)
                    elif new_state == "NORMAL":
                        self._affected_components.discard(component)

        self._pending_health_updates.clear()


async def _attempt_recovery(self) -> None:
    """Recovery with exponential backoff."""
    if self._recovery_attempts >= _MAX_RECOVERY_ATTEMPTS:
        self.logger.error("Max recovery attempts exceeded")
        return

    # Calculate backoff delay
    recovery_delay = _BACKOFF_FACTOR**self._recovery_attempts
    await asyncio.sleep(recovery_delay)

    self._recovery_attempts += 1
    try:
        components_to_reset = list(self._affected_components)
        reset_results = await asyncio.gather(
            *[self._reset_component(c) for c in components_to_reset],
            return_exceptions=True,
        )

        # Process results
        for component, result in zip(components_to_reset, reset_results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to reset {component}: {result}")
            elif result:
                self._affected_components.discard(component)

        if not self._affected_components:
            self._recovery_attempts = 0
            self._last_error = None
            self.logger.info("System recovered successfully")

    except Exception as e:
        self._last_error = str(e)
        self.logger.error(f"Recovery attempt failed: {e}")


async def _batch_health_update(self):
    """Process batched health updates efficiently."""
    async with self._health_update_lock:
        if not self._pending_health_updates:
            return

        async with self._state_lock:
            for component, score in self._pending_health_updates.items():
                self._component_health[component] = max(0.0, min(1.0, score))

                if score < 0.3:
                    new_state = "ERROR"
                elif score < 0.7:
                    new_state = "WARNING"
                else:
                    new_state = "NORMAL"

                old_state = self._component_states.get(component)
                if old_state != new_state:
                    self._component_states[component] = new_state
                    if new_state == "ERROR":
                        self._affected_components.add(component)
                    elif new_state == "NORMAL":
                        self._affected_components.discard(component)

        self._pending_health_updates.clear()


async def update_component_health(self, component: str, health_score: float) -> None:
    """Queue health update for batch processing."""
    self._pending_health_updates[component] = max(0.0, min(1.0, health_score))

    # Process batch if enough updates or time elapsed
    if len(self._pending_health_updates) >= 3 or (
        self._pending_health_updates and time.time() - self._last_check > 1.0
    ):
        await self._batch_health_update()


async def get_status(self) -> Dict[str, Any]:
    """Get status with caching."""
    now = time.time()
    if self._last_state_cache and now - self._last_cache_time < _STATE_CACHE_TTL:
        return self._last_state_cache.copy()

    try:
        async with asyncio.timeout(_STATE_LOCK_TIMEOUT):
            async with self._state_lock:
                self._last_state_cache = {
                    "state": self._state,
                    "triggered": self._shutdown_in_progress,
                    "emergency_triggered": self.emergency_triggered,
                    "affected_components": list(self._affected_components),
                    "component_states": self._component_states.copy(),
                    "component_health": self._component_health.copy(),
                    "recovery_attempts": self._recovery_attempts,
                    "last_error": self._last_error,
                    "last_check": self._last_check,
                    "warning_triggered": self._warning_triggered,
                    "warning_start_time": self._warning_start_time,
                    "system_health": await self._calculate_system_health(),
                }
                self._last_cache_time = now
                return self._last_state_cache.copy()

    except asyncio.TimeoutError:
        self.logger.error("Status retrieval timed out")
        return {
            "state": "UNKNOWN",
            "triggered": True,
            "emergency_triggered": True,
            "error": "Status retrieval timed out",
        }


async def _cleanup(self) -> None:
    """Cleanup resources with improved error handling."""
    try:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self.logger.warning("Monitor task cleanup timed out")

        # Clear all state
        self._affected_components.clear()
        self._component_states.clear()
        self._component_health.clear()
        self._pending_health_updates.clear()
        self._component_health_array.fill(0)
        self._recovery_attempts = 0
        self._last_error = None
        self.initialized = False
        self._state = "NORMAL"
        self._shutdown_complete.clear()
        self._warning_start_time = None
        self._warning_triggered = False
        self._last_state_cache = None
        self._last_cache_time = 0

    except Exception as e:
        self.logger.error(f"Cleanup failed: {e}")

    finally:
        # Ensure these critical flags are reset even if cleanup fails
        self._shutdown_in_progress = False
        self.emergency_triggered = False


async def _calculate_drawdown(self) -> Decimal:
    """Calculate current drawdown using NumericHandler."""
    try:
        if hasattr(self.ctx, "portfolio_manager"):
            return await self.ctx.portfolio_manager.get_current_drawdown()
        return self.nh.create_decimal("0")
    except Exception as e:
        await handle_error_async(e, "CircuitBreaker._calculate_drawdown", self.logger)
        return self.nh.create_decimal("0")


async def _calculate_daily_loss(self) -> Decimal:
    """Calculate current daily loss using NumericHandler."""
    try:
        if hasattr(self.ctx, "portfolio_manager"):
            return await self.ctx.portfolio_manager.get_daily_loss()
        return self.nh.create_decimal("0")
    except Exception as e:
        await handle_error_async(e, "CircuitBreaker._calculate_daily_loss", self.logger)
        return self.nh.create_decimal("0")


@property
def triggered(self) -> bool:
    """Returns True if the circuit breaker has been triggered."""
    return self.emergency_triggered or self._shutdown_in_progress
