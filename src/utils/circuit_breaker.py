import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIPPED = "TRIPPED"


class CircuitBreaker:
    """Circuit breaker for handling system failures."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        max_recovery_attempts: int = 3,
        warning_threshold: int = 2,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before tripping
            recovery_timeout: Seconds to wait before recovery attempt
            max_recovery_attempts: Maximum number of recovery attempts
            warning_threshold: Number of failures before warning
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.max_recovery_attempts = max_recovery_attempts
        self.warning_threshold = warning_threshold

        self.failure_count = 0
        self.recovery_attempts = 0
        self.last_failure_time = None
        self.last_recovery_time = None
        self.state = CircuitState.NORMAL
        self.triggered = False

        self._monitor_task = None
        self._component_lock = asyncio.Lock()
        self.logger = logger

    async def record_failure(self):
        """Record a failure and update circuit state."""
        async with self._component_lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                await self.trip()
            elif self.failure_count >= self.warning_threshold:
                self.state = CircuitState.WARNING

    async def trip(self):
        """Trip the circuit breaker."""
        self.state = CircuitState.TRIPPED
        self.triggered = True
        self.logger.warning("Circuit breaker tripped")

        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def reset(self):
        """Reset the circuit breaker."""
        async with self._component_lock:
            self.failure_count = 0
            self.recovery_attempts = 0
            self.last_failure_time = None
            self.last_recovery_time = None
            self.state = CircuitState.NORMAL
            self.triggered = False

            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None

            self.logger.info("Circuit breaker reset")

    async def attempt_recovery(self):
        """Attempt to recover from tripped state."""
        async with self._component_lock:
            if self.state != CircuitState.TRIPPED:
                return

            if self.recovery_attempts >= self.max_recovery_attempts:
                self.logger.error("Max recovery attempts reached")
                return

            self.recovery_attempts += 1
            self.last_recovery_time = datetime.now()

            try:
                # Add recovery logic here
                await self.reset()
                self.logger.info("Recovery successful")

            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}")

    async def _monitor_loop(self):
        """Monitor loop for recovery attempts."""
        while True:
            try:
                if self.state == CircuitState.TRIPPED:
                    if self.last_recovery_time:
                        time_since_recovery = datetime.now() - self.last_recovery_time
                        if time_since_recovery.seconds >= self.recovery_timeout:
                            await self.attempt_recovery()
                    else:
                        await self.attempt_recovery()

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def check_state(self) -> CircuitState:
        """Get current circuit state."""
        async with self._component_lock:
            return self.state

    def is_tripped(self) -> bool:
        """Check if circuit is tripped."""
        return self.state == CircuitState.TRIPPED
