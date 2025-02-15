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
from typing import Any, Dict, Optional

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
        self.triggered = False
        self.last_check = time.time()
        self._monitor_task: Optional[asyncio.Task] = None
        self.nh = NumericHandler()
        self._lock = asyncio.Lock()
        self.emergency_triggered = False

    # initialize the circuit breaker components
    async def initialize(self) -> bool:
        """Initialize circuit breaker components."""
        try:
            if self.initialized:
                return True
            if (
                not self.ctx.portfolio_manager
                or not self.ctx.portfolio_manager.initialized
            ):
                self.logger.error("Portfolio manager must be initialized first")
                return False
            self.risk_limits = self.ctx.portfolio_manager.risk_limits
            self._monitor_task = asyncio.create_task(self.monitor_loop())
            self.initialized = True
            self.logger.info("CircuitBreaker initialized successfully.")
            return True
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker.initialize", self.logger)
            return False

    # continuously monitor the portfolio and enforce circuit breakers
    async def monitor_loop(self):
        """Continuously monitor portfolio and enforce circuit breakers"""
        while not self.triggered and self.ctx.running:
            try:
                if (
                    not self.ctx.portfolio_manager
                    or not self.ctx.portfolio_manager.risk_limits
                ):
                    await asyncio.sleep(60)
                    continue

                await self.check_conditions()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                await handle_error_async(e, "CircuitBreaker.monitor_loop", self.logger)
                await asyncio.sleep(60)  # Wait before retrying

    # check various risk conditions and enforce circuit breakers
    async def check_conditions(self) -> None:
        """Check various risk conditions and enforce circuit breakers"""
        try:
            if not self.initialized:
                await self.initialize()
                if not self.initialized:
                    return

            current_drawdown = await self._calculate_drawdown()
            try:
                emergency_stop_pct = Decimal(self.risk_limits["emergency_stop_pct"])
            except Exception:
                emergency_stop_pct = Decimal("0.05")

            if current_drawdown >= emergency_stop_pct:
                await self.trigger_emergency_stop(
                    f"Emergency stop triggered: drawdown {current_drawdown} >= {emergency_stop_pct}"
                )
                return

            daily_loss = await self._calculate_daily_loss()
            try:
                max_daily_loss = Decimal(self.risk_limits["max_daily_loss"])
            except Exception:
                max_daily_loss = Decimal("0.03")

            if daily_loss >= max_daily_loss:
                await self.trigger_emergency_stop(
                    f"Daily loss limit exceeded: {daily_loss} >= {max_daily_loss}"
                )
                return

        except Exception as e:
            await handle_error_async(e, "CircuitBreaker.check_conditions", self.logger)

    async def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        try:
            self.triggered = True
            self.logger.error(f"Circuit breaker triggered: {reason}")
            # Implement emergency stop logic here

        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker.trigger_emergency_stop", self.logger
            )

    # calculate the current drawdown of the portfolio
    async def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown"""
        try:
            if not self.ctx.portfolio_manager:
                return Decimal("0")

            initial_balance = self.ctx.portfolio_manager.balance
            current_balance = await self._get_current_balance()

            if initial_balance <= 0:
                return Decimal("0")

            return (initial_balance - current_balance) / initial_balance

        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker._calculate_drawdown", self.logger
            )
            return Decimal("0")

    async def _get_current_balance(self) -> Decimal:
        """Get current portfolio balance"""
        try:
            return (
                self.ctx.portfolio_manager.balance
                if self.ctx.portfolio_manager
                else Decimal("0")
            )
        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker._get_current_balance", self.logger
            )
            return Decimal("0")

    # calculate the daily loss of the portfolio
    async def _calculate_daily_loss(self) -> Decimal:
        """Calculate daily loss"""
        try:
            if not self.ctx.portfolio_manager:
                return Decimal("0")

            initial_balance = self.ctx.portfolio_manager.balance
            current_balance = await self._get_current_balance()

            if initial_balance <= 0:
                return Decimal("0")

            daily_loss = (initial_balance - current_balance) / initial_balance
            return daily_loss

        except Exception as e:
            await handle_error_async(
                e, "CircuitBreaker._calculate_daily_loss", self.logger
            )
            return Decimal("0")

    # check if emergency stop conditions are met
    async def check_emergency_stop(self):
        """Check if emergency stop conditions are met."""
        async with self._lock:
            try:
                portfolio = self.ctx.portfolio_manager
                if (
                    portfolio.current_drawdown
                    >= self.ctx.risk_manager.risk_limits["emergency_stop_pct"]
                ):
                    self.emergency_triggered = True
                    self.logger.warning("Emergency stop triggered due to drawdown.")
                    # Add logic to halt trading, close positions, etc.
            except Exception as e:
                await handle_error_async(
                    e, "CircuitBreaker.check_emergency_stop", self.logger
                )

    # verify the exchange connectivity
    async def _check_exchange(self) -> bool:
        """Verify exchange connectivity"""
        try:
            start_time = time.time()
            await self.ctx.exchange_interface.exchange.ping()  # This is correct, using the new structure
            return True
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker._check_exchange", self.logger)
            return False
