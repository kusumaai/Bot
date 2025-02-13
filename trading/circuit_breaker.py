#!/usr/bin/env python3
"""
Module: trading/circuit_breaker.py
Circuit breaker implementation for risk management
"""

from decimal import Decimal
from typing import Dict, Any, Optional
import logging
import asyncio
import time

from utils.error_handler import handle_error_async
from bot_types.base_types import RiskLimits
from utils.numeric_handler import NumericHandler

class CircuitBreaker:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self.triggered = False
        self.last_check = time.time()
        self._monitor_task: Optional[asyncio.Task] = None
        self.nh = NumericHandler()

    async def initialize(self) -> bool:
        """Initialize circuit breaker"""
        try:
            if self.initialized:
                return True
            if not self.ctx.portfolio_manager or not self.ctx.portfolio_manager.initialized:
                self.logger.error("Portfolio manager must be initialized first")
                return False
            self.risk_limits = self.ctx.portfolio_manager.risk_limits
            self._monitor_task = asyncio.create_task(self.monitor_loop())
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize circuit breaker: {e}")
            return False

    async def monitor_loop(self):
        """Continuously monitor portfolio and enforce circuit breakers"""
        while not self.triggered and self.ctx.running:
            try:
                await self.check_conditions()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                await handle_error_async(e, "CircuitBreaker.monitor_loop", self.logger)
                await asyncio.sleep(60)  # Wait before retrying

    async def check_conditions(self) -> None:
        """Check various risk conditions and enforce circuit breakers"""
        try:
            if not self.initialized:
                await self.initialize()
                if not self.initialized:
                    return

            current_drawdown = await self._calculate_drawdown()
            if current_drawdown >= self.risk_limits.emergency_stop_pct:
                await self.trigger_emergency_stop(
                    f"Emergency stop triggered: drawdown {current_drawdown} >= {self.risk_limits.emergency_stop_pct}"
                )
                return

            # Check daily loss limit
            daily_loss = await self._calculate_daily_loss()
            if daily_loss >= self.risk_limits.max_daily_loss:
                await self.trigger_emergency_stop(
                    f"Daily loss limit exceeded: {daily_loss} >= {self.risk_limits.max_daily_loss}"
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
            await handle_error_async(e, "CircuitBreaker.trigger_emergency_stop", self.logger)

    async def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown"""
        try:
            if not self.ctx.portfolio_manager:
                return Decimal('0')
            
            initial_balance = self.ctx.portfolio_manager.balance
            current_balance = await self._get_current_balance()
            
            if initial_balance <= 0:
                return Decimal('0')
                
            return (initial_balance - current_balance) / initial_balance
            
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker._calculate_drawdown", self.logger)
            return Decimal('0')

    async def _get_current_balance(self) -> Decimal:
        """Get current portfolio balance"""
        try:
            return self.ctx.portfolio_manager.balance if self.ctx.portfolio_manager else Decimal('0')
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker._get_current_balance", self.logger)
            return Decimal('0')

    async def _calculate_daily_loss(self) -> Decimal:
        """Calculate daily loss"""
        try:
            if not self.ctx.portfolio_manager:
                return Decimal('0')
            
            initial_balance = self.ctx.portfolio_manager.balance
            current_balance = await self._get_current_balance()
            
            if initial_balance <= 0:
                return Decimal('0')
                
            daily_loss = (initial_balance - current_balance) / initial_balance
            return daily_loss
            
        except Exception as e:
            await handle_error_async(e, "CircuitBreaker._calculate_daily_loss", self.logger)
            return Decimal('0')