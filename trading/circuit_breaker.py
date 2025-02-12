#!/usr/bin/env python3
"""
Module: trading/circuit_breaker.py 
Production circuit breaker system
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum


from utils.numeric import NumericHandler
from risk.limits import RiskLimits
from risk.position import Position
from risk.manager import RiskManager
from utils.error_handler import handle_error, CircuitBreakerError
from database.queries import DatabaseQueries

class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING" 
    TRIGGERED = "TRIGGERED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    CLOSED = "CLOSED"

class CircuitBreaker:
    def __init__(
        self,
        db_queries: DatabaseQueries,
        logger: logging.Logger,
        max_drawdown: Decimal = Decimal('-0.1'),
        max_daily_loss: Decimal = Decimal('-0.03'),
        max_trades_per_hour: int = 10,
        cooldown_minutes: int = 60
    ):
        self.db = db_queries
        self.logger = logger
        self.max_drawdown = max_drawdown 
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_hour = max_trades_per_hour
        self.cooldown_minutes = cooldown_minutes
        
        self._circuit_open = False
        self._last_reset: Optional[datetime] = None
        self._trade_counts: Dict[str, List[datetime]] = {}
        self.nh = NumericHandler()
        
        # State tracking
        self.state = CircuitBreakerState.NORMAL
        self.last_state_change = time.time()
        self.error_counts: Dict[str, int] = {}
        self.triggered_reasons: List[str] = []
        self._last_error_reset = time.time()
        
        # Performance tracking
        self.error_window = timedelta(minutes=5)
        self.degradation_threshold = Decimal('2.0')
        
        # Start monitoring
        asyncio.create_task(self.monitor_loop())

    async def check_circuit(self, symbol: str) -> bool:
        """Check if trading should be allowed"""
        try:
            if self._circuit_open:
                if await self._should_reset_circuit():
                    await self._reset_circuit()
                else:
                    raise CircuitBreakerError("Circuit breaker is active")
            
            await self._check_conditions(symbol)
            return True
            
        except CircuitBreakerError as e:
            self.logger.warning(f"Circuit breaker check failed: {str(e)}")
            raise
    
    async def record_trade(self, symbol: str) -> None:
        """Record trade for rate limiting"""
        now = datetime.utcnow()
        if symbol not in self._trade_counts:
            self._trade_counts[symbol] = []
        self._trade_counts[symbol].append(now)
        
        # Clean old records
        hour_ago = now - timedelta(hours=1)
        self._trade_counts[symbol] = [t for t in self._trade_counts[symbol] if t > hour_ago]
    
    async def _check_conditions(self, symbol: str) -> None:
        """Check all circuit breaker conditions"""
        await asyncio.gather(
            self._check_drawdown(),
            self._check_daily_loss(),
            self._check_trade_frequency(symbol)
        )
    
    async def _check_drawdown(self) -> None:
        """Check current drawdown"""
        query = """
            SELECT (
                MAX(close) - (
                    SELECT close FROM candles 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT 1
                )
            ) / MAX(close) as drawdown
            FROM candles WHERE symbol = ? AND timestamp >= ?
        """
        
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        result = await self.db.execute(query, [symbol, symbol, day_start.timestamp()], fetch=True)
        
        if result and result[0]['drawdown'] < self.max_drawdown:
            self._circuit_open = True
            raise CircuitBreakerError(f"Maximum drawdown exceeded: {result[0]['drawdown']}")
    
    async def _check_daily_loss(self) -> None:
        """Check daily loss limit"""
        query = """
            SELECT SUM(pnl) / initial_balance as daily_pnl
            FROM trades WHERE timestamp >= ?
        """
        
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.db.execute(query, [day_start.timestamp()], fetch=True)
        
        if result and result[0]['daily_pnl'] < self.max_daily_loss:
            self._circuit_open = True
            raise CircuitBreakerError(f"Maximum daily loss exceeded: {result[0]['daily_pnl']}")
    
    async def _check_trade_frequency(self, symbol: str) -> None:
        """Check trade frequency limits"""
        if symbol in self._trade_counts:
            trades_last_hour = len(self._trade_counts[symbol])
            if trades_last_hour >= self.max_trades_per_hour:
                raise CircuitBreakerError(f"Maximum trades per hour exceeded: {trades_last_hour}")
    
    async def _should_reset_circuit(self) -> bool:
        """Check if circuit breaker should reset"""
        if not self._last_reset:
            return False
            
        cooldown_passed = (datetime.utcnow() - self._last_reset > timedelta(minutes=self.cooldown_minutes))
        if not cooldown_passed:
            return False
            
        try:
            await self._check_conditions(symbol)
            return True
        except CircuitBreakerError:
            return False
    
    async def _reset_circuit(self) -> None:
        """Reset circuit breaker"""
        self._circuit_open = False
        self._last_reset = datetime.utcnow()
        self.logger.info("Circuit breaker reset")

    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                if self.state == CircuitBreakerState.NORMAL:
                    trading_violation = await self.check_trading_limits()
                    if trading_violation:
                        await self.trigger(trading_violation)
                        continue
                        
                    technical_violation = await self.check_technical_limits()
                    if technical_violation:
                        await self.trigger(technical_violation)
                        continue
                
                elif self.state == CircuitBreakerState.TRIGGERED:
                    if await self.test_recovery():
                        self.state = CircuitBreakerState.NORMAL
                        self.last_state_change = time.time()
                        self.logger.info("Circuit breaker closed - resuming normal operation")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                handle_error(e, "CircuitBreaker.monitor_loop", logger=self.logger)
                await asyncio.sleep(5)

    async def trigger(self, reason: str) -> None:
        """Trigger circuit breaker"""
        if self.state != CircuitBreakerState.TRIGGERED:
            self.state = CircuitBreakerState.TRIGGERED
            self.last_state_change = time.time()
            self.triggered_reasons.append(reason)
            
            self.logger.critical(f"Circuit breaker triggered: {reason}")
            
            try:
                if self.ctx.config.get("close_positions_on_trigger", True):
                    await self.close_all_positions()
                self.log_state()
            except Exception as e:
                handle_error(e, "CircuitBreaker.trigger", logger=self.logger)

    async def close_all_positions(self) -> None:
        """Close all open positions"""
        try:
            positions = await self.ctx.position_manager.get_open_positions()
            for pos in positions:
                try:
                    await self.ctx.exchange_interface.close_position(pos)
                except Exception as e:
                    handle_error(e, "CircuitBreaker.close_position", logger=self.logger)
        except Exception as e:
            handle_error(e, "CircuitBreaker.close_all_positions", logger=self.logger)

    async def test_recovery(self) -> bool:
        """Test if system can return to normal"""
        try:
            # Check recovery time
            if time.time() - self.last_state_change < self.cooldown_minutes * 60:
                return False
                
            # Check limits
            if await self.check_trading_limits() or await self.check_technical_limits():
                return False
                
            # Test exchange connection
            try:
                await self.ctx.exchange_interface.ping()
            except:
                return False
                
            return True
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.test_recovery", logger=self.logger)
            return False

    def log_state(self) -> None:
        """Log circuit breaker state"""
        state = {
            'state': self.state.value,
            'last_change': datetime.fromtimestamp(self.last_state_change).isoformat(),
            'triggered_reasons': self.triggered_reasons,
            'error_counts': self.error_counts
        }
        self.logger.info(f"Circuit breaker state: {json.dumps(state, indent=2)}")

    def record_error(self, error_type: str) -> None:
        """Record error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics"""
        self.daily_high_balance = Decimal('0')
        self.daily_pnl = Decimal('0')
        self.triggered_reasons = []