#!/usr/bin/env python3
"""
Module: trading/circuit_breaker.py
Production circuit breaker system
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from enum import Enum
from utils.numeric import NumericHandler

from risk.limits import RiskLimits
from risk.position import Position
from risk.manager import RiskManager
from utils.error_handler import handle_error

from database.queries import DatabaseQueries
from utils.error_handler import CircuitBreakerError

class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"

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
        
        # Initialize with config values
        self.nh = NumericHandler()
        self.state = CircuitBreakerState.NORMAL
        self.last_state_change = time.time()
        self.error_counts: Dict[str, int] = {}
        self.triggered_reasons: List[str] = []
        
        # Load thresholds from config with proper decimal handling
        self.config = db_queries.config.get("circuit_breaker", {})
        self.max_drawdown = Decimal(str(self.config.get("max_drawdown_pct", "10"))) / Decimal("100")
        self.max_position_loss = Decimal(str(self.config.get("max_position_loss_pct", "5"))) / Decimal("100")
        self.position_correlation_limit = Decimal(str(self.config.get("position_correlation_limit", "0.7")))
        self.max_order_errors = self.config.get("max_order_errors", 3)
        self.recovery_time = self.config.get("recovery_time_minutes", 30)
        
        # State tracking
        self.last_error_reset = time.time()
        
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
        """Record a new trade for rate limiting"""
        now = datetime.utcnow()
        
        if symbol not in self._trade_counts:
            self._trade_counts[symbol] = []
            
        self._trade_counts[symbol].append(now)
        
        # Clean old trade records
        hour_ago = now - timedelta(hours=1)
        self._trade_counts[symbol] = [
            t for t in self._trade_counts[symbol]
            if t > hour_ago
        ]
    
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
                (SELECT close FROM candles 
                 WHERE symbol = ? 
                 ORDER BY timestamp DESC LIMIT 1) /
                (SELECT MAX(close) FROM candles 
                 WHERE symbol = ? AND 
                 timestamp >= ?) - 1
            ) as drawdown
        """
        
        day_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        result = await self.db.execute(
            query,
            [symbol, symbol, day_start.timestamp()],
            fetch=True
        )
        
        if result and result[0]['drawdown'] < self.max_drawdown:
            self._circuit_open = True
            raise CircuitBreakerError(
                f"Maximum drawdown exceeded: {result[0]['drawdown']}"
            )
    
    async def _check_daily_loss(self) -> None:
        """Check daily loss limit"""
        query = """
            SELECT SUM(
                CASE 
                    WHEN direction = 'long' THEN 
                        (exit_price - entry_price) * size
                    ELSE 
                        (entry_price - exit_price) * size
                END
            ) / initial_balance as daily_pnl
            FROM trades
            WHERE timestamp >= ?
        """
        
        day_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        result = await self.db.execute(
            query,
            [day_start.timestamp()],
            fetch=True
        )
        
        if result and result[0]['daily_pnl'] < self.max_daily_loss:
            self._circuit_open = True
            raise CircuitBreakerError(
                f"Maximum daily loss exceeded: {result[0]['daily_pnl']}"
            )
    
    async def _check_trade_frequency(self, symbol: str) -> None:
        """Check trade frequency limits"""
        if symbol in self._trade_counts:
            trades_last_hour = len(self._trade_counts[symbol])
            if trades_last_hour >= self.max_trades_per_hour:
                raise CircuitBreakerError(
                    f"Maximum trades per hour exceeded: {trades_last_hour}"
                )
    
    async def _should_reset_circuit(self) -> bool:
        """Check if circuit breaker should be reset"""
        if not self._last_reset:
            return False
            
        cooldown_passed = (
            datetime.utcnow() - self._last_reset >
            timedelta(minutes=self.cooldown_minutes)
        )
        
        if not cooldown_passed:
            return False
            
        # Check if conditions have improved
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

    async def check_conditions(self) -> bool:
        """Check if trading should be allowed"""
        try:
            # Get current portfolio value
            portfolio_value = await self.ctx.portfolio_manager.get_total_value()
            
            # Update daily high water mark
            self.daily_high_balance = max(self.daily_high_balance, portfolio_value)
            
            # Calculate daily PnL
            self.daily_pnl = portfolio_value - self.daily_high_balance
            
            # Check daily loss threshold
            if self.daily_pnl <= -self.max_daily_loss * self.daily_high_balance:
                self.triggered_reasons.append("Daily loss threshold exceeded")
                await self._update_state(CircuitBreakerState.TRIGGERED)
                return False
                
            return self.state != CircuitBreakerState.TRIGGERED
            
        except Exception as e:
            self.ctx.logger.error(f"Circuit breaker check failed: {e}")
            return False
            
    async def _update_state(self, new_state: CircuitBreakerState) -> None:
        """Update circuit breaker state with logging"""
        if new_state != self.state:
            self.state = new_state
            self.last_state_change = time.time()
            self.log_state()

    def check_trading_limits(self) -> Optional[str]:
        """Check trading-based circuit breaker conditions"""
        try:
            # Get latest portfolio status
            portfolio = self.ctx.position_manager.get_portfolio_status()
            current_balance = Decimal(str(portfolio['current_balance']))
            
            # Update daily tracking
            if current_balance > self.daily_high_balance:
                self.daily_high_balance = current_balance
            
            # Check daily loss
            self.daily_pnl = current_balance - self.daily_high_balance
            if self.daily_pnl <= self.max_daily_loss:
                return f"Daily loss limit reached: {float(self.daily_pnl):.2%}"
                
            # Check drawdown
            if self.daily_high_balance > 0:
                drawdown = (self.daily_high_balance - current_balance) / self.daily_high_balance
                if drawdown >= self.max_drawdown:
                    return f"Maximum drawdown reached: {float(drawdown):.2%}"
                
            # Check individual positions
            for position in portfolio['positions']:
                position_return = Decimal(str(position['unrealized_pnl'])) / Decimal(str(position['size']))
                if position_return <= -self.max_position_loss:
                    return f"Position loss limit reached for {position['symbol']}: {float(position_return):.2%}"
                    
            # Check position correlation
            if len(portfolio['positions']) > 1:
                correlation = self.calculate_position_correlation()
                if correlation > float(self.position_correlation_limit):
                    return f"Position correlation limit exceeded: {correlation:.2f}"
                    
            return None
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.check_trading_limits", logger=self.ctx.logger)
            return "Error checking trading limits"

    def check_technical_limits(self) -> Optional[str]:
        """Check technical circuit breaker conditions"""
        try:
            # Get latest health report
            health = self.ctx.health_monitor.get_health_report()
            
            # Check critical components
            unhealthy_components = [
                name for name, status in health['components'].items()
                if not status['healthy']
            ]
            if unhealthy_components:
                return f"Unhealthy components: {', '.join(unhealthy_components)}"
                
            # Check error rates
            now = time.time()
            if now - self.last_error_reset > 300:  # Reset every 5 minutes
                self.error_counts = {}
                self.last_error_reset = now
                
            total_errors = sum(self.error_counts.values())
            if total_errors >= self.max_order_errors:
                return f"Error rate too high: {total_errors} errors in 5 minutes"
                
            # Check system resources
            sys_metrics = health['system_metrics']
            if sys_metrics['memory_used_pct'] > 95:
                return "Critical memory usage"
            if sys_metrics['disk_used_pct'] > 95:
                return "Critical disk usage"
                
            return None
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.check_technical_limits", logger=self.ctx.logger)
            return "Error checking technical limits"

    def calculate_position_correlation(self) -> float:
        """Calculate correlation between position returns"""
        try:
            positions = self.ctx.position_manager.get_portfolio_status()['positions']
            if len(positions) <= 1:
                return 0.0
                
            # Get return series for positions
            returns = {}
            for pos in positions:
                symbol = pos['symbol']
                with self.ctx.db_pool.connection() as conn:
                    prices = pd.read_sql_query(
                        """
                        SELECT close 
                        FROM candles 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 100
                        """,
                        conn,
                        params=[symbol]
                    )['close'].values
                    
                    if len(prices) > 1:
                        returns[symbol] = np.diff(np.log(prices))
            
            # Calculate correlations between all pairs
            max_correlation = 0.0
            for symbol1, ret1 in returns.items():
                for symbol2, ret2 in returns.items():
                    if symbol1 < symbol2:  # Only calculate each pair once
                        correlation = abs(np.corrcoef(ret1, ret2)[0,1])
                        max_correlation = max(max_correlation, correlation)
            
            return max_correlation
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.calculate_position_correlation", logger=self.ctx.logger)
            return 1.0  # Return max correlation on error to be safe

    async def open_circuit(self, reason: str) -> None:
        """Open the circuit breaker and halt trading"""
        if self.state != CircuitBreakerState.TRIGGERED:
            self.state = CircuitBreakerState.TRIGGERED
            self.last_state_change = time.time()
            self.triggered_reasons.append(reason)
            
            self.ctx.logger.critical(f"Circuit breaker opened: {reason}")
            
            try:
                # Cancel all pending orders
                if hasattr(self.ctx, 'exchange_interface'):
                    await self.ctx.exchange_interface.cancel_all_orders()
                
                # Close all positions if configured
                if self.config.get("close_positions_on_trigger", True):
                    await self.close_all_positions()
                
                # Log final state
                self.log_state()
                
            except Exception as e:
                handle_error(e, "CircuitBreaker.open_circuit", logger=self.ctx.logger)

    async def close_all_positions(self) -> None:
        """Close all open positions"""
        try:
            positions = self.ctx.position_manager.get_portfolio_status()['positions']
            
            for pos in positions:
                try:
                    await self.ctx.exchange_interface.close_position(
                        pos['symbol'],
                        Decimal(str(pos['size']))
                    )
                    self.ctx.logger.info(f"Closed position {pos['symbol']} size {pos['size']}")
                except Exception as e:
                    handle_error(e, "CircuitBreaker.close_position", logger=self.ctx.logger)
                    
        except Exception as e:
            handle_error(e, "CircuitBreaker.close_all_positions", logger=self.ctx.logger)

    async def test_recovery(self) -> bool:
        """Test if system can return to normal operation"""
        try:
            # Check if minimum recovery time elapsed
            if time.time() - self.last_state_change < self.recovery_time * 60:
                return False
            
            # Verify system health
            health = self.ctx.health_monitor.get_health_report()
            if not health['healthy']:
                return False
            
            # Check trading limits
            if self.check_trading_limits():
                return False
                
            # Check technical limits
            if self.check_technical_limits():
                return False
                
            # Test market connection
            if hasattr(self.ctx, 'exchange_interface'):
                try:
                    await self.ctx.exchange_interface.ping()
                except Exception:
                    return False
                    
            return True
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.test_recovery", logger=self.ctx.logger)
            return False

    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                if self.state == CircuitBreakerState.NORMAL:
                    # Check circuit breaker conditions
                    trading_violation = self.check_trading_limits()
                    if trading_violation:
                        await self.open_circuit(trading_violation)
                        continue
                        
                    technical_violation = self.check_technical_limits()
                    if technical_violation:
                        await self.open_circuit(technical_violation)
                        continue
                
                elif self.state == CircuitBreakerState.TRIGGERED:
                    # Check if we can try recovery
                    if await self.test_recovery():
                        self.state = CircuitBreakerState.NORMAL
                        self.last_state_change = time.time()
                        self.ctx.logger.info("Circuit breaker closed - resuming normal operation")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                handle_error(e, "CircuitBreaker.monitor_loop", logger=self.ctx.logger)
                await asyncio.sleep(5)  # Back off on error

    async def test_trading(self) -> bool:
        """Test trading functionality with minimal risk"""
        try:
            if not hasattr(self.ctx, 'exchange_interface'):
                return False
                
            # Verify order placement
            test_order = await self.ctx.exchange_interface.place_test_order()
            if not test_order:
                return False
                
            # Verify order cancellation
            if not await self.ctx.exchange_interface.cancel_order(test_order['id']):
                return False
                
            return True
            
        except Exception as e:
            handle_error(e, "CircuitBreaker.test_trading", logger=self.ctx.logger)
            return False

    def log_state(self) -> None:
        """Log current circuit breaker state"""
        state = {
            'state': self.state,
            'last_change': datetime.fromtimestamp(self.last_state_change).isoformat(),
            'triggered_reasons': self.triggered_reasons,
            'daily_pnl': float(self.daily_pnl),
            'error_counts': self.error_counts
        }
        
        self.ctx.logger.info(f"Circuit breaker state: {json.dumps(state, indent=2)}")

    def record_error(self, error_type: str) -> None:
        """Record error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics"""
        self.daily_high_balance = Decimal('0')
        self.daily_pnl = Decimal('0')
        self.triggered_reasons = []