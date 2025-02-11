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

from risk.limits import RiskLimits
from risk.position import Position
from risk.manager import RiskManager

class CircuitBreakerState:
    CLOSED = "closed"     # Normal operation
    HALF_OPEN = "half"    # Testing if system can resume
    OPEN = "open"         # Trading halted

class CircuitBreaker:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = time.time()
        self.triggered_reasons: List[str] = []
        
        # Load thresholds from config
        self.config = ctx.config.get("circuit_breaker", {})
        self.max_daily_loss = Decimal(str(self.config.get("max_daily_loss", -1000)))
        self.max_drawdown = Decimal(str(self.config.get("max_drawdown_pct", 10))) / 100
        self.max_position_loss = Decimal(str(self.config.get("max_position_loss_pct", 5))) / 100
        self.position_correlation_limit = self.config.get("position_correlation_limit", 0.7)
        self.max_order_errors = self.config.get("max_order_errors", 3)
        self.recovery_time = self.config.get("recovery_time_minutes", 30)
        
        # State tracking
        self.daily_high_balance = Decimal('0')
        self.daily_pnl = Decimal('0')
        self.error_counts: Dict[str, int] = {}
        self.last_error_reset = time.time()
        
        # Start monitoring
        asyncio.create_task(self.monitor_loop())

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
                return f"Daily loss limit reached: {self.daily_pnl}"
                
            # Check drawdown
            drawdown = (self.daily_high_balance - current_balance) / self.daily_high_balance
            if drawdown >= self.max_drawdown:
                return f"Maximum drawdown reached: {drawdown:.2%}"
                
            # Check individual positions
            for position in portfolio['positions']:
                position_return = Decimal(str(position['unrealized_pnl'])) / Decimal(str(position['size']))
                if position_return <= -self.max_position_loss:
                    return f"Position loss limit reached for {position['symbol']}: {position_return:.2%}"
                    
            # Check position correlation
            if len(portfolio['positions']) > 1:
                correlation = self.calculate_position_correlation()
                if correlation > self.position_correlation_limit:
                    return f"Position correlation limit exceeded: {correlation:.2f}"
                    
            return None
            
        except Exception as e:
            self.ctx.logger.error(f"Error in trading limits check: {str(e)}")
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
            if len(unhealthy_components) >= 2:
                return f"Multiple critical components unhealthy: {', '.join(unhealthy_components)}"
                
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
            self.ctx.logger.error(f"Error in technical limits check: {str(e)}")
            return "Error checking technical limits"

    def calculate_position_correlation(self) -> float:
        """Calculate correlation between position returns"""
        try:
            positions = self.ctx.position_manager.get_portfolio_status()['positions']
            if len(positions) <= 1:
                return 0.0
                
            # Get return series for each position# Get return series for positions
            returns = {}
            for pos in positions:
                symbol = pos['symbol']
                with self.ctx.db_pool.connection() as conn:
                    query = """
                        SELECT close 
                        FROM candles 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    """
                    cursor = conn.execute(query, [symbol])
                    prices = [row[0] for row in cursor.fetchall()]
                    if len(prices) > 1:
                        # Calculate log returns
                        returns[symbol] = np.diff(np.log(prices))
            
            # Calculate correlations between all pairs
            max_correlation = 0.0
            for symbol1, ret1 in returns.items():
                for symbol2, ret2 in returns.items():
                    if symbol1 != symbol2:
                        correlation = abs(np.corrcoef(ret1, ret2)[0,1])
                        max_correlation = max(max_correlation, correlation)
            
            return max_correlation
            
        except Exception as e:
            self.ctx.logger.error(f"Error calculating position correlation: {str(e)}")
            return 1.0  # Return max correlation on error to be safe

    async def open_circuit(self, reason: str) -> None:
        """Open the circuit breaker and halt trading"""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
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
                self.ctx.logger.error(f"Error during circuit breaker open: {str(e)}")

    async def close_all_positions(self) -> None:
        """Close all open positions"""
        try:
            positions = self.ctx.position_manager.get_portfolio_status()['positions']
            
            for pos in positions:
                try:
                    success = await self.ctx.exchange_interface.close_position(
                        pos['symbol'],
                        pos['size']
                    )
                    
                    if success:
                        self.ctx.logger.info(
                            f"Closed position {pos['symbol']} size {pos['size']}"
                        )
                    else:
                        self.ctx.logger.error(
                            f"Failed to close position {pos['symbol']}"
                        )
                        
                except Exception as e:
                    self.ctx.logger.error(
                        f"Error closing position {pos['symbol']}: {str(e)}"
                    )
                    
        except Exception as e:
            self.ctx.logger.error(f"Error in close_all_positions: {str(e)}")

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
            self.ctx.logger.error(f"Error in recovery test: {str(e)}")
            return False

    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                if self.state == CircuitBreakerState.CLOSED:
                    # Check circuit breaker conditions
                    trading_violation = self.check_trading_limits()
                    if trading_violation:
                        await self.open_circuit(trading_violation)
                        continue
                        
                    technical_violation = self.check_technical_limits()
                    if technical_violation:
                        await self.open_circuit(technical_violation)
                        continue
                
                elif self.state == CircuitBreakerState.OPEN:
                    # Check if we can try recovery
                    if await self.test_recovery():
                        self.state = CircuitBreakerState.HALF_OPEN
                        self.last_state_change = time.time()
                        self.ctx.logger.info("Circuit breaker entering half-open state")
                
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    # Test with reduced trading
                    trading_ok = await self.test_trading()
                    if trading_ok:
                        self.state = CircuitBreakerState.CLOSED
                        self.last_state_change = time.time()
                        self.ctx.logger.info("Circuit breaker closed - resuming normal operation")
                    else:
                        self.state = CircuitBreakerState.OPEN
                        self.last_state_change = time.time()
                        self.ctx.logger.warning("Circuit breaker recovery failed - returning to open state")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.ctx.logger.error(f"Error in circuit breaker monitor: {str(e)}")
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
            self.ctx.logger.error(f"Error in trading test: {str(e)}")
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