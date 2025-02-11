#!/usr/bin/env python3
"""
Module: risk/portfolio.py
Portfolio management with proper risk tracking
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
import time
import threading
from datetime import datetime, timedelta
import asyncio

from utils.error_handler import handle_error
from .position import Position
from .limits import RiskLimits
from utils.numeric import NumericHandler

@dataclass
class PortfolioStats:
    """Portfolio performance statistics"""
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    drawdown: Decimal
    peak_value: Decimal
    daily_pnl: Decimal
    total_exposure: Decimal
    leverage: Decimal
    position_count: int

class PortfolioManager:
    def __init__(self, risk_limits: Any):
        self.nh = NumericHandler()
        self.risk_limits = risk_limits
        self.positions: Dict[str, Dict] = {}
        self.balance = Decimal(0)
        self.peak_balance = Decimal(0)
        self.daily_starting_balance = Decimal(0)
        self.realized_pnl = Decimal(0)
        self._portfolio_value: Decimal = Decimal(0)
        self._last_update: float = 0
        self._update_interval: float = 0.1  # 100ms
        self.lock = threading.Lock()
        self._last_daily_reset = datetime.now().date()
        self._lock = asyncio.Lock()
        self._last_value = Decimal('0')
        self._high_water_mark = Decimal('0')
        self._position_updates: List[Dict] = []
        
    def calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value including unrealized PnL"""
        try:
            now = time.time()
            if now - self._last_update >= self._update_interval:
                with self.lock:
                    self._portfolio_value = self.balance + sum(
                        pos['size'] * pos['current_price'] for pos in self.positions.values()
                    )
                    if self._portfolio_value > self.peak_balance:
                        self.peak_balance = self._portfolio_value
                    self._last_update = now
                    
                    # Check for daily reset
                    current_date = datetime.now().date()
                    if current_date > self._last_daily_reset:
                        self.daily_starting_balance = self._portfolio_value
                        self._last_daily_reset = current_date
                        
            return self._portfolio_value
            
        except Exception as e:
            handle_error(e, "PortfolioManager.calculate_portfolio_value", logger=None)
            return self._portfolio_value

    def calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak value"""
        try:
            portfolio_value = self.calculate_portfolio_value()
            if self.peak_balance == 0:
                return Decimal(0)
            return (self.peak_balance - portfolio_value) / self.peak_balance
            
        except Exception as e:
            handle_error(e, "PortfolioManager.calculate_drawdown", logger=None)
            return Decimal(0)

    def get_portfolio_stats(self) -> PortfolioStats:
        """Get comprehensive portfolio statistics"""
        try:
            with self.lock:
                total_value = self.calculate_portfolio_value()
                unrealized_pnl = sum(pos['size'] * pos['current_price'] for pos in self.positions.values())
                total_exposure = sum(
                    pos['size'] * pos['current_price'] for pos in self.positions.values()
                )
                leverage = total_exposure / total_value if total_value > 0 else Decimal(0)
                daily_pnl = total_value - self.daily_starting_balance
                
                return PortfolioStats(
                    total_value=total_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=self.realized_pnl,
                    drawdown=self.calculate_drawdown(),
                    peak_value=self.peak_balance,
                    daily_pnl=daily_pnl,
                    total_exposure=total_exposure,
                    leverage=leverage,
                    position_count=len(self.positions)
                )
                
        except Exception as e:
            handle_error(e, "PortfolioManager.get_portfolio_stats", logger=None)
            return PortfolioStats(
                total_value=Decimal(0),
                unrealized_pnl=Decimal(0),
                realized_pnl=Decimal(0),
                drawdown=Decimal(0),
                peak_value=Decimal(0),
                daily_pnl=Decimal(0),
                total_exposure=Decimal(0),
                leverage=Decimal(0),
                position_count=0
            )

    async def add_position(self, 
                          symbol: str, 
                          size: Decimal, 
                          entry_price: Decimal) -> bool:
        """Add new position with thread safety"""
        async with self._lock:
            try:
                # Validate against risk limits
                if len(self.positions) >= self.risk_limits.max_positions:
                    return False
                    
                position_value = size * entry_price
                total_value = await self.get_total_value()
                
                if position_value / total_value > self.risk_limits.max_position_size:
                    return False
                
                self.positions[symbol] = {
                    'size': self.nh.to_decimal(size),
                    'entry_price': self.nh.to_decimal(entry_price),
                    'current_price': self.nh.to_decimal(entry_price),
                    'unrealized_pnl': Decimal('0'),
                    'last_update': datetime.utcnow()
                }
                
                await self._update_portfolio_metrics()
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to add position: {e}")
                return False
                
    async def update_position(self, 
                            symbol: str, 
                            current_price: Decimal) -> None:
        """Update position with new price data"""
        async with self._lock:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            old_price = position['current_price']
            new_price = self.nh.to_decimal(current_price)
            
            # Update position metrics
            position['current_price'] = new_price
            position['unrealized_pnl'] = (
                position['size'] * (new_price - position['entry_price'])
            )
            position['last_update'] = datetime.utcnow()
            
            # Record update for analysis
            self._position_updates.append({
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'price_change': new_price - old_price,
                'unrealized_pnl': position['unrealized_pnl']
            })
            
            await self._update_portfolio_metrics()
            
    async def get_total_value(self) -> Decimal:
        """Get current portfolio value"""
        async with self._lock:
            total = Decimal('0')
            for pos in self.positions.values():
                total += pos['size'] * pos['current_price']
            return total
            
    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio-wide metrics"""
        current_value = await self.get_total_value()
        self._last_value = current_value
        self._high_water_mark = max(self._high_water_mark, current_value)

    def close_position(self, symbol: str, exit_price: Decimal) -> Optional[Position]:
        """Close position and update realized PnL"""
        try:
            with self.lock:
                if symbol not in self.positions:
                    return None
                    
                position = self.positions[symbol]
                position.update(exit_price)
                self.realized_pnl += position.unrealized_pnl
                self.balance += position.unrealized_pnl
                del self.positions[symbol]
                return position
                
        except Exception as e:
            handle_error(e, "PortfolioManager.close_position", logger=None)
            return None

    def update_position(self, symbol: str, current_price: Decimal) -> bool:
        """Update position with new price"""
        try:
            with self.lock:
                if symbol not in self.positions:
                    return False
                    
                self.positions[symbol].update(current_price)
                return True
                
        except Exception as e:
            handle_error(e, "PortfolioManager.update_position", logger=None)
            return False 