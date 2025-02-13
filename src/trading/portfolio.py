#!/usr/bin/env python3
"""
Module: trading/portfolio.py
Portfolio management with proper risk tracking
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
import time
import asyncio
import logging
import threading
from datetime import datetime
from collections import deque

from utils.error_handler import handle_error
from trading.position import Position
from risk.limits import RiskLimits
from utils.numeric_handler import NumericHandler
from trading.exceptions import PortfolioError

@dataclass
class PortfolioStats:
    """Portfolio performance statistics"""
    # Required fields (no defaults)
    total_value: Decimal
    cash_balance: Decimal
    position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    
    # Optional fields (with defaults)
    margin_used: Decimal = Decimal('0')
    free_margin: Decimal = Decimal('0')
    risk_ratio: float = 0.0
    exposure: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioManager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.positions = []
        self.risk_limits = ctx.risk_limits
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self.positions: Dict[str, Position] = {}
        self.balance = Decimal('0')
        self.peak_balance = Decimal('0')
        self.daily_starting_balance = Decimal('0')
        self.realized_pnl = Decimal('0')
        self._portfolio_value: Decimal = Decimal('0')
        self._last_update: float = 0
        self._update_interval: float = 0.1  # 100ms
        self.lock = threading.Lock()
        self._last_daily_reset = datetime.now().date()
        self._last_value: Decimal = Decimal('0')
        self._high_water_mark: Decimal = Decimal('0')
        self._position_updates: deque = deque(maxlen=1000)  # Limit to prevent unbounded growth
        self.nh = NumericHandler()
        self.initialized = False
        
    async def initialize(self):
        self.initialized = True
        return True

    async def get_total_value(self):
        return Decimal("10000")

    def calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value including unrealized PnL"""
        try:
            now = time.time()
            if now - self._last_update >= self._update_interval:
                with self.lock:
                    self._portfolio_value = self.balance + sum(
                        pos.size * pos.current_price for pos in self.positions.values()
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
            handle_error(e, "PortfolioManager.calculate_portfolio_value", logger=self.logger)
            return self._portfolio_value

    def calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak value"""
        try:
            portfolio_value = self.calculate_portfolio_value()
            if self.peak_balance == 0:
                return Decimal('0')
            return (self.peak_balance - portfolio_value) / self.peak_balance
            
        except Exception as e:
            handle_error(e, "PortfolioManager.calculate_drawdown", logger=self.logger)
            return Decimal('0')

    def get_portfolio_stats(self) -> PortfolioStats:
        """Get comprehensive portfolio statistics"""
        try:
            with self.lock:
                total_value = self.calculate_portfolio_value()
                unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
                total_exposure = sum(
                    pos.size * pos.current_price for pos in self.positions.values()
                )
                leverage = total_exposure / total_value if total_value > 0 else Decimal('0')
                daily_pnl = total_value - self.daily_starting_balance
                
                return PortfolioStats(
                    total_value=total_value,
                    cash_balance=self.balance,
                    position_value=total_value - self.balance,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=self.realized_pnl,
                    margin_used=Decimal('0'),
                    free_margin=Decimal('0'),
                    risk_ratio=leverage,
                    exposure=total_exposure,
                    metadata={}
                )
                
        except Exception as e:
            handle_error(e, "PortfolioManager.get_portfolio_stats", logger=self.logger)
            return PortfolioStats(
                total_value=Decimal('0'),
                cash_balance=Decimal('0'),
                position_value=Decimal('0'),
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                margin_used=Decimal('0'),
                free_margin=Decimal('0'),
                risk_ratio=0.0,
                exposure=0.0,
                metadata={}
            )

    async def add_position(
        self, 
        symbol: str, 
        size: Decimal, 
        entry_price: Decimal
    ) -> bool:
        """Add new position with thread safety"""
        async with self._lock:
            try:
                if not isinstance(symbol, str):
                    raise PortfolioError("Symbol must be a string.")

                if size <= Decimal('0') or entry_price <= Decimal('0'):
                    raise PortfolioError("Size and entry price must be positive.")

                if len(self.positions) >= self.risk_limits['max_positions']:
                    self.logger.warning(f"Max positions limit reached: {self.risk_limits['max_positions']}")
                    return False

                position_value = size * entry_price
                total_value = self.calculate_portfolio_value()
                
                if total_value > Decimal('0') and (position_value / total_value) > self.risk_limits['max_position_size']:
                    self.logger.warning("Position size exceeds max position size limit.")
                    return False
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    direction="long"  # Assuming long; adjust as needed
                )
                
                await self._update_portfolio_metrics()
                return True
                
            except PortfolioError as e:
                self.logger.error(f"Failed to add position: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in add_position: {e}")
                return False

    async def update_position_price(
        self, 
        symbol: str, 
        current_price: Decimal
    ) -> None:
        """Update position with new price data"""
        async with self._lock:
            try:
                if not isinstance(symbol, str):
                    raise PortfolioError("Symbol must be a string.")

                if current_price <= Decimal('0'):
                    raise PortfolioError("Current price must be positive.")

                if symbol not in self.positions:
                    self.logger.warning(f"Attempted to update non-existent position: {symbol}")
                    return
                
                position = self.positions[symbol]
                old_price = position.current_price
                new_price = self.nh.to_decimal(current_price)
                
                # Update position metrics
                position.current_price = new_price
                position.unrealized_pnl = (
                    position.size * (new_price - position.entry_price)
                )
                position.last_update = datetime.utcnow()
                
                # Record update for analysis
                self._position_updates.append({
                    'symbol': symbol,
                    'timestamp': datetime.utcnow(),
                    'price_change': new_price - old_price,
                    'unrealized_pnl': position.unrealized_pnl
                })
                
                await self._update_portfolio_metrics()
                
            except PortfolioError as e:
                self.logger.error(f"Failed to update position: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in update_position_price: {e}")

    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio-wide metrics"""
        try:
            current_value = await self.get_total_value()
            self._last_value = current_value
            if current_value > self._high_water_mark:
                self._high_water_mark = current_value
                self.logger.info(f"New high water mark: {self._high_water_mark}")
        except Exception as e:
            self.logger.error(f"Failed to update portfolio metrics: {e}")

    def close_position(
        self, 
        symbol: str, 
        exit_price: Decimal
    ) -> Optional[Position]:
        """Close position and update realized PnL"""
        try:
            with self.lock:
                if symbol not in self.positions:
                    self.logger.warning(f"Attempted to close non-existent position: {symbol}")
                    return None
                    
                position = self.positions[symbol]
                position.close(exit_price)
                self.realized_pnl += position.unrealized_pnl
                self.balance += position.unrealized_pnl
                del self.positions[symbol]
                self.logger.info(f"Position closed for {symbol} at {exit_price}. Realized PnL: {position.unrealized_pnl}")
                return position
                
        except Exception as e:
            handle_error(e, "PortfolioManager.close_position", logger=self.logger)
            return None

    async def update_position(
        self, 
        symbol: str, 
        current_price: Decimal
    ) -> bool:
        """Update position with new price"""
        async with self._lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning(f"Attempted to update non-existent position: {symbol}")
                    return False
                    
                position = self.positions[symbol]
                position.update_price(current_price)
                self.logger.info(f"Position updated for {symbol}: Current Price = {current_price}")
                await self._update_portfolio_metrics()
                return True
                
            except Exception as e:
                handle_error(e, "PortfolioManager.update_position", logger=self.logger)
                return False 