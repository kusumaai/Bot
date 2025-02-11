#!/usr/bin/env python3
"""
Module: risk/position.py
Core position tracking with risk metrics
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
import time
from utils.error_handler import handle_error

@dataclass
class Position:
    """Trading position with standardized decimal handling and risk tracking"""
    symbol: str
    direction: str
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal = Decimal(0)
    max_favorable_excursion: Decimal = Decimal(0)
    trailing_stop: Optional[Decimal] = None
    last_update_time: float = time.time()
    
    def update(self, current_price: Decimal) -> None:
        """Update position with new price and track metrics"""
        try:
            self.current_price = current_price
            price_diff = current_price - self.entry_price
            multiplier = Decimal(1) if self.direction == "long" else Decimal(-1)
            self.unrealized_pnl = price_diff * self.size * multiplier
            
            # Update excursion tracking
            pnl_pct = self.unrealized_pnl / (self.entry_price * self.size)
            self.max_adverse_excursion = min(self.max_adverse_excursion, pnl_pct)
            self.max_favorable_excursion = max(self.max_favorable_excursion, pnl_pct)
            
            # Update trailing stop if set
            if self.trailing_stop is not None:
                if self.direction == "long" and current_price > self.stop_loss + self.trailing_stop:
                    self.stop_loss = current_price - self.trailing_stop
                elif self.direction == "short" and current_price < self.stop_loss - self.trailing_stop:
                    self.stop_loss = current_price + self.trailing_stop
                    
            self.last_update_time = time.time()
            
        except Exception as e:
            handle_error(e, f"Position.update: {self.symbol}", logger=None)

    def should_close(self) -> Tuple[bool, Optional[str]]:
        """Check if position should be closed"""
        try:
            # Check stop loss
            if self.direction == "long" and self.current_price <= self.stop_loss:
                return True, "stop_loss"
            if self.direction == "short" and self.current_price >= self.stop_loss:
                return True, "stop_loss"
                
            # Check take profit
            if self.direction == "long" and self.current_price >= self.take_profit:
                return True, "take_profit"
            if self.direction == "short" and self.current_price <= self.take_profit:
                return True, "take_profit"
                
            return False, None
            
        except Exception as e:
            handle_error(e, f"Position.should_close: {self.symbol}", logger=None)
            return False, None

    def get_duration(self) -> float:
        """Get position duration in seconds"""
        return time.time() - self.entry_time

    def get_return(self) -> Decimal:
        """Get position return percentage"""
        try:
            return self.unrealized_pnl / (self.entry_price * self.size)
        except Exception:
            return Decimal(0) 