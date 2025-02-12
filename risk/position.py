#!/usr/bin/env python3
"""
Module: risk/position.py
Core position tracking with risk metrics
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, Tuple
import time
from utils.error_handler import handle_error
from trading.exceptions import PositionUpdateError
import asyncio
import logging
from datetime import datetime

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
    closed: bool = False
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None

    def __post_init__(self):
        if self.entry_price <= Decimal('0') or self.size <= Decimal('0'):
            raise ValueError("Entry price and size must be positive.")

    async def update(self, current_price: Decimal) -> None:
        """Update position with new price and track metrics"""
        async with asyncio.Lock():
            try:
                if current_price <= Decimal('0'):
                    raise PositionUpdateError("Current price must be positive.")

                old_price = self.current_price
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
                
            except InvalidOperation as e:
                logging.getLogger(__name__).error(f"Invalid operation during position update: {e}")
                raise PositionUpdateError(f"Invalid operation: {e}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Unexpected error during position update: {e}")
                raise PositionUpdateError(f"Unexpected error: {e}")

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

    def update_price(self, new_price: Decimal) -> None:
        """Update the current price and recalculate unrealized PnL."""
        self.current_price = new_price
        if self.direction == "long":
            self.unrealized_pnl = self.size * (self.current_price - self.entry_price)
        elif self.direction == "short":
            self.unrealized_pnl = self.size * (self.entry_price - self.current_price)
        self.last_update_time = time.time()

    def close(self, exit_price: Decimal) -> None:
        """Close the position by setting exit price and updating PnL."""
        self.exit_price = exit_price
        if self.direction == "long":
            self.unrealized_pnl = self.size * (self.exit_price - self.entry_price)
        elif self.direction == "short":
            self.unrealized_pnl = self.size * (self.entry_price - self.exit_price)
        self.closed = True
        self.exit_time = datetime.utcnow() 