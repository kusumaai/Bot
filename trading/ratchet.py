#!/usr/bin/env python3
"""
Module: trading/ratchet.py
Implements a dynamic stop-loss ratcheting system
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class RatchetState:
    """Tracks the state of a trade's ratchet system"""
    entry_price: float
    current_stop: float
    highest_price: float
    lowest_price: float
    current_level: int = 0  # Current ratchet level achieved

class RatchetManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.trades_state: Dict[str, RatchetState] = {}
        
        # Load config
        self.thresholds = ctx.config.get("ratchet_thresholds", [2, 4, 6])
        self.lock_ins = ctx.config.get("ratchet_lock_ins", [1, 2, 3])
        self.emergency_stop = ctx.config.get("emergency_stop_pct", -3)
        
        # Validate config
        if len(self.thresholds) != len(self.lock_ins):
            raise ValueError("Ratchet thresholds and lock-ins must have same length")
        if not all(x < y for x, y in zip(self.thresholds[:-1], self.thresholds[1:])):
            raise ValueError("Ratchet thresholds must be in ascending order")

    def initialize_trade(self, trade_id: str, entry_price: float) -> None:
        """Initialize ratchet tracking for a new trade"""
        self.trades_state[trade_id] = RatchetState(
            entry_price=entry_price,
            current_stop=entry_price * (1 + self.emergency_stop / 100),
            highest_price=entry_price,
            lowest_price=entry_price
        )

    def update_price(self, trade_id: str, current_price: float) -> Optional[Tuple[float, str]]:
        """
        Update trade state with new price and return new stop if triggered
        Returns: (new_stop_price, reason) or None if no stop change
        """
        if trade_id not in self.trades_state:
            return None
            
        state = self.trades_state[trade_id]
        
        # Update high/low
        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)
        
        # Calculate current profit percentage
        profit_pct = (current_price - state.entry_price) / state.entry_price * 100
        
        # Check if we hit emergency stop
        if profit_pct <= self.emergency_stop:
            return (current_price, "emergency_stop")
            
        # Find appropriate ratchet level
        new_level = None
        for i, threshold in enumerate(self.thresholds):
            if profit_pct >= threshold and i > state.current_level:
                new_level = i
                
        if new_level is not None:
            state.current_level = new_level
            new_stop = state.entry_price * (1 + self.lock_ins[new_level] / 100)
            if new_stop > state.current_stop:
                state.current_stop = new_stop
                return (new_stop, f"ratchet_level_{new_level}")
                
        return None

    def get_current_stop(self, trade_id: str) -> Optional[float]:
        """Get current stop price for a trade"""
        state = self.trades_state.get(trade_id)
        return state.current_stop if state else None

    def remove_trade(self, trade_id: str) -> None:
        """Clean up when trade is closed"""
        self.trades_state.pop(trade_id, None)

    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for a trade"""
        state = self.trades_state.get(trade_id)
        if not state:
            return {}
            
        return {
            "entry_price": state.entry_price,
            "current_stop": state.current_stop,
            "highest_price": state.highest_price,
            "lowest_price": state.lowest_price,
            "current_level": state.current_level,
            "max_profit_pct": (state.highest_price - state.entry_price) / state.entry_price * 100,
            "max_drawdown_pct": (state.lowest_price - state.highest_price) / state.highest_price * 100
        }

def integrate_with_position_manager(position_manager: Any) -> None:
    """
    Integrate ratchet system with position manager
    Example usage with bot.py
    """
    original_update = position_manager.update_trade_stop
    
    def new_update_stop(trade_id: str, new_stop: float, ctx: Any) -> None:
        """Enhanced stop update with ratchet system"""
        if not hasattr(ctx, 'ratchet_manager'):
            ctx.ratchet_manager = RatchetManager(ctx)
            
        # Update through ratchet system
        result = ctx.ratchet_manager.update_price(trade_id, new_stop)
        if result:
            new_stop, reason = result
            ctx.logger.info(f"Ratchet system triggered: {reason} for trade {trade_id}")
            original_update(trade_id, new_stop, ctx)
            
    position_manager.update_trade_stop = new_update_stop

if __name__ == "__main__":
    # Example usage and testing
    class DummyContext:
        def __init__(self):
            self.config = {
                "ratchet_thresholds": [2, 4, 6],
                "ratchet_lock_ins": [1, 2, 3],
                "emergency_stop_pct": -3
            }
    
    ctx = DummyContext()
    manager = RatchetManager(ctx)
    
    # Simulate a trade
    trade_id = "test_trade"
    entry_price = 100.0
    
    print("Initializing trade...")
    manager.initialize_trade(trade_id, entry_price)
    
    # Test price movements
    test_prices = [101, 102, 103, 104, 105, 106, 105, 104, 103]
    
    for price in test_prices:
        result = manager.update_price(trade_id, price)
        if result:
            new_stop, reason = result
            print(f"Price: {price}, New Stop: {new_stop:.2f}, Reason: {reason}")
            
        metrics = manager.get_trade_metrics(trade_id)
        print(f"Current metrics: {metrics}")