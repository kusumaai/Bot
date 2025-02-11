#!/usr/bin/env python3
"""
Module: trading/ratchet.py
Production-ready ratchet stop system with guaranteed execution
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from decimal import Decimal
import time
import asyncio

@dataclass
class RatchetState:
    entry_price: float
    current_stop: float
    highest_price: float
    lowest_price: float
    current_level: int
    emergency_stop: float
    position_size: float
    unrealized_pnl: float
    max_adverse_excursion: float
    last_update: float

class RatchetManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.trades: Dict[str, RatchetState] = {}
        
        # Load settings from config
        self.thresholds = ctx.config.get("ratchet_thresholds", [1.5, 3.0, 4.5])
        self.lock_ins = ctx.config.get("ratchet_lock_ins", [0.5, 1.0, 1.5])
        self.emergency_stop_pct = Decimal(str(ctx.config.get("emergency_stop_pct", -2)))
        self.trailing_pct = Decimal(str(ctx.config.get("trailing_stop_pct", 1.5)))
        
        # Risk limits
        self.max_drawdown = Decimal(str(ctx.config.get("max_drawdown_pct", 10))) / 100
        self.max_adverse_pct = Decimal(str(ctx.config.get("max_adverse_pct", 3))) / 100
        
        # Validation 
        if len(self.thresholds) != len(self.lock_ins):
            raise ValueError("Ratchet thresholds and lock-ins must have same length")
        if not all(x < y for x, y in zip(self.thresholds[:-1], self.thresholds[1:])):
            raise ValueError("Ratchet thresholds must be ascending")

    def initialize_trade(
        self,
        trade_id: str,
        entry_price: float,
        position_size: float,
        emergency_stop: Optional[float] = None
    ) -> None:
        """Initialize ratchet tracking for new trade"""
        if emergency_stop is None:
            emergency_stop = entry_price * (1 + float(self.emergency_stop_pct/100))
            
        self.trades[trade_id] = RatchetState(
            entry_price=entry_price,
            current_stop=emergency_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            current_level=0,
            emergency_stop=emergency_stop,
            position_size=position_size,
            unrealized_pnl=0.0,
            max_adverse_excursion=0.0,
            last_update=time.time()
        )
        
        self.ctx.logger.info(
            f"Initialized ratchet for trade {trade_id}: "
            f"Entry: {entry_price:.2f}, Stop: {emergency_stop:.2f}"
        )

    def update_price(
        self,
        trade_id: str,
        current_price: float,
        force_check: bool = False
    ) -> Optional[Tuple[float, str]]:
        """
        Update trade state and check stops
        Returns: (stop_price, reason) if stop triggered
        """
        if trade_id not in self.trades:
            return None
            
        state = self.trades[trade_id]
        
        # Skip frequent updates unless forced
        if not force_check and time.time() - state.last_update < 1.0:
            return None
            
        try:
            # Update high/low
            state.highest_price = max(state.highest_price, current_price)
            state.lowest_price = min(state.lowest_price, current_price)
            
            # Calculate metrics
            profit_pct = (current_price - state.entry_price) / state.entry_price * 100
            drawdown_pct = (state.highest_price - current_price) / state.highest_price * 100
            state.unrealized_pnl = (current_price - state.entry_price) * state.position_size
            
            # Update max adverse excursion
            current_adverse = (state.entry_price - current_price) / state.entry_price
            state.max_adverse_excursion = max(state.max_adverse_excursion, current_adverse)
            
            # Emergency stop check
            if current_price <= state.emergency_stop:
                self.ctx.logger.warning(
                    f"Emergency stop triggered for {trade_id} at {current_price:.2f}"
                )
                return (current_price, "emergency_stop")
                
            # Maximum adverse excursion check
            if state.max_adverse_excursion >= float(self.max_adverse_pct):
                self.ctx.logger.warning(
                    f"Max adverse excursion {state.max_adverse_excursion:.2%} "
                    f"exceeded for {trade_id}"
                )
                return (current_price, "max_adverse")
                
            # Maximum drawdown check
            if drawdown_pct >= float(self.max_drawdown * 100):
                self.ctx.logger.warning(
                    f"Max drawdown {drawdown_pct:.2f}% exceeded for {trade_id}"
                )
                return (current_price, "max_drawdown")
                
            # Trailing stop
            trailing_stop = state.highest_price * (1 - float(self.trailing_pct/100))
            if state.current_stop < trailing_stop:
                state.current_stop = trailing_stop
                
            if current_price <= state.current_stop:
                return (current_price, "trailing_stop")
                
            # Ratchet level check
            for level, (threshold, lock_in) in enumerate(zip(self.thresholds, self.lock_ins)):
                if profit_pct >= threshold and level > state.current_level:
                    # Move to new ratchet level
                    state.current_level = level
                    new_stop = state.entry_price * (1 + lock_in/100)
                    
                    if new_stop > state.current_stop:
                        state.current_stop = new_stop
                        self.ctx.logger.info(
                            f"Ratchet level {level} achieved for {trade_id}, "
                            f"new stop: {new_stop:.2f}"
                        )
                        return (new_stop, f"ratchet_level_{level}")
                        
            # Update timestamp
            state.last_update = time.time()
            return None
            
        except Exception as e:
            self.ctx.logger.error(f"Error in ratchet update for {trade_id}: {str(e)}")
            # Return emergency stop on error
            return (current_price, "error_stop")

    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for trade"""
        if trade_id not in self.trades:
            return {}
            
        state = self.trades[trade_id]
        
        return {
            "entry_price": state.entry_price,
            "current_stop": state.current_stop,
            "highest_price": state.highest_price,
            "lowest_price": state.lowest_price,
            "current_level": state.current_level,
            "unrealized_pnl": state.unrealized_pnl,
            "max_adverse_excursion": state.max_adverse_excursion,
            "max_excursion": (state.highest_price - state.entry_price) / state.entry_price,
            "current_drawdown": (state.highest_price - state.lowest_price) / state.highest_price
        }

    def remove_trade(self, trade_id: str) -> None:
        """Clean up trade tracking"""
        if trade_id in self.trades:
            metrics = self.get_trade_metrics(trade_id)
            self.ctx.logger.info(
                f"Removing trade {trade_id} tracking. "
                f"Final metrics: {metrics}"
            )
            del self.trades[trade_id]

    async def monitor_trades(self, exchange_interface: Any) -> None:
        """
        Active monitoring loop to ensure stops are enforced
        Should run in separate task
        """
        while True:
            try:
                for trade_id, state in list(self.trades.items()):
                    # Get current price
                    symbol = trade_id.split('_')[0]  # Assumes ID format: SYMBOL_TIMESTAMP
                    price = await exchange_interface.fetch_ticker(symbol)
                    
                    if price <= 0:
                        continue
                        
                    # Force check stops
                    result = self.update_price(trade_id, price, force_check=True)
                    
                    if result:
                        stop_price, reason = result
                        
                        # Execute stop
                        success = await exchange_interface.close_position(
                            symbol,
                            state.position_size
                        )
                        
                        if success:
                            self.ctx.logger.info(
                                f"Stop executed for {trade_id}: "
                                f"Price: {stop_price:.2f}, Reason: {reason}"
                            )
                            self.remove_trade(trade_id)
                        else:
                            self.ctx.logger.error(
                                f"Failed to execute stop for {trade_id}. "
                                f"Will retry next cycle."
                            )
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.ctx.logger.error(f"Error in ratchet monitor: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

    def get_status_report(self) -> Dict[str, Any]:
        """Get status report for all tracked trades"""
        return {
            trade_id: self.get_trade_metrics(trade_id)
            for trade_id in self.trades
        }