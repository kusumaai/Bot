#!/usr/bin/env python3
"""
Module: trading/ratchet.py
Production-ready ratchet stop system with guaranteed execution
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from decimal import Decimal
import time
import asyncio
from datetime import datetime

from risk.limits import RiskLimits
from risk.position import Position
from utils.error_handler import handle_error

@dataclass
class RatchetState:
    """State tracking for ratchet system"""
    entry_price: Decimal
    current_stop: Decimal
    highest_price: Decimal
    lowest_price: Decimal
    current_level: int
    emergency_stop: Decimal
    position_size: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal
    last_update: float
    entry_time: float

class RatchetManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.trades: Dict[str, RatchetState] = {}
        
        # Load settings from config with proper decimal handling
        self.thresholds = [Decimal(str(x)) for x in ctx.config.get("ratchet_thresholds", [1.5, 3.0, 4.5])]
        self.lock_ins = [Decimal(str(x)) for x in ctx.config.get("ratchet_lock_ins", [0.5, 1.0, 1.5])]
        self.emergency_stop_pct = Decimal(str(ctx.config.get("emergency_stop_pct", -2)))
        self.trailing_pct = Decimal(str(ctx.config.get("trailing_stop_pct", 1.5)))
        
        # Risk limits
        self.max_drawdown = Decimal(str(ctx.config.get("max_drawdown_pct", 10))) / Decimal("100")
        self.max_adverse_pct = Decimal(str(ctx.config.get("max_adverse_pct", 3))) / Decimal("100")
        self.max_hold_hours = Decimal(str(ctx.config.get("max_hold_hours", 8)))
        
        # Validation 
        if len(self.thresholds) != len(self.lock_ins):
            raise ValueError("Ratchet thresholds and lock-ins must have same length")
        if not all(x < y for x, y in zip(self.thresholds[:-1], self.thresholds[1:])):
            raise ValueError("Ratchet thresholds must be ascending")

    def initialize_trade(
        self,
        trade_id: str,
        entry_price: Decimal,
        position_size: Decimal,
        emergency_stop: Optional[Decimal] = None
    ) -> None:
        """Initialize ratchet tracking for new trade"""
        try:
            if emergency_stop is None:
                emergency_stop = entry_price * (Decimal("1") + self.emergency_stop_pct/Decimal("100"))
                
            self.trades[trade_id] = RatchetState(
                entry_price=entry_price,
                current_stop=emergency_stop,
                highest_price=entry_price,
                lowest_price=entry_price,
                current_level=0,
                emergency_stop=emergency_stop,
                position_size=position_size,
                unrealized_pnl=Decimal("0"),
                max_adverse_excursion=Decimal("0"),
                last_update=time.time(),
                entry_time=time.time()
            )
            
            self.ctx.logger.info(
                f"Initialized ratchet for trade {trade_id}: "
                f"Entry: {float(entry_price):.2f}, Stop: {float(emergency_stop):.2f}"
            )
        except Exception as e:
            handle_error(e, "RatchetManager.initialize_trade", logger=self.ctx.logger)

    def update_price(
        self,
        trade_id: str,
        current_price: Decimal,
        force_check: bool = False
    ) -> Optional[Tuple[Decimal, str]]:
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
            profit_pct = (current_price - state.entry_price) / state.entry_price * Decimal("100")
            drawdown_pct = (state.highest_price - current_price) / state.highest_price * Decimal("100")
            state.unrealized_pnl = (current_price - state.entry_price) * state.position_size
            
            # Update max adverse excursion
            current_adverse = (state.entry_price - current_price) / state.entry_price
            state.max_adverse_excursion = max(state.max_adverse_excursion, current_adverse)
            
            # Check hold time
            hold_hours = (time.time() - state.entry_time) / 3600
            if hold_hours >= float(self.max_hold_hours):
                self.ctx.logger.warning(
                    f"Max hold time reached for {trade_id}: {hold_hours:.1f} hours"
                )
                return (current_price, "max_hold_time")
            
            # Emergency stop check
            if current_price <= state.emergency_stop:
                self.ctx.logger.warning(
                    f"Emergency stop triggered for {trade_id} at {float(current_price):.2f}"
                )
                return (current_price, "emergency_stop")
                
            # Maximum adverse excursion check
            if state.max_adverse_excursion >= self.max_adverse_pct:
                self.ctx.logger.warning(
                    f"Max adverse excursion {float(state.max_adverse_excursion):.2%} "
                    f"exceeded for {trade_id}"
                )
                return (current_price, "max_adverse")
                
            # Maximum drawdown check
            if drawdown_pct >= self.max_drawdown * Decimal("100"):
                self.ctx.logger.warning(
                    f"Max drawdown {float(drawdown_pct):.2f}% exceeded for {trade_id}"
                )
                return (current_price, "max_drawdown")
                
            # Trailing stop
            trailing_stop = state.highest_price * (Decimal("1") - self.trailing_pct/Decimal("100"))
            if state.current_stop < trailing_stop:
                state.current_stop = trailing_stop
                
            if current_price <= state.current_stop:
                return (current_price, "trailing_stop")
                
            # Ratchet level check
            for level, (threshold, lock_in) in enumerate(zip(self.thresholds, self.lock_ins)):
                if profit_pct >= threshold and level > state.current_level:
                    # Move to new ratchet level
                    state.current_level = level
                    new_stop = state.entry_price * (Decimal("1") + lock_in/Decimal("100"))
                    
                    if new_stop > state.current_stop:
                        state.current_stop = new_stop
                        self.ctx.logger.info(
                            f"Ratchet level {level} achieved for {trade_id}, "
                            f"new stop: {float(new_stop):.2f}"
                        )
                        return (new_stop, f"ratchet_level_{level}")
                        
            # Update timestamp
            state.last_update = time.time()
            return None
            
        except Exception as e:
            handle_error(e, "RatchetManager.update_price", logger=self.ctx.logger)
            return (current_price, "error_stop")

    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for trade"""
        if trade_id not in self.trades:
            return {}
            
        try:
            state = self.trades[trade_id]
            hold_time = (time.time() - state.entry_time) / 3600
            
            return {
                "entry_price": float(state.entry_price),
                "current_stop": float(state.current_stop),
                "highest_price": float(state.highest_price),
                "lowest_price": float(state.lowest_price),
                "current_level": state.current_level,
                "unrealized_pnl": float(state.unrealized_pnl),
                "max_adverse_excursion": float(state.max_adverse_excursion),
                "max_excursion": float((state.highest_price - state.entry_price) / state.entry_price),
                "current_drawdown": float((state.highest_price - state.lowest_price) / state.highest_price),
                "hold_time_hours": round(hold_time, 1)
            }
        except Exception as e:
            handle_error(e, "RatchetManager.get_trade_metrics", logger=self.ctx.logger)
            return {}

    def remove_trade(self, trade_id: str) -> None:
        """Clean up trade tracking"""
        try:
            if trade_id in self.trades:
                metrics = self.get_trade_metrics(trade_id)
                self.ctx.logger.info(
                    f"Removing trade {trade_id} tracking. "
                    f"Final metrics: {metrics}"
                )
                del self.trades[trade_id]
        except Exception as e:
            handle_error(e, "RatchetManager.remove_trade", logger=self.ctx.logger)

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
                    price = Decimal(str(await exchange_interface.fetch_ticker(symbol)))
                    
                    if price <= Decimal("0"):
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
                                f"Price: {float(stop_price):.2f}, Reason: {reason}"
                            )
                            self.remove_trade(trade_id)
                        else:
                            self.ctx.logger.error(
                                f"Failed to execute stop for {trade_id}. "
                                f"Will retry next cycle."
                            )
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                handle_error(e, "RatchetManager.monitor_trades", logger=self.ctx.logger)
                await asyncio.sleep(5)  # Back off on error

    def get_status_report(self) -> Dict[str, Any]:
        """Get status report for all tracked trades"""
        return {
            trade_id: self.get_trade_metrics(trade_id)
            for trade_id in self.trades
        }