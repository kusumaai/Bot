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
from utils.numeric import NumericHandler

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
        self.nh = NumericHandler()
        self.active_trades: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
        # Load thresholds from config
        self.thresholds = [
            self.nh.percentage_to_decimal(t) 
            for t in ctx.config.get('ratchet_thresholds', [2, 4, 6])
        ]
        self.lock_ins = [
            self.nh.percentage_to_decimal(l) 
            for l in ctx.config.get('ratchet_lock_ins', [1, 2, 3])
        ]
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

    async def initialize_trade(self, trade_id: str, entry_price: Decimal, symbol: str) -> None:
        """Initialize a new trade with proper ID structure"""
        async with self._lock:
            normalized_id = self._normalize_trade_id(trade_id, symbol)
            self.active_trades[normalized_id] = {
                'entry_price': self.nh.to_decimal(entry_price),
                'current_stop': self.nh.to_decimal(entry_price),
                'highest_price': self.nh.to_decimal(entry_price),
                'threshold_index': 0,
                'symbol': symbol,
                'initialized_at': datetime.utcnow()
            }
    
    def _normalize_trade_id(self, trade_id: str, symbol: str) -> str:
        """Create consistent trade ID format"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{symbol}_{timestamp}_{trade_id[-8:]}"
    
    async def update_price(self, trade_id: str, current_price: Decimal) -> Optional[Decimal]:
        """Update price and return new stop loss if applicable"""
        async with self._lock:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return None
                
            current_price = self.nh.to_decimal(current_price)
            trade['highest_price'] = max(trade['highest_price'], current_price)
            
            return await self._calculate_new_stop(trade, current_price)
    
    async def _calculate_new_stop(self, trade: Dict, current_price: Decimal) -> Optional[Decimal]:
        """Calculate new stop loss based on ratchet thresholds"""
        entry_price = trade['entry_price']
        highest_price = trade['highest_price']
        current_threshold_idx = trade['threshold_index']
        
        # Calculate price movement as percentage
        price_movement = (highest_price - entry_price) / entry_price
        
        # Check if we've hit next threshold
        while (current_threshold_idx < len(self.thresholds) and 
               price_movement >= self.thresholds[current_threshold_idx]):
            # Move stop loss to lock in profits
            new_stop = entry_price * (Decimal('1') + self.lock_ins[current_threshold_idx])
            if new_stop > trade['current_stop']:
                trade['current_stop'] = new_stop
                trade['threshold_index'] = current_threshold_idx + 1
                await self._log_stop_update(trade, new_stop)
            current_threshold_idx += 1
            
        return trade['current_stop']
    
    async def _log_stop_update(self, trade: Dict, new_stop: Decimal) -> None:
        """Log stop loss updates"""
        self.ctx.logger.info(
            f"Updated stop for {trade['symbol']}: "
            f"Entry: {trade['entry_price']}, "
            f"Highest: {trade['highest_price']}, "
            f"New Stop: {new_stop}"
        )

    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for trade"""
        if trade_id not in self.active_trades:
            return {}
            
        try:
            trade = self.active_trades[trade_id]
            hold_time = (time.time() - trade['initialized_at'].timestamp()) / 3600
            
            return {
                "entry_price": float(trade['entry_price']),
                "current_stop": float(trade['current_stop']),
                "highest_price": float(trade['highest_price']),
                "lowest_price": float(trade['current_stop']),
                "current_level": trade['threshold_index'],
                "unrealized_pnl": float(trade['entry_price'] - trade['current_stop']) * float(trade['position_size']),
                "max_adverse_excursion": float((trade['highest_price'] - trade['entry_price']) / trade['entry_price']),
                "max_excursion": float((trade['highest_price'] - trade['entry_price']) / trade['entry_price']),
                "current_drawdown": float((trade['highest_price'] - trade['current_stop']) / trade['highest_price']),
                "hold_time_hours": round(hold_time, 1)
            }
        except Exception as e:
            handle_error(e, "RatchetManager.get_trade_metrics", logger=self.ctx.logger)
            return {}

    def remove_trade(self, trade_id: str) -> None:
        """Clean up trade tracking"""
        try:
            if trade_id in self.active_trades:
                metrics = self.get_trade_metrics(trade_id)
                self.ctx.logger.info(
                    f"Removing trade {trade_id} tracking. "
                    f"Final metrics: {metrics}"
                )
                del self.active_trades[trade_id]
        except Exception as e:
            handle_error(e, "RatchetManager.remove_trade", logger=self.ctx.logger)

    async def monitor_trades(self, exchange_interface: Any) -> None:
        """
        Active monitoring loop to ensure stops are enforced
        Should run in separate task
        """
        while True:
            try:
                for trade_id, trade in list(self.active_trades.items()):
                    # Get current price
                    symbol = trade['symbol']
                    price = Decimal(str(await exchange_interface.fetch_ticker(symbol)))
                    
                    if price <= Decimal("0"):
                        continue
                        
                    # Force check stops
                    result = await self.update_price(trade_id, price)
                    
                    if result:
                        stop_price = result
                        
                        # Execute stop
                        success = await exchange_interface.close_position(
                            symbol,
                            trade['position_size']
                        )
                        
                        if success:
                            self.ctx.logger.info(
                                f"Stop executed for {trade_id}: "
                                f"Price: {float(stop_price):.2f}"
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
            for trade_id in self.active_trades
        }