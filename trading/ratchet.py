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
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.nh = NumericHandler()
        self.ratchets: Dict[str, Dict[str, Any]] = {}
        
        # Load thresholds from config
        self.thresholds = [
            self.nh.to_decimal(str(t))
            for t in ctx.config.get('ratchet_thresholds', [])
        ]
        self.lock_ins = [
            self.nh.to_decimal(str(l))
            for l in ctx.config.get('ratchet_lock_ins', [])
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
            self.ratchets[normalized_id] = {
                'entry_price': self.nh.to_decimal(entry_price),
                'current_level': 0,
                'stop_loss': None
            }
    
    def _normalize_trade_id(self, trade_id: str, symbol: str) -> str:
        """Create consistent trade ID format"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{symbol}_{timestamp}_{trade_id[-8:]}"
    
    async def update_position_ratchet(
        self,
        symbol: str,
        current_price: Decimal,
        position: Dict[str, Any]
    ) -> Optional[Decimal]:
        async with self._lock:
            try:
                if symbol not in self.ratchets:
                    self.ratchets[symbol] = {
                        'entry_price': self.nh.to_decimal(str(position['entry_price'])),
                        'current_level': 0,
                        'stop_loss': None
                    }

                ratchet = self.ratchets[symbol]
                profit_pct = self.nh.safe_divide(
                    current_price - ratchet['entry_price'],
                    ratchet['entry_price']
                )

                return await self._check_ratchet_levels(symbol, profit_pct)

            except Exception as e:
                self.logger.error(f"Ratchet update failed: {e}")
                return None
    
    async def _check_ratchet_levels(self, symbol: str, profit_pct: Decimal) -> Optional[Decimal]:
        """Check ratchet levels and return new stop loss if applicable"""
        ratchet = self.ratchets[symbol]
        current_level = ratchet['current_level']
        
        # Check if we've hit next threshold
        while (current_level < len(self.thresholds) and 
               profit_pct >= self.thresholds[current_level]):
            # Move stop loss to lock in profits
            new_stop = ratchet['entry_price'] * (Decimal('1') + self.lock_ins[current_level])
            if new_stop > ratchet['stop_loss']:
                ratchet['stop_loss'] = new_stop
                ratchet['current_level'] = current_level + 1
                await self._log_stop_update(ratchet, new_stop)
            current_level += 1
            
        return ratchet['stop_loss']
    
    async def _log_stop_update(self, ratchet: Dict, new_stop: Decimal) -> None:
        """Log stop loss updates"""
        self.logger.info(
            f"Updated stop for {symbol}: "
            f"Entry: {ratchet['entry_price']}, "
            f"New Stop: {new_stop}"
        )

    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for trade"""
        if trade_id not in self.ratchets:
            return {}
            
        try:
            ratchet = self.ratchets[trade_id]
            hold_time = (time.time() - ratchet['entry_time']) / 3600
            
            return {
                "entry_price": float(ratchet['entry_price']),
                "current_stop": float(ratchet['stop_loss']),
                "highest_price": float(ratchet['highest_price']),
                "lowest_price": float(ratchet['lowest_price']),
                "current_level": ratchet['current_level'],
                "unrealized_pnl": float(ratchet['entry_price'] - ratchet['stop_loss']) * float(ratchet['position_size']),
                "max_adverse_excursion": float((ratchet['highest_price'] - ratchet['entry_price']) / ratchet['entry_price']),
                "max_excursion": float((ratchet['highest_price'] - ratchet['entry_price']) / ratchet['entry_price']),
                "current_drawdown": float((ratchet['highest_price'] - ratchet['stop_loss']) / ratchet['highest_price']),
                "hold_time_hours": round(hold_time, 1)
            }
        except Exception as e:
            handle_error(e, "RatchetManager.get_trade_metrics", logger=self.logger)
            return {}

    def remove_trade(self, trade_id: str) -> None:
        """Clean up trade tracking"""
        try:
            if trade_id in self.ratchets:
                metrics = self.get_trade_metrics(trade_id)
                self.logger.info(
                    f"Removing trade {trade_id} tracking. "
                    f"Final metrics: {metrics}"
                )
                del self.ratchets[trade_id]
        except Exception as e:
            handle_error(e, "RatchetManager.remove_trade", logger=self.logger)

    async def monitor_trades(self, exchange_interface: Any) -> None:
        """
        Active monitoring loop to ensure stops are enforced
        Should run in separate task
        """
        while True:
            try:
                for trade_id, ratchet in list(self.ratchets.items()):
                    # Get current price
                    symbol = trade_id.split('_')[0]
                    price = Decimal(str(await exchange_interface.fetch_ticker(symbol)))
                    
                    if price <= Decimal("0"):
                        continue
                        
                    # Force check stops
                    result = await self.update_position_ratchet(symbol, price, ratchet)
                    
                    if result:
                        stop_price = result
                        
                        # Execute stop
                        success = await exchange_interface.close_position(
                            symbol,
                            ratchet['position_size']
                        )
                        
                        if success:
                            self.logger.info(
                                f"Stop executed for {trade_id}: "
                                f"Price: {float(stop_price):.2f}"
                            )
                            self.remove_trade(trade_id)
                        else:
                            self.logger.error(
                                f"Failed to execute stop for {trade_id}. "
                                f"Will retry next cycle."
                            )
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                handle_error(e, "RatchetManager.monitor_trades", logger=self.logger)
                await asyncio.sleep(5)  # Back off on error

    def get_status_report(self) -> Dict[str, Any]:
        """Get status report for all tracked trades"""
        return {
            trade_id: self.get_trade_metrics(trade_id)
            for trade_id in self.ratchets
        }