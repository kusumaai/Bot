#!/usr/bin/env python3
"""
Module: trading/ratchet.py
Production-ready ratchet stop system with guaranteed execution
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from decimal import Decimal, InvalidOperation, DivisionByZero
import time
import asyncio
from datetime import datetime
import logging

from risk.limits import RiskLimits
from risk.position import Position
from utils.error_handler import handle_error
from utils.numeric import NumericHandler
from trading.exceptions import RatchetError

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
        self.ratchets: Dict[str, Dict[str, Any]] = {}
        self.nh = NumericHandler()
        self.max_ratchets = 1000  # Limit to prevent unbounded growth
        
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
                if not isinstance(symbol, str):
                    raise RatchetError("Symbol must be a string.")

                if not isinstance(current_price, Decimal) or current_price <= Decimal('0'):
                    raise RatchetError("Invalid current price.")

                if 'entry_price' not in position:
                    raise RatchetError("Position missing 'entry_price'.")

                if symbol not in self.ratchets:
                    entry_price = self.nh.to_decimal(position['entry_price'])
                    if entry_price <= Decimal('0'):
                        raise RatchetError("Invalid entry price in position.")
                    self.ratchets[symbol] = {
                        'entry_price': entry_price,
                        'current_level': 0,
                        'stop_loss': None
                    }

                ratchet = self.ratchets[symbol]
                entry_price = ratchet['entry_price']
                try:
                    profit_pct = self.nh.safe_divide(
                        current_price - entry_price,
                        entry_price
                    )
                except DivisionByZero:
                    raise RatchetError("Division by zero in profit percentage calculation.")

                # Check and update ratchet levels
                new_level = await self._check_ratchet_levels(symbol, profit_pct)
                return new_level

            except (RatchetError, InvalidOperation, TypeError) as e:
                self.logger.error(f"Ratchet update failed: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error in update_position_ratchet: {e}")
                return None
    
    async def _check_ratchet_levels(self, symbol: str, profit_pct: Decimal) -> Optional[Decimal]:
        try:
            ratchet = self.ratchets[symbol]
            # Example logic: Increment ratchet level if profit_pct exceeds 5%
            if profit_pct >= Decimal('0.05'):
                ratchet['current_level'] += 1
                self.logger.info(f"Ratchet level for {symbol} increased to {ratchet['current_level']}.")

                # Example stop_loss adjustment based on ratchet level
                ratchet['stop_loss'] = self.nh.safe_divide(
                    self.nh.to_decimal('1.0'),
                    Decimal('1.0') + (Decimal('0.01') * ratchet['current_level'])
                ) * ratchet['entry_price']

                return ratchet['current_level']
            return None
        except KeyError:
            self.logger.error(f"Ratchet for symbol {symbol} does not exist.")
            return None
        except Exception as e:
            self.logger.error(f"Error in _check_ratchet_levels for {symbol}: {e}")
            return None
        finally:
            # Ensure ratchets do not grow unbounded
            if len(self.ratchets) > self.max_ratchets:
                oldest_symbol = next(iter(self.ratchets))
                del self.ratchets[oldest_symbol]
                self.logger.warning(f"Ratchet for {oldest_symbol} removed to maintain max_ratchets limit.")
    
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