from typing import Any, Dict
from decimal import Decimal
import asyncio
import logging

from utils.error_handler import handle_error

class RatchetManager:
    """Manages trailing stops and ratchet mechanisms for open trades"""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def initialize_trade(
        self, 
        trade_id: str, 
        entry_price: float, 
        take_profit: float, 
        stop_loss: float
    ) -> None:
        """Initialize a new trade for ratchet management"""
        async with self._lock:
            self.active_trades[trade_id] = {
                "entry_price": Decimal(str(entry_price)),
                "take_profit": Decimal(str(take_profit)),
                "stop_loss": Decimal(str(stop_loss)),
                "trailing_stop": Decimal(str(stop_loss)),
                "status": "active"
            }
            self.logger.info(f"Trade {trade_id} initialized for ratchet management.")

    async def finalize_trade(self, position: Any) -> None:
        """Finalize trade upon closure"""
        async with self._lock:
            trade_id = position.symbol  # Assuming symbol as trade ID; adjust as necessary
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
                self.logger.info(f"Trade {trade_id} finalized and removed from ratchet management.")

    async def update_trailing_stop(self, trade_id: str, current_price: Decimal) -> None:
        """Update trailing stop based on current price"""
        async with self._lock:
            try:
                if trade_id not in self.active_trades:
                    return
                
                trade = self.active_trades[trade_id]
                if trade["status"] != "active":
                    return
                
                if trade["take_profit"] and current_price >= trade["take_profit"]:
                    trade["status"] = "take_profit_reached"
                    self.logger.info(f"Trade {trade_id} reached take profit at {current_price}")
                    await self.ctx.exchange_interface.close_position(trade_id, current_price)
                
                elif current_price > trade["trailing_stop"]:
                    # Adjust the trailing stop upwards for long positions
                    profit = current_price - trade["entry_price"]
                    trail_percentage = Decimal('0.02')  # Example trail percentage
                    new_trailing_stop = current_price - (profit * trail_percentage)
                    
                    if new_trailing_stop > trade["trailing_stop"]:
                        trade["trailing_stop"] = new_trailing_stop
                        self.logger.info(f"Trade {trade_id} trailing stop updated to {new_trailing_stop}")
                
            except Exception as e:
                handle_error(e, f"RatchetManager.update_trailing_stop for {trade_id}", logger=self.logger) 