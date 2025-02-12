import asyncio
import time
from typing import Any, Dict, Optional
from decimal import Decimal
from trading.position import Position
from trading.numeric_handler import NumericHandler
from trading.exceptions import PositionError

class PositionManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.positions: Dict[str, Position] = {}
        self.nh = NumericHandler()

    async def open_position(self, order: Dict[str, Any]) -> Optional[Position]:
        async with self._lock:
            try:
                symbol = order['symbol']
                if symbol in self.positions:
                    raise PositionError(f"Position already exists for {symbol}")
                
                position = Position(
                    symbol=symbol,
                    side=order['side'],
                    entry_price=Decimal(str(order['price'])),
                    size=Decimal(str(order['amount'])),
                    timestamp=int(time.time() * 1000)
                )
                
                self.positions[symbol] = position
                await self._store_position(position)
                return position
                
            except Exception as e:
                self.logger.error(f"Failed to open position: {e}")
                return None

    async def close_position(self, symbol: str) -> Optional[Position]:
        async with self._lock:
            try:
                position = self.positions.pop(symbol, None)
                if position:
                    await self._update_position(position, closed=True)
                return position
            except Exception as e:
                self.logger.error(f"Failed to close position: {e}")
                return None 