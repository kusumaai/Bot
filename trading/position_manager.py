import asyncio
import time
from typing import Any, Dict, Optional
from decimal import Decimal, InvalidOperation
from trading.position import Position
from trading.numeric_handler import NumericHandler
from trading.exceptions import PositionError, InvalidOrderError
from database.queries import DatabaseError

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
                symbol = order.get('symbol')
                side = order.get('side')
                price = order.get('price')
                amount = order.get('amount')

                if not all([symbol, side, price, amount]):
                    raise InvalidOrderError("Missing required order parameters.")

                if not isinstance(symbol, str) or not isinstance(side, str):
                    raise InvalidOrderError("Invalid types for 'symbol' or 'side'.")

                try:
                    entry_price = Decimal(str(price))
                    size = Decimal(str(amount))
                except (InvalidOperation, TypeError):
                    raise InvalidOrderError("Invalid numeric values for 'price' or 'amount'.")

                if symbol in self.positions:
                    raise PositionError(f"Position already exists for {symbol}")
                
                position = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    size=size,
                    timestamp=int(time.time() * 1000)
                )
                
                self.positions[symbol] = position
                await self._store_position(position)
                return position
            except InvalidOrderError as e:
                self.logger.error(f"Failed to open position: {e}")
                return None
            except PositionError as e:
                self.logger.error(f"Position error: {e}")
                return None
            except DatabaseError as e:
                self.logger.error(f"Database error while opening position: {e}")
                self.positions.pop(symbol, None)
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error in open_position: {e}")
                return None

    async def close_position(self, symbol: str, exit_price: Decimal) -> Optional[Position]:
        async with self._lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning(f"No existing position for symbol: {symbol}")
                    return None

                position = self.positions[symbol]
                position.update(exit_price)
                await self._update_position(position, closed=True)
                del self.positions[symbol]
                return position
            except Exception as e:
                self.logger.error(f"Unexpected error in close_position: {e}")
                return None

    async def _store_position(self, position: Position) -> None:
        try:
            await self.ctx.database.queries.store_trade({
                'id': str(position.timestamp),  # Assuming timestamp as unique ID
                'symbol': position.symbol,
                'entry_price': str(position.entry_price),
                'size': str(position.size),
                'side': position.side,
                'strategy': 'PositionManager',
                'metadata': {}
            })
        except DatabaseError as e:
            self.logger.error(f"Failed to store position: {e}")
            async with self._lock:
                self.positions.pop(position.symbol, None)

    async def _update_position(self, position: Position, closed: bool = False) -> None:
        try:
            status = 'CLOSED' if closed else 'OPEN'
            await self.ctx.database.queries.update_position_status(
                position_id=str(position.timestamp),  # Assuming timestamp is used as ID
                status=status,
                metadata={'exit_price': str(position.current_price)} if closed else None
            )
        except DatabaseError as e:
            self.logger.error(f"Failed to update position status: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in _update_position: {e}") 