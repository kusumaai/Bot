import asyncio
import time
from typing import Any, Dict, Optional
from decimal import Decimal, InvalidOperation
from utils.numeric_handler import NumericHandler
from utils.exceptions import PositionError, InvalidOrderError, DatabaseError
from utils.error_handler import handle_error_async
from signals.market_state import prepare_market_state
import pandas as pd
from dataclasses import dataclass, field
from bot_types.base_types import Position
from bot_types.base_types import MarketState

@dataclass
class Position:
    # Required fields (no defaults)
    symbol: str
    side: str
    entry_price: Decimal
    size: Decimal
    timestamp: int
    
    # Optional fields (with defaults)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
            except (PositionError, InvalidOrderError, DatabaseError) as e:
                self.logger.error(f"Failed to open position: {e}")
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
            await self.ctx.db_queries.insert_trade({
                "id": position.symbol,
                "symbol": position.symbol,
                "entry_price": float(position.entry_price),
                "size": float(position.size),
                "side": position.side,
                "strategy": "manual",
                "metadata": {}
            })
        except DatabaseError as e:
            await handle_error_async(e, "PositionManager._store_position", self.logger)
            raise

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

    async def update_position(self, symbol: str, market_data: pd.DataFrame) -> None:
        try:
            market_state = prepare_market_state(market_data)
            position = self.positions.get(symbol)
            if position:
                await position.update_market_state(market_state)
        except Exception as e:
            await handle_error_async(e, "PositionManager.update_position", self.logger) 