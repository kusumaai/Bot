from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any
import asyncio
import logging

from utils.exceptions import PositionUpdateError

@dataclass
class Position:
    # Required fields (no defaults)
    symbol: str
    side: str
    entry_price: Decimal
    size: Decimal
    timestamp: int
    
    # Optional fields (with defaults)
    current_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    closed: bool = False
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._lock = asyncio.Lock()
        if self.entry_price <= Decimal('0') or self.size <= Decimal('0'):
            raise ValueError("Entry price and size must be positive.")

    async def update(self, current_price: Decimal) -> None:
        async with self._lock:
            try:
                if current_price <= Decimal('0'):
                    raise PositionUpdateError("Current price must be positive.")

                self.current_price = current_price
                self.unrealized_pnl = self.size * (self.current_price - self.entry_price) if self.side.lower() == 'buy' else self.size * (self.entry_price - self.current_price)
                
                # Additional update logic as needed

            except InvalidOperation as e:
                logging.getLogger(__name__).error(f"Invalid operation during position update: {e}")
                raise PositionUpdateError(f"Invalid operation: {e}") from e
            except Exception as e:
                logging.getLogger(__name__).error(f"Unexpected error during position update: {e}")
                raise PositionUpdateError(f"Unexpected error: {e}") from e

    def close_position(self, realized_pnl: Decimal):
        self.realized_pnl += realized_pnl
        self.closed = True 