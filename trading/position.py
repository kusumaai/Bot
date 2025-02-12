from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional
import asyncio
import logging

from utils.exceptions import PositionUpdateError

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: Decimal
    size: Decimal
    timestamp: int
    current_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    closed: bool = False

    def __post_init__(self):
        if self.entry_price <= Decimal('0') or self.size <= Decimal('0'):
            raise ValueError("Entry price and size must be positive.")

    async def update(self, current_price: Decimal) -> None:
        async with asyncio.Lock():
            try:
                if current_price <= Decimal('0'):
                    raise PositionUpdateError("Current price must be positive.")

                old_price = self.current_price
                self.current_price = current_price
                self.unrealized_pnl = self.size * (self.current_price - self.entry_price)
                # Additional update logic as needed

            except InvalidOperation as e:
                logging.getLogger(__name__).error(f"Invalid operation during position update: {e}")
                raise PositionUpdateError(f"Invalid operation: {e}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Unexpected error during position update: {e}")
                raise PositionUpdateError(f"Unexpected error: {e}")

    def update_market_data(self, current_price: Decimal):
        self.current_price = current_price
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size if self.side.lower() == 'buy' else (self.entry_price - self.current_price) * self.size

    def close_position(self, realized_pnl: Decimal):
        self.realized_pnl += realized_pnl 