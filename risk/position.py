from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional

@dataclass
class Position:
    """Trading position with standardized decimal handling"""
    symbol: str
    direction: str
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal = Decimal(0)
    max_favorable_excursion: Decimal = Decimal(0)
    
    def update(self, current_price: Decimal) -> None:
        self.current_price = current_price
        price_diff = current_price - self.entry_price
        multiplier = Decimal(1) if self.direction == "long" else Decimal(-1)
        self.unrealized_pnl = price_diff * self.size * multiplier 