from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any

@dataclass
class PositionInfo:
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    timestamp: int
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    strategy: str
    metadata: Dict[str, Any] = None
    # Add other relevant fields as needed 