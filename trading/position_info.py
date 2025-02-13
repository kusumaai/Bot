from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any

@dataclass
class PositionInfo:
    # Required fields (no defaults)
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
    
    # Optional fields (with defaults)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Add other relevant fields as needed 