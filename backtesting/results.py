from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional


@dataclass
class TradeResult:
    # Required fields (no defaults)
    symbol: str
    entry_time: datetime
    entry_price: Decimal
    direction: str
    size: Decimal
    
    # Optional fields (with defaults)
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    pnl: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)