from dataclasses import dataclass, field
from typing import Dict, Any
from decimal import Decimal

@dataclass
class SignalPerformance:
    # Required fields (no defaults)
    signal_id: str
    symbol: str
    direction: str
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    
    # Optional fields (with defaults)
    hold_time: float = 0.0
    max_adverse_excursion: Decimal = Decimal('0')
    max_favorable_excursion: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)