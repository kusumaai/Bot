"""
trading/types.py - Core trading type definitions
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from decimal import Decimal
import time

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