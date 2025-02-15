#! /usr/bin/env python3
#src/execution/fills.py
"""
Module: src.execution
Provides fill information management.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict


@dataclass
class FillInfo:
    # Required fields (no defaults)
    symbol: str
    side: str
    size: Decimal
    price: Decimal
    timestamp: float
    order_id: str
    
    # Optional fields (with defaults)
    fees: Decimal = Decimal('0')
    liquidity: str = 'taker'
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
    # ... rest of the class needs implementation ... 
    #TODO: Implement the class
