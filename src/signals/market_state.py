#! /usr/bin/env python3
#src/signals/market_state.py
"""
Module: src.signals
Provides market state analysis functions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
from decimal import Decimal
from bot_types.base_types import MarketState

@dataclass
class MarketState:
    trend: str
    volatility: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "volume": self.volume
        }

def prepare_market_state(data: Dict[str, Any]) -> MarketState:
    """Create MarketState from raw data"""
    return MarketState(
        symbol=data["symbol"],
        price=Decimal(str(data["price"])),
        volume=Decimal(str(data["volume"])),
        timestamp=float(data["timestamp"]),
        metadata=data.get("metadata", {})
    )

@dataclass
class MarketCondition:
    # Required fields (no defaults)
    symbol: str
    timestamp: float
    price: Decimal
    volume: Decimal
    trend: str
    volatility: float
    
    # Optional fields (with defaults)
    liquidity: float = 0.0
    spread: Decimal = Decimal('0')
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 