#! /usr/bin/env python3
# src/signals/market_state.py
"""
Module: src.signals
Provides market state analysis functions.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.bot_types.base_types import MarketState


@dataclass
class MarketState:
    trend: str
    volatility: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "volume": self.volume,
        }


def prepare_market_state(data, ctx=None):
    """Create MarketState from raw data"""
    import pandas as pd

    # Safely determine the symbol
    if isinstance(data, pd.DataFrame):
        symbol = (
            data["symbol"].iloc[0]
            if ("symbol" in data.columns and not data.empty)
            else "UNKNOWN"
        )
    else:
        symbol = data.get("symbol", "UNKNOWN")

    return MarketState(
        symbol=symbol,
        price=Decimal(str(data["price"])),
        volume=Decimal(str(data["volume"])),
        timestamp=float(data["timestamp"]),
        metadata=data.get("metadata", {}),
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
    spread: Decimal = Decimal("0")
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
