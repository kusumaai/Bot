#!/usr/bin/env python3
"""
Position-related type definitions to avoid circular imports
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

from utils.numeric_handler import NumericHandler

from .base_types import Validatable, ValidationResult


@dataclass
class PositionInfo:
    """Basic position information"""

    symbol: str
    size: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal]
    unrealized_pnl: Optional[Decimal]
    realized_pnl: Optional[Decimal]
    side: str
    timestamp: float


@dataclass
class PositionValidationConfig:
    """Configuration for position validation thresholds."""

    max_position_size: Decimal
    min_position_size: Decimal
    max_drawdown: Decimal
    max_leverage: Decimal
    max_position_duration: int  # in seconds
    min_stop_distance: Decimal  # minimum distance for stop loss as percentage
    min_profit_distance: Decimal  # minimum distance for take profit as percentage
    max_daily_positions: int
    max_positions_per_symbol: int

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PositionValidationConfig":
        """Create config from dictionary with proper error handling."""
        try:
            nh = NumericHandler()
            return cls(
                max_position_size=nh.to_decimal(config.get("max_position_size", "1.0")),
                min_position_size=nh.to_decimal(
                    config.get("min_position_size", "0.01")
                ),
                max_drawdown=nh.to_decimal(config.get("max_drawdown", "0.5")),
                max_leverage=nh.to_decimal(config.get("max_leverage", "10.0")),
                max_position_duration=int(config.get("max_position_duration", 86400)),
                min_stop_distance=nh.to_decimal(
                    config.get("min_stop_distance", "0.01")
                ),
                min_profit_distance=nh.to_decimal(
                    config.get("min_profit_distance", "0.01")
                ),
                max_daily_positions=int(config.get("max_daily_positions", 10)),
                max_positions_per_symbol=int(config.get("max_positions_per_symbol", 1)),
            )
        except Exception as e:
            raise ValueError(f"Invalid position validation config: {e}")
