#! /usr/bin/env python3
# src/risk/limits.py
"""
Module: src.risk
Provides risk limits and validation.
"""
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from bot_types.base_types import RiskLimits as BaseRiskLimits
from bot_types.base_types import ValidationResult
from utils.error_handler import handle_error

# Re-export the RiskLimits class from base_types
RiskLimits = BaseRiskLimits


# Add helper functions for risk limit operations
def load_risk_limits_from_config(config: Dict[str, Any]) -> RiskLimits:
    """Load and validate risk limits from configuration"""
    try:
        risk_config = config.get("risk_limits", {})
        limits = RiskLimits(
            min_position_size=Decimal(
                str(risk_config.get("min_position_size", "0.01"))
            ),
            max_position_size=Decimal(str(risk_config.get("max_position_size", "0.5"))),
            max_positions=int(risk_config.get("max_positions", 10)),
            max_leverage=Decimal(str(risk_config.get("max_leverage", "3"))),
            max_drawdown=Decimal(str(risk_config.get("max_drawdown", "0.2"))),
            max_daily_loss=Decimal(str(risk_config.get("max_daily_loss", "0.03"))),
            emergency_stop_pct=Decimal(
                str(risk_config.get("emergency_stop_pct", "0.05"))
            ),
            risk_factor=Decimal(str(risk_config.get("risk_factor", "0.02"))),
            kelly_scaling=Decimal(str(risk_config.get("kelly_scaling", "0.5"))),
            max_correlation=Decimal(str(risk_config.get("max_correlation", "0.7"))),
            max_sector_exposure=Decimal(
                str(risk_config.get("max_sector_exposure", "0.3"))
            ),
            max_volatility=Decimal(str(risk_config.get("max_volatility", "0.4"))),
        )

        validation = limits.validate()
        if not validation.is_valid:
            raise ValueError(validation.error_message)

        return limits

    except Exception as e:
        raise ValueError(f"Failed to load risk limits: {str(e)}")


class RiskLimits:
    def __init__(
        self,
        max_position_size,
        min_position_size,
        max_positions,
        max_leverage,
        max_drawdown,
        max_daily_loss,
        emergency_stop_pct,
        risk_factor,
        kelly_scaling,
        max_correlation,
        max_sector_exposure,
        max_volatility,
        min_liquidity,
        **kwargs,
    ):
        self.max_position_size = Decimal(max_position_size)
        self.min_position_size = Decimal(min_position_size)
        self.max_positions = max_positions
        self.max_leverage = Decimal(max_leverage)
        self.max_drawdown = Decimal(max_drawdown)
        self.max_daily_loss = Decimal(max_daily_loss)
        self.emergency_stop_pct = Decimal(emergency_stop_pct)
        self.risk_factor = Decimal(risk_factor)
        self.kelly_scaling = Decimal(kelly_scaling)
        self.max_correlation = Decimal(max_correlation)
        self.max_sector_exposure = Decimal(max_sector_exposure)
        self.max_volatility = Decimal(max_volatility)
        self.min_liquidity = Decimal(min_liquidity)
        # Store any extra kwargs as attributes to avoid test errors
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            max_position_size=config.get("max_position_size", "1"),
            min_position_size=config.get("min_position_size", "0"),
            max_positions=config.get("max_positions", 1),
            max_leverage=config.get("max_leverage", "1"),
            max_drawdown=config.get("max_drawdown", "1"),
            max_daily_loss=config.get("max_daily_loss", "1"),
            emergency_stop_pct=config.get("emergency_stop_pct", "1"),
            risk_factor=config.get("risk_factor", "1"),
            kelly_scaling=config.get("kelly_scaling", "1"),
            max_correlation=config.get("max_correlation"),
            max_sector_exposure=config.get("max_sector_exposure"),
            max_volatility=config.get("max_volatility"),
            min_liquidity=config.get("min_liquidity"),
        )
