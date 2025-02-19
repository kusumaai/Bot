#! /usr/bin/env python3
# src/risk/limits.py
"""
Module: src.risk
Provides risk limits and validation.
"""
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from src.bot_types.base_types import RiskLimits as BaseRiskLimits
from src.bot_types.base_types import ValidationResult
from src.utils.error_handler import handle_error

# Re-export the RiskLimits class from base_types
RiskLimits = BaseRiskLimits

logger = logging.getLogger(__name__)


# Add helper functions for risk limit operations
def load_risk_limits_from_config(config: Dict[str, Any]) -> RiskLimits:
    """Load and validate risk limits from configuration"""
    try:
        risk_config = config.get("risk_limits", {})
        limits = RiskLimits(
            min_position_size=Decimal(
                str(
                    risk_config.get(
                        "min_position_size", config.get("min_position_size", "0.01")
                    )
                )
            ),
            max_position_size=Decimal(
                str(
                    risk_config.get(
                        "max_position_size", config.get("max_position_size", "0.5")
                    )
                )
            ),
            max_positions=int(
                risk_config.get("max_positions", config.get("max_positions", 10))
            ),
            max_leverage=Decimal(
                str(risk_config.get("max_leverage", config.get("max_leverage", "3")))
            ),
            max_drawdown=Decimal(
                str(risk_config.get("max_drawdown", config.get("max_drawdown", "0.2")))
            ),
            max_daily_loss=Decimal(
                str(
                    risk_config.get(
                        "max_daily_loss", config.get("max_daily_loss", "0.03")
                    )
                )
            ),
            emergency_stop_pct=Decimal(
                str(
                    risk_config.get(
                        "emergency_stop_pct", config.get("emergency_stop_pct", "0.05")
                    )
                )
            ),
            risk_factor=Decimal(
                str(risk_config.get("risk_factor", config.get("risk_factor", "0.02")))
            ),
            kelly_scaling=Decimal(
                str(
                    risk_config.get("kelly_scaling", config.get("kelly_scaling", "0.5"))
                )
            ),
            max_correlation=Decimal(
                str(
                    risk_config.get(
                        "max_correlation", config.get("max_correlation", "0.7")
                    )
                )
            ),
            max_sector_exposure=Decimal(
                str(
                    risk_config.get(
                        "max_sector_exposure", config.get("max_sector_exposure", "0.3")
                    )
                )
            ),
            max_volatility=Decimal(
                str(
                    risk_config.get(
                        "max_volatility", config.get("max_volatility", "0.4")
                    )
                )
            ),
            min_liquidity=Decimal(
                str(
                    risk_config.get(
                        "min_liquidity", config.get("min_liquidity", "0.0001")
                    )
                )
            ),
        )

        validation = limits.validate()
        if not validation.is_valid:
            raise ValueError(validation.error_message)

        return limits

    except Exception as e:
        raise ValueError(f"Failed to load risk limits: {str(e)}")


class RiskLimits:
    """Class to manage trading risk limits."""

    def __init__(
        self,
        min_position_size: Decimal,
        max_position_size: Decimal,
        max_leverage: Decimal,
        max_drawdown: Decimal,
        max_daily_trades: int,
        max_open_positions: int,
        max_position_value: Decimal,
        emergency_stop_pct: Decimal,
        risk_factor: Decimal = Decimal("0.02"),
        max_risk_per_trade: Decimal = Decimal("0.01"),
        max_correlation: Decimal = Decimal("0.7"),
        min_liquidity: Decimal = Decimal("1000"),
        max_volatility: Decimal = Decimal("0.5"),
        max_daily_loss: Decimal = Decimal("0.03"),
        max_sector_exposure: Decimal = Decimal("0.3"),
        kelly_scaling: Decimal = Decimal("0.5"),
    ):
        """Initialize risk limits."""
        self.logger = logging.getLogger(__name__)

        # Position limits
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.max_daily_trades = max_daily_trades
        self.max_open_positions = max_open_positions
        self.max_position_value = max_position_value
        self.emergency_stop_pct = emergency_stop_pct

        # Risk parameters
        self.risk_factor = risk_factor
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.kelly_scaling = kelly_scaling

        # Market condition limits
        self.max_correlation = max_correlation
        self.min_liquidity = min_liquidity
        self.max_volatility = max_volatility
        self.max_sector_exposure = max_sector_exposure

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RiskLimits":
        """Create RiskLimits from configuration dictionary."""
        try:
            return cls(
                min_position_size=Decimal(str(config.get("min_position_size", "0.01"))),
                max_position_size=Decimal(str(config.get("max_position_size", "1.0"))),
                max_leverage=Decimal(str(config.get("max_leverage", "3.0"))),
                max_drawdown=Decimal(str(config.get("max_drawdown", "0.2"))),
                max_daily_trades=int(config.get("max_daily_trades", 10)),
                max_open_positions=int(config.get("max_open_positions", 5)),
                max_position_value=Decimal(
                    str(config.get("max_position_value", "1000.0"))
                ),
                emergency_stop_pct=Decimal(
                    str(config.get("emergency_stop_pct", "0.15"))
                ),
                risk_factor=Decimal(str(config.get("risk_factor", "0.02"))),
                max_risk_per_trade=Decimal(
                    str(config.get("max_risk_per_trade", "0.01"))
                ),
                max_correlation=Decimal(str(config.get("max_correlation", "0.7"))),
                min_liquidity=Decimal(str(config.get("min_liquidity", "1000"))),
                max_volatility=Decimal(str(config.get("max_volatility", "0.5"))),
                max_daily_loss=Decimal(str(config.get("max_daily_loss", "0.03"))),
                max_sector_exposure=Decimal(
                    str(config.get("max_sector_exposure", "0.3"))
                ),
                kelly_scaling=Decimal(str(config.get("kelly_scaling", "0.5"))),
            )
        except Exception as e:
            logger.error(f"Failed to create RiskLimits from config: {e}")
            raise ValueError(f"Invalid risk limits configuration: {e}")
