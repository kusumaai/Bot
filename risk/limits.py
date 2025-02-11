#!/usr/bin/env python3
"""
Module: risk/limits.py
Core risk limits and validation
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple

from utils.error_handler import handle_error

@dataclass(frozen=True)
class RiskLimits:
    """Risk management limits with validation"""
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    max_correlation: Decimal
    max_drawdown: Decimal
    emergency_stop_pct: Decimal
    max_daily_loss: Decimal
    trailing_stop_pct: Decimal
    max_adverse_pct: Decimal
    kelly_scaling: Decimal
    risk_factor: Decimal

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional['RiskLimits']:
        """Create RiskLimits from config with validation"""
        try:
            # Required fields with defaults
            limits = cls(
                max_position_size=Decimal(str(config.get("max_position_size", 0.1))),
                max_positions=config.get("max_positions", 3),
                max_leverage=Decimal(str(config.get("max_leverage", 2.0))),
                max_correlation=Decimal(str(config.get("max_correlation", 0.7))),
                max_drawdown=Decimal(str(config.get("max_drawdown_pct", 10))) / 100,
                emergency_stop_pct=Decimal(str(config.get("emergency_stop_pct", 5))) / 100,
                max_daily_loss=Decimal(str(config.get("max_daily_loss", 3))) / 100,
                trailing_stop_pct=Decimal(str(config.get("trailing_stop_pct", 1.5))) / 100,
                max_adverse_pct=Decimal(str(config.get("max_adverse_pct", 3))) / 100,
                kelly_scaling=Decimal(str(config.get("kelly_scaling", 0.5))),
                risk_factor=Decimal(str(config.get("risk_factor", 0.1)))
            )

            validation_error = limits.validate()
            if validation_error:
                raise ValueError(validation_error)

            return limits

        except Exception as e:
            handle_error(e, "RiskLimits.from_config", logger=None)
            return None

    def validate(self) -> Optional[str]:
        """Validate risk limits are within acceptable ranges"""
        try:
            validations: List[Tuple[bool, str]] = [
                (self.max_position_size <= Decimal('0.5'), 
                 "max_position_size cannot exceed 0.5"),
                (self.max_daily_loss <= Decimal('0.03'), 
                 "max_daily_loss cannot exceed 3%"),
                (self.max_drawdown <= Decimal('0.2'), 
                 "max_drawdown cannot exceed 20%"),
                (self.emergency_stop_pct <= Decimal('0.05'), 
                 "emergency_stop_pct cannot exceed 5%"),
                (self.max_leverage <= Decimal('3'), 
                 "max_leverage cannot exceed 3x"),
                (self.max_correlation <= Decimal('0.8'),
                 "max_correlation cannot exceed 0.8"),
                (self.trailing_stop_pct <= Decimal('0.05'),
                 "trailing_stop_pct cannot exceed 5%"),
                (self.max_adverse_pct <= Decimal('0.1'),
                 "max_adverse_pct cannot exceed 10%"),
                (self.kelly_scaling <= Decimal('1'),
                 "kelly_scaling cannot exceed 1.0"),
                (self.risk_factor <= Decimal('0.2'),
                 "risk_factor cannot exceed 0.2")
            ]

            for condition, message in validations:
                if not condition:
                    return message
            return None

        except Exception as e:
            handle_error(e, "RiskLimits.validate", logger=None)
            return str(e)