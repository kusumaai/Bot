#!/usr/bin/env python3
"""
Module: risk/limits.py
Core risk limits and validation
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
import json

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
            min_position_size=Decimal(str(risk_config.get("min_position_size", "0.01"))),
            max_position_size=Decimal(str(risk_config.get("max_position_size", "0.5"))),
            max_positions=int(risk_config.get("max_positions", 10)),
            max_leverage=Decimal(str(risk_config.get("max_leverage", "3"))),
            max_drawdown=Decimal(str(risk_config.get("max_drawdown", "0.2"))),
            max_daily_loss=Decimal(str(risk_config.get("max_daily_loss", "0.03"))),
            emergency_stop_pct=Decimal(str(risk_config.get("emergency_stop_pct", "0.05"))),
            risk_factor=Decimal(str(risk_config.get("risk_factor", "0.02"))),
            kelly_scaling=Decimal(str(risk_config.get("kelly_scaling", "0.5"))),
            max_correlation=Decimal(str(risk_config.get("max_correlation", "0.7"))),
            max_sector_exposure=Decimal(str(risk_config.get("max_sector_exposure", "0.3"))),
            max_volatility=Decimal(str(risk_config.get("max_volatility", "0.4")))
        )
        
        validation = limits.validate()
        if not validation.is_valid:
            raise ValueError(validation.error_message)
            
        return limits
        
    except Exception as e:
        raise ValueError(f"Failed to load risk limits: {str(e)}")
