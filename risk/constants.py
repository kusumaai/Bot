#!/usr/bin/env python3
"""
Module: risk/constants.py
Core trading constants and decimal handling utilities
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Optional

# Core precision constants
PRICE_PRECISION = Decimal('0.00000001')  # 8 decimal places
SIZE_PRECISION = Decimal('0.00000001')   # 8 decimal places
PNL_PRECISION = Decimal('0.00000001')    # 8 decimal places
PERCENTAGE_PRECISION = Decimal('0.0001')  # 4 decimal places for percentages

# Risk management constants
MAX_ALLOWED_POSITION_SIZE = Decimal('0.5')      # 50% of portfolio
MAX_ALLOWED_DAILY_LOSS = Decimal('1000')        # $1000 USD
MAX_ALLOWED_DRAWDOWN = Decimal('0.20')          # 20%
MAX_ALLOWED_EMERGENCY_STOP = Decimal('0.05')    # 5%

def normalize_decimal(value: Union[Decimal, float, str, int], 
                     precision: Decimal,
                     min_value: Optional[Decimal] = None,
                     max_value: Optional[Decimal] = None) -> Decimal:
    """
    Normalize a value to a specific decimal precision with optional bounds checking.
    
    Args:
        value: Value to normalize
        precision: Decimal precision to normalize to
        min_value: Optional minimum allowed value
        max_value: Optional maximum allowed value
        
    Returns:
        Normalized decimal value
        
    Raises:
        ValueError: If value is outside allowed bounds
    """
    try:
        normalized = Decimal(str(value)).quantize(precision, rounding=ROUND_HALF_UP)
        
        if min_value is not None and normalized < min_value:
            raise ValueError(f"Value {normalized} below minimum {min_value}")
            
        if max_value is not None and normalized > max_value:
            raise ValueError(f"Value {normalized} above maximum {max_value}")
            
        return normalized
        
    except Exception as e:
        raise ValueError(f"Failed to normalize value {value}: {str(e)}")

def validate_risk_limits(value: Decimal, limit_type: str) -> bool:
    """Validate risk management values against maximum allowed limits"""
    limits = {
        'position_size': MAX_ALLOWED_POSITION_SIZE,
        'daily_loss': MAX_ALLOWED_DAILY_LOSS,
        'drawdown': MAX_ALLOWED_DRAWDOWN,
        'emergency_stop': MAX_ALLOWED_EMERGENCY_STOP
    }
    
    if limit_type not in limits:
        raise ValueError(f"Unknown limit type: {limit_type}")
        
    return value <= limits[limit_type] 