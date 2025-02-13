#!/usr/bin/env python3
"""
Base type definitions and shared interfaces
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
import time
from datetime import datetime

@dataclass
class ValidationResult:
    """Standard validation result structure"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Validatable:
    """Interface for validatable objects"""
    def validate(self) -> ValidationResult:
        raise NotImplementedError("Validate method must be implemented")

    def __post_init__(self):
        validation = self.validate()
        if not validation.is_valid:
            raise ValueError(validation.error_message)

@dataclass
class BaseValidationResult:
    """Base validation result structure"""
    is_valid: bool
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketState:
    """Core market state data"""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position(Validatable):
    """Trading position with standardized decimal handling and risk tracking"""
    # Required fields (no defaults)
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal
    
    # Optional fields (with defaults)
    max_adverse_excursion: Decimal = Decimal(0)
    max_favorable_excursion: Decimal = Decimal(0)
    trailing_stop: Optional[Decimal] = None
    last_update_time: float = field(default_factory=time.time)
    closed: bool = False
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None

    def validate(self) -> ValidationResult:
        """Validate position data"""
        try:
            if self.entry_price <= Decimal('0'):
                return ValidationResult(
                    is_valid=False,
                    error_message="Entry price must be positive"
                )
            
            if self.size <= Decimal('0'):
                return ValidationResult(
                    is_valid=False,
                    error_message="Position size must be positive"
                )
                
            if self.direction not in ['long', 'short']:
                return ValidationResult(
                    is_valid=False,
                    error_message="Direction must be 'long' or 'short'"
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Position validation failed: {str(e)}"
            ) 

@dataclass
class RiskLimits(Validatable):
    """Risk management limits and parameters"""
    min_position_size: Decimal
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    emergency_stop_pct: Decimal
    risk_factor: Decimal
    kelly_scaling: Decimal
    max_correlation: Decimal
    max_sector_exposure: Decimal
    max_volatility: Decimal

    def validate(self) -> ValidationResult:
        """Validate risk limits are within acceptable ranges"""
        try:
            if self.max_position_size > Decimal('0.5'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_position_size cannot exceed 0.5 (50%)"
                )
            if self.min_position_size < Decimal('0'):
                return ValidationResult(
                    is_valid=False,
                    error_message="min_position_size cannot be negative"
                )
            if self.max_daily_loss > Decimal('0.03'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_daily_loss cannot exceed 0.03 (3%)"
                )
            if self.max_drawdown > Decimal('0.2'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_drawdown cannot exceed 0.2 (20%)"
                )
            if self.emergency_stop_pct > Decimal('0.15'):
                return ValidationResult(
                    is_valid=False,
                    error_message="emergency_stop_pct cannot exceed 0.15 (15%)"
                )
            if self.risk_factor > Decimal('0.02'):
                return ValidationResult(
                    is_valid=False,
                    error_message="risk_factor cannot exceed 0.02 (2%)"
                )
            if self.max_correlation > Decimal('0.7'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_correlation cannot exceed 0.7 (70%)"
                )
            if self.max_sector_exposure > Decimal('0.3'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_sector_exposure cannot exceed 0.3 (30%)"
                )
            if self.max_volatility > Decimal('0.4'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_volatility cannot exceed 0.4 (40%)"
                )
            if self.max_leverage > Decimal('3'):
                return ValidationResult(
                    is_valid=False,
                    error_message="max_leverage cannot exceed 3x"
                )
            if self.kelly_scaling > Decimal('0.5'):
                return ValidationResult(
                    is_valid=False,
                    error_message="kelly_scaling cannot exceed 0.5 (50%)"
                )
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Risk limits validation failed: {str(e)}"
            ) 