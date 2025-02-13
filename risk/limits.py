#!/usr/bin/env python3
"""
Module: risk/limits.py
Core risk limits and validation
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
import json
from types.base_types import BaseValidationResult
from utils.error_handler import handle_error

@dataclass
class RiskLimits:
    """Risk management limits and thresholds"""
    
    # Required fields (no defaults)
    # Position Limits
    max_position_size: Decimal
    min_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    
    # Loss Limits
    max_drawdown: Decimal
    max_daily_loss: Decimal
    emergency_stop_pct: Decimal
    
    # Risk Scaling
    risk_factor: Decimal
    kelly_scaling: Decimal
    
    # Correlation Limits
    max_correlation: Decimal
    max_sector_exposure: Decimal
    
    # Volatility Limits
    max_volatility: Decimal
    
    # Optional fields (with defaults)
    min_liquidity: Decimal = Decimal('100000')
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RiskLimits':
        """Create RiskLimits from configuration dictionary"""
        return cls(
            max_position_size=Decimal(str(config.get('max_position_size', '0.1'))),
            min_position_size=Decimal(str(config.get('min_position_size', '0.01'))),
            max_positions=int(config.get('max_positions', 3)),
            max_leverage=Decimal(str(config.get('max_leverage', '2.0'))),
            max_drawdown=Decimal(str(config.get('max_drawdown', '0.1'))),
            max_daily_loss=Decimal(str(config.get('max_daily_loss', '0.03'))),
            emergency_stop_pct=Decimal(str(config.get('emergency_stop_pct', '-3.0'))),
            risk_factor=Decimal(str(config.get('risk_factor', '0.01'))),
            kelly_scaling=Decimal(str(config.get('kelly_scaling', '0.5'))),
            max_correlation=Decimal(str(config.get('max_correlation', '0.7'))),
            max_sector_exposure=Decimal(str(config.get('max_sector_exposure', '0.3'))),
            max_volatility=Decimal(str(config.get('max_volatility', '0.05'))),
            min_liquidity=Decimal(str(config.get('min_liquidity', '100000')))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert limits to dictionary for storage/transmission"""
        return {
            'max_position_size': str(self.max_position_size),
            'min_position_size': str(self.min_position_size),
            'max_positions': self.max_positions,
            'max_leverage': str(self.max_leverage),
            'max_drawdown': str(self.max_drawdown),
            'max_daily_loss': str(self.max_daily_loss),
            'emergency_stop_pct': str(self.emergency_stop_pct),
            'risk_factor': str(self.risk_factor),
            'kelly_scaling': str(self.kelly_scaling),
            'max_correlation': str(self.max_correlation),
            'max_sector_exposure': str(self.max_sector_exposure),
            'max_volatility': str(self.max_volatility),
            'min_liquidity': str(self.min_liquidity)
        }
    
    def validate_position_size(self, size: Decimal) -> bool:
        """Validate if position size is within limits"""
        return self.min_position_size <= size <= self.max_position_size
    
    def validate_total_exposure(self, current_positions: int) -> bool:
        """Validate if new position would exceed max positions"""
        return current_positions < self.max_positions
    
    def validate_drawdown(self, current_drawdown: Decimal) -> bool:
        """Validate if drawdown is within limits"""
        return abs(current_drawdown) <= abs(self.max_drawdown)
    
    def validate_daily_loss(self, daily_loss: Decimal) -> bool:
        """Validate if daily loss is within limits"""
        return abs(daily_loss) <= abs(self.max_daily_loss)
    
    def validate_emergency_stop(self, current_loss: Decimal) -> bool:
        """Check if emergency stop loss has been hit"""
        return current_loss > self.emergency_stop_pct
    
    def calculate_position_size(
        self,
        account_size: Decimal,
        risk_per_trade: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate safe position size based on risk parameters"""
        try:
            risk = risk_per_trade if risk_per_trade else self.risk_factor
            size = account_size * risk * self.kelly_scaling
            
            if size > self.max_position_size * account_size:
                return self.max_position_size * account_size
            elif size < Decimal('0.0'):
                return Decimal('0')
            return size
        except Exception as e:
            handle_error(e, "RiskLimits.calculate_position_size", logger=None)
            return Decimal('0')

    def validate(self) -> Optional[str]:
        """Validate risk limits are within acceptable ranges"""
        try:
            validations: List[Tuple[bool, str]] = [
                (self.max_position_size <= Decimal('0.5'), 
                 "max_position_size cannot exceed 0.5 (50%)"),
                (self.max_daily_loss <= Decimal('0.03'), 
                 "max_daily_loss cannot exceed 0.03 (3%)"),
                (self.max_drawdown <= Decimal('0.2'), 
                 "max_drawdown cannot exceed 0.2 (20%)"),
                (self.emergency_stop_pct <= Decimal('0.05'), 
                 "emergency_stop_pct cannot exceed 0.05 (5%)"),
                (self.max_leverage <= Decimal('3'), 
                 "max_leverage cannot exceed 3 (3x)"),
                (self.max_correlation <= Decimal('0.8'),
                 "max_correlation cannot exceed 0.8 (80%)"),
                (self.risk_factor <= Decimal('0.2'),
                 "risk_factor cannot exceed 0.2 (20%)"),
                (self.kelly_scaling <= Decimal('1'),
                 "kelly_scaling cannot exceed 1.0")
            ]

            for condition, message in validations:
                if not condition:
                    return message
            return None

        except Exception as e:
            handle_error(e, "RiskLimits.validate", logger=None)
            return str(e)
