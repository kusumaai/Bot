#!/usr/bin/env python3
"""
Module: config/risk_config.py
Risk configuration management
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional
from bot_types.base_types import RiskLimits, ValidationResult

@dataclass
class RiskConfig:
    """Trading risk configuration"""
    max_leverage: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    emergency_stop_pct: Decimal
    trailing_stop_pct: Decimal
    max_hold_hours: Decimal
    max_position_pct: Decimal
    initial_balance: Decimal
    ratchet_thresholds: List[Decimal]
    ratchet_lock_ins: List[Decimal]

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'RiskConfig':
        """Create RiskConfig from dictionary with validation"""
        try:
            # Validate and sort ratchet parameters
            thresholds = sorted([Decimal(str(t)) for t in config.get('ratchet_thresholds', [])])
            lock_ins = sorted([Decimal(str(l)) for l in config.get('ratchet_lock_ins', [])])
            if len(thresholds) != len(lock_ins):
                raise ValueError("Ratchet thresholds and lock-ins must have the same length")
            if not all(x < y for x, y in zip(thresholds[:-1], thresholds[1:])):
                raise ValueError("Ratchet thresholds must be ascending")

            # Create config with validated values
            risk_config = RiskConfig(
                max_leverage=Decimal(str(config.get('max_leverage', '3.0'))),
                max_drawdown=Decimal(str(config.get('max_drawdown', '0.2'))),
                max_daily_loss=Decimal(str(config.get('max_daily_loss', '0.03'))),
                emergency_stop_pct=Decimal(str(config.get('emergency_stop_pct', '0.05'))),  # Updated to 5%
                trailing_stop_pct=Decimal(str(config.get('trailing_stop_pct', '0.015'))),
                max_hold_hours=Decimal(str(config.get('max_hold_hours', '8'))),
                max_position_pct=Decimal(str(config.get('max_position_pct', '50'))) / Decimal('100'),
                initial_balance=Decimal(str(config.get('initial_balance', '10000'))),
                ratchet_thresholds=thresholds,
                ratchet_lock_ins=lock_ins
            )

            # Validate against RiskLimits
            limits = RiskLimits(
                min_position_size=Decimal('0.01'),
                max_position_size=risk_config.max_position_pct,
                max_positions=int(config.get('max_positions', 10)),
                max_leverage=risk_config.max_leverage,
                max_drawdown=risk_config.max_drawdown,
                max_daily_loss=risk_config.max_daily_loss,
                emergency_stop_pct=risk_config.emergency_stop_pct,
                risk_factor=Decimal(str(config.get('risk_factor', '0.02'))),
                kelly_scaling=Decimal(str(config.get('kelly_scaling', '0.5'))),
                max_correlation=Decimal(str(config.get('max_correlation', '0.7'))),
                max_sector_exposure=Decimal(str(config.get('max_sector_exposure', '0.3'))),
                max_volatility=Decimal(str(config.get('max_volatility', '0.4')))
            )
            
            validation = limits.validate()
            if not validation.is_valid:
                raise ValueError(f"Risk config validation failed: {validation.error_message}")

            return risk_config

        except Exception as e:
            raise ValueError(f"Invalid risk configuration: {str(e)}") from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format"""
        return {
            'max_leverage': str(self.max_leverage),
            'max_drawdown': str(self.max_drawdown),
            'max_daily_loss': str(self.max_daily_loss),
            'emergency_stop_pct': str(self.emergency_stop_pct),
            'trailing_stop_pct': str(self.trailing_stop_pct),
            'max_hold_hours': str(self.max_hold_hours),
            'max_position_pct': str(self.max_position_pct * Decimal('100')),
            'initial_balance': str(self.initial_balance),
            'ratchet_thresholds': [str(t) for t in self.ratchet_thresholds],
            'ratchet_lock_ins': [str(l) for l in self.ratchet_lock_ins]
        } 

    def validate_risk_limits(self, limits: RiskLimits) -> ValidationResult:
        """Validate risk limits against RiskConfig"""
        return limits.validate()

