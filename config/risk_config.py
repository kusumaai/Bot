from dataclasses import dataclass
import yaml
from decimal import Decimal
from typing import List, Optional, Dict, Any

@dataclass
class RiskConfig:
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    emergency_stop_pct: Decimal
    position_timeout_hours: int
    max_correlation: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    kelly_scaling: Decimal
    risk_factor: Decimal
    ratchet_thresholds: List[Decimal]
    ratchet_lock_ins: List[Decimal]
    trailing_stop_pct: Decimal
    max_adverse_pct: Decimal
    max_hold_hours: Decimal
    max_position_pct: Decimal
    initial_balance: Decimal

    @classmethod
    def from_yaml(cls, path: str) -> 'RiskConfig':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required = ['max_position_size', 'max_positions', 'max_leverage', 
                   'emergency_stop_pct', 'position_timeout_hours']
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Missing required fields in config: {missing}")
            
        # Validate thresholds and lock-ins
        thresholds = sorted([Decimal(str(t)) for t in config.get('ratchet_thresholds', [])])
        lock_ins = sorted([Decimal(str(l)) for l in config.get('ratchet_lock_ins', [])])
        if len(thresholds) != len(lock_ins):
            raise ValueError("Ratchet thresholds and lock-ins must have the same length.")
        if not all(x < y for x, y in zip(thresholds[:-1], thresholds[1:])):
            raise ValueError("Ratchet thresholds must be ascending.")

        return cls(
            max_position_size=Decimal(str(config['max_position_size'])),
            max_positions=config['max_positions'],
            max_leverage=Decimal(str(config['max_leverage'])),
            emergency_stop_pct=Decimal(str(config['emergency_stop_pct'])),
            position_timeout_hours=config['position_timeout_hours'],
            max_correlation=Decimal(str(config.get('max_correlation', '0.7'))),
            max_drawdown=Decimal(str(config.get('max_drawdown', '0.1'))),
            max_daily_loss=Decimal(str(config.get('max_daily_loss', '0.03'))),
            kelly_scaling=Decimal(str(config.get('kelly_scaling', '0.5'))),
            risk_factor=Decimal(str(config.get('risk_factor', '0.1'))),
            ratchet_thresholds=thresholds,
            ratchet_lock_ins=lock_ins,
            trailing_stop_pct=Decimal(str(config.get('trailing_stop_pct', '1.5'))),
            max_adverse_pct=Decimal(str(config.get('max_adverse_pct', '3'))) / Decimal('100'),
            max_hold_hours=Decimal(str(config.get('max_hold_hours', '8'))),
            max_position_pct=Decimal(str(config.get('max_position_pct', '10'))) / Decimal('100'),
            initial_balance=Decimal(str(config.get('initial_balance', '10000')))
        )

    def validate_limits(self) -> Optional[str]:
        """Validate risk limits are within acceptable ranges"""
        try:
            validations = [
                (self.max_position_size <= Decimal('0.5'), 
                 "max_position_size cannot exceed 0.5"),
                (self.max_daily_loss <= Decimal('0.03'), 
                 "max_daily_loss cannot exceed 3%"),
                (self.max_drawdown <= Decimal('0.2'), 
                 "max_drawdown cannot exceed 20%"),
                (self.emergency_stop_pct <= Decimal('5'), 
                 "emergency_stop_pct cannot exceed 5%"),
                (self.max_leverage <= Decimal('3'), 
                 "max_leverage cannot exceed 3x")
            ]
            
            for condition, message in validations:
                if not condition:
                    return message
            return None
            
        except Exception as e:
            return f"Risk limit validation failed: {str(e)}"

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'RiskConfig':
        try:
            thresholds = sorted([Decimal(str(t)) for t in config.get('ratchet_thresholds', [])])
            lock_ins = sorted([Decimal(str(l)) for l in config.get('ratchet_lock_ins', [])])
            if len(thresholds) != len(lock_ins):
                raise ValueError("Ratchet thresholds and lock-ins must have the same length.")
            if not all(x < y for x, y in zip(thresholds[:-1], thresholds[1:])):
                raise ValueError("Ratchet thresholds must be ascending.")

            return RiskConfig(
                max_leverage=Decimal(str(config.get('max_leverage', '2.0'))),
                max_drawdown=Decimal(str(config.get('max_drawdown', '0.1'))),
                max_daily_loss=Decimal(str(config.get('max_daily_loss', '0.03'))),
                ratchet_thresholds=thresholds,
                ratchet_lock_ins=lock_ins,
                emergency_stop_pct=Decimal(str(config.get("emergency_stop_pct", "-2"))),
                trailing_stop_pct=Decimal(str(config.get("trailing_stop_pct", "1.5"))),
                max_hold_hours=Decimal(str(config.get("max_hold_hours", "8"))),
                max_position_pct=Decimal(str(config.get("max_position_pct", '10'))) / Decimal('100'),
                initial_balance=Decimal(str(config.get("initial_balance", '10000')))
            )
        except Exception as e:
            raise ValueError(f"Invalid risk configuration: {e}") from e 