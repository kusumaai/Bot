#! /usr/bin/env python3
#src/signals/validation.py
"""
Module: src.signals
Provides signal validation.
"""
from dataclasses import dataclass
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime

from signals.base_types import BaseSignal
from bot_types.base_types import Validatable, ValidationResult

@dataclass
class SignalValidator(Validatable):
    """Centralized signal validation"""
    
    def validate_signal(self, signal: BaseSignal) -> ValidationResult:
        """Validate trading signal"""
        try:
            validations = [
                (signal.direction in ['long', 'short'],
                 "Invalid signal direction"),
                (Decimal('0') <= signal.strength <= Decimal('1'),
                 "Signal strength must be between 0 and 1"),
                (signal.timestamp <= datetime.now(),
                 "Signal timestamp cannot be in future"),
                (self._validate_signal_metadata(signal.metadata),
                 "Invalid signal metadata")
            ]
            
            for condition, message in validations:
                if not condition:
                    return ValidationResult(is_valid=False, error_message=message)
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Signal validation failed: {str(e)}"
            )
    
    def _validate_signal_metadata(self, metadata: Dict[str, Any]) -> bool:
        required_fields = ['timeframe', 'model_version', 'probability']
        return all(field in metadata for field in required_fields) 