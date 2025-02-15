#! /usr/bin/env python3
#src/execution/validation.py
"""
Module: src.execution
Provides validation management.
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict
from src.bot_types.base_types import Validatable, ValidationResult

#order validator class that validates orders
@dataclass
class OrderValidator(Validatable):
    """Centralized order validation"""
    
    def validate_order(self, order: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters"""
        try:
            required_fields = ['symbol', 'side', 'type', 'size']
            missing_fields = [f for f in required_fields if f not in order]
            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            validations = [
                (order['side'] in ['buy', 'sell'],
                 "Invalid order side"),
                (order['type'] in ['market', 'limit', 'stop', 'take_profit'],
                 "Invalid order type"),
                (Decimal(str(order['size'])) > Decimal('0'),
                 "Order size must be positive"),
                (not order.get('price') or Decimal(str(order['price'])) > Decimal('0'),
                 "Order price must be positive"),
                (self._validate_order_limits(order),
                 "Order exceeds limits")
            ]
            
            for condition, message in validations:
                if not condition:
                    return ValidationResult(is_valid=False, error_message=message)
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Order validation failed: {str(e)}"
            ) 