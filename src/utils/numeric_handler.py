#! /usr/bin/env python3
#src/utils/numeric_handler.py
"""
Module: src.utils
Provides numerical handling functionality.
Maybe needs to merge with numeric.py
"""
from decimal import Decimal, InvalidOperation, DivisionByZero, ROUND_HALF_UP
import logging
from typing import Any, Optional, Union
from utils.exceptions import MathError
from utils.error_handler import handle_error_async
#numeric handler    
class NumericHandler:
    """Handles numerical operations and conversions"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.PRECISION = 8
        self.ROUNDING = Decimal('1.' + '0' * self.PRECISION)
#to decimal
    def to_decimal(self, value: Union[str, float, int, Decimal]) -> Optional[Decimal]:
        """Convert any numeric value to Decimal, handling errors"""
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Failed to convert value to Decimal: {value} - {e}")
            return None
#safe divide
    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        if denominator == Decimal("0"):
            raise DivisionByZero("Division by zero.")
        return numerator / denominator
#percentage to decimal
    def percentage_to_decimal(self, value) -> Decimal:
        try:
            return Decimal(value) / Decimal("100")
        except (InvalidOperation, ValueError) as e:
            raise e

    def convert_to_decimal(self, value) -> Decimal:
        """Converts a value to Decimal safely."""
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise MathError(f"Invalid operation during conversion to Decimal: {e}")
#round value
    def round_value(self, value: Decimal, precision: str = '0.0001') -> Decimal:
        """Rounds the decimal value to the specified precision."""
        try:
            return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)
        except InvalidOperation as e:
            raise MathError(f"Invalid operation during rounding: {e}")
#normalize
    def normalize(self, value: Decimal, precision: Optional[int] = None) -> Decimal:
        precision = precision or self.PRECISION
        quantize_str = '1.' + '0' * precision
        return value.quantize(Decimal(quantize_str))

    def round_decimal(self, value: Any, places: int = 8) -> Decimal:
        """Round Decimal to specified places"""
        decimal_val = self.to_decimal(value)
        if decimal_val is not None:
            return decimal_val.quantize(Decimal(f'0.{"0" * places}'))
        else:
            return Decimal('0')
#round value
    def round_value(self, value: Decimal, precision: str = '0.0001') -> Decimal:
        """Rounds the decimal value to the specified precision."""
        try:
            return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)
        except InvalidOperation as e:
            raise MathError(f"Invalid operation during rounding: {e}")
#convert to decimal
    def convert_to_decimal(self, value) -> Decimal:
        """Converts a value to Decimal safely."""
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise MathError(f"Invalid operation during conversion to Decimal: {e}")
#add other numerical methods as needed
    #def name(one, two):
    #    """
    #    Purpose: one
    #    """
