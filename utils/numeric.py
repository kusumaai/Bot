from decimal import Decimal
from typing import Union, Any

class NumericHandler:
    @staticmethod
    def to_decimal(value: Any) -> Decimal:
        """Convert any numeric value to Decimal safely"""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @staticmethod
    def percentage_to_decimal(value: Any) -> Decimal:
        """Convert percentage to decimal (e.g., 3% -> 0.03)"""
        decimal_val = NumericHandler.to_decimal(value)
        return decimal_val / Decimal('100') if decimal_val > 1 else decimal_val

    @staticmethod
    def round_decimal(value: Any, places: int = 8) -> Decimal:
        """Round Decimal to specified places"""
        return NumericHandler.to_decimal(value).quantize(Decimal(f'0.{"0" * places}'))
