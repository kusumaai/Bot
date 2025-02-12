from decimal import Decimal, InvalidOperation, DivisionByZero
import logging
from typing import Any, Optional, Union

from utils.exceptions import MathError

class NumericHandler:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.PRECISION = 8
        self.ROUNDING = Decimal('1.' + '0' * self.PRECISION)

    def to_decimal(self, value: Union[str, float, int, Decimal]) -> Optional[Decimal]:
        """Convert any numeric value to Decimal, handling errors"""
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Failed to convert value to Decimal: {value} - {e}")
            return None

    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        try:
            if denominator == Decimal('0'):
                raise MathError("Division by zero.")
            return numerator / denominator
        except InvalidOperation as e:
            raise MathError(f"Invalid operation in division: {e}") from e

    def percentage_to_decimal(self, percentage: Decimal) -> Decimal:
        try:
            return percentage / Decimal('100')
        except InvalidOperation as e:
            raise MathError(f"Invalid percentage value: {e}") from e

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

    # Add other numerical methods as needed 