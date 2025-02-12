from decimal import Decimal
from typing import Union, Any, Optional
from decimal import ROUND_HALF_UP

class NumericHandler:
    def __init__(self):
        self.PRECISION = 8
        self.ROUNDING = ROUND_HALF_UP
        
    def to_decimal(self, value: Union[str, float, int, Decimal]) -> Decimal:
        """Convert any numeric value to Decimal safely"""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def normalize(self, value: Decimal, precision: Optional[int] = None) -> Decimal:
        precision = precision or self.PRECISION
        return value.quantize(
            Decimal(f'0.{"0" * precision}'),
            rounding=self.ROUNDING
        )

    def safe_divide(
        self,
        numerator: Union[Decimal, str, float],
        denominator: Union[Decimal, str, float],
        default: Decimal = Decimal('0')
    ) -> Decimal:
        try:
            num = self.to_decimal(numerator)
            den = self.to_decimal(denominator)
            if den == Decimal('0'):
                return default
            return self.normalize(num / den)
        except Exception:
            return default

    @staticmethod
    def percentage_to_decimal(value: Any) -> Decimal:
        """Convert percentage to decimal (e.g., 3% -> 0.03)"""
        decimal_val = NumericHandler.to_decimal(value)
        return decimal_val / Decimal('100') if decimal_val > 1 else decimal_val

    @staticmethod
    def round_decimal(value: Any, places: int = 8) -> Decimal:
        """Round Decimal to specified places"""
        return NumericHandler.to_decimal(value).quantize(Decimal(f'0.{"0" * places}'))
