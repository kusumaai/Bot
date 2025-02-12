from decimal import Decimal, InvalidOperation, DivisionByZero
import logging

class NumericHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def to_decimal(self, value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Conversion to Decimal failed: {e}")
            raise

    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        try:
            return numerator / denominator
        except DivisionByZero as e:
            self.logger.error(f"Division by zero: {e}")
            return Decimal('0')
        except InvalidOperation as e:
            self.logger.error(f"Invalid operation during division: {e}")
            return Decimal('0')

    def percentage_to_decimal(self, percentage: Decimal) -> Decimal:
        try:
            return percentage / Decimal('100')
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Invalid percentage conversion: {e}")
            return Decimal('0') 