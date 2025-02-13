from decimal import Decimal, DivisionByZero, InvalidOperation

class TradingMathHandler:
    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        try:
            return numerator / denominator
        except DivisionByZero:
            return Decimal('0')  # Or handle as per business logic
        except InvalidOperation:
            return Decimal('0')  # Or handle as needed 