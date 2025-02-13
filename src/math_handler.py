from decimal import Decimal, InvalidOperation

class MathHandler:
    def calculate_expected_value(self, probability: Decimal, odds: Decimal) -> Decimal:
        try:
            return (probability * odds) - (Decimal('1') - probability)
        except InvalidOperation as e:
            raise e 