from typing import Dict
from decimal import Decimal, DivisionByZero, InvalidOperation

class TradingMathHandler:
    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        try:
            return numerator / denominator
        except DivisionByZero:
            return Decimal('0')  # Or handle as per business logic
        except InvalidOperation:
            return Decimal('0')  # Or handle as needed 
        
    def calculate_expected_value(self, probability: Decimal, odds: Decimal) -> Decimal:
        try:
            return (probability * odds) - (Decimal('1') - probability)
        except InvalidOperation as e:
            raise e 
        
    def calculate_kelly_position_size(self, probability: Decimal, expected_value: Decimal, price: Decimal) -> Decimal:
        try:
            return (probability * expected_value) / price
        except InvalidOperation as e:
            raise e 
        
    def calculate_position_correlation(self, symbol: str, correlations: Dict[str, Decimal]) -> Decimal:
        try:
            return correlations[symbol]
        except KeyError as e:
            raise e  
        
    
        
        
