from decimal import Decimal, InvalidOperation, DivisionByZero
import logging

from utils.numeric_handler import NumericHandler
from trading.exceptions import MathError

class MathHandler:
    def __init__(self):
        self.nh = NumericHandler()
        self.logger = logging.getLogger(__name__)

    def calculate_kelly_fraction(self, win_prob: Decimal, win_loss_ratio: Decimal) -> Decimal:
        try:
            numerator = win_prob - (Decimal('1') - win_prob) / win_loss_ratio
            return self.nh.safe_divide(numerator, Decimal('1'))
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            raise MathError(f"Error calculating Kelly fraction: {e}")

    def calculate_position_size(self, account_size: Decimal, risk_per_trade: Decimal, stop_loss: Decimal) -> Decimal:
        try:
            position_size = account_size * risk_per_trade
            return self.nh.safe_divide(position_size, stop_loss)
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Error calculating position size: {e}")
            raise MathError(f"Error calculating position size: {e}")

    def calculate_expected_value(self, win_prob: Decimal, win_amount: Decimal, loss_amount: Decimal) -> Decimal:
        try:
            expected_value = (win_prob * win_amount) - ((Decimal('1') - win_prob) * loss_amount)
            return expected_value
        except InvalidOperation as e:
            self.logger.error(f"Error calculating expected value: {e}")
            raise MathError(f"Error calculating expected value: {e}") 