from decimal import Decimal
import pytest
from src.trading.math import MathHandler
from src.utils.exceptions import MathError

@pytest.fixture
def math_handler():
    return MathHandler()

def test_calculate_kelly_fraction(math_handler):
    kelly = math_handler.calculate_kelly_fraction(Decimal('0.6'), Decimal('2'))
    assert kelly == Decimal('0.2')

    with pytest.raises(MathError):
        math_handler.calculate_kelly_fraction(Decimal('0.6'), Decimal('0'))

def test_calculate_position_size(math_handler):
    size = math_handler.calculate_position_size(Decimal('10000'), Decimal('0.01'), Decimal('100'))
    assert size == Decimal('1')

    with pytest.raises(MathError):
        math_handler.calculate_position_size(Decimal('10000'), Decimal('0.01'), Decimal('0'))

def test_calculate_expected_value(math_handler):
    ev = math_handler.calculate_expected_value(Decimal('0.6'), Decimal('200'), Decimal('100'))
    assert ev == Decimal('20.0')

    with pytest.raises(MathError):
        math_handler.calculate_expected_value(Decimal('0.6'), Decimal('invalid'), Decimal('100')) 