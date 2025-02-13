from decimal import Decimal
import pytest
from src.utils.numeric_handler import NumericHandler
from src.utils.exceptions import MathError

@pytest.fixture
def numeric_handler():
    return NumericHandler()

def test_safe_divide(numeric_handler):
    result = numeric_handler.safe_divide(Decimal('10'), Decimal('2'))
    assert result == Decimal('5')

    with pytest.raises(MathError):
        numeric_handler.safe_divide(Decimal('10'), Decimal('0'))

def test_to_decimal(numeric_handler):
    assert numeric_handler.to_decimal("123.45") == Decimal("123.45")
    assert numeric_handler.to_decimal(123.45) == Decimal("123.45")
    assert numeric_handler.to_decimal("abc") is None

def test_percentage_to_decimal(numeric_handler):
    result = numeric_handler.percentage_to_decimal(Decimal('50'))
    assert result == Decimal('0.5')

    with pytest.raises(MathError):
        numeric_handler.percentage_to_decimal(Decimal('invalid')) 