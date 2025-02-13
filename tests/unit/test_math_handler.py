import pytest
from src.utils.numeric_handler import NumericHandler
from src.utils.exceptions import MathError
from decimal import Decimal, InvalidOperation

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

def test_calculate_kelly_fraction(math_handler):
    win_rate = Decimal('0.6')
    win_loss_ratio = Decimal('2.0')
    kelly_fraction = math_handler.calculate_kelly_fraction(win_rate, win_loss_ratio)
    assert kelly_fraction == Decimal('0.2')  # Update expected value if the calculation is correct 

@pytest.fixture
def math_handler():
    return MathHandler()

@pytest.mark.asyncio
async def test_calculate_expected_value(math_handler):
    probability = Decimal('0.6')
    odds = Decimal('1.5')
    try:
        expected_value = math_handler.calculate_expected_value(probability, odds)
        expected = (probability * odds) - (1 - probability)
        assert expected_value == expected
    except InvalidOperation:
        pytest.fail("InvalidOperation raised unexpectedly!")

@pytest.mark.asyncio
async def test_calculate_expected_value_invalid():
    math_handler = MathHandler()
    probability = "invalid"  # Invalid input
    odds = Decimal('1.5')
    with pytest.raises(InvalidOperation):
        math_handler.calculate_expected_value(probability, odds) 