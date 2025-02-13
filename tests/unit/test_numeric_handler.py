import logging
import pytest
from decimal import Decimal, InvalidOperation

from utils.numeric_handler import NumericHandler


@pytest.fixture
def numeric_handler():
    """Provide a NumericHandler instance."""
    return NumericHandler()


def test_to_decimal_valid_values(numeric_handler):
    """Test conversion of valid numeric values to Decimal."""
    assert numeric_handler.to_decimal('100.5') == Decimal('100.5')
    assert numeric_handler.to_decimal(100.5) == Decimal('100.5')
    assert numeric_handler.to_decimal(Decimal('100.5')) == Decimal('100.5')


def test_to_decimal_invalid_values(numeric_handler, caplog):
    """Test conversion of invalid numeric values to Decimal."""
    with caplog.at_level(logging.ERROR):
        assert numeric_handler.to_decimal('invalid') is None
        assert "Failed to convert value to Decimal: invalid - [<class 'decimal.ConversionSyntax'>]" in caplog.text
        
        assert numeric_handler.to_decimal(None) is None
        assert "Failed to convert value to Decimal: None - [<class 'decimal.ConversionSyntax'>]" in caplog.text


def test_numeric_rounding():
    nh = NumericHandler()
    result = nh.round_value(Decimal('3.14159'), 2)
    assert result == Decimal('3.14')


def test_numeric_conversion():
    nh = NumericHandler()
    result = nh.convert_to_decimal("123.456")
    assert result == Decimal("123.456") 