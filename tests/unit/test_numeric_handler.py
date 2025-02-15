#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#test_numeric_handler.py
"""Test numeric handler"""
import logging
import pytest
from decimal import Decimal, InvalidOperation
from src.utils.numeric_handler import NumericHandler
from src.utils.exceptions import MathError
from unittest.mock import MagicMock

#fixture for numeric handler
@pytest.fixture
def numeric_handler():
    mock_logger = MagicMock()
    return NumericHandler(logger=mock_logger)

#test to decimal valid values
def test_to_decimal_valid_values(numeric_handler):
    """Test conversion of valid numeric values to Decimal."""
    assert numeric_handler.to_decimal('100.5') == Decimal('100.5')
    assert numeric_handler.to_decimal(100.5) == Decimal('100.5')
    assert numeric_handler.to_decimal(Decimal('100.5')) == Decimal('100.5')

#test to decimal invalid values
def test_to_decimal_invalid_values(numeric_handler, caplog):
    """Test conversion of invalid numeric values to Decimal."""
    with caplog.at_level(logging.ERROR):
        assert numeric_handler.to_decimal('invalid') is None
        assert "Failed to convert value to Decimal: invalid - [<class 'decimal.ConversionSyntax'>]" in caplog.text
        
        assert numeric_handler.to_decimal(None) is None
        assert "Failed to convert value to Decimal: None - [<class 'decimal.ConversionSyntax'>]" in caplog.text

#test numeric rounding
def test_numeric_rounding():
    nh = NumericHandler()
    result = nh.round_value(Decimal('3.14159'), 2)
    assert result == Decimal('3.14')

#test numeric conversion
def test_numeric_conversion():
    nh = NumericHandler()
    result = nh.convert_to_decimal("123.456")
    assert result == Decimal("123.456")

#test convert to decimal valid
def test_convert_to_decimal_valid(numeric_handler):
    assert numeric_handler.convert_to_decimal('100') == Decimal('100')
    assert numeric_handler.convert_to_decimal(100) == Decimal('100')

#test convert to decimal invalid
def test_convert_to_decimal_invalid(numeric_handler):
    with pytest.raises(MathError, match="Invalid operation during conversion to Decimal:"):
        numeric_handler.convert_to_decimal('invalid')

#test round value
def test_round_value(numeric_handler):
    value = Decimal('123.4567')
    rounded = numeric_handler.round_value(value, '0.01')
    assert rounded == Decimal('123.46')
    #test round value invalid
    with pytest.raises(MathError, match="Invalid operation during rounding:"):
        numeric_handler.round_value(Decimal('123.4567'), 'invalid_precision') 