#! /usr/bin/env python3
# tests/unit/test_utils_numeric_handler.py
"""
Module: tests.unit
Provides unit testing functionality for the numeric handler module.
"""
import logging
from decimal import Decimal, DivisionByZero, InvalidOperation

import pytest

from utils.numeric_handler import NumericHandler


@pytest.fixture
def numeric_handler():
    """Provide a NumericHandler instance."""
    return NumericHandler()


def test_to_decimal_valid(numeric_handler):
    """Test successful conversion to Decimal."""
    assert numeric_handler.to_decimal("100.5") == Decimal("100.5")
    assert numeric_handler.to_decimal(100.5) == Decimal("100.5")
    assert numeric_handler.to_decimal(Decimal("100.5")) == Decimal("100.5")


def test_to_decimal_invalid(numeric_handler, caplog):
    """Test conversion failures to Decimal."""
    with caplog.at_level(logging.ERROR):
        assert numeric_handler.to_decimal("invalid") is None
        assert "Failed to convert value to Decimal: invalid - " in caplog.text

        assert numeric_handler.to_decimal(None) is None
        assert "Failed to convert value to Decimal: None - " in caplog.text


def test_calculate_percentage_change(numeric_handler):
    """Test calculation of percentage change."""
    old_value = Decimal("100")
    new_value = Decimal("110")
    change = numeric_handler.calculate_percentage_change(old_value, new_value)
    assert change == Decimal("0.10")  # 10% increase

    old_value = Decimal("200")
    new_value = Decimal("180")
    change = numeric_handler.calculate_percentage_change(old_value, new_value)
    assert change == Decimal("-0.10")  # 10% decrease


def test_calculate_percentage_change_division_by_zero(numeric_handler):
    """Test percentage change calculation with division by zero."""
    old_value = Decimal("0")
    new_value = Decimal("100")
    with pytest.raises(DivisionByZero):
        numeric_handler.calculate_percentage_change(old_value, new_value)
