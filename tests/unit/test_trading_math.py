#! /usr/bin/env python3
# tests/unit/test_trading_math_handler.py
"""
Module: tests.unit
Provides unit testing functionality for the trading math handler module.
"""
from decimal import Decimal

import pytest

from trading.math import MathHandler
from utils.exceptions import MathError
from utils.numeric_handler import NumericHandler


@pytest.fixture
def numeric_handler():
    return NumericHandler()


@pytest.fixture
def math_handler():
    return MathHandler()


@pytest.mark.asyncio
async def test_safe_divide(math_handler):
    a = Decimal("10")
    b = Decimal("2")
    assert math_handler.safe_divide(a, b) == Decimal("5")

    b = Decimal("0")
    assert math_handler.safe_divide(a, b) == Decimal("0")


@pytest.mark.asyncio
async def test_safe_divide_invalid(math_handler):
    a = Decimal("10")
    b = "invalid"
    with pytest.raises(TypeError):
        math_handler.safe_divide(a, b)


def test_to_decimal(numeric_handler):
    assert numeric_handler.to_decimal("123.45") == Decimal("123.45")
    assert numeric_handler.to_decimal(123.45) == Decimal("123.45")
    assert numeric_handler.to_decimal("abc") is None


def test_percentage_to_decimal(numeric_handler):
    result = numeric_handler.percentage_to_decimal(Decimal("50"))
    assert result == Decimal("0.5")

    with pytest.raises(MathError):
        numeric_handler.percentage_to_decimal(Decimal("invalid"))
