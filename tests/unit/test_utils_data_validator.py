#! /usr/bin/env python3
# tests/unit/test_utils_data_validator.py
"""
Module: tests.unit
Provides unit testing functionality for the data validator module.
"""
from decimal import Decimal

import pytest

from utils.data_validator import DataValidator
from utils.error_handler import ValidationError


@pytest.fixture
def data_validator():
    """Provide a DataValidator instance."""
    return DataValidator()


def test_validate_trade_parameters_valid(data_validator):
    """Test validation of valid trade parameters."""
    trade_params = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": Decimal("50000"),
        "amount": Decimal("0.1"),
    }
    is_valid, error = data_validator.validate_trade_parameters(trade_params)
    assert is_valid is True
    assert error is None


def test_validate_trade_parameters_invalid_price(data_validator):
    """Test validation fails with invalid price."""
    trade_params = {
        "symbol": "ETH/USDT",
        "side": "sell",
        "price": Decimal("-3000"),
        "amount": Decimal("1"),
    }
    is_valid, error = data_validator.validate_trade_parameters(trade_params)
    assert is_valid is False
    assert error == "Trade price must be positive"


def test_validate_trade_parameters_invalid_side(data_validator):
    """Test validation fails with invalid side."""
    trade_params = {
        "symbol": "SOL/USDT",
        "side": "hold",
        "price": Decimal("100"),
        "amount": Decimal("10"),
    }
    is_valid, error = data_validator.validate_trade_parameters(trade_params)
    assert is_valid is False
    assert error == "Trade side must be 'buy' or 'sell'"


def test_validate_trade_parameters_missing_field(data_validator):
    """Test validation fails with missing fields."""
    trade_params = {
        "symbol": "ADA/USDT",
        "side": "buy",
        # Missing price and amount
    }
    is_valid, error = data_validator.validate_trade_parameters(trade_params)
    assert is_valid is False
    assert error == "Missing required trade parameters: price, amount"


def test_validate_trade_parameters_non_decimal_amount(data_validator):
    """Test validation fails with non-decimal amount."""
    trade_params = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": Decimal("50000"),
        "amount": "0.1",  # Should be Decimal
    }
    with pytest.raises(AttributeError):
        data_validator.validate_trade_parameters(trade_params)
