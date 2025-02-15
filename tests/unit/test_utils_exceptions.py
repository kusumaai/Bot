#! /usr/bin/env python3
# tests/unit/test_utils_exceptions.py
"""
Module: tests.unit
Provides unit testing functionality for the exceptions module.
"""
import pytest

from utils.exceptions import (
    BackTestError,
    DatabaseError,
    ExchangeError,
    MathError,
    RatchetError,
    RiskError,
    ValidationError,
)


def test_backtest_error():
    with pytest.raises(BackTestError):
        raise BackTestError("Simulated backtesting failure")


def test_validation_error():
    with pytest.raises(ValidationError):
        raise ValidationError("Simulated validation error")


def test_database_error():
    with pytest.raises(DatabaseError):
        raise DatabaseError("Simulated database error")


def test_exchange_error():
    with pytest.raises(ExchangeError):
        raise ExchangeError("Simulated exchange error")


def test_math_error():
    with pytest.raises(MathError):
        raise MathError("Simulated math error")


def test_ratchet_error():
    with pytest.raises(RatchetError):
        raise RatchetError("Simulated ratchet error")


def test_risk_error():
    with pytest.raises(RiskError):
        raise RiskError("Simulated risk error")
