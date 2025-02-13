import pytest
from src.utils.exceptions import BackTestError, ValidationError

def test_backtest_error():
    with pytest.raises(BackTestError):
        raise BackTestError("Simulated backtesting failure")

def test_validation_error():
    with pytest.raises(ValidationError):
        raise ValidationError("Simulated validation error")