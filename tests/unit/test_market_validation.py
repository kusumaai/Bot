"""
Test suite for enhanced market data validation functionality.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.bot_types.base_types import ValidationResult
from src.risk.limits import RiskLimits
from src.risk.validation import MarketDataValidation
from src.utils.error_handler import ValidationError


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_logger")


@pytest.fixture
def risk_limits():
    """Create test risk limits."""
    return RiskLimits(
        emergency_stop_pct=Decimal("0.1"),
        max_position_size=Decimal("1000"),
        max_volatility=Decimal("0.2"),
    )


@pytest.fixture
def validator(logger, risk_limits):
    """Create a market data validator instance."""
    return MarketDataValidation(risk_limits=risk_limits, logger=logger)


@pytest.fixture
def valid_market_data():
    """Create valid market data."""
    return {
        "symbol": "BTC/USDT",
        "price": "50000.00",
        "volume": "10.5",
        "timestamp": datetime.now().timestamp(),
        "bid": "49990.00",
        "ask": "50010.00",
    }


def test_basic_validation_chain(validator, valid_market_data):
    """Test basic validation chain functionality."""
    # Test valid data
    result = validator.validate_market_data(valid_market_data)
    assert result.is_valid
    assert not result.error_message

    # Test missing required field
    invalid_data = valid_market_data.copy()
    del invalid_data["price"]
    result = validator.validate_market_data(invalid_data)
    assert not result.is_valid
    assert "Missing required fields" in result.error_message

    # Test invalid numeric values
    invalid_data = valid_market_data.copy()
    invalid_data["price"] = "invalid"
    result = validator.validate_market_data(invalid_data)
    assert not result.is_valid
    assert "Invalid numeric values" in result.error_message

    # Test negative price
    invalid_data = valid_market_data.copy()
    invalid_data["price"] = "-1"
    result = validator.validate_market_data(invalid_data)
    assert not result.is_valid
    assert "Invalid price" in result.error_message

    # Test zero volume
    invalid_data = valid_market_data.copy()
    invalid_data["volume"] = "0"
    result = validator.validate_market_data(invalid_data)
    assert not result.is_valid
    assert "Invalid volume" in result.error_message


def test_enhanced_validation_chain(validator, valid_market_data):
    """Test enhanced validation chain functionality."""
    # Test stale data
    stale_data = valid_market_data.copy()
    stale_data["timestamp"] = (datetime.now() - timedelta(seconds=10)).timestamp()
    result = validator.validate_market_data(stale_data)
    assert not result.is_valid
    assert "Stale market data" in result.error_message

    # Test excessive price change
    price_change_data = valid_market_data.copy()
    price_change_data["last_price"] = "45000.00"  # 10%+ change
    result = validator.validate_market_data(price_change_data)
    assert not result.is_valid
    assert "Excessive price change" in result.error_message

    # Test excessive volume change
    volume_change_data = valid_market_data.copy()
    volume_change_data["last_volume"] = "5.0"  # >50% change
    result = validator.validate_market_data(volume_change_data)
    assert not result.is_valid
    assert "Excessive volume change" in result.error_message

    # Test excessive spread
    spread_data = valid_market_data.copy()
    spread_data["bid"] = "47500.00"  # >5% spread
    spread_data["ask"] = "50000.00"
    result = validator.validate_market_data(spread_data)
    assert not result.is_valid
    assert "Excessive spread" in result.error_message


def test_risk_limit_validation(validator, valid_market_data):
    """Test risk limit validation functionality."""
    # Test emergency stop
    emergency_data = valid_market_data.copy()
    emergency_data["drawdown"] = str(validator.risk_limits.emergency_stop_pct)
    result = validator.validate_market_data(emergency_data)
    assert not result.is_valid
    assert "Emergency stop triggered" in result.error_message

    # Test position size limit
    position_data = valid_market_data.copy()
    position_data["position_size"] = str(validator.risk_limits.max_position_size + 1)
    result = validator.validate_market_data(position_data)
    assert not result.is_valid
    assert "Position size exceeds limit" in result.error_message

    # Test volatility limit
    volatility_data = valid_market_data.copy()
    volatility_data["volatility"] = str(
        validator.risk_limits.max_volatility + Decimal("0.1")
    )
    result = validator.validate_market_data(volatility_data)
    assert not result.is_valid
    assert "Volatility exceeds limit" in result.error_message


def test_validation_history(validator, valid_market_data):
    """Test validation history tracking."""
    # Generate some validation history
    validator.validate_market_data(valid_market_data)  # Success
    invalid_data = valid_market_data.copy()
    del invalid_data["price"]
    validator.validate_market_data(invalid_data)  # Failure

    # Get validation stats
    stats = validator.get_validation_stats()
    assert stats["total"] == 2
    assert 0 < stats["success_rate"] < 1
    assert stats["recent_failures"] > 0
    assert "thresholds" in stats


def test_validation_threshold_management(validator):
    """Test validation threshold management."""
    # Test setting valid threshold
    new_spread = Decimal("0.1")
    validator.set_validation_threshold("max_spread", new_spread)
    assert validator._validation_thresholds["max_spread"] == new_spread

    # Test setting invalid threshold
    with pytest.raises(ValueError):
        validator.set_validation_threshold("invalid_threshold", 0.5)


def test_validation_error_handling(validator, valid_market_data):
    """Test validation error handling."""
    # Test exception in numeric conversion
    invalid_data = valid_market_data.copy()
    invalid_data["price"] = object()  # Will cause conversion error
    result = validator.validate_market_data(invalid_data)
    assert not result.is_valid
    assert "validation failed" in result.error_message

    # Test history trimming
    original_size = validator._max_history_size
    validator._max_history_size = 2
    for _ in range(5):
        validator.validate_market_data(valid_market_data)
    assert len(validator._validation_history) == 2
    validator._max_history_size = original_size


def test_empty_data_handling(validator):
    """Test handling of empty market data."""
    result = validator.validate_market_data({})
    assert not result.is_valid
    assert "Empty market data received" in result.error_message

    result = validator.validate_market_data(None)
    assert not result.is_valid
    assert "Empty market data received" in result.error_message


def test_validation_stats_empty_history(validator):
    """Test validation stats with empty history."""
    stats = validator.get_validation_stats()
    assert stats["total"] == 0
    assert stats["success_rate"] == 0
    assert stats["recent_failures"] == 0
    assert isinstance(stats["thresholds"], dict)
