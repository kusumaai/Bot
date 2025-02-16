"""
Test suite for enhanced signal validation functionality.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from bot_types.base_types import ValidationResult
from signals.base_types import BaseSignal
from signals.validation import SignalValidator


class TestSignal(BaseSignal):
    """Test signal implementation."""

    def __init__(
        self,
        direction: str,
        strength: Decimal,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ):
        self.direction = direction
        self.strength = strength
        self.timestamp = timestamp
        self.metadata = metadata


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_logger")


@pytest.fixture
def validator(logger):
    """Create a signal validator instance."""
    return SignalValidator(logger=logger)


@pytest.fixture
def valid_metadata():
    """Create valid signal metadata."""
    return {"timeframe": "5m", "model_version": "1.0.0", "probability": "0.85"}


@pytest.fixture
def valid_signal(valid_metadata):
    """Create a valid test signal."""
    return TestSignal(
        direction="long",
        strength=Decimal("0.75"),
        timestamp=datetime.now(),
        metadata=valid_metadata,
    )


def test_basic_validation_chain(validator, valid_signal):
    """Test basic validation chain functionality."""
    result = validator.validate_signal(valid_signal)
    assert result.is_valid
    assert not result.error_message

    # Test invalid direction
    invalid_signal = TestSignal(
        direction="invalid",
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=valid_signal.metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Invalid signal direction" in result.error_message

    # Test invalid strength
    invalid_signal = TestSignal(
        direction="long",
        strength=Decimal("-0.1"),
        timestamp=valid_signal.timestamp,
        metadata=valid_signal.metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Signal strength must be between 0 and 1" in result.error_message

    # Test future timestamp
    invalid_signal = TestSignal(
        direction="long",
        strength=valid_signal.strength,
        timestamp=datetime.now() + timedelta(minutes=5),
        metadata=valid_signal.metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Signal timestamp cannot be in future" in result.error_message


def test_metadata_validation(validator, valid_signal):
    """Test metadata validation functionality."""
    # Test missing required field
    invalid_metadata = valid_signal.metadata.copy()
    del invalid_metadata["timeframe"]
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=invalid_metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Invalid signal metadata" in result.error_message

    # Test invalid probability
    invalid_metadata = valid_signal.metadata.copy()
    invalid_metadata["probability"] = "0.5"  # Below threshold
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=invalid_metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Invalid signal metadata" in result.error_message

    # Test invalid model version
    invalid_metadata = valid_signal.metadata.copy()
    invalid_metadata["model_version"] = ""
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=invalid_metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Invalid signal metadata" in result.error_message

    # Test invalid timeframe
    invalid_metadata = valid_signal.metadata.copy()
    invalid_metadata["timeframe"] = "invalid"
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=invalid_metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Invalid signal metadata" in result.error_message


def test_enhanced_validation_chain(validator, valid_signal):
    """Test enhanced validation chain functionality."""
    # Test signal strength threshold
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=Decimal("0.05"),  # Below threshold
        timestamp=valid_signal.timestamp,
        metadata=valid_signal.metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Signal strength below threshold" in result.error_message

    # Test signal age
    old_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=datetime.now() - timedelta(minutes=10),  # Too old
        metadata=valid_signal.metadata,
    )
    result = validator.validate_signal(old_signal)
    assert not result.is_valid
    assert "Signal too old" in result.error_message


def test_consecutive_failures(validator, valid_signal):
    """Test consecutive failures tracking."""
    # Create an invalid signal that will fail validation
    invalid_signal = TestSignal(
        direction="invalid",
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=valid_signal.metadata,
    )

    # Generate multiple failures
    for _ in range(validator._validation_thresholds["max_consecutive_failures"]):
        result = validator.validate_signal(invalid_signal)
        assert not result.is_valid

    # Verify that the next validation fails due to too many consecutive failures
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Too many consecutive failures" in result.error_message

    # Verify that a successful validation resets the failure count
    result = validator.validate_signal(valid_signal)
    assert result.is_valid


def test_validation_history(validator, valid_signal):
    """Test validation history tracking."""
    # Generate some validation history
    validator.validate_signal(valid_signal)  # Success
    invalid_signal = TestSignal(
        direction="invalid",
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=valid_signal.metadata,
    )
    validator.validate_signal(invalid_signal)  # Failure

    # Get validation stats
    stats = validator.get_validation_stats()
    assert stats["total"] == 2
    assert 0 < stats["success_rate"] < 1
    assert stats["recent_failures"] > 0
    assert "thresholds" in stats


def test_validation_threshold_management(validator):
    """Test validation threshold management."""
    # Test setting valid threshold
    new_strength = Decimal("0.2")
    validator.set_validation_threshold("min_strength", new_strength)
    assert validator._validation_thresholds["min_strength"] == new_strength

    # Test setting invalid threshold
    with pytest.raises(ValueError):
        validator.set_validation_threshold("invalid_threshold", 0.5)


def test_validation_error_handling(validator, valid_signal):
    """Test validation error handling."""
    # Test exception in metadata validation
    invalid_metadata = valid_signal.metadata.copy()
    invalid_metadata["probability"] = "invalid"
    invalid_signal = TestSignal(
        direction=valid_signal.direction,
        strength=valid_signal.strength,
        timestamp=valid_signal.timestamp,
        metadata=invalid_metadata,
    )
    result = validator.validate_signal(invalid_signal)
    assert not result.is_valid
    assert "Signal validation failed" in result.error_message

    # Test history trimming
    original_size = validator._max_history_size
    validator._max_history_size = 2
    for _ in range(5):
        validator.validate_signal(valid_signal)
    assert len(validator._validation_history) == 2
    validator._max_history_size = original_size
