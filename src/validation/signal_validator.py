import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from validation.validation_result import ValidationResult


class SignalValidator:
    """Validates trading signals."""

    def __init__(self, logger=None):
        """Initialize the signal validator."""
        self.logger = logger or logging.getLogger(__name__)
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
        self._validation_history = []

    def validate_signal(self, signal: Any) -> ValidationResult:
        """
        Validate a trading signal.

        Args:
            signal: The signal to validate

        Returns:
            ValidationResult with validation status and error message
        """
        try:
            # Check consecutive failures first
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._consecutive_failures += 1
                return ValidationResult(False, "Too many consecutive failures")

            # Validate basic signal properties
            if not self._validate_signal_basics(signal):
                self._consecutive_failures += 1
                return ValidationResult(False, self._get_validation_error(signal))

            # Validate metadata
            try:
                if not isinstance(signal.metadata.get("probability"), Decimal):
                    self._consecutive_failures += 1
                    return ValidationResult(
                        False, "Signal validation failed: Invalid probability value"
                    )
            except Exception as e:
                self.logger.error(f"Metadata validation error: {e.__class__.__name__}")
                self._consecutive_failures += 1
                return ValidationResult(False, f"Signal validation failed: {str(e)}")

            # Signal is valid, reset consecutive failures
            self._consecutive_failures = 0
            return ValidationResult(True, "")

        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            self._consecutive_failures += 1
            return ValidationResult(False, f"Signal validation failed: {str(e)}")

    def _validate_signal_basics(self, signal: Any) -> bool:
        """Validate basic signal properties."""
        if not hasattr(signal, "direction") or signal.direction not in ["buy", "sell"]:
            return False
        if not hasattr(signal, "strength") or not isinstance(signal.strength, Decimal):
            return False
        if not hasattr(signal, "timestamp") or not isinstance(
            signal.timestamp, datetime
        ):
            return False
        if not hasattr(signal, "metadata") or not isinstance(signal.metadata, dict):
            return False
        return True

    def _get_validation_error(self, signal: Any) -> str:
        """Get specific validation error message."""
        if not hasattr(signal, "direction") or signal.direction not in ["buy", "sell"]:
            return "Invalid signal direction"
        if not hasattr(signal, "strength") or not isinstance(signal.strength, Decimal):
            return "Invalid signal strength"
        if not hasattr(signal, "timestamp") or not isinstance(
            signal.timestamp, datetime
        ):
            return "Invalid signal timestamp"
        if not hasattr(signal, "metadata") or not isinstance(signal.metadata, dict):
            return "Invalid signal metadata"
        return "Unknown validation error"
