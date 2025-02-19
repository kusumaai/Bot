from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass
class ValidationResult:
    """Result of a signal validation."""

    is_valid: bool
    error_message: str


class SignalValidator:
    def __init__(self, validation_thresholds: dict, logger):
        self._validation_thresholds = validation_thresholds
        self.logger = logger
        self._consecutive_failures = 0

    def validate_signal(self, signal: Any) -> ValidationResult:
        """
        Validate a trading signal.

        :param signal: The signal to validate
        :return: ValidationResult with validation status and error message
        """
        try:
            # Reset consecutive failures if signal is valid
            if self._validate_signal_basics(signal):
                self._consecutive_failures = 0
                return ValidationResult(is_valid=True, error_message="")

            # Increment consecutive failures
            self._consecutive_failures += 1

            # Check for too many consecutive failures
            if (
                self._consecutive_failures
                >= self._validation_thresholds["max_consecutive_failures"]
            ):
                return ValidationResult(
                    is_valid=False, error_message="Too many consecutive failures"
                )

            # Return specific validation error
            return ValidationResult(
                is_valid=False, error_message=self._get_validation_error(signal)
            )

        except Exception as e:
            self.logger.error(f"Signal validation failed: {e.__class__.__name__}")
            return ValidationResult(
                is_valid=False, error_message=f"Signal validation failed: {str(e)}"
            )

    def _validate_signal_basics(self, signal: Any) -> bool:
        """Validate basic signal properties."""
        try:
            # Check signal direction
            if not hasattr(signal, "direction") or signal.direction not in [
                "buy",
                "sell",
            ]:
                return False

            # Check signal strength
            if (
                not hasattr(signal, "strength")
                or signal.strength < self._validation_thresholds["min_strength"]
            ):
                return False

            # Check signal timestamp
            if not hasattr(signal, "timestamp") or not isinstance(
                signal.timestamp, datetime
            ):
                return False

            # Check signal age
            age = datetime.now() - signal.timestamp
            if (
                age.total_seconds()
                > self._validation_thresholds["max_age_minutes"] * 60
            ):
                return False

            # Check signal metadata
            if not hasattr(signal, "metadata") or not isinstance(signal.metadata, dict):
                return False

            # Validate probability if present
            if "probability" in signal.metadata:
                try:
                    prob = Decimal(str(signal.metadata["probability"]))
                    if prob < self._validation_thresholds["min_probability"]:
                        return False
                except Exception:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Basic signal validation failed: {e}")
            return False

    def _get_validation_error(self, signal: Any) -> str:
        """Get specific validation error message."""
        if not hasattr(signal, "direction") or signal.direction not in ["buy", "sell"]:
            return "Invalid signal direction"

        if (
            not hasattr(signal, "strength")
            or signal.strength < self._validation_thresholds["min_strength"]
        ):
            return "Invalid signal strength"

        if not hasattr(signal, "timestamp") or not isinstance(
            signal.timestamp, datetime
        ):
            return "Invalid signal timestamp"

        age = datetime.now() - signal.timestamp
        if age.total_seconds() > self._validation_thresholds["max_age_minutes"] * 60:
            return "Signal too old"

        if not hasattr(signal, "metadata") or not isinstance(signal.metadata, dict):
            return "Invalid signal metadata"

        if "probability" in signal.metadata:
            try:
                prob = Decimal(str(signal.metadata["probability"]))
                if prob < self._validation_thresholds["min_probability"]:
                    return "Signal probability too low"
            except Exception:
                return "Invalid signal probability"

        return "Unknown validation error"
