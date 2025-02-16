#! /usr/bin/env python3
# src/signals/validation.py
"""
Module: src.signals
Provides signal validation with comprehensive validation chains and error recovery.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from bot_types.base_types import Validatable, ValidationResult
from signals.base_types import BaseSignal
from utils.error_handler import handle_error


@dataclass
class SignalValidator(Validatable):
    """Centralized signal validation with enhanced validation chains"""

    logger: Optional[logging.Logger] = None
    _validation_history: List[Dict[str, Any]] = None
    _max_history_size: int = 1000
    _validation_thresholds: Dict[str, Any] = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        if self._validation_history is None:
            self._validation_history = []
        if self._validation_thresholds is None:
            self._validation_thresholds = {
                "min_strength": Decimal("0.1"),
                "max_age_minutes": 5,
                "min_probability": Decimal("0.6"),
                "max_consecutive_failures": 3,
            }

    def validate_signal(self, signal: BaseSignal) -> ValidationResult:
        """Validate trading signal with comprehensive checks"""
        try:
            # Basic validation chain
            basic_validations = [
                (signal.direction in ["long", "short"], "Invalid signal direction"),
                (
                    Decimal("0") <= signal.strength <= Decimal("1"),
                    "Signal strength must be between 0 and 1",
                ),
                (
                    signal.timestamp <= datetime.now(),
                    "Signal timestamp cannot be in future",
                ),
                (
                    self._validate_signal_metadata(signal.metadata),
                    "Invalid signal metadata",
                ),
            ]

            for condition, message in basic_validations:
                if not condition:
                    self._record_validation_failure(signal, message)
                    return ValidationResult(is_valid=False, error_message=message)

            # Enhanced validation chain
            enhanced_validations = self._perform_enhanced_validations(signal)
            if not enhanced_validations.is_valid:
                self._record_validation_failure(
                    signal, enhanced_validations.error_message
                )
                return enhanced_validations

            # Record successful validation
            self._record_validation_success(signal)
            return ValidationResult(is_valid=True)

        except Exception as e:
            error_msg = f"Signal validation failed: {str(e)}"
            handle_error(e, "SignalValidator.validate_signal", self.logger)
            self._record_validation_failure(signal, error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)

    def _validate_signal_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate signal metadata with enhanced checks"""
        try:
            # Required fields check
            required_fields = ["timeframe", "model_version", "probability"]
            if not all(field in metadata for field in required_fields):
                return False

            # Probability threshold check
            probability = Decimal(str(metadata["probability"]))
            if probability < self._validation_thresholds["min_probability"]:
                return False

            # Model version format check
            if (
                not isinstance(metadata["model_version"], str)
                or not metadata["model_version"].strip()
            ):
                return False

            # Timeframe format check
            valid_timeframes = {"1m", "5m", "15m", "1h", "4h", "1d"}
            if metadata["timeframe"] not in valid_timeframes:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Metadata validation error: {e}")
            return False

    def _perform_enhanced_validations(self, signal: BaseSignal) -> ValidationResult:
        """Perform additional validation checks"""
        try:
            # Signal strength threshold
            if signal.strength < self._validation_thresholds["min_strength"]:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Signal strength below threshold: {signal.strength}",
                )

            # Signal age check
            age = datetime.now() - signal.timestamp
            if age > timedelta(minutes=self._validation_thresholds["max_age_minutes"]):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Signal too old: {age.total_seconds() / 60:.1f} minutes",
                )

            # Check consecutive failures for this signal type
            recent_failures = self._get_recent_failures(signal)
            if (
                recent_failures
                >= self._validation_thresholds["max_consecutive_failures"]
            ):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Too many consecutive failures: {recent_failures}",
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            self.logger.error(f"Enhanced validation error: {e}")
            return ValidationResult(
                is_valid=False, error_message=f"Enhanced validation failed: {str(e)}"
            )

    def _record_validation_failure(self, signal: BaseSignal, error_message: str):
        """Record validation failure with details"""
        self._validation_history.append(
            {
                "timestamp": datetime.now(),
                "signal_type": signal.__class__.__name__,
                "success": False,
                "error": error_message,
                "metadata": signal.metadata,
            }
        )
        self._trim_history()

    def _record_validation_success(self, signal: BaseSignal):
        """Record successful validation"""
        self._validation_history.append(
            {
                "timestamp": datetime.now(),
                "signal_type": signal.__class__.__name__,
                "success": True,
                "metadata": signal.metadata,
            }
        )
        self._trim_history()

    def _get_recent_failures(self, signal: BaseSignal) -> int:
        """Count recent consecutive failures for this signal type"""
        signal_type = signal.__class__.__name__
        consecutive = 0

        for entry in reversed(self._validation_history):
            if entry["signal_type"] != signal_type:
                continue
            if entry["success"]:
                break
            consecutive += 1

        return consecutive

    def _trim_history(self):
        """Trim validation history to maximum size"""
        if len(self._validation_history) > self._max_history_size:
            self._validation_history = self._validation_history[
                -self._max_history_size :
            ]

    def set_validation_threshold(self, threshold_name: str, value: Any):
        """Update validation threshold"""
        if threshold_name not in self._validation_thresholds:
            raise ValueError(f"Unknown threshold: {threshold_name}")
        self._validation_thresholds[threshold_name] = value

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self._validation_history:
            return {"total": 0, "success_rate": 0, "recent_failures": 0}

        total = len(self._validation_history)
        successes = sum(1 for entry in self._validation_history if entry["success"])
        recent = self._validation_history[-100:]  # Last 100 entries
        recent_failures = sum(1 for entry in recent if not entry["success"])

        return {
            "total": total,
            "success_rate": successes / total if total > 0 else 0,
            "recent_failures": recent_failures,
            "thresholds": dict(self._validation_thresholds),
        }
