from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Represents the result of a validation operation."""

    is_valid: bool
    error_message: str = ""
