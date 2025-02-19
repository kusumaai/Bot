#!/usr/bin/env python3
# src/utils/data_validator.py
"""
Module: src.utils
Provides data validation functionality.
"""

import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.exceptions import ValidationError
from utils.logger import get_logger
from utils.numeric_handler import NumericHandler

logger = get_logger(__name__)
nh = NumericHandler()


class DataValidator:
    """Validates data structures and types."""

    def __init__(self, logger=None):
        """Initialize data validator."""
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def validate_decimal(
        value: Union[str, float, Decimal],
        field_name: str,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None,
        allow_zero: bool = False,
    ) -> Decimal:
        """
        Validate and convert a value to Decimal.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_value: Optional minimum value
            max_value: Optional maximum value
            allow_zero: Whether to allow zero value

        Returns:
            Validated Decimal value

        Raises:
            ValidationError: If validation fails
        """
        try:
            dec_value = nh.to_decimal(value)

            if not allow_zero and dec_value == Decimal("0"):
                raise ValidationError(f"{field_name} cannot be zero")

            if min_value is not None and dec_value < min_value:
                raise ValidationError(f"{field_name} must be greater than {min_value}")

            if max_value is not None and dec_value > max_value:
                raise ValidationError(f"{field_name} must be less than {max_value}")

            return dec_value

        except InvalidOperation:
            raise ValidationError(f"Invalid {field_name} value: {value}")
        except Exception as e:
            raise ValidationError(f"Error validating {field_name}: {e}")

    @staticmethod
    def validate_string(
        value: Any,
        field_name: str,
        allowed_values: Optional[List[str]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Validate a string value.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            allowed_values: Optional list of allowed values
            min_length: Optional minimum length
            max_length: Optional maximum length

        Returns:
            Validated string value

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")

        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters"
            )

        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters"
            )

        if allowed_values is not None and value not in allowed_values:
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(allowed_values)}"
            )

        return value

    @staticmethod
    def validate_dict(
        data: Dict[str, Any],
        required_fields: List[str],
        field_name: str = "data",
    ) -> None:
        """
        Validate a dictionary has required fields.

        Args:
            data: Dictionary to validate
            required_fields: List of required field names
            field_name: Name of dictionary for error messages

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(f"{field_name} must be a dictionary")

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"{field_name} missing required fields: {', '.join(missing_fields)}"
            )

    @staticmethod
    def validate_list(
        data: List[Any],
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        item_type: Optional[type] = None,
    ) -> None:
        """
        Validate a list.

        Args:
            data: List to validate
            field_name: Name of list for error messages
            min_length: Optional minimum length
            max_length: Optional maximum length
            item_type: Optional type that all items must be

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, list):
            raise ValidationError(f"{field_name} must be a list")

        if min_length is not None and len(data) < min_length:
            raise ValidationError(f"{field_name} must have at least {min_length} items")

        if max_length is not None and len(data) > max_length:
            raise ValidationError(f"{field_name} must have at most {max_length} items")

        if item_type is not None:
            invalid_items = [
                i for i, item in enumerate(data) if not isinstance(item, item_type)
            ]
            if invalid_items:
                raise ValidationError(
                    f"{field_name} items at indices {invalid_items} must be of type {item_type.__name__}"
                )

    def validate_trade_parameters(
        self, trade_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate trade parameters.

        Args:
            trade_params: Dictionary containing trade parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ["symbol", "side", "price", "amount"]
            missing_fields = [
                field for field in required_fields if field not in trade_params
            ]
            if missing_fields:
                return (
                    False,
                    f"Missing required trade parameters: {', '.join(missing_fields)}",
                )

            # Validate side
            if trade_params["side"] not in ["buy", "sell"]:
                return False, "Trade side must be 'buy' or 'sell'"

            # Validate amount and price
            try:
                amount = Decimal(str(trade_params["amount"]))
                price = Decimal(str(trade_params["price"]))
            except (TypeError, ValueError):
                raise AttributeError("Invalid decimal conversion")

            if amount <= Decimal("0"):
                return False, "Trade amount must be positive"
            if price <= Decimal("0"):
                return False, "Trade price must be positive"

            return True, None

        except Exception as e:
            self.logger.error(f"Trade parameter validation error: {e}")
            return False, str(e)
