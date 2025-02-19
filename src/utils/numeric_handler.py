#! /usr/bin/env python3
# src/utils/numeric_handler.py
"""
Module: src.utils
Provides numerical handling functionality.
"""
import logging
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation
from typing import Any, Optional, Union

from src.utils.error_handler import handle_error_async
from src.utils.exceptions import MathError

logger = logging.getLogger(__name__)


class NumericHandler:
    """Handles numerical operations and conversions"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the numeric handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.PRECISION = 8
        self.ROUNDING = ROUND_HALF_UP

    def to_decimal(self, value: Any) -> Optional[Decimal]:
        """
        Convert a value to a Decimal.

        Args:
            value: Value to convert

        Returns:
            Decimal value or None if conversion fails
        """
        try:
            if isinstance(value, Decimal):
                return value
            if value is None:
                return None
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as e:
            error_msg = f"Failed to convert value to Decimal: {value} - [{e.__class__.__name__}]"
            self.logger.error(error_msg)
            return None

    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        """
        Safely divide two Decimal values.

        Args:
            numerator: The numerator
            denominator: The denominator

        Returns:
            Result of division

        Raises:
            DivisionByZero: If denominator is zero
        """
        if denominator == Decimal("0"):
            raise DivisionByZero("Division by zero.")
        return numerator / denominator

    def percentage_to_decimal(self, value: Any) -> Decimal:
        """
        Convert a percentage to a decimal value.

        Args:
            value: Percentage value

        Returns:
            Decimal value

        Raises:
            InvalidOperation: If conversion fails
        """
        try:
            return Decimal(str(value)) / Decimal("100")
        except (InvalidOperation, ValueError) as e:
            raise e

    def convert_to_decimal(self, value: Any) -> Decimal:
        """
        Convert a value to Decimal safely.

        Args:
            value: Value to convert

        Returns:
            Decimal value

        Raises:
            MathError: If conversion fails
        """
        try:
            if isinstance(value, Decimal):
                return value
            if value is None:
                raise MathError("Cannot convert None to Decimal")
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as e:
            error_msg = f"Invalid operation during conversion to Decimal: {str(e)}"
            self.logger.error(error_msg)
            raise MathError(error_msg)

    def round_value(
        self, value: Decimal, precision: Union[str, int] = "0.0001"
    ) -> Decimal:
        """
        Rounds the decimal value to the specified precision.

        Args:
            value: The Decimal value to round
            precision: String representation of precision (e.g. "0.0001" for 4 decimal places)
                      or integer number of decimal places

        Returns:
            Rounded Decimal value

        Raises:
            MathError: If rounding fails
        """
        try:
            if not isinstance(value, Decimal):
                value = self.convert_to_decimal(value)

            # Handle integer precision
            if isinstance(precision, int):
                quantize_str = f"0.{'0' * precision}"
            else:
                # Handle string precision
                precision = str(precision)
                if not precision.replace(".", "").replace("0", "").strip() == "":
                    if precision.count(".") > 1:
                        raise MathError(f"Invalid precision format: {precision}")
                    try:
                        # Try to convert to Decimal to validate format
                        Decimal(precision)
                        quantize_str = precision
                    except InvalidOperation:
                        raise MathError(f"Invalid precision format: {precision}")
                else:
                    quantize_str = precision

            return value.quantize(Decimal(quantize_str), rounding=self.ROUNDING)
        except (InvalidOperation, ValueError) as e:
            error_msg = f"Invalid operation during rounding: {str(e)}"
            self.logger.error(error_msg)
            raise MathError(error_msg)

    def normalize(self, value: Decimal, precision: Optional[int] = None) -> Decimal:
        """
        Normalize a Decimal value to a specified precision.

        Args:
            value: Value to normalize
            precision: Number of decimal places (defaults to self.PRECISION)

        Returns:
            Normalized Decimal value
        """
        try:
            if not isinstance(value, Decimal):
                value = self.convert_to_decimal(value)
            precision = precision or self.PRECISION
            quantize_str = f"0.{'0' * precision}"
            return value.quantize(Decimal(quantize_str), rounding=self.ROUNDING)
        except (InvalidOperation, ValueError) as e:
            error_msg = f"Invalid operation during normalization: {str(e)}"
            self.logger.error(error_msg)
            raise MathError(error_msg)

    def round_decimal(self, value: Any, places: int = 8) -> Decimal:
        """Round Decimal to specified places"""
        decimal_val = self.to_decimal(value)
        if decimal_val is not None:
            return decimal_val.quantize(Decimal(f'0.{"0" * places}'))
        else:
            return Decimal("0")


# add other numerical methods as needed
# def name(one, two):
#    """
#    Purpose: one
#    """
