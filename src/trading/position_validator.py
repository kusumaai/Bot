#!/usr/bin/env python3
# src/trading/position_validator.py
"""
Module: src.trading
Provides unified position validation functionality.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

from src.bot_types.base_types import Validatable, ValidationResult
from src.bot_types.position_types import PositionValidationConfig
from src.utils.error_handler import handle_error, handle_error_async
from src.utils.exceptions import PositionError
from src.utils.logger import get_logger
from src.utils.numeric_handler import NumericHandler

logger = get_logger(__name__)


class PositionValidator:
    """Validates position operations against configured thresholds."""

    def __init__(self, max_position_size: Decimal, min_position_size: Decimal):
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def validate_position(self, position: Any) -> ValidationResult:
        """
        Validate a position against configured thresholds.

        Args:
            position: Position to validate

        Returns:
            ValidationResult indicating if position is valid
        """
        try:
            if not position.symbol:
                return ValidationResult(False, "Position must have a symbol")

            if position.size == 0:
                return ValidationResult(False, "Position size cannot be zero")

            if position.entry_price <= 0:
                return ValidationResult(False, "Entry price must be positive")

            if position.current_price is not None and position.current_price <= 0:
                return ValidationResult(False, "Current price must be positive if set")

            if position.side not in ["buy", "sell"]:
                return ValidationResult(False, f"Invalid side: {position.side}")

            return ValidationResult(True)

        except Exception as e:
            logger.error(f"Position validation error: {str(e)}")
            return ValidationResult(False, str(e))

    def validate_position_update(
        self, position: Any, new_price: Decimal
    ) -> ValidationResult:
        """
        Validate a position update.

        Args:
            position: Position to validate update for
            new_price: New price to validate

        Returns:
            ValidationResult indicating if update is valid
        """
        try:
            if new_price <= 0:
                return ValidationResult(False, "Update price must be positive")

            validation = self.validate_position(position)
            if not validation.is_valid:
                return validation

            return ValidationResult(True)

        except Exception as e:
            logger.error(f"Position update validation error: {str(e)}")
            return ValidationResult(False, str(e))

    def validate_stop_loss(
        self, position: Any, stop_price: Decimal
    ) -> ValidationResult:
        """
        Validate a stop loss price for a position.

        Args:
            position: Position to validate stop loss for
            stop_price: Stop loss price to validate

        Returns:
            ValidationResult indicating if stop loss is valid
        """
        try:
            if stop_price <= 0:
                return ValidationResult(False, "Stop loss must be positive")

            if position.side == "buy" and stop_price >= position.current_price:
                return ValidationResult(
                    False, "Stop loss must be below current price for long positions"
                )

            if position.side == "sell" and stop_price <= position.current_price:
                return ValidationResult(
                    False, "Stop loss must be above current price for short positions"
                )

            return ValidationResult(True)

        except Exception as e:
            logger.error(f"Stop loss validation error: {str(e)}")
            return ValidationResult(False, str(e))

    def validate_take_profit(
        self, position: Any, take_profit_price: Decimal
    ) -> ValidationResult:
        """
        Validate a take profit price for a position.

        Args:
            position: Position to validate take profit for
            take_profit_price: Take profit price to validate

        Returns:
            ValidationResult indicating if take profit is valid
        """
        try:
            if take_profit_price <= 0:
                return ValidationResult(False, "Take profit must be positive")

            if position.side == "buy" and take_profit_price <= position.current_price:
                return ValidationResult(
                    False, "Take profit must be above current price for long positions"
                )

            if position.side == "sell" and take_profit_price >= position.current_price:
                return ValidationResult(
                    False, "Take profit must be below current price for short positions"
                )

            return ValidationResult(True)

        except Exception as e:
            logger.error(f"Take profit validation error: {str(e)}")
            return ValidationResult(False, str(e))

    async def validate_new_position(
        self,
        symbol: str,
        side: str,
        size: Union[str, float, Decimal],
        entry_price: Union[str, float, Decimal],
        stop_loss: Optional[Union[str, float, Decimal]] = None,
        take_profit: Optional[Union[str, float, Decimal]] = None,
        leverage: Optional[Union[str, float, Decimal]] = None,
        existing_positions: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for opening a new position.

        Args:
            symbol: Trading symbol
            side: Position side (buy/sell)
            size: Position size
            entry_price: Entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            leverage: Optional leverage
            existing_positions: Optional dict of existing positions

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Convert inputs to Decimal
            nh = NumericHandler()
            size_dec = nh.to_decimal(size)
            entry_price_dec = nh.to_decimal(entry_price)
            stop_loss_dec = nh.to_decimal(stop_loss) if stop_loss is not None else None
            take_profit_dec = (
                nh.to_decimal(take_profit) if take_profit is not None else None
            )
            leverage_dec = nh.to_decimal(leverage) if leverage is not None else None

            # Basic validation
            if not symbol:
                return False, "Symbol is required"

            if side.lower() not in ["buy", "sell"]:
                return False, f"Invalid side: {side}"

            if size_dec <= 0:
                return False, "Size must be positive"

            if entry_price_dec <= 0:
                return False, "Entry price must be positive"

            # Size limits
            if size_dec < self.min_position_size:
                return False, f"Size {size_dec} below minimum {self.min_position_size}"

            if size_dec > self.max_position_size:
                return False, f"Size {size_dec} above maximum {self.max_position_size}"

            # Optional validations
            if stop_loss_dec is not None:
                if stop_loss_dec <= 0:
                    return False, "Stop loss must be positive"

                if side.lower() == "buy" and stop_loss_dec >= entry_price_dec:
                    return (
                        False,
                        "Stop loss must be below entry price for long positions",
                    )

                if side.lower() == "sell" and stop_loss_dec <= entry_price_dec:
                    return (
                        False,
                        "Stop loss must be above entry price for short positions",
                    )

            if take_profit_dec is not None:
                if take_profit_dec <= 0:
                    return False, "Take profit must be positive"

                if side.lower() == "buy" and take_profit_dec <= entry_price_dec:
                    return (
                        False,
                        "Take profit must be above entry price for long positions",
                    )

                if side.lower() == "sell" and take_profit_dec >= entry_price_dec:
                    return (
                        False,
                        "Take profit must be below entry price for short positions",
                    )

            if leverage_dec is not None:
                if leverage_dec <= 0:
                    return False, "Leverage must be positive"

            return True, None

        except Exception as e:
            logger.error(f"New position validation error: {str(e)}")
            return False, str(e)

    async def validate_position_close(
        self, position: Any, exit_price: Union[str, float, Decimal]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for closing a position.

        Args:
            position: Position to close
            exit_price: Exit price

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Convert exit price to Decimal
            nh = NumericHandler()
            exit_price_dec = nh.to_decimal(exit_price)

            # Basic validation
            if exit_price_dec <= 0:
                return False, "Exit price must be positive"

            # Validate position
            validation = self.validate_position(position)
            if not validation.is_valid:
                return False, validation.error_message

            return True, None

        except Exception as e:
            logger.error(f"Position close validation error: {str(e)}")
            return False, str(e)
