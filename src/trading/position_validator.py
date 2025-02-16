#! /usr/bin/env python3
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
from src.trading.position import Position
from src.utils.error_handler import handle_error, handle_error_async
from src.utils.exceptions import PositionError
from src.utils.logger import get_logger
from src.utils.numeric_handler import NumericHandler

logger = get_logger(__name__)


@dataclass
class PositionValidationConfig:
    """Configuration for position validation thresholds."""

    max_position_size: Decimal
    min_position_size: Decimal
    max_drawdown: Decimal
    max_leverage: Decimal
    max_position_duration: int  # in seconds
    min_stop_distance: Decimal  # minimum distance for stop loss as percentage
    min_profit_distance: Decimal  # minimum distance for take profit as percentage
    max_daily_positions: int
    max_positions_per_symbol: int

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PositionValidationConfig":
        """Create config from dictionary with proper error handling."""
        try:
            nh = NumericHandler()
            return cls(
                max_position_size=nh.to_decimal(config.get("max_position_size", "1.0")),
                min_position_size=nh.to_decimal(
                    config.get("min_position_size", "0.01")
                ),
                max_drawdown=nh.to_decimal(config.get("max_drawdown", "0.5")),
                max_leverage=nh.to_decimal(config.get("max_leverage", "10.0")),
                max_position_duration=int(config.get("max_position_duration", 86400)),
                min_stop_distance=nh.to_decimal(
                    config.get("min_stop_distance", "0.01")
                ),
                min_profit_distance=nh.to_decimal(
                    config.get("min_profit_distance", "0.01")
                ),
                max_daily_positions=int(config.get("max_daily_positions", 10)),
                max_positions_per_symbol=int(config.get("max_positions_per_symbol", 1)),
            )
        except Exception as e:
            raise ValueError(f"Invalid position validation config: {e}")


class PositionValidator:
    """Validates position operations and state."""

    def __init__(self, max_position_size: Decimal, min_position_size: Decimal):
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def validate_position(self, position: Position) -> ValidationResult:
        """
        Validates a position's state.

        Args:
            position: The position to validate

        Returns:
            ValidationResult indicating if position is valid
        """
        if not isinstance(position, Position):
            return ValidationResult(False, f"Invalid position type: {type(position)}")

        if not position.symbol:
            return ValidationResult(False, "Position must have a symbol")

        if position.size == 0:
            return ValidationResult(False, "Position size cannot be zero")

        if abs(position.size) > self.max_position_size:
            return ValidationResult(
                False,
                f"Position size {position.size} exceeds maximum {self.max_position_size}",
            )

        if abs(position.size) < self.min_position_size:
            return ValidationResult(
                False,
                f"Position size {position.size} below minimum {self.min_position_size}",
            )

        if not position.entry_price or position.entry_price <= 0:
            return ValidationResult(False, "Position must have a valid entry price")

        if position.current_price and position.current_price <= 0:
            return ValidationResult(False, "Current price must be positive if set")

        return ValidationResult(True)

    def validate_position_update(
        self, position: Position, new_price: Decimal
    ) -> ValidationResult:
        """
        Validates a position update operation.

        Args:
            position: The position being updated
            new_price: The new price to update to

        Returns:
            ValidationResult indicating if update is valid
        """
        if not isinstance(position, Position):
            return ValidationResult(False, f"Invalid position type: {type(position)}")

        if new_price <= 0:
            return ValidationResult(False, "Update price must be positive")

        # Validate base position state first
        base_validation = self.validate_position(position)
        if not base_validation.is_valid:
            return base_validation

        # Additional update-specific validation can be added here

        return ValidationResult(True)

    def validate_stop_loss(
        self, position: Position, stop_price: Decimal
    ) -> ValidationResult:
        """
        Validates a stop loss placement.

        Args:
            position: The position to place stop loss on
            stop_price: The stop loss price

        Returns:
            ValidationResult indicating if stop loss is valid
        """
        if not isinstance(position, Position):
            return ValidationResult(False, f"Invalid position type: {type(position)}")

        if stop_price <= 0:
            return ValidationResult(False, "Stop price must be positive")

        # Long positions must have stop below current price
        if position.size > 0 and stop_price >= position.current_price:
            return ValidationResult(
                False, "Stop loss must be below current price for long positions"
            )

        # Short positions must have stop above current price
        if position.size < 0 and stop_price <= position.current_price:
            return ValidationResult(
                False, "Stop loss must be above current price for short positions"
            )

        return ValidationResult(True)

    def validate_take_profit(
        self, position: Position, take_profit_price: Decimal
    ) -> ValidationResult:
        """
        Validates a take profit placement.

        Args:
            position: The position to place take profit on
            take_profit_price: The take profit price

        Returns:
            ValidationResult indicating if take profit is valid
        """
        if not isinstance(position, Position):
            return ValidationResult(False, f"Invalid position type: {type(position)}")

        if take_profit_price <= 0:
            return ValidationResult(False, "Take profit price must be positive")

        # Long positions must have take profit above current price
        if position.size > 0 and take_profit_price <= position.current_price:
            return ValidationResult(
                False, "Take profit must be above current price for long positions"
            )

        # Short positions must have take profit below current price
        if position.size < 0 and take_profit_price >= position.current_price:
            return ValidationResult(
                False, "Take profit must be below current price for short positions"
            )

        return ValidationResult(True)

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
        """Validate parameters for a new position."""
        try:
            # Validate basic parameters
            if not symbol or not isinstance(symbol, str):
                return False, "Invalid symbol"

            if side not in ["buy", "sell"]:
                return False, f"Invalid side: {side}"

            # Validate numeric values
            size = self.nh.to_decimal(size)
            if size is None or size <= 0:
                return False, "Size must be positive"

            entry_price = self.nh.to_decimal(entry_price)
            if entry_price is None or entry_price <= 0:
                return False, "Entry price must be positive"

            # Validate against configuration if available
            if self.config:
                # Check position size limits
                if size < self.config.min_position_size:
                    return (
                        False,
                        f"Position size below minimum ({self.config.min_position_size})",
                    )
                if size > self.config.max_position_size:
                    return (
                        False,
                        f"Position size exceeds maximum ({self.config.max_position_size})",
                    )

                # Check leverage
                if leverage is not None:
                    leverage = self.nh.to_decimal(leverage)
                    if leverage > self.config.max_leverage:
                        return (
                            False,
                            f"Leverage exceeds maximum ({self.config.max_leverage})",
                        )

                # Check existing positions
                if existing_positions:
                    if len(existing_positions) >= self.config.max_daily_positions:
                        return (
                            False,
                            f"Maximum daily positions ({self.config.max_daily_positions}) reached",
                        )

                    symbol_positions = sum(
                        1 for pos in existing_positions.values() if pos.symbol == symbol
                    )
                    if symbol_positions >= self.config.max_positions_per_symbol:
                        return (
                            False,
                            f"Maximum positions for {symbol} ({self.config.max_positions_per_symbol}) reached",
                        )

            # Validate stop loss if provided
            if stop_loss is not None:
                stop_loss = self.nh.to_decimal(stop_loss)
                if stop_loss <= 0:
                    return False, "Stop loss must be positive"

                if self.config:
                    stop_distance = abs(entry_price - stop_loss) / entry_price
                    if stop_distance < self.config.min_stop_distance:
                        return (
                            False,
                            f"Stop loss too close to entry price (minimum {float(self.config.min_stop_distance)*100}%)",
                        )

            # Validate take profit if provided
            if take_profit is not None:
                take_profit = self.nh.to_decimal(take_profit)
                if take_profit <= 0:
                    return False, "Take profit must be positive"

                if self.config:
                    profit_distance = abs(take_profit - entry_price) / entry_price
                    if profit_distance < self.config.min_profit_distance:
                        return (
                            False,
                            f"Take profit too close to entry price (minimum {float(self.config.min_profit_distance)*100}%)",
                        )

            return True, None

        except Exception as e:
            await handle_error_async(e, "validate_new_position", self.logger)
            return False, str(e)

    async def validate_position_close(
        self, position: Any, exit_price: Union[str, float, Decimal]
    ) -> Tuple[bool, Optional[str]]:
        """Validate position closing parameters."""
        try:
            if not hasattr(position, "entry_price") or not hasattr(position, "size"):
                return False, "Invalid position object"

            exit_price = self.nh.to_decimal(exit_price)
            if exit_price is None or exit_price <= 0:
                return False, "Exit price must be positive"

            return True, None

        except Exception as e:
            await handle_error_async(e, "validate_position_close", self.logger)
            return False, str(e)
