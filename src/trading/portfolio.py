# src/trading/position.py
"""
Module: src.trading
Provides position management functionality with proper concurrency control.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Set, Union

from bot_types.base_types import (
    MarketState,
    PositionInfo,
    Validatable,
    ValidationResult,
)
from signals.market_state import prepare_market_state
from trading.position_validator import PositionValidationConfig, PositionValidator
from utils.error_handler import handle_error_async
from utils.exceptions import PositionError
from utils.logger import get_logger
from utils.numeric_handler import NumericHandler

logger = get_logger(__name__)


@dataclass
class Position(Validatable):
    """
    Represents a trading position with validation.
    This is the single source of truth for position state in the system.
    """

    symbol: str
    size: Decimal
    entry_price: Decimal
    timestamp: float
    side: str
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = field(default_factory=lambda: Decimal("0"))
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    last_update: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    strategy: Optional[str] = None

    # Exchange state reconciliation
    last_exchange_sync: Optional[float] = None
    exchange_state_valid: bool = False
    reconciliation_attempts: int = 0
    max_reconciliation_attempts: int = 3
    reconciliation_threshold: float = 300  # 5 minutes

    def __post_init__(self):
        """Validate initial state and perform necessary conversions."""
        nh = NumericHandler()
        self.side = self.side.lower()
        self.size = nh.to_decimal(self.size)
        self.entry_price = nh.to_decimal(self.entry_price)

        if self.stop_loss is not None:
            self.stop_loss = nh.to_decimal(self.stop_loss)
        if self.take_profit is not None:
            self.take_profit = nh.to_decimal(self.take_profit)
        if self.current_price is not None:
            self.current_price = nh.to_decimal(self.current_price)

        if self.side not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {self.side}")

        if self.size.is_zero():
            raise ValueError("Position size cannot be zero")

        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")

        if self.current_price is not None and self.current_price <= 0:
            raise ValueError("Current price must be positive if set")

    def validate(self) -> ValidationResult:
        """Validate the position state."""
        return self._validate()

    def _validate(self) -> ValidationResult:
        """Internal validation logic."""
        if not self.symbol:
            return ValidationResult(False, "Position must have a symbol")

        if self.size.is_zero():
            return ValidationResult(False, "Position size cannot be zero")

        if self.entry_price <= 0:
            return ValidationResult(False, "Entry price must be positive")

        if self.current_price is not None and self.current_price <= 0:
            return ValidationResult(False, "Current price must be positive if set")

        if self.side not in ["buy", "sell"]:
            return ValidationResult(False, f"Invalid side: {self.side}")

        # Validate stop loss
        if self.stop_loss:
            if self.stop_loss <= 0:
                return ValidationResult(False, "Stop loss must be positive")

            if self.side == "buy" and self.stop_loss >= self.current_price:
                return ValidationResult(
                    False, "Stop loss must be below current price for long positions"
                )

            if self.side == "sell" and self.stop_loss <= self.current_price:
                return ValidationResult(
                    False, "Stop loss must be above current price for short positions"
                )

        # Validate take profit
        if self.take_profit:
            if self.take_profit <= 0:
                return ValidationResult(False, "Take profit must be positive")

            if self.side == "buy" and self.take_profit <= self.current_price:
                return ValidationResult(
                    False, "Take profit must be above current price for long positions"
                )

            if self.side == "sell" and self.take_profit >= self.current_price:
                return ValidationResult(
                    False, "Take profit must be below current price for short positions"
                )

        return ValidationResult(True)

    async def update(
        self, new_price: Union[Decimal, str, float, int], current_time: float
    ) -> None:
        """
        Update position with new price and recalculate state.

        Args:
            new_price: The new current price
            current_time: Current timestamp

        Raises:
            PositionError: If position update fails validation
        """
        nh = NumericHandler()
        new_price = nh.to_decimal(new_price)

        if new_price <= 0:
            raise PositionError("Update price must be positive")

        # Check if we need to reconcile with exchange
        if (
            not self.exchange_state_valid
            and self.reconciliation_attempts < self.max_reconciliation_attempts
        ):
            await self._reconcile_with_exchange()

        if not self.exchange_state_valid:
            logger.warning("Exchange state invalid, attempting reconciliation")

        # Update state
        self.current_price = new_price
        self.last_update = current_time

        # Recalculate PnL
        if self.side == "buy":
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.size

        # Validate updated state
        validation = self.validate()
        if not validation.is_valid:
            raise PositionError(
                f"Invalid position state after update: {validation.error_message}"
            )

    async def close_position(self, exit_price: Union[Decimal, str, float, int]) -> None:
        """
        Close the position and calculate realized PNL.

        Args:
            exit_price: The price at which the position is closed
        """
        nh = NumericHandler()
        exit_price = nh.to_decimal(exit_price)

        if self.closed:
            logger.warning(f"Position {self.symbol} is already closed.")
            return

        await self.update(exit_price, time.time())
        self.realized_pnl = self.unrealized_pnl
        self.closed = True
        self.unrealized_pnl = None
        logger.info(f"Position {self.symbol} closed. Realized PNL: {self.realized_pnl}")

    async def _reconcile_with_exchange(self) -> None:
        """Reconcile local position state with exchange state."""
        try:
            if (
                self.last_exchange_sync
                and time.time() - self.last_exchange_sync
                < self.reconciliation_threshold
            ) or self.reconciliation_attempts >= self.max_reconciliation_attempts:
                return

            exchange_position = await self._fetch_exchange_position()
            if not exchange_position:
                logger.warning(f"No position found on exchange for {self.symbol}")
                self.exchange_state_valid = False
                return

            if not self._validate_exchange_state(exchange_position):
                logger.error(f"Position state mismatch with exchange for {self.symbol}")
                self.exchange_state_valid = False
                return

            self.exchange_state_valid = True
            self.last_exchange_sync = time.time()
            self.reconciliation_attempts = 0

        except Exception as e:
            logger.error(f"Failed to reconcile position state: {str(e)}")
            self.reconciliation_attempts += 1
            self.exchange_state_valid = False

    async def _fetch_exchange_position(self) -> Optional[Dict[str, Any]]:
        """Fetch position from exchange."""
        try:
            if not hasattr(self, "ctx") or not self.ctx.exchange_interface:
                logger.error("No exchange interface available")
                return None

            position = await self.ctx.exchange_interface.get_position(self.symbol)
            if not position:
                return None

            return {
                "symbol": position["symbol"],
                "size": Decimal(str(position["size"])),
                "side": position["side"].lower(),
                "entry_price": Decimal(str(position["entry_price"])),
            }

        except Exception as e:
            logger.error(f"Error fetching position from exchange: {e}")
            return None

    def _validate_exchange_state(self, exchange_position: Dict[str, Any]) -> bool:
        """
        Validate local state matches exchange state.

        Args:
            exchange_position: Position data from exchange

        Returns:
            bool indicating if states match
        """
        try:
            nh = NumericHandler()
            size_diff = abs(nh.to_decimal(exchange_position["size"]) - self.size)

            if (
                exchange_position["symbol"] != self.symbol
                or size_diff > nh.to_decimal("0.0001")
                or exchange_position["side"].lower() != self.side
            ):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating exchange state: {str(e)}")
            return False
