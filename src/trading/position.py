#!/usr/bin/env python3
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
from typing import Any, Dict, Optional, Set

import pandas as pd

from bot_types.base_types import MarketState, Validatable, ValidationResult
from bot_types.position_types import PositionInfo, PositionValidationConfig
from signals.market_state import prepare_market_state
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
    realized_pnl: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    last_update: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    # Exchange state reconciliation
    last_exchange_sync: Optional[float] = None
    exchange_state_valid: bool = False
    reconciliation_attempts: int = 0
    max_reconciliation_attempts: int = 3
    reconciliation_threshold: float = 300  # 5 minutes

    def __post_init__(self):
        """Validate initial state"""
        self.side = self.side.lower()
        if self.side not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {self.side}")

        if self.size == 0:
            raise ValueError("Position size cannot be zero")

        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")

        if self.current_price is not None and self.current_price <= 0:
            raise ValueError("Current price must be positive if set")

    def validate(self) -> ValidationResult:
        """
        Validate the position state.

        Returns:
            ValidationResult indicating if position is valid
        """
        try:
            if not self.symbol:
                return ValidationResult(False, "Position must have a symbol")

            if self.size == 0:
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
                        False,
                        "Stop loss must be below current price for long positions",
                    )

                if self.side == "sell" and self.stop_loss <= self.current_price:
                    return ValidationResult(
                        False,
                        "Stop loss must be above current price for short positions",
                    )

            # Validate take profit
            if self.take_profit:
                if self.take_profit <= 0:
                    return ValidationResult(False, "Take profit must be positive")

                if self.side == "buy" and self.take_profit <= self.current_price:
                    return ValidationResult(
                        False,
                        "Take profit must be above current price for long positions",
                    )

                if self.side == "sell" and self.take_profit >= self.current_price:
                    return ValidationResult(
                        False,
                        "Take profit must be below current price for short positions",
                    )

            return ValidationResult(True)

        except Exception as e:
            logger.error(f"Position validation error: {str(e)}")
            return ValidationResult(False, str(e))

    async def update(self, new_price: Decimal, current_time: float) -> None:
        """
        Update position with new price and recalculate state.

        Args:
            new_price: The new current price
            current_time: Current timestamp

        Raises:
            PositionError: If position update fails validation
        """
        # Validate new price
        if new_price <= 0:
            raise PositionError("Update price must be positive")

        # Check if we need to reconcile with exchange
        if (
            not self.exchange_state_valid
            and self.reconciliation_attempts < self.max_reconciliation_attempts
        ):
            await self._reconcile_with_exchange()

        if not self.exchange_state_valid:
            raise PositionError("Cannot update position - exchange state invalid")

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

    async def _reconcile_with_exchange(self) -> None:
        """
        Reconcile local position state with exchange state.

        This ensures our position tracking stays accurate.
        """
        try:
            # Skip if recently reconciled
            if (
                self.last_exchange_sync
                and time.time() - self.last_exchange_sync
                < self.reconciliation_threshold
            ):
                return

            exchange_position = await self._fetch_exchange_position()
            if not exchange_position:
                logger.warning(f"No position found on exchange for {self.symbol}")
                self.exchange_state_valid = False
                return

            # Validate key fields match
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

    async def _fetch_exchange_position(self):
        """Fetch position from exchange"""
        # TODO: Implement exchange API call
        pass

    def _validate_exchange_state(self, exchange_position: Dict) -> bool:
        """
        Validate local state matches exchange state

        Args:
            exchange_position: Position data from exchange

        Returns:
            bool indicating if states match
        """
        try:
            # Compare key fields
            if (
                exchange_position["symbol"] != self.symbol
                or abs(float(exchange_position["size"]) - float(self.size)) > 0.0001
                or exchange_position["side"].lower() != self.side
            ):
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating exchange state: {str(e)}")
            return False


class PositionManager:
    """Manages trading positions with proper concurrency control and transaction safety."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.positions: Dict[str, Position] = {}
        self.db = ctx.db_connection if hasattr(ctx, "db_connection") else None
        self._position_updates: Dict[str, Set[str]] = {}

        # Initialize validator with config from context if available
        config = None
        if hasattr(ctx, "config") and "position_limits" in ctx.config:
            config = PositionValidationConfig.from_dict(ctx.config["position_limits"])
        self._validator = PositionValidator(config=config, logger=self.logger)

    async def open_position(self, order: Dict[str, Any]) -> Optional[Position]:
        """Open a new position with proper transaction management."""
        symbol = order.get("symbol")
        if not symbol:
            raise PositionError("Order must have a symbol")

        async with self._lock:
            try:
                # Basic parameter validation
                required_fields = ["side", "price", "amount"]
                if any(field not in order for field in required_fields):
                    raise PositionError(f"Missing required fields: {required_fields}")

                side = order["side"]
                price = order["price"]
                amount = order["amount"]
                stop_loss = order.get("stop_loss")
                take_profit = order.get("take_profit")
                leverage = order.get("leverage")

                # Validate new position
                is_valid, error_msg = await self._validator.validate_new_position(
                    symbol=symbol,
                    side=side,
                    size=amount,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    existing_positions=self.positions,
                )
                if not is_valid:
                    self.logger.error(f"Failed to validate new position: {error_msg}")
                    return None

                # Create position object
                position = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=Decimal(str(price)),
                    size=Decimal(str(amount)),
                    timestamp=int(time.time() * 1000),
                    stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
                    take_profit=Decimal(str(take_profit)) if take_profit else None,
                    strategy=order.get("strategy"),
                    metadata=order.get("metadata", {}),
                )

                # Store position with transaction safety
                if self.db:
                    async with self.db.transaction():
                        self.positions[symbol] = position
                        await self._store_position(position)
                else:
                    self.positions[symbol] = position
                    await self._store_position(position)

                self._position_updates[symbol] = set()
                return position

            except Exception as e:
                self.logger.error(f"Failed to open position: {e}")
                return None

    async def close_position(
        self, symbol: str, exit_price: Decimal
    ) -> Optional[Position]:
        """Close a position with proper transaction management."""
        async with self._lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning(f"No existing position for symbol: {symbol}")
                    return None

                position = self.positions[symbol]
                if position.closed:
                    self.logger.warning(f"Position already closed: {symbol}")
                    return None

                # Validate close parameters
                is_valid, error_msg = await self._validator.validate_position_close(
                    position=position, exit_price=exit_price
                )
                if not is_valid:
                    self.logger.error(f"Failed to validate position close: {error_msg}")
                    return None

                # Close position with transaction safety
                if self.db:
                    async with self.db.transaction():
                        await position.update(exit_price, time.time())
                        await position.close_position(position.unrealized_pnl)
                        await self._update_position(position, closed=True)
                        del self.positions[symbol]
                else:
                    await position.update(exit_price, time.time())
                    await position.close_position(position.unrealized_pnl)
                    await self._update_position(position, closed=True)
                    del self.positions[symbol]

                if symbol in self._position_updates:
                    del self._position_updates[symbol]

                return position

            except Exception as e:
                self.logger.error(f"Unexpected error in close_position: {e}")
                return None

    async def _store_position(self, position: Position) -> None:
        """Store position with proper error handling."""
        try:
            await self.ctx.db_queries.insert_trade(
                {
                    "id": position.symbol,
                    "symbol": position.symbol,
                    "entry_price": float(position.entry_price),
                    "size": float(position.size),
                    "side": position.side,
                    "strategy": position.strategy or "manual",
                    "metadata": position.metadata,
                    "timestamp": position.timestamp,
                    "status": "OPEN",
                }
            )
        except Exception as e:
            await handle_error_async(e, "PositionManager._store_position", self.logger)
            raise

    async def _update_position(self, position: Position, closed: bool = False) -> None:
        """Update position with proper error handling."""
        try:
            status = "CLOSED" if closed else "OPEN"
            metadata = (
                {
                    "exit_price": str(position.current_price),
                    "realized_pnl": str(position.realized_pnl),
                    "close_time": int(time.time() * 1000),
                }
                if closed
                else None
            )

            await self.ctx.db_queries.update_position_status(
                position_id=str(position.timestamp), status=status, metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Failed to update position status: {e}")
            raise

    async def update_position(self, symbol: str, market_data: pd.DataFrame) -> None:
        """Update position with proper concurrency control."""
        if symbol not in self.positions:
            return

        update_id = f"market_{time.time()}"
        if symbol in self._position_updates:
            self._position_updates[symbol].add(update_id)

        try:
            position = self.positions[symbol]
            if position.closed:
                return

            market_state = prepare_market_state(market_data)

            # Validate update
            is_valid, error_msg = await self._validator.validate_position_update(
                position=position,
                new_price=market_state.price,
                current_time=time.time(),
            )
            if not is_valid:
                self.logger.warning(f"Position update validation failed: {error_msg}")
                return

            await position.update(market_state.price, time.time())

        except Exception as e:
            await handle_error_async(e, "PositionManager.update_position", self.logger)
        finally:
            if symbol in self._position_updates:
                self._position_updates[symbol].discard(update_id)

    async def get_position_info(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get the position info for a given symbol.
        """
        return self.positions.get(symbol)

    async def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all positions.
        """
        return self.positions

    async def get_open_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        """
        return {symbol: pos for symbol, pos in self.positions.items() if not pos.closed}

    async def get_closed_positions(self) -> Dict[str, Position]:
        """
        Get all closed positions.
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos.closed}

    async def get_position_by_id(self, id: str) -> Optional[Position]:
        """
        Get a position by its ID.
        """
        return self.positions.get(id)

    async def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get a position by its symbol.
        """
        return self.positions.get(symbol)

    async def get_position_by_timestamp(self, timestamp: int) -> Dict[str, Position]:
        """
        Get all positions matching a given timestamp.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.timestamp == timestamp
        }

    async def get_position_by_side(self, side: str) -> Dict[str, Position]:
        """
        Get all positions by their side.
        """
        return {
            symbol: pos for symbol, pos in self.positions.items() if pos.side == side
        }

    async def get_position_by_strategy(self, strategy: str) -> Dict[str, Position]:
        """
        Get all positions by their strategy.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.strategy == strategy
        }

    async def get_position_by_entry_price(
        self, entry_price: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their entry price.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.entry_price == entry_price
        }

    async def get_position_by_size(self, size: Decimal) -> Dict[str, Position]:
        """
        Get all positions by their size.
        """
        return {
            symbol: pos for symbol, pos in self.positions.items() if pos.size == size
        }

    async def get_position_by_status(self, status: str) -> Dict[str, Position]:
        """
        Get all positions by their status.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if ("CLOSED" if pos.closed else "OPEN") == status
        }

    async def get_position_by_pnl(self, pnl: Decimal) -> Dict[str, Position]:
        """
        Get all positions by their PnL.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.realized_pnl == pnl
        }

    async def get_position_by_unrealized_pnl(
        self, unrealized_pnl: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their unrealized PnL.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.unrealized_pnl == unrealized_pnl
        }

    async def get_position_by_realized_pnl(
        self, realized_pnl: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their realized PnL.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.realized_pnl == realized_pnl
        }

    async def get_position_by_pnl_percentage(
        self, pnl_percentage: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their PnL percentage.
        (Assumes each position has an attribute or computed property 'pnl_percentage'.)
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if getattr(pos, "pnl_percentage", None) == pnl_percentage
        }

    async def get_position_by_stop_loss(
        self, stop_loss: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their stop loss.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.stop_loss == stop_loss
        }

    async def get_position_by_take_profit(
        self, take_profit: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their take profit.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.take_profit == take_profit
        }

    async def get_position_by_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Position]:
        """
        Get all positions by their metadata.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.metadata == metadata
        }

    async def get_position_by_current_price(
        self, current_price: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their current price.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.current_price == current_price
        }

    async def get_position_by_sentiment(self, sentiment: str) -> Dict[str, Position]:
        """
        Get all positions by their sentiment.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.sentiment == sentiment
        }

    async def get_position_by_sentiment_score(
        self, sentiment_score: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their sentiment score.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.sentiment_score == sentiment_score
        }

    async def get_position_by_sentiment_magnitude(
        self, sentiment_magnitude: Decimal
    ) -> Dict[str, Position]:
        """
        Get all positions by their sentiment magnitude.
        """
        return {
            symbol: pos
            for symbol, pos in self.positions.items()
            if pos.sentiment_magnitude == sentiment_magnitude
        }
