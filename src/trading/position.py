#!/usr/bin/env python3
# src/trading/position.py
"""
Module: src.trading
Provides position management functionality.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import pandas as pd

from bot_types.base_types import (  # MarketState type remains in base types
    MarketState,
    PositionInfo,
)
from signals.market_state import prepare_market_state
from utils.error_handler import handle_error_async


@dataclass
class Position:
    # Required fields (no defaults)
    symbol: str
    side: str
    entry_price: Decimal
    size: Decimal
    timestamp: int

    # Additional fields from previous versions
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    closed: bool = False

    # Optional fields (with defaults)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Additional optional fields for extended features
    strategy: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[Decimal] = None
    sentiment_magnitude: Optional[Decimal] = None

    def __post_init__(self):
        self._lock = asyncio.Lock()
        if self.entry_price <= Decimal("0") or self.size <= Decimal("0"):
            raise ValueError("Entry price and size must be positive.")

    async def update(self, current_price: Decimal) -> None:
        """
        Update the position with a new current price and recalculate unrealized PnL.
        """
        async with self._lock:
            try:
                if current_price <= Decimal("0"):
                    raise Exception("Current price must be positive.")
                self.current_price = current_price
                if self.side.lower() == "buy":
                    self.unrealized_pnl = self.size * (
                        self.current_price - self.entry_price
                    )
                else:
                    self.unrealized_pnl = self.size * (
                        self.entry_price - self.current_price
                    )
            except InvalidOperation as e:
                logging.getLogger(__name__).error(
                    f"Invalid operation during position update: {e}"
                )
                raise e
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Unexpected error during position update: {e}"
                )
                raise e

    def close_position(self, realized_pnl: Decimal) -> None:
        """
        Close the position by updating the realized PnL and marking it as closed.
        """
        self.realized_pnl += realized_pnl
        self.closed = True

    async def update_price(self, new_price: Decimal) -> None:
        """
        Update the current price of the position.
        """
        try:
            self.current_price = new_price
        except Exception as e:
            await handle_error_async(
                e, "Position.update_price", logging.getLogger(__name__)
            )
            raise e

    async def close(self, close_price: Decimal) -> None:
        """
        Close the position at the given price.
        """
        try:
            self.current_price = close_price
            self.closed = True
        except Exception as e:
            await handle_error_async(e, "Position.close", logging.getLogger(__name__))
            raise e

    async def update_market_state(self, market_state: MarketState) -> None:
        """
        Update the position using a MarketState object (e.g., update current price).
        """
        try:
            await self.update(market_state.price)
        except Exception as e:
            await handle_error_async(
                e, "Position.update_market_state", logging.getLogger(__name__)
            )
            raise e


class PositionManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.positions: Dict[str, Position] = {}
        from utils.numeric_handler import NumericHandler

        self.nh = NumericHandler()

    async def open_position(self, order: Dict[str, Any]) -> Optional[Position]:
        async with self._lock:
            try:
                symbol = order.get("symbol")
                side = order.get("side")
                price = order.get("price")
                amount = order.get("amount")

                if not all([symbol, side, price, amount]):
                    raise Exception("Missing required order parameters.")

                if not isinstance(symbol, str) or not isinstance(side, str):
                    raise Exception("Invalid types for 'symbol' or 'side'.")

                try:
                    entry_price = Decimal(str(price))
                    size = Decimal(str(amount))
                except (InvalidOperation, TypeError):
                    raise Exception("Invalid numeric values for 'price' or 'amount'.")

                if symbol in self.positions:
                    raise Exception(f"Position already exists for {symbol}")

                position = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    size=size,
                    timestamp=int(time.time() * 1000),
                )
                self.positions[symbol] = position
                await self._store_position(position)
                return position
            except Exception as e:
                self.logger.error(f"Failed to open position: {e}")
                return None

    async def close_position(
        self, symbol: str, exit_price: Decimal
    ) -> Optional[Position]:
        async with self._lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning(f"No existing position for symbol: {symbol}")
                    return None

                position = self.positions[symbol]
                await position.update(exit_price)
                await self._update_position(position, closed=True)
                del self.positions[symbol]
                return position
            except Exception as e:
                self.logger.error(f"Unexpected error in close_position: {e}")
                return None

    async def _store_position(self, position: Position) -> None:
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
                }
            )
        except Exception as e:
            await handle_error_async(e, "PositionManager._store_position", self.logger)
            raise e

    async def _update_position(self, position: Position, closed: bool = False) -> None:
        try:
            status = "CLOSED" if closed else "OPEN"
            await self.ctx.database.queries.update_position_status(
                position_id=str(position.timestamp),  # Assuming timestamp is used as ID
                status=status,
                metadata=(
                    {"exit_price": str(position.current_price)} if closed else None
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to update position status: {e}")

    async def update_position(self, symbol: str, market_data: pd.DataFrame) -> None:
        try:
            from signals.market_state import prepare_market_state

            market_state = prepare_market_state(market_data)
            position = self.positions.get(symbol)
            if position:
                await position.update_market_state(market_state)
        except Exception as e:
            await handle_error_async(e, "PositionManager.update_position", self.logger)

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
