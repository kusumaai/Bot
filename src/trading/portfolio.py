#! /usr/bin/env python3
# src/trading/portfolio.py
"""
Module: src.trading
Provides portfolio management functionality.
"""
import asyncio
import logging
import threading

# import the necessary libraries
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Optional

from risk.limits import RiskLimits
from trading.position import Position
from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import PortfolioError
from utils.numeric_handler import NumericHandler


@dataclass
class PortfolioStats:
    """Portfolio performance statistics"""

    total_value: Decimal
    cash_balance: Decimal
    position_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin_used: Decimal = Decimal("0")
    free_margin: Decimal = Decimal("0")
    risk_ratio: float = 0.0
    exposure: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# portfolio manager class that manages the portfolio for the trading bot in order to track the portfolio value, unrealized PnL, and other metrics providing a comprehensive view of the portfolio's performance
class PortfolioManager:
    def __init__(self, ctx, logger=None):
        """
        Initialize PortfolioManager.
        The provided context (ctx) should include:
          - risk_limits: A dictionary of risk limits (e.g., 'max_positions', 'max_position_size').
          - db_queries (optional): An object with a method get_account_balance() for fetching account balance.
        """
        self.ctx = ctx
        self.risk_limits = getattr(ctx, "risk_limits", {})
        self.db_queries = getattr(ctx, "db_queries", None)
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Attributes from the first script
        self.current_balance = Decimal("0")

        # Attributes from the second script
        self.positions: Dict[str, Position] = {}
        self.balance = Decimal("0")
        self.peak_balance = Decimal("0")
        self.daily_starting_balance = Decimal("0")
        self.realized_pnl = Decimal("0")
        self._portfolio_value: Decimal = Decimal("0")
        self._last_update: float = 0
        self._update_interval: float = 0.1  # 100ms
        self.lock = threading.Lock()
        self._last_daily_reset = datetime.now().date()
        self._last_value: Decimal = Decimal("0")
        self._high_water_mark: Decimal = Decimal("0")
        self._position_updates: deque = deque(maxlen=1000)
        self.nh = NumericHandler()
        self._async_lock = asyncio.Lock()  # For async operations
        self.initialized = False

        # Initialize trade history for simulation monitoring
        self.trade_history = []

    # initialize the portfolio manager
    async def initialize(self):
        if self.ctx.config.get("paper_mode", False):
            from decimal import Decimal

            starting_balance = Decimal(
                str(self.ctx.config.get("paper_starting_balance", "10000"))
            )
            self.balance = starting_balance
            self.current_balance = starting_balance
            self.peak_balance = starting_balance
            self.logger.info(f"Paper mode: Set starting balance to {starting_balance}")

        self.initialized = True
        return True

    # update the current balance from the database
    async def update_current_balance(self) -> None:
        """
        Updates the current balance from the database.
        If db_queries is not provided in the context, a warning is logged and the update is skipped.
        """
        if self.db_queries is None:
            self.logger.warning(
                "db_queries not provided in context; skipping balance update."
            )
            return
        try:
            balance = await self.db_queries.get_account_balance()
            self.current_balance = Decimal(balance)
        except Exception as e:
            await handle_error_async(
                e, "PortfolioManager.update_current_balance", self.logger
            )
            raise PortfolioError(f"Error updating current balance: {e}")

    # get the total value of the portfolio
    async def get_total_value(self):
        """
        Return the total account value.
        This is a placeholder and should be replaced with the actual logic if available.
        """
        return Decimal("10000")

    # calculate the portfolio value for the portfolio manager including unrealized PnL
    def calculate_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value including unrealized PnL.
        The value is updated if the specified interval has passed.
        """
        try:
            now = time.time()
            if now - self._last_update >= self._update_interval:
                with self.lock:
                    self._portfolio_value = self.balance + sum(
                        pos.size * pos.current_price for pos in self.positions.values()
                    )
                    if self._portfolio_value > self.peak_balance:
                        self.peak_balance = self._portfolio_value
                    self._last_update = now

                    # Daily reset check
                    current_date = datetime.now().date()
                    if current_date > self._last_daily_reset:
                        self.daily_starting_balance = self._portfolio_value
                        self._last_daily_reset = current_date

            return self._portfolio_value
        except Exception as e:
            handle_error(
                e, "PortfolioManager.calculate_portfolio_value", logger=self.logger
            )
            return self._portfolio_value

    # calculate the drawdown
    def calculate_drawdown(self) -> Decimal:
        """
        Calculate current drawdown from the portfolio's peak value.
        """
        try:
            portfolio_value = self.calculate_portfolio_value()
            if self.peak_balance == 0:
                return Decimal("0")
            return (self.peak_balance - portfolio_value) / self.peak_balance
        except Exception as e:
            handle_error(e, "PortfolioManager.calculate_drawdown", logger=self.logger)
            return Decimal("0")

    # get the portfolio stats
    def get_portfolio_stats(self) -> PortfolioStats:
        """
        Get comprehensive portfolio statistics including total value, PnL, and exposure.
        """
        try:
            with self.lock:
                total_value = self.calculate_portfolio_value()
                unrealized_pnl = sum(
                    pos.unrealized_pnl for pos in self.positions.values()
                )
                total_exposure = sum(
                    pos.size * pos.current_price for pos in self.positions.values()
                )
                leverage = (
                    total_exposure / total_value if total_value > 0 else Decimal("0")
                )
                # daily pnl = current portfolio value - starting balance
                daily_pnl = total_value - self.daily_starting_balance
                # return the portfolio stats
                return PortfolioStats(
                    total_value=total_value,
                    cash_balance=self.balance,
                    position_value=total_value - self.balance,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=self.realized_pnl,
                    margin_used=Decimal("0"),
                    free_margin=Decimal("0"),
                    risk_ratio=float(leverage),
                    exposure=total_exposure,
                    metadata={},
                )
        except Exception as e:
            handle_error(e, "PortfolioManager.get_portfolio_stats", logger=self.logger)
            return PortfolioStats(
                total_value=Decimal("0"),
                cash_balance=Decimal("0"),
                position_value=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                margin_used=Decimal("0"),
                free_margin=Decimal("0"),
                risk_ratio=0.0,
                exposure=Decimal("0"),
                metadata={},
            )

    # add a new position to the portfolio
    async def add_position(
        self, symbol: str, size: Decimal, entry_price: Decimal
    ) -> bool:
        """
        Add a new position with thread safety.
        Validates input and risk limits before adding the position.
        """
        async with self._async_lock:
            try:
                if not isinstance(symbol, str):
                    raise PortfolioError("Symbol must be a string.")
                if size <= Decimal("0") or entry_price <= Decimal("0"):
                    raise PortfolioError("Size and entry price must be positive.")
                if len(self.positions) >= self.risk_limits.get("max_positions", 10):
                    self.logger.warning(
                        f"Max positions limit reached: {self.risk_limits.get('max_positions', 10)}"
                    )
                    return False

                position_value = size * entry_price
                total_value = self.calculate_portfolio_value()
                if total_value > Decimal("0") and (
                    position_value / total_value
                ) > self.risk_limits.get("max_position_size", Decimal("0.1")):
                    self.logger.warning(
                        "Position size exceeds max position size limit."
                    )
                    return False

                self.positions[symbol] = Position(
                    symbol=symbol,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    direction="long",  # Assuming long; adjust as needed
                )
                await self._update_portfolio_metrics()
                return True
            except PortfolioError as e:
                self.logger.error(f"Failed to add position: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in add_position: {e}")
                return False

    # update the position with a new price
    async def update_position_price(self, symbol: str, current_price: Decimal) -> None:
        """
        Update an existing position with new price data.
        Also updates the position's unrealized PnL and records the change.
        """
        async with self._async_lock:
            try:
                if not isinstance(symbol, str):
                    raise PortfolioError("Symbol must be a string.")
                if current_price <= Decimal("0"):
                    raise PortfolioError("Current price must be positive.")
                if symbol not in self.positions:
                    self.logger.warning(
                        f"Attempted to update non-existent position: {symbol}"
                    )
                    return
                # update the position with a new price
                position = self.positions[symbol]
                old_price = position.current_price
                new_price = self.nh.to_decimal(current_price)
                position.current_price = new_price
                position.unrealized_pnl = position.size * (
                    new_price - position.entry_price
                )
                position.last_update = datetime.utcnow()
                self._position_updates.append(
                    {
                        "symbol": symbol,
                        "timestamp": datetime.utcnow(),
                        "price_change": new_price - old_price,
                        "unrealized_pnl": position.unrealized_pnl,
                    }
                )
                await self._update_portfolio_metrics()
            except PortfolioError as e:
                self.logger.error(f"Failed to update position: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in update_position_price: {e}")

    # update portfolio metrics
    async def _update_portfolio_metrics(self) -> None:
        """
        Update portfolio-wide metrics.
        This includes updating the high water mark if a new portfolio value is reached.
        """
        try:
            current_value = await self.get_total_value()
            self._last_value = current_value
            if current_value > self._high_water_mark:
                self._high_water_mark = current_value
                self.logger.info(f"New high water mark: {self._high_water_mark}")
        except Exception as e:
            self.logger.error(f"Failed to update portfolio metrics: {e}")

    # close a position
    def close_position(self, symbol: str, exit_price: Decimal) -> Optional[Position]:
        """
        Close an existing position, update realized PnL, and adjust the portfolio balance.
        """
        try:
            with self.lock:
                if symbol not in self.positions:
                    self.logger.warning(
                        f"Attempted to close non-existent position: {symbol}"
                    )
                    return None
                position = self.positions[symbol]
                position.close(exit_price)
                self.realized_pnl += position.unrealized_pnl
                self.balance += position.unrealized_pnl
                del self.positions[symbol]
                self.logger.info(
                    f"Position closed for {symbol} at {exit_price}. Realized PnL: {position.unrealized_pnl}"
                )
                return position
        except Exception as e:
            handle_error(e, "PortfolioManager.close_position", logger=self.logger)
            return None

    # update a position with a new price
    async def update_position(self, symbol: str, current_price: Decimal) -> bool:
        """
        Update a position with a new price.
        Uses the position's update_price() method and updates portfolio metrics.
        """
        async with self._async_lock:
            try:
                if symbol not in self.positions:
                    self.logger.warning(
                        f"Attempted to update non-existent position: {symbol}"
                    )
                    return False
                position = self.positions[symbol]
                position.update_price(current_price)
                self.logger.info(
                    f"Position updated for {symbol}: Current Price = {current_price}"
                )
                await self._update_portfolio_metrics()
                return True
            except Exception as e:
                handle_error(e, "PortfolioManager.update_position", logger=self.logger)
                return False

    # calculate the percentage change between two values
    def calculate_percentage_change(
        self, old_value: Decimal, new_value: Decimal
    ) -> Decimal:
        """
        Calculate the percentage change between two values.
        Returns the result quantized to two decimal places.
        """
        try:
            change = (new_value - old_value) / old_value
            return change.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except Exception as e:
            handle_error(
                e, "PortfolioManager.calculate_percentage_change", logger=self.logger
            )
            return Decimal("0")

    # get the total positions in the portfolio
    def get_total_positions(self) -> int:
        """
        Get the total number of positions in the portfolio.
        """
        return len(self.positions)

    def simulate_trade(self, trade_details: dict) -> None:
        """Simulate execution of a trade in paper mode, record the trade, and update the portfolio balance."""
        from decimal import Decimal

        # Record the trade details for later analysis
        self.trade_history.append(trade_details)

        # Update portfolio balance based on trade side
        side = trade_details.get("side")
        amount = Decimal(str(trade_details.get("amount", "0")))
        price = Decimal(str(trade_details.get("price", "0")))

        if side == "buy":
            cost = amount * price
            self.current_balance -= cost
            self.logger.info(
                f"Simulated BUY trade executed: cost={cost}, new balance={self.current_balance}"
            )
        elif side == "sell":
            proceeds = amount * price
            self.current_balance += proceeds
            self.logger.info(
                f"Simulated SELL trade executed: proceeds={proceeds}, new balance={self.current_balance}"
            )
        else:
            self.logger.warning(f"Unknown trade side: {side}")

    def get_trade_summary(self) -> dict:
        """Return summary of simulated trades, including starting balance, current balance, net profit and trade history."""
        from decimal import Decimal

        net_profit = self.current_balance - self.balance
        summary = {
            "starting_balance": self.balance,
            "current_balance": self.current_balance,
            "net_profit": net_profit,
            "total_trades": len(self.trade_history),
            "trade_history": self.trade_history,
        }
        return summary
