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
        self._update_interval: float = 5  # 5 seconds
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

        self._lock = asyncio.Lock()
        self._last_exchange_sync = 0
        self._sync_interval = 60  # 1 minute
        self._max_sync_deviation = Decimal("0.01")  # 1% maximum deviation
        self._sync_retries = 3
        self._sync_backoff = 5  # 5 seconds between retries
        self._exchange_balance = Decimal("0")
        self._balance_synced = False
        self._position_syncs: Dict[str, float] = {}
        self._failed_syncs = 0
        self._max_failed_syncs = 3

        if not hasattr(self, "risk_limits") or self.risk_limits is None:
            self.risk_limits = {"max_position_size": Decimal("0.1")}

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
        """Update current balance with exchange reconciliation."""
        try:
            if not hasattr(self.ctx, "exchange_interface"):
                raise PortfolioError("No exchange interface available")

            exchange_balance = await self.ctx.exchange_interface.get_balance()
            if exchange_balance is None:
                raise PortfolioError("Failed to fetch exchange balance")

            async with self._lock:
                old_balance = self.balance
                self.balance = exchange_balance
                self._exchange_balance = exchange_balance
                self._balance_synced = True

                # Log significant deviations
                if old_balance > 0:
                    deviation = abs(old_balance - exchange_balance) / old_balance
                    if deviation > self._max_sync_deviation:
                        self.logger.warning(
                            f"Balance deviation of {deviation:.2%} detected. "
                            f"Old: {old_balance}, New: {exchange_balance}"
                        )
                        # Trigger risk assessment on significant deviation
                        if hasattr(self.ctx, "risk_manager"):
                            await self.ctx.risk_manager.assess_portfolio_risk()

        except Exception as e:
            self.logger.error(f"Failed to update balance: {e}")
            self._balance_synced = False
            raise PortfolioError("Balance update failed") from e

    # get the total value of the portfolio
    async def get_total_value(self):
        """
        Return the total account value.
        This is a placeholder and should be replaced with the actual logic if available.
        """
        return Decimal("10000")

    # calculate the portfolio value for the portfolio manager including unrealized PnL
    async def calculate_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value with exchange state reconciliation.

        Returns:
            Decimal: Total portfolio value

        Raises:
            PortfolioError: If exchange sync fails repeatedly
        """
        try:
            now = time.time()
            needs_sync = (
                not self._balance_synced
                or now - self._last_exchange_sync >= self._sync_interval
            )

            async with self._lock:
                if needs_sync:
                    await self._reconcile_with_exchange()

                # Calculate local portfolio value
                local_value = self.balance + sum(
                    pos.size * pos.current_price
                    for pos in self.positions.values()
                    if pos.current_price is not None
                )

                # Verify against exchange state if recently synced
                if self._balance_synced:
                    exchange_value = self._exchange_balance + sum(
                        pos.size * pos.current_price
                        for pos in self.positions.values()
                        if pos.exchange_state_valid and pos.current_price is not None
                    )

                    # Check for significant deviation
                    if local_value > 0 and exchange_value > 0:
                        deviation = abs(local_value - exchange_value) / local_value
                        if deviation > self._max_sync_deviation:
                            self.logger.warning(
                                f"Portfolio value deviation of {deviation:.2%} detected. "
                                f"Local: {local_value}, Exchange: {exchange_value}"
                            )
                            # Force immediate resync
                            await self._reconcile_with_exchange()
                            # Use exchange value in case of deviation
                            local_value = exchange_value

                self._portfolio_value = local_value
                self._last_update = now
                return local_value

        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            if self._balance_synced:
                # Fall back to last known exchange state
                return self._exchange_balance
            raise PortfolioError("Failed to calculate portfolio value") from e

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
        self,
        symbol: str,
        size: Decimal,
        entry_price: Decimal,
        trade_id: str = None,
        side: str = None,
    ) -> dict:
        """
        Add a new position to the portfolio with proper atomicity.
        All validation and updates happen within a single atomic operation.
        """
        if not isinstance(symbol, str):
            raise PortfolioError("Symbol must be a string.")
        if size <= Decimal("0") or entry_price <= Decimal("0"):
            raise PortfolioError("Size and entry price must be positive.")

        async with self._async_lock:
            # Check position limits
            if len(self.positions) >= self.risk_limits.get("max_positions", 10):
                raise PortfolioError(
                    f"Max positions limit reached: {self.risk_limits.get('max_positions', 10)}"
                )

            # Calculate values while holding the lock
            current_total = self.calculate_portfolio_value()  # This is now thread-safe
            position_value = size * entry_price

            # Validate position size against current portfolio value
            if current_total > Decimal("0"):
                max_position_size = self.risk_limits.get(
                    "max_position_size", Decimal("0.1")
                )
                if (position_value / current_total) > max_position_size:
                    raise PortfolioError(
                        f"Position size ({position_value / current_total:.2%}) exceeds maximum allowed ({max_position_size:.2%})"
                    )

            # Create and add the position atomically
            position_id = (
                trade_id
                if trade_id is not None
                else f"{symbol}_{int(time.time()*1000)}"
            )
            new_position = {
                "id": position_id,
                "symbol": symbol,
                "size": size,
                "entry_price": entry_price,
                "current_price": entry_price,
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "side": side or "long",
                "timestamp": int(time.time() * 1000),
                "status": "OPEN",
            }

            # Update portfolio state atomically
            self.positions[position_id] = Position(**new_position)
            self._last_update = 0  # Force portfolio value recalculation

            # Update portfolio metrics while still holding the lock
            await self._update_portfolio_metrics()

            return new_position

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
    async def update_position(self, position_id: str, current_price: Decimal) -> bool:
        async with self._async_lock:
            for pos in self.positions.values():
                if pos["id"] == position_id:
                    pos["current_price"] = current_price
                    pos["unrealized_pnl"] = pos["size"] * (
                        current_price - pos["entry_price"]
                    )
                    await self._update_portfolio_metrics()
                    return True
            self.logger.warning(
                f"Attempted to update non-existent position with id: {position_id}"
            )
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

    async def simulate_trade(self, trade_details: dict) -> None:
        """Simulate execution of a trade in paper mode, record the trade, and update the open position accordingly."""
        from decimal import Decimal

        # Record the trade details for monitoring
        self.trade_history.append(trade_details)
        side = trade_details.get("side", "").lower()
        symbol = trade_details.get("symbol")
        amount = Decimal(str(trade_details.get("amount", "0")))
        price = Decimal(str(trade_details.get("price", "0")))

        if side == "buy":
            # Open a new position
            success = await self.add_position(symbol, amount, price)
            if success:
                self.logger.info(
                    f"Simulated BUY executed: Opened position for {symbol} (amount={amount} at {price})."
                )
            else:
                self.logger.error(f"Failed to open position for {symbol}.")
        elif side == "sell":
            # Close an existing position if available
            closed_position = self.close_position(symbol, price)
            if closed_position:
                self.logger.info(
                    f"Simulated SELL executed: Closed position for {symbol} at {price}."
                )
            else:
                self.logger.warning(f"No open position to close for {symbol}.")
        else:
            self.logger.warning(f"Unknown trade side: {side}")

        # Record the trade in the database as an open trade (if not closed)
        try:
            from accounting.accounting import record_new_trade

            order = {"id": trade_details.get("id", str(len(self.trade_history)))}
            signal = {
                "symbol": symbol,
                "direction": side,
                "entry_price": price,
                "sl": trade_details.get("sl", 0),
                "tp": trade_details.get("tp", 0),
                "exchange": "paper",
            }
            ev = trade_details.get("ev", 0.0)
            kelly_frac = trade_details.get("kelly_frac", 0.0)
            position_size = trade_details.get("amount", 0)
            record_new_trade(
                order,
                signal,
                ev,
                kelly_frac,
                position_size,
                self.ctx,
                trade_source="paper",
            )
            self.logger.info(
                "Simulated trade recorded in database as an open paper trade."
            )
        except Exception as e:
            self.logger.error("Failed to record simulated trade in database: " + str(e))

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

    async def open_position(self, position_data: dict) -> dict:
        """Opens a new position. Supports two calling conventions:
        1) open_position(symbol, size, entry_price, trade_id=None)
        2) open_position(position_dict) where position_dict contains keys 'symbol', 'size', 'entry_price', and optionally 'id'.
        Returns the position dict on success.
        """
        # Ensure side is present in position_data
        if "side" not in position_data:
            position_data["side"] = "buy"  # default to buy if not provided
        # Check portfolio limits
        if len(self.positions) >= self.MAX_POSITIONS:
            from utils.exceptions import PortfolioError

            raise PortfolioError("Portfolio limit reached")
        # Call add_position with the side (explicitly passing it)
        position = self.add_position(
            position_data["symbol"],
            position_data["size"],
            position_data["entry_price"],
            position_data.get("id"),
            position_data["side"],
        )
        # Mark position as open
        position["status"] = "open"
        return position

    async def validate_position_size(self, position: dict) -> bool:
        """Validate position size against risk limits. Raises PortfolioError if invalid."""
        from utils.exceptions import PortfolioError

        max_size = self.risk_limits.get("max_position_size", Decimal("0.1"))
        if "size" not in position:
            raise PortfolioError("Position size not provided.")
        if position["size"] > max_size:
            raise PortfolioError("Position size exceeds maximum allowed.")
        return True

    async def calculate_portfolio_metrics(self) -> dict:
        stats = self.get_portfolio_stats()
        metrics = {
            "total_value": stats.total_value,
            "cash_balance": stats.cash_balance,
            "position_value": stats.position_value,
            "unrealized_pnl": stats.unrealized_pnl,
            "realized_pnl": stats.realized_pnl,
            "risk_ratio": stats.risk_ratio,
            "exposure": stats.exposure,
            "margin_used": stats.margin_used,
            "free_margin": stats.free_margin,
            "position_count": len(self.positions),
        }
        return metrics

    async def calculate_correlation(self, position: dict) -> Decimal:
        # Dummy implementation: return 0.9 to simulate high correlation
        return Decimal("0.9")

    async def validate_correlation(self, new_position: dict) -> None:
        # For demonstration, assume calculate_correlation is a method that returns a Decimal correlation value
        correlation = await self.calculate_correlation(new_position)
        max_correlation = self.risk_limits.get("max_correlation", Decimal("0.8"))
        if correlation > max_correlation:
            from utils.exceptions import PortfolioError

            raise PortfolioError(
                f"High correlation for {new_position.get('symbol', 'UNKNOWN')} exceeds maximum allowed."
            )
        return

    async def handle_emergency_closure(self, position_id: str) -> dict:
        # Handle emergency closure of a position
        if position_id in self.positions:
            self.positions[position_id]["status"] = "closed"
            # Additional emergency closure logic can be placed here
            return self.positions[position_id]
        else:
            from utils.exceptions import PortfolioError

            raise PortfolioError(
                f"Position with id {position_id} not found for emergency closure"
            )

    async def _reconcile_with_exchange(self) -> None:
        """
        Reconcile local portfolio state with exchange state.

        This ensures our portfolio calculations stay accurate.
        """
        try:
            if not hasattr(self.ctx, "exchange_interface"):
                raise PortfolioError("No exchange interface available")

            for attempt in range(self._sync_retries):
                try:
                    # Get exchange balance
                    exchange_balance = await self.ctx.exchange_interface.get_balance()
                    if exchange_balance is None:
                        raise PortfolioError("Failed to fetch exchange balance")

                    # Get exchange positions
                    exchange_positions = (
                        await self.ctx.exchange_interface.get_positions()
                    )

                    async with self._lock:
                        # Update balance
                        self._exchange_balance = exchange_balance
                        self.balance = exchange_balance
                        self._balance_synced = True

                        # Update positions
                        for symbol, pos in self.positions.items():
                            exch_pos = exchange_positions.get(symbol)
                            if exch_pos:
                                # Position exists on exchange
                                if abs(pos.size - exch_pos["size"]) > Decimal(
                                    "0.00001"
                                ):
                                    self.logger.warning(
                                        f"Position size mismatch for {symbol}. "
                                        f"Local: {pos.size}, Exchange: {exch_pos['size']}"
                                    )
                                    pos.size = exch_pos["size"]
                                pos.exchange_state_valid = True
                            else:
                                # Position doesn't exist on exchange
                                if pos.size != 0:
                                    self.logger.warning(
                                        f"Phantom position detected for {symbol}. "
                                        f"Local size: {pos.size}"
                                    )
                                    await self._handle_phantom_position(pos)

                        self._last_exchange_sync = time.time()
                        self._failed_syncs = 0
                        return

                except Exception as e:
                    if attempt < self._sync_retries - 1:
                        await asyncio.sleep(self._sync_backoff * (attempt + 1))
                        continue
                    raise e

            self._failed_syncs += 1
            if self._failed_syncs >= self._max_failed_syncs:
                # Trigger circuit breaker on repeated sync failures
                if hasattr(self.ctx, "circuit_breaker"):
                    await self.ctx.circuit_breaker.trigger_emergency_stop(
                        "Maximum portfolio sync failures reached"
                    )

        except Exception as e:
            self.logger.error(f"Failed to reconcile with exchange: {e}")
            self._balance_synced = False
            raise PortfolioError("Exchange reconciliation failed") from e

    async def _handle_phantom_position(self, position: Position) -> None:
        """Handle detection of a phantom position."""
        try:
            self.logger.error(
                f"Handling phantom position for {position.symbol} "
                f"with size {position.size}"
            )

            # Log the incident for audit
            if hasattr(self.ctx, "db_queries"):
                await self.ctx.db_queries.log_portfolio_incident(
                    {
                        "type": "phantom_position",
                        "symbol": position.symbol,
                        "size": str(position.size),
                        "timestamp": time.time(),
                    }
                )

            # Clear the phantom position
            position.size = Decimal("0")
            position.current_price = None
            position.unrealized_pnl = Decimal("0")
            position.exchange_state_valid = False

            # Trigger risk assessment
            if hasattr(self.ctx, "risk_manager"):
                await self.ctx.risk_manager.assess_portfolio_risk()

        except Exception as e:
            self.logger.error(f"Error handling phantom position: {e}")
            raise PortfolioError("Failed to handle phantom position") from e
