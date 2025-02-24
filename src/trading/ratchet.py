#! /usr/bin/env python3
# src/trading/ratchet.py
"""
Module: src.trading
Provides ratchet functionality.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from execution.exchange_interface import ExchangeInterface
from risk.limits import RiskLimits
from trading.position import Position
from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import RatchetError
from utils.numeric_handler import NumericHandler


# ratchet state class that represents the state of the ratchet
@dataclass
class RatchetState:
    """State tracking for ratchet system"""

    entry_price: Decimal
    current_stop: Decimal
    highest_price: Decimal
    lowest_price: Decimal
    current_level: int
    emergency_stop: Decimal
    position_size: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal
    last_update: float
    entry_time: float


# ratchet manager class that manages the ratchet
class RatchetManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self._lock = asyncio.Lock()
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.nh = NumericHandler()
        self.risk_limits = None
        self.max_ratchets = 1000  # Limit to prevent unbounded growth

        # Load thresholds from config
        self.thresholds = [
            self.nh.to_decimal(str(t)) for t in ctx.config.get("ratchet_thresholds", [])
        ]
        self.lock_ins = [
            self.nh.to_decimal(str(l)) for l in ctx.config.get("ratchet_lock_ins", [])
        ]
        self.emergency_stop_pct = Decimal(str(ctx.config.get("emergency_stop_pct", -2)))
        self.trailing_pct = Decimal(str(ctx.config.get("trailing_stop_pct", 1.5)))

        # Risk limits
        self.max_drawdown = Decimal(
            str(ctx.config.get("max_drawdown_pct", 10))
        ) / Decimal("100")
        self.max_adverse_pct = Decimal(
            str(ctx.config.get("max_adverse_pct", 3))
        ) / Decimal("100")
        self.max_hold_hours = Decimal(str(ctx.config.get("max_hold_hours", 8)))

        # Validation
        if len(self.thresholds) != len(self.lock_ins):
            raise ValueError("Ratchet thresholds and lock-ins must have same length")
        if not all(x < y for x, y in zip(self.thresholds[:-1], self.thresholds[1:])):
            raise ValueError("Ratchet thresholds must be ascending")

    # safe decimal function that safely converts a value to a decimal
    def safe_decimal(self, value) -> Decimal:
        """Safely convert value to Decimal"""
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except (TypeError, InvalidOperation):
            self.logger.warning(f"Could not convert {value} to Decimal")
            return Decimal("0")

    # initialize the ratchet manager
    async def initialize(self) -> bool:
        """Initialize ratchet manager"""
        try:
            if self.initialized:
                return True

            if (
                not self.ctx.portfolio_manager
                or not self.ctx.portfolio_manager.initialized
            ):
                self.logger.error("Portfolio manager must be initialized first")
                return False

            self.risk_limits = self.ctx.portfolio_manager.risk_limits
            self.initialized = True
            return True

        except Exception as e:
            await handle_error_async(e, "RatchetManager.initialize", self.logger)
            return False

    # update the trailing stops for all active trades
    async def update_trailing_stops(self) -> None:
        """Update trailing stops for all active trades with enhanced error handling"""
        try:
            if not self.initialized:
                await self.initialize()
                if not self.initialized:
                    return

            async with self._lock:
                failed_updates = []
                for trade_id, trade_data in list(self.active_trades.items()):
                    try:
                        current_price = await self._get_current_price(trade_id)
                        if not current_price:
                            failed_updates.append(trade_id)
                            self.logger.error(
                                f"Failed to fetch price for trade {trade_id}"
                            )
                            continue

                        # Check emergency stop conditions
                        drawdown = self._calculate_trade_drawdown(
                            trade_data, current_price
                        )
                        if drawdown >= self.risk_limits.emergency_stop_pct:
                            await self._close_trade(
                                trade_id,
                                f"Emergency stop triggered: drawdown {drawdown} >= {self.risk_limits.emergency_stop_pct}",
                            )
                            continue

                        # Update trailing stop if price has moved favorably
                        await self._update_single_trailing_stop(
                            trade_id, trade_data, current_price
                        )

                    except Exception as e:
                        failed_updates.append(trade_id)
                        await handle_error_async(
                            e,
                            f"RatchetManager.update_trailing_stops for trade {trade_id}",
                            self.logger,
                        )

                # Handle failed updates
                if failed_updates:
                    error_msg = (
                        f"Failed to update trailing stops for trades: {failed_updates}"
                    )
                    self.logger.error(error_msg)

                    # If we have too many failed updates, trigger circuit breaker
                    if len(failed_updates) >= len(self.active_trades) * Decimal(
                        "0.5"
                    ):  # 50% failure threshold
                        if hasattr(self.ctx, "circuit_breaker"):
                            await self.ctx.circuit_breaker.trigger_emergency_stop(
                                f"Critical failure: Multiple trailing stop updates failed - {error_msg}"
                            )
                        else:
                            # If no circuit breaker, close affected trades
                            for trade_id in failed_updates:
                                await self._close_trade(
                                    trade_id,
                                    "Emergency close: Failed to update trailing stop",
                                )

        except Exception as e:
            await handle_error_async(
                e, "RatchetManager.update_trailing_stops", self.logger
            )
            if hasattr(self.ctx, "circuit_breaker"):
                await self.ctx.circuit_breaker.trigger_emergency_stop(
                    f"Critical system error in trailing stop updates: {str(e)}"
                )

    async def _get_current_price(self, trade_id: str) -> Optional[Decimal]:
        """Get current price for a trade with enhanced error handling and retries"""
        MAX_RETRIES = 3
        RETRY_DELAY = 1  # seconds

        for attempt in range(MAX_RETRIES):
            try:
                if not self.ctx.market_data:
                    raise ValueError("Market data service not available")

                data = await self.ctx.market_data.fetch_market_data()
                if not data or "price" not in data:
                    raise ValueError("Invalid market data response")

                price = self.nh.to_decimal(data.get("price", 0))
                if price <= 0:
                    raise ValueError(f"Invalid price value: {price}")

                return price

            except Exception as e:
                self.logger.warning(
                    f"Price fetch attempt {attempt + 1}/{MAX_RETRIES} failed for {trade_id}: {str(e)}"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    self.logger.error(f"All price fetch attempts failed for {trade_id}")
                    return None

        return None

    def _calculate_trade_drawdown(
        self, trade_data: Dict[str, Any], current_price: Decimal
    ) -> Decimal:
        """Calculate drawdown for a single trade"""
        try:
            entry_price = self.nh.to_decimal(trade_data["entry_price"])
            if entry_price <= 0:
                return Decimal("0")

            return abs((current_price - entry_price) / entry_price)
        except Exception as e:
            self.logger.error(f"Error calculating trade drawdown: {str(e)}")
            return Decimal("0")

    async def _close_trade(self, trade_id: str, reason: str) -> None:
        """Close a trade due to risk limit breach"""
        try:
            self.logger.warning(f"Closing trade {trade_id}: {reason}")
            if trade_id in self.active_trades:
                # Implement trade closing logic here
                del self.active_trades[trade_id]
        except Exception as e:
            await handle_error_async(e, "RatchetManager._close_trade", self.logger)

    async def initialize_trade(
        self,
        trade_id: str,
        entry_price: Decimal,
        take_profit: Decimal,
        stop_loss: Decimal,
    ) -> bool:
        """Initialize a new trade with ratchet tracking"""
        try:
            if not self.initialized:
                await self.initialize()

            entry_price = self.safe_decimal(entry_price)
            take_profit = self.safe_decimal(take_profit)
            stop_loss = self.safe_decimal(stop_loss)

            if entry_price <= 0:
                raise RatchetError("Invalid entry price")

            self.active_trades[trade_id] = {
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "current_stop": stop_loss,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "current_level": 0,
                "entry_time": time.time(),
                "position_size": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "max_adverse_excursion": Decimal("0"),
            }

            self.logger.info(f"Initialized trade {trade_id}")
            return True

        except Exception as e:
            await handle_error_async(e, "RatchetManager.initialize_trade", self.logger)
            return False

    # normalize the trade id
    def _normalize_trade_id(self, trade_id: str, symbol: str) -> str:
        return f"{symbol}_{trade_id}"

    # update the position ratchet
    async def update_position_ratchet(
        self, symbol: str, current_price: Decimal, additional_data: Dict[str, Any]
    ) -> Decimal:
        normalized_id = self._get_trade_id(symbol)
        if normalized_id not in self.active_trades:
            raise RatchetError(f"No active trade found for symbol: {symbol}")

        trade = self.active_trades[normalized_id]
        new_stop = current_price * Decimal("0.99")  # Example logic for updating stop
        trade["current_stop"] = new_stop
        self.logger.info(f"Updated ratchet for {normalized_id}: new_stop={new_stop}")
        return new_stop

    # get the trade id
    def _get_trade_id(self, symbol: str) -> Optional[str]:
        for trade_id, details in self.active_trades.items():
            if details["symbol"] == symbol:
                return trade_id
        return None

    # monitor the trades
    async def monitor_trades(self, exchange: ExchangeInterface):
        while True:
            try:
                for trade_id, trade in list(self.active_trades.items()):
                    symbol = trade["symbol"]
                    ticker = await exchange.fetch_ticker(symbol)
                    if ticker is None:
                        continue
                    current_price = Decimal(str(ticker))
                    if current_price < trade["current_stop"]:
                        await exchange.close_position(symbol, trade["current_stop"])
                        del self.active_trades[trade_id]
                        self.logger.info(
                            f"Closed position for {trade_id} due to stop loss."
                        )
                    else:
                        await self.update_position_ratchet(symbol, current_price, {})
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                await handle_error_async(
                    e, "RatchetManager.monitor_trades", self.logger
                )
                await asyncio.sleep(5)  # Back off on error

    # get the trade metrics
    def get_trade_metrics(self, trade_id: str) -> Dict[str, Any]:
        """Get current metrics for trade"""
        if trade_id not in self.active_trades:
            return {}

        try:
            trade = self.active_trades[trade_id]
            hold_time = (time.time() - trade["entry_time"]) / 3600

            return {
                "entry_price": float(trade["entry_price"]),
                "current_stop": float(trade["current_stop"]),
                "highest_price": float(trade["highest_price"]),
                "lowest_price": float(trade["lowest_price"]),
                "current_level": trade["current_level"],
                "unrealized_pnl": float(trade["entry_price"] - trade["current_stop"])
                * float(trade["position_size"]),
                "max_adverse_excursion": float(
                    (trade["highest_price"] - trade["entry_price"])
                    / trade["entry_price"]
                ),
                "max_excursion": float(
                    (trade["highest_price"] - trade["entry_price"])
                    / trade["entry_price"]
                ),
                "current_drawdown": float(
                    (trade["highest_price"] - trade["current_stop"])
                    / trade["highest_price"]
                ),
                "hold_time_hours": round(hold_time, 1),
            }
        except Exception as e:
            handle_error(e, "RatchetManager.get_trade_metrics", logger=self.logger)
            return {}

    # remove the trade
    def remove_trade(self, trade_id: str) -> None:
        """Clean up trade tracking"""
        try:
            if trade_id in self.active_trades:
                metrics = self.get_trade_metrics(trade_id)
                self.logger.info(
                    f"Removing trade {trade_id} tracking. " f"Final metrics: {metrics}"
                )
                del self.active_trades[trade_id]
        except Exception as e:
            handle_error(e, "RatchetManager.remove_trade", logger=self.logger)

    # get the status report
    def get_status_report(self) -> Dict[str, Any]:
        """Get status report for all tracked trades"""
        return {
            trade_id: self.get_trade_metrics(trade_id)
            for trade_id in self.active_trades
        }
