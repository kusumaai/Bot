#!/usr/bin/env python3
# src/risk/manager.py
"""
Module: src.risk
Provides risk management functionality.
"""

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from trading.portfolio import PortfolioManager
from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import (
    MathError,
    RiskCalculationError,
    RiskManagerError,
    ValidationError,
)
from utils.numeric_handler import NumericHandler

from .limits import RiskLimits
from .validation import MarketDataValidation


class RiskManager:
    """Risk manager class"""

    def __init__(
        self, risk_limits: RiskLimits, db_queries: Any, logger: logging.Logger
    ):
        self.risk_limits = risk_limits
        self.db_queries = db_queries
        self.logger = logger
        self._lock = asyncio.Lock()
        self.initialized = False
        self.nh = NumericHandler(logger)
        self.portfolio = PortfolioManager(db_queries, logger)

        # These will be initialized during initialize()
        self.position_limits = None
        self._last_risk_check = time.time()
        self._daily_loss_start = datetime.now().date()
        self._daily_loss = Decimal("0")

    async def check_risk_limits(self, portfolio_value: Decimal) -> bool:
        """Check if any risk limits are breached"""
        try:
            async with self._lock:
                # Reset daily loss counter if new day
                current_date = datetime.now().date()
                if current_date > self._daily_loss_start:
                    self._daily_loss_start = current_date
                    self._daily_loss = Decimal("0")

                # Calculate current drawdown
                drawdown = await self._calculate_drawdown(portfolio_value)
                if drawdown >= self.risk_limits.emergency_stop_pct:
                    self.logger.error(f"Emergency stop limit breached: {drawdown}")
                    return False

                # Check daily loss
                if self._daily_loss >= self.risk_limits.max_daily_loss:
                    self.logger.error(f"Daily loss limit breached: {self._daily_loss}")
                    return False

                return True

        except Exception as e:
            await handle_error_async(e, "RiskManager.check_risk_limits", self.logger)
            return False

    def validate_position(
        self, symbol: str, size: Decimal, price: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Validate a new position against all risk limits"""
        try:
            # Check max positions
            if len(self.portfolio.positions) >= self.risk_limits["max_positions"]:
                return False, "Maximum positions limit reached"

            # Check position size
            position_value = size * price
            portfolio_value = self.portfolio.calculate_portfolio_value()
            if (
                portfolio_value > Decimal("0")
                and (position_value / portfolio_value)
                > self.risk_limits["max_position_size"]
            ):
                return False, "Position size exceeds maximum allowed"

            # Check leverage
            total_exposure = (
                sum(p.size * p.current_price for p in self.portfolio.positions.values())
                + position_value
            )

            if (
                portfolio_value > Decimal("0")
                and (total_exposure / portfolio_value)
                > self.risk_limits["max_position_size"]
            ):
                return False, "Leverage limit exceeded"

            # Check drawdown
            if self.portfolio.calculate_drawdown() > self.risk_limits["max_drawdown"]:
                return False, "Maximum drawdown reached"

            # Check correlation if multiple positions
            if len(self.portfolio.positions) > 0:
                correlation = self._calculate_position_correlation(symbol)
                if correlation > self.risk_limits["max_position_size"]:
                    return False, "Position correlation too high"

            return True, None

        except Exception as e:
            handle_error(e, "RiskManager.validate_position", logger=self.logger)
            return False, f"Error validating position: {str(e)}"

    def calculate_position_params(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position parameters with risk limits"""
        try:
            if not self.validator.validate_market_data(signal["symbol"], market_data):
                self.logger.warning(
                    f"Market data validation failed for {signal['symbol']}"
                )
                return {"size": Decimal("0")}

            current_price = Decimal(str(market_data["current_price"]))

            # Calculate expected value
            ev = self._calculate_expected_value(signal, market_data)
            if ev <= 0:
                self.logger.info(
                    f"Expected value non-positive for {signal['symbol']}. Skipping trade."
                )
                return {"size": Decimal("0")}

            # Calculate position size with Kelly
            size = self._calculate_kelly_position_size(
                Decimal(str(signal.get("probability", "0.5"))), ev, current_price
            )

            # Apply risk factor scaling
            size *= self.risk_limits["max_position_size"]

            # Calculate stops based on volatility (ATR)
            atr = Decimal(
                str(market_data.get("ATR_14", self.risk_limits["max_position_size"]))
            )
            stop_loss = current_price * (Decimal("1") - atr)
            take_profit = current_price * (Decimal("1") + atr * Decimal("2"))

            return {
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": self.risk_limits["max_position_size"],
            }

        except Exception as e:
            handle_error(e, "RiskManager.calculate_position_params", logger=self.logger)
            return {}

    def update_positions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update all positions and return any that need closing"""
        try:
            positions_to_close = []

            for symbol, position in self.portfolio.positions.items():
                if symbol not in market_data:
                    continue

                current_price = Decimal(str(market_data[symbol]["current_price"]))
                position.update_price(current_price)

                # Check stop conditions
                if self._check_stop_conditions(position):
                    positions_to_close.append(
                        {"symbol": symbol, "reason": "stop_loss", "position": position}
                    )

                # Check max adverse excursion
                if position.unrealized_pnl < -self.risk_limits["max_position_size"]:
                    positions_to_close.append(
                        {
                            "symbol": symbol,
                            "reason": "max_adverse_excursion",
                            "position": position,
                        }
                    )

            return positions_to_close

        except Exception as e:
            handle_error(e, "RiskManager.update_positions", logger=self.logger)
            return []

    def _calculate_position_correlation(self, new_symbol: str) -> Decimal:
        """Calculate correlation between positions (placeholder implementation)"""
        # Placeholder: implement actual correlation logic using historical price data
        return Decimal("0")

    def _calculate_expected_value(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate trade expected value"""
        try:
            probability = Decimal(str(signal.get("probability", "0.5")))
            potential_profit = Decimal(str(signal.get("potential_profit", "0")))
            potential_loss = Decimal(str(signal.get("potential_loss", "0")))

            return (probability * potential_profit) - (
                (Decimal("1") - probability) * potential_loss
            )

        except Exception as e:
            handle_error(e, "RiskManager._calculate_expected_value", logger=self.logger)
            return Decimal("0")

    def _calculate_kelly_position_size(
        self, probability: Decimal, expected_value: Decimal, price: Decimal
    ) -> Decimal:
        """Calculate position size using Kelly Criterion"""
        try:
            if expected_value <= Decimal("0"):
                return Decimal("0")

            kelly_fraction = probability - (
                (Decimal("1") - probability) / (expected_value / price)
            )
            kelly_fraction = max(Decimal("0"), min(kelly_fraction, Decimal("1")))

            # Apply Kelly scaling factor
            return kelly_fraction * self.risk_limits["max_position_size"]

        except Exception as e:
            handle_error(
                e, "RiskManager._calculate_kelly_position_size", logger=self.logger
            )
            return Decimal("0")

    def _check_stop_conditions(self, position: Any) -> bool:
        """Check if position should be stopped out"""
        return (
            position.side == "long" and position.current_price <= position.stop_loss
        ) or (position.side == "short" and position.current_price >= position.stop_loss)

    async def calculate_position_size(
        self, signal: Dict[str, Any], current_price: Decimal
    ) -> Decimal:
        """Calculate safe position size with all risk checks."""
        try:
            price = self.nh.convert_to_decimal(signal.get("price"))
            account_size = await self.portfolio.get_total_value()
            risk_factor = self.nh.percentage_to_decimal(self.risk_limits.risk_factor)

            position_size = account_size * risk_factor
            max_allowed = self.risk_limits.max_position_size * account_size
            return min(position_size, max_allowed).quantize(Decimal("0.0001"))
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return Decimal("0")
        except Exception as e:
            self.logger.error(f"Unexpected error in calculate_position_size: {e}")
            return Decimal("0")

    async def calculate_kelly_fraction(
        self, probability: Decimal, odds: Decimal
    ) -> Decimal:
        """Calculates the Kelly fraction."""
        try:
            if odds <= Decimal("0"):
                return Decimal("0")
            kelly = (probability * (odds + Decimal("1")) - Decimal("1")) / odds
            return max(Decimal("0"), min(kelly, Decimal("1")))
        except Exception as e:
            await handle_error_async(e, "calculate_kelly_fraction", self.logger)
            raise MathError(f"Error calculating Kelly fraction: {e}")

    async def validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate an order before execution"""
        async with self._lock:
            try:
                amount = order.get("amount")
                symbol = order.get("symbol")

                if amount is None or symbol is None:
                    raise RiskManagerError("Missing 'amount' or 'symbol' in order.")

                position_size = Decimal(str(amount))

                if not await self._check_position_limits(symbol, position_size):
                    self.logger.warning("Position limits check failed.")
                    return False

                if not await self._check_risk_limits(position_size):
                    self.logger.warning("Risk limits check failed.")
                    return False

                return True

            except (RiskManagerError, InvalidOperation, TypeError) as e:
                self.logger.error(f"Order validation failed: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in validate_order: {e}")
                return False

    async def _check_position_limits(self, symbol: str, size: Decimal) -> bool:
        """Check position limits for a given symbol"""
        limits = self.position_limits.get(symbol, {})
        try:
            min_size = Decimal(str(limits.get("min_qty", "0")))
            max_size = Decimal(str(limits.get("max_qty", "Infinity")))
            return min_size <= size <= max_size
        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Error checking position limits for {symbol}: {e}")
            return False

    async def _check_risk_limits(self, position_size: Decimal) -> bool:
        """Check overall risk limits"""
        try:
            total_value = await self.portfolio.get_total_value()
            if total_value <= Decimal("0"):
                self.logger.warning("Total portfolio value is zero or negative.")
                return False
            position_ratio = self.nh.safe_divide(position_size, total_value)
            return position_ratio <= self.risk_limits["max_position_size"]
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in _check_risk_limits: {e}")
            return False

    async def get_account_size(self) -> Decimal:
        return await self.portfolio.get_total_value()

    async def initialize(self) -> bool:
        """Initialize risk manager"""
        try:
            if self.initialized:
                return True
            self.position_limits = self._load_position_limits()
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize risk manager: {e}")
            return False

    def _load_position_limits(self) -> Dict[str, Any]:
        """Load position limits from configuration"""
        try:
            # Get position limits from config
            position_limits = {}

            # Load from config if available
            if hasattr(self.ctx, "config") and "position_limits" in self.ctx.config:
                raw_limits = self.ctx.config["position_limits"]

                # Convert string values to Decimal
                for symbol, limits in raw_limits.items():
                    position_limits[symbol] = {
                        "min_qty": Decimal(str(limits.get("min_qty", "0"))),
                        "max_qty": Decimal(str(limits.get("max_qty", "0"))),
                    }

            # Add default limits if none configured
            if not position_limits:
                self.logger.warning(
                    "No position limits found in config, using defaults"
                )
                position_limits = {
                    "default": {
                        "min_qty": Decimal("0.001"),
                        "max_qty": Decimal("100.0"),
                    }
                }

            return position_limits

        except (InvalidOperation, TypeError) as e:
            self.logger.error(f"Error loading position limits: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error loading position limits: {e}")
            return {}

    async def check_system_readiness(self) -> Tuple[bool, Dict[str, bool]]:
        """Check if system components are ready for risk management"""
        readiness = {
            "position_limits": False,
            "risk_limits": False,
            "portfolio": False,
            "overall": False,
        }

        try:
            # Check position limits loaded
            if self.position_limits and len(self.position_limits) > 0:
                readiness["position_limits"] = True

            # Check risk limits initialized
            if self.risk_limits:
                readiness["risk_limits"] = True

            # Check portfolio manager initialized
            if self.portfolio and self.portfolio.initialized:
                readiness["portfolio"] = True

            # Overall readiness requires all components
            readiness["overall"] = all(
                ready
                for component, ready in readiness.items()
                if component != "overall"
            )

            # Log missing components if not ready
            if not readiness["overall"]:
                missing = [
                    comp
                    for comp, ready in readiness.items()
                    if not ready and comp != "overall"
                ]
                self.logger.info(
                    f"Risk manager not ready. Missing: {', '.join(missing)}"
                )

            return readiness["overall"], readiness

        except Exception as e:
            self.logger.warning(f"Could not check risk system readiness: {e}")
            return False, readiness

    async def validate_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """Validates risk metrics against defined risk limits."""
        try:
            # Validate value against max limit
            if metrics["value"] > self.risk_limits.max_value:
                raise ValidationError("Risk metric value exceeds maximum allowed.")

            # Validate drawdown percentage
            if metrics.get("drawdown", 0) > self.risk_limits.max_drawdown:
                raise ValidationError("Drawdown exceeds maximum allowed threshold.")

            # Validate leverage ratio
            if metrics.get("leverage", 0) > self.risk_limits.max_leverage:
                raise ValidationError("Leverage ratio exceeds maximum allowed.")

            # Validate position concentration
            if (
                metrics.get("position_concentration", 0)
                > self.risk_limits.max_position_size
            ):
                raise ValidationError(
                    "Position concentration exceeds diversification limits."
                )

            # Validate daily loss
            if metrics.get("daily_loss", 0) > self.risk_limits.max_daily_loss:
                raise ValidationError("Daily loss exceeds maximum allowed threshold.")

            # Validate system metrics
            if metrics.get("cpu_usage", 0) > 85 or metrics.get("memory_usage", 0) > 90:
                raise ValidationError("System resource usage exceeds safe thresholds.")
        except KeyError as e:
            await handle_error_async(
                e, "RiskManager.validate_risk_metrics", self.logger
            )
            raise ValidationError(f"Missing key in metrics: {e}")
        except Exception as e:
            await handle_error_async(
                e, "RiskManager.validate_risk_metrics", self.logger
            )
            raise ValidationError(f"Error validating risk metrics: {e}")
