#! /usr/bin/env python3
# src/risk/validation.py
"""
Module: src.risk
Provides comprehensive market data validation with enhanced validation chains.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.bot_types.base_types import MarketState, Validatable, ValidationResult
from src.signals.market_state import prepare_market_state
from src.utils.error_handler import ValidationError, handle_error
from src.utils.numeric_handler import NumericHandler
from trading.position import Position
from utils.helpers import vol_estimate

from .limits import RiskLimits


@dataclass
class MarketDataValidation(Validatable):
    """Market data validation with enhanced validation chains"""

    risk_limits: RiskLimits
    logger: Optional[logging.Logger] = None
    _validation_history: List[Dict[str, Any]] = field(default_factory=list)
    _max_history_size: int = 1000
    _validation_thresholds: Dict[str, Any] = field(default_factory=dict)
    _validation_timeouts: Dict[str, float] = field(default_factory=dict)
    _volatility_cache: Dict[str, Tuple[float, Decimal]] = field(default_factory=dict)
    _cache_timeout: int = 300  # 5 minutes

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self.nh = NumericHandler()
        self._init_validation_timeouts()
        self._init_base_thresholds()

    def _init_validation_timeouts(self):
        """Initialize validation timeouts"""
        self._validation_timeouts.update(
            {
                "basic_validation": 2.0,  # 2 seconds
                "enhanced_validation": 3.0,  # 3 seconds
                "risk_validation": 2.0,  # 2 seconds
                "total_validation": 10.0,  # 10 seconds total
            }
        )

    def _init_base_thresholds(self):
        """Initialize base thresholds that will be adjusted dynamically"""
        self._validation_thresholds.update(
            {
                "min_price": Decimal("0"),
                "min_volume": Decimal("0"),
                "base_price_change": Decimal("0.02"),  # 2% base price change
                "base_volume_change": Decimal("0.1"),  # 10% base volume change
                "base_spread": Decimal("0.01"),  # 1% base spread
                "staleness_threshold": self._calculate_staleness_threshold(),
                "min_tick_size": Decimal("0.00000001"),
                "max_consecutive_failures": 3,
                "volatility_scaling_factor": Decimal(
                    "2.5"
                ),  # Scale thresholds by 2.5x volatility
                "min_volatility": Decimal("0.001"),  # Minimum volatility floor
            }
        )

    def _calculate_staleness_threshold(self) -> float:
        """Calculate dynamic staleness threshold based on market volatility"""
        # Default to 5 seconds, but should be adjusted based on market conditions
        base_threshold = 5.0
        # TODO: Implement dynamic adjustment based on market volatility
        return base_threshold

    async def get_dynamic_thresholds(
        self, symbol: str, data: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Calculate dynamic thresholds based on asset volatility"""
        try:
            # Get current volatility
            current_vol = self._get_current_volatility(data)

            # Use cached volatility if current not available
            if current_vol <= self._validation_thresholds["min_volatility"]:
                current_vol = await self._get_cached_volatility(symbol)

            # Scale thresholds based on volatility
            scaling = (
                max(current_vol, self._validation_thresholds["min_volatility"])
                * self._validation_thresholds["volatility_scaling_factor"]
            )

            return {
                "max_price_change": self._validation_thresholds["base_price_change"]
                * scaling,
                "max_volume_change": self._validation_thresholds["base_volume_change"]
                * scaling,
                "max_spread": self._validation_thresholds["base_spread"] * scaling,
            }
        except Exception as e:
            self.logger.error(f"Error calculating dynamic thresholds: {e}")
            # Fallback to base thresholds
            return {
                "max_price_change": self._validation_thresholds["base_price_change"],
                "max_volume_change": self._validation_thresholds["base_volume_change"],
                "max_spread": self._validation_thresholds["base_spread"],
            }

    def _get_current_volatility(self, data: Dict[str, Any]) -> Decimal:
        """Extract current volatility from market data"""
        try:
            if "volatility" in data:
                return self.nh.to_decimal(data["volatility"])
            if "market_state" in data and hasattr(data["market_state"], "volatility"):
                return self.nh.to_decimal(str(data["market_state"].volatility))
            return self._validation_thresholds["min_volatility"]
        except Exception as e:
            self.logger.error(f"Error getting current volatility: {e}")
            return self._validation_thresholds["min_volatility"]

    async def _get_cached_volatility(self, symbol: str) -> Decimal:
        """Get cached volatility for symbol with async update"""
        try:
            current_time = time.time()
            if symbol in self._volatility_cache:
                cache_time, vol = self._volatility_cache[symbol]
                if current_time - cache_time < self._cache_timeout:
                    return vol

            # Calculate new volatility using helper function
            new_vol = await vol_estimate(symbol, self.ctx)
            self._volatility_cache[symbol] = (current_time, new_vol)
            return new_vol
        except Exception as e:
            self.logger.error(f"Error getting cached volatility: {e}")
            return self._validation_thresholds["min_volatility"]

    async def validate_market_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate market data with comprehensive checks and timeouts"""
        try:
            if not data:
                self._record_validation_failure("Empty market data received")
                return ValidationResult(
                    is_valid=False, error_message="Empty market data received"
                )

            # Overall validation timeout
            try:
                async with asyncio.timeout(
                    self._validation_timeouts["total_validation"]
                ):
                    # Basic validation chain with timeout
                    basic_result = await self._validate_with_timeout(
                        self._validate_basic_requirements,
                        data,
                        self._validation_timeouts["basic_validation"],
                    )
                    if not basic_result.is_valid:
                        self._record_validation_failure(basic_result.error_message)
                        return basic_result

                    # Enhanced validation chain with timeout
                    enhanced_result = await self._validate_with_timeout(
                        self._validate_enhanced_requirements,
                        data,
                        self._validation_timeouts["enhanced_validation"],
                    )
                    if not enhanced_result.is_valid:
                        self._record_validation_failure(enhanced_result.error_message)
                        return enhanced_result

                    # Risk limit validation chain with timeout
                    risk_result = await self._validate_with_timeout(
                        self._validate_risk_limits,
                        data,
                        self._validation_timeouts["risk_validation"],
                    )
                    if not risk_result.is_valid:
                        self._record_validation_failure(risk_result.error_message)
                        return risk_result

                    self._record_validation_success()
                    return ValidationResult(is_valid=True)

            except asyncio.TimeoutError:
                error_msg = "Market data validation timed out"
                self._record_validation_failure(error_msg)
                return ValidationResult(is_valid=False, error_message=error_msg)

        except Exception as e:
            error_msg = f"Market data validation failed: {str(e)}"
            handle_error(e, "MarketDataValidation.validate_market_data", self.logger)
            self._record_validation_failure(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)

    async def _validate_with_timeout(
        self, validation_func, data: Dict[str, Any], timeout: float
    ) -> ValidationResult:
        """Execute validation function with timeout"""
        try:
            async with asyncio.timeout(timeout):
                if asyncio.iscoroutinefunction(validation_func):
                    return await validation_func(data)
                return validation_func(data)
        except asyncio.TimeoutError:
            return ValidationResult(
                is_valid=False,
                error_message=f"{validation_func.__name__} timed out after {timeout}s",
            )

    def update_validation_timeout(self, validation_type: str, timeout: float):
        """Update validation timeout for a specific validation type"""
        if validation_type not in self._validation_timeouts:
            raise ValueError(f"Unknown validation type: {validation_type}")
        self._validation_timeouts[validation_type] = timeout

    def update_staleness_threshold(self, threshold: float):
        """Update staleness threshold"""
        self._validation_thresholds["staleness_threshold"] = threshold

    def _validate_basic_requirements(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate basic market data requirements"""
        try:
            # Required fields check
            required_fields = ["symbol", "price", "volume", "timestamp"]
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required fields: {missing_fields}",
                )

            # Numeric value validation
            price_val = data.get("price")
            if price_val is None:
                price_val = data.get("current_price", 0)
            try:
                price = self.nh.to_decimal(price_val)
                volume = self.nh.to_decimal(data.get("volume", 0))
            except (ValueError, TypeError) as e:
                return ValidationResult(
                    is_valid=False, error_message=f"Invalid numeric values: {str(e)}"
                )

            # Basic value checks
            if price <= self._validation_thresholds["min_price"]:
                return ValidationResult(
                    is_valid=False, error_message=f"Invalid price: {price}"
                )

            if volume < self._validation_thresholds["min_volume"]:
                return ValidationResult(
                    is_valid=False, error_message=f"Invalid volume: {volume}"
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Basic validation failed: {str(e)}"
            )

    async def _validate_enhanced_requirements(
        self, data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate enhanced market data requirements with dynamic thresholds"""
        try:
            symbol = data.get("symbol", "")
            if not symbol:
                return ValidationResult(
                    is_valid=False, error_message="Missing symbol in market data"
                )

            # Get dynamic thresholds for this symbol
            thresholds = await self.get_dynamic_thresholds(symbol, data)

            # Timestamp freshness
            timestamp = float(data.get("timestamp", 0))
            if (
                time.time() - timestamp
                > self._validation_thresholds["staleness_threshold"]
            ):
                return ValidationResult(
                    is_valid=False, error_message="Stale market data"
                )

            # Price change validation with dynamic threshold
            if "last_price" in data:
                last_price = self.nh.to_decimal(data["last_price"])
                current_price = self.nh.to_decimal(data["price"])
                price_change = abs(current_price - last_price) / last_price
                if price_change > thresholds["max_price_change"]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Excessive price change: {price_change:.2%} > {thresholds['max_price_change']:.2%}",
                    )

            # Volume change validation with dynamic threshold
            if "last_volume" in data:
                last_volume = self.nh.to_decimal(data["last_volume"])
                current_volume = self.nh.to_decimal(data["volume"])
                volume_change = abs(current_volume - last_volume) / last_volume
                if volume_change > thresholds["max_volume_change"]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Excessive volume change: {volume_change:.2%} > {thresholds['max_volume_change']:.2%}",
                    )

            # Spread validation with dynamic threshold
            if "bid" in data and "ask" in data:
                bid = self.nh.to_decimal(data["bid"])
                ask = self.nh.to_decimal(data["ask"])
                spread = (ask - bid) / bid
                if spread > thresholds["max_spread"]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Excessive spread: {spread:.2%} > {thresholds['max_spread']:.2%}",
                    )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Enhanced validation failed: {str(e)}"
            )

    def _validate_risk_limits(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate market data against risk limits"""
        try:
            # Emergency stop check
            if "drawdown" in data:
                drawdown = self.nh.to_decimal(data["drawdown"])
                if drawdown >= self.risk_limits.emergency_stop_pct:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Emergency stop triggered: drawdown {drawdown:.2%}",
                    )

            # Position size check
            if "position_size" in data:
                position_size = self.nh.to_decimal(data["position_size"])
                if position_size > self.risk_limits.max_position_size:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Position size exceeds limit: {position_size}",
                    )

            # Volatility check
            if "volatility" in data:
                volatility = self.nh.to_decimal(data["volatility"])
                if volatility > self.risk_limits.max_volatility:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Volatility exceeds limit: {volatility:.2%}",
                    )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Risk limit validation failed: {str(e)}"
            )

    def _record_validation_failure(self, error_message: str):
        """Record validation failure"""
        self._validation_history.append(
            {"timestamp": datetime.now(), "success": False, "error": error_message}
        )
        self._trim_history()

    def _record_validation_success(self):
        """Record successful validation"""
        self._validation_history.append({"timestamp": datetime.now(), "success": True})
        self._trim_history()

    def _trim_history(self):
        """Trim validation history to maximum size"""
        if len(self._validation_history) > self._max_history_size:
            self._validation_history = self._validation_history[
                -self._max_history_size :
            ]

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self._validation_history:
            return {
                "total": 0,
                "success_rate": 0,
                "recent_failures": 0,
                "thresholds": dict(self._validation_thresholds),
            }

        total = len(self._validation_history)
        successes = sum(1 for entry in self._validation_history if entry["success"])
        recent = self._validation_history[-100:]  # Last 100 entries
        recent_failures = sum(1 for entry in recent if not entry["success"])

        return {
            "total": total,
            "success_rate": successes / total if total > 0 else 0,
            "recent_failures": recent_failures,
            "thresholds": dict(self._validation_thresholds),
        }

    def set_validation_threshold(self, threshold_name: str, value: Any):
        """Update validation threshold"""
        if threshold_name not in self._validation_thresholds:
            raise ValueError(f"Unknown threshold: {threshold_name}")
        self._validation_thresholds[threshold_name] = value


class OrderValidator(Validatable):
    """Centralized order validation"""

    def validate_order(self, order: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters"""
        try:
            required_fields = ["symbol", "side", "type", "size"]
            missing_fields = [f for f in required_fields if f not in order]
            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required fields: {missing_fields}",
                )

            validations = [
                (order["side"] in ["buy", "sell"], "Invalid order side"),
                (
                    order["type"] in ["market", "limit", "stop", "take_profit"],
                    "Invalid order type",
                ),
                (
                    Decimal(str(order["size"])) > Decimal("0"),
                    "Order size must be positive",
                ),
                (
                    not order.get("price")
                    or Decimal(str(order["price"])) > Decimal("0"),
                    "Order price must be positive",
                ),
                (self._validate_order_limits(order), "Order exceeds limits"),
            ]

            for condition, message in validations:
                if not condition:
                    return ValidationResult(is_valid=False, error_message=message)
            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Order validation failed: {str(e)}"
            )


def validate_market_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate market data freshness and integrity with comprehensive OHLCV checks"""
    try:
        # Validate required fields
        required_fields = ["symbol", "price", "volume", "timestamp"]
        if "ohlcv" in data:
            required_fields.extend(["open", "high", "low", "close"])
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                error_message=f"Missing required fields: {missing_fields}",
            )

        # Validate numeric values and OHLCV relationships
        try:
            price = Decimal(str(data["price"]))
            volume = Decimal(str(data["volume"]))

            # Basic value validation
            if price <= 0 or volume < 0:
                return ValidationResult(
                    is_valid=False, error_message="Invalid price or volume"
                )

            # Validate OHLCV relationships if present
            if "ohlcv" in data:
                ohlcv_result = validate_ohlcv_relationships(data)
                if not ohlcv_result.is_valid:
                    return ohlcv_result

            # Validate price consistency if both price and close are present
            if "close" in data and abs(price - Decimal(str(data["close"]))) > Decimal(
                "0.0001"
            ):
                return ValidationResult(
                    is_valid=False,
                    error_message="Price inconsistency between current price and close price",
                )

        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False, error_message=f"Invalid numeric values: {str(e)}"
            )

        # Validate timestamp freshness
        timestamp = float(data.get("timestamp", 0))
        if time.time() - timestamp > 5:  # 5 second staleness check
            return ValidationResult(is_valid=False, error_message="Stale data")

        return ValidationResult(is_valid=True)

    except Exception as e:
        return ValidationResult(
            is_valid=False, error_message=f"Market data validation failed: {str(e)}"
        )


def validate_ohlcv_relationships(data: Dict[str, Any]) -> ValidationResult:
    """Validate OHLCV data relationships and integrity"""
    try:
        # Convert all OHLCV values to Decimal for precise comparison
        open_price = Decimal(str(data["open"]))
        high_price = Decimal(str(data["high"]))
        low_price = Decimal(str(data["low"]))
        close_price = Decimal(str(data["close"]))
        volume = Decimal(str(data["volume"]))

        # Validate basic value constraints
        if any(
            price <= 0 for price in [open_price, high_price, low_price, close_price]
        ):
            return ValidationResult(
                is_valid=False, error_message="All OHLCV prices must be positive"
            )

        if volume < 0:
            return ValidationResult(
                is_valid=False, error_message="Volume cannot be negative"
            )

        # Validate price relationships
        validations = [
            (
                high_price >= open_price,
                "High price must be greater than or equal to open price",
            ),
            (
                high_price >= close_price,
                "High price must be greater than or equal to close price",
            ),
            (
                high_price >= low_price,
                "High price must be greater than or equal to low price",
            ),
            (
                low_price <= open_price,
                "Low price must be less than or equal to open price",
            ),
            (
                low_price <= close_price,
                "Low price must be less than or equal to close price",
            ),
        ]

        for condition, message in validations:
            if not condition:
                return ValidationResult(is_valid=False, error_message=message)

        # Validate price range sanity
        price_range = high_price - low_price
        avg_price = (high_price + low_price) / 2
        if price_range > avg_price * Decimal("0.5"):  # 50% range check
            return ValidationResult(
                is_valid=False,
                error_message=f"Suspicious price range: {price_range/avg_price:.2%} of average price",
            )

        # Check for tick alignment if tick size is known
        tick_size = data.get("tick_size")
        if tick_size:
            tick_size = Decimal(str(tick_size))
            for price in [open_price, high_price, low_price, close_price]:
                if (price % tick_size) != 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Price {price} not aligned with tick size {tick_size}",
                    )

        return ValidationResult(is_valid=True)

    except (ValueError, TypeError) as e:
        return ValidationResult(
            is_valid=False, error_message=f"OHLCV validation failed: {str(e)}"
        )


def validate_risk_parameters(params: Dict[str, Any]) -> ValidationResult:
    """Validate risk parameters against limits"""
    try:
        required_params = [
            ("position_size", Decimal("0.5")),  # Max 50% position size
            ("leverage", Decimal("3")),  # Max 3x leverage
            ("stop_loss_pct", Decimal("0.1")),  # Max 10% stop loss
            ("take_profit_pct", Decimal("0.3")),  # Max 30% take profit
        ]

        for param, max_value in required_params:
            if param not in params:
                return ValidationResult(
                    is_valid=False, error_message=f"Missing required parameter: {param}"
                )

            try:
                value = Decimal(str(params[param]))
                if value <= 0 or value > max_value:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid {param}: {value} (max: {max_value})",
                    )
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False, error_message=f"Invalid numeric value for {param}"
                )

        return ValidationResult(is_valid=True)

    except Exception as e:
        handle_error(e, "validate_risk_parameters")
        return ValidationResult(is_valid=False, error_message=str(e))


def validate_portfolio_limits(
    portfolio_value: Decimal,
    total_exposure: Decimal,
    max_leverage: Decimal,
    drawdown: Decimal,
    max_drawdown: Decimal,
) -> ValidationResult:
    """Validate portfolio-wide risk limits"""
    try:
        # Check leverage
        if portfolio_value > 0:
            current_leverage = total_exposure / portfolio_value
            if current_leverage > max_leverage:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Leverage {current_leverage} exceeds maximum {max_leverage}",
                )

        # Check drawdown
        if drawdown > max_drawdown:
            return ValidationResult(
                is_valid=False,
                error_message=f"Drawdown {drawdown} exceeds maximum {max_drawdown}",
            )

        return ValidationResult(is_valid=True)

    except Exception as e:
        handle_error(e, "validate_portfolio_limits")
        return ValidationResult(is_valid=False, error_message=str(e))
