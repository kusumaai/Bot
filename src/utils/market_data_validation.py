#! /usr/bin/env python3
# src/utils/market_data_validation.py
"""
Module: src.utils
Provides unified market data validation functionality.
"""
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils.error_handler import handle_error, handle_error_async
from src.utils.exceptions import MarketDataValidationError, ValidationError
from src.utils.numeric_handler import NumericHandler


class MarketDataValidator:
    """Unified validator for all market data and trading related validations."""

    def __init__(self, risk_limits=None, logger=None):
        """Initialize the MarketDataValidator.

        Args:
            risk_limits: Optional risk limits configuration
            logger: Optional logger instance
        """
        self.risk_limits = risk_limits
        self.logger = logger or logging.getLogger(__name__)
        self.nh = NumericHandler()

    async def validate_trade_parameters(
        self, trade_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validates trade parameters with risk limits if available."""
        try:
            # Check required fields
            required_fields = ["symbol", "side", "amount", "price"]
            for field in required_fields:
                if field not in trade_params:
                    return False, f"Missing required field: {field}"

            # Validate side
            if trade_params["side"] not in ["buy", "sell"]:
                return False, f"Invalid trade side: {trade_params['side']}"

            # Validate amount
            amount = self.nh.to_decimal(trade_params["amount"])
            if amount is None or amount <= 0:
                return False, "Trade amount must be positive"

            # Validate price
            price = self.nh.to_decimal(trade_params["price"])
            if price is None or price <= 0:
                return False, "Trade price must be positive"

            # Risk limit validations if available
            if self.risk_limits:
                if amount > self.risk_limits.max_position_size:
                    return False, "Trade amount exceeds maximum position size"

                # Add other risk limit validations as needed
                if hasattr(self.risk_limits, "max_trade_value"):
                    trade_value = amount * price
                    if trade_value > self.risk_limits.max_trade_value:
                        return False, "Trade value exceeds maximum allowed"

            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_trade_parameters", self.logger)
            return False, str(e)

    def validate_market_data(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Validates market data structure and content."""
        try:
            # Handle DataFrame input
            if isinstance(data, pd.DataFrame):
                return self._validate_dataframe(data)

            # Handle dictionary input
            elif isinstance(data, dict):
                return self._validate_dict(data)

            return False, "Invalid data type for market data validation"

        except Exception as e:
            handle_error(e, "validate_market_data", self.logger)
            return False, str(e)

    def _validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validates market data in DataFrame format."""
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]

        if not all(col in df.columns for col in required_cols):
            return False, "Missing required columns in market data"

        if df.empty or len(df) < 2:
            return False, "Insufficient market data"

        if df.isnull().any().any():
            return False, "Market data contains null values"

        if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
            return False, "Market data contains negative values"

        return True, None

    def _validate_dict(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates market data in dictionary format."""
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]

        if not all(field in data for field in required_fields):
            return False, "Missing required fields in market data"

        for field in ["open", "high", "low", "close", "volume"]:
            value = self.nh.to_decimal(data[field])
            if value is None or value < 0:
                return False, f"Invalid {field} value in market data"

        return True, None

    async def validate_order_params(
        self,
        symbol: str,
        side: str,
        amount: Union[str, float, Decimal],
        price: Optional[Union[str, float, Decimal]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validates order parameters."""
        try:
            if side not in ["buy", "sell"]:
                return False, f"Invalid side: {side}"

            amount = self.nh.to_decimal(amount)
            if amount is None or amount <= 0:
                return False, "Amount must be positive"

            if price is not None:
                price = self.nh.to_decimal(price)
                if price is None or price <= 0:
                    return False, "Price must be positive"

            # Additional risk validations if available
            if self.risk_limits and price is not None:
                trade_value = amount * price
                if hasattr(self.risk_limits, "min_trade_value"):
                    if trade_value < self.risk_limits.min_trade_value:
                        return False, "Trade value below minimum allowed"

                if hasattr(self.risk_limits, "max_trade_value"):
                    if trade_value > self.risk_limits.max_trade_value:
                        return False, "Trade value exceeds maximum allowed"

            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_order_params", self.logger)
            return False, str(e)

    async def validate_correlation(
        self, symbol: str, correlations: Dict[str, Decimal]
    ) -> Tuple[bool, Optional[str]]:
        """Validates market correlations against risk limits."""
        try:
            if not self.risk_limits or not hasattr(self.risk_limits, "max_correlation"):
                return True, None  # Skip correlation check if no limits set

            for correlated_symbol, correlation in correlations.items():
                if correlation > self.risk_limits.max_correlation:
                    return (
                        False,
                        f"Correlation for {correlated_symbol} exceeds maximum allowed",
                    )
            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_correlation", self.logger)
            return False, f"Error validating correlations: {e}"

    async def validate_liquidity(
        self, symbol: str, liquidity: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Validates market liquidity against risk limits."""
        try:
            if not self.risk_limits or not hasattr(self.risk_limits, "min_liquidity"):
                return True, None  # Skip liquidity check if no limits set

            if liquidity < self.risk_limits.min_liquidity:
                return False, f"Liquidity for {symbol} is below minimum required"
            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_liquidity", self.logger)
            return False, f"Error validating liquidity: {e}"

    def validate_candle(self, candle: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates a single candle data point."""
        try:
            required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(field in candle for field in required_fields):
                return False, "Missing required fields in candle data"

            # Validate price relationships
            high = self.nh.to_decimal(candle["high"])
            low = self.nh.to_decimal(candle["low"])
            open_price = self.nh.to_decimal(candle["open"])
            close = self.nh.to_decimal(candle["close"])

            if any(x is None for x in [high, low, open_price, close]):
                return False, "Invalid price values in candle"

            if not (
                high >= low
                and high >= open_price
                and high >= close
                and low <= open_price
                and low <= close
            ):
                return False, "Invalid price relationships in candle"

            volume = self.nh.to_decimal(candle["volume"])
            if volume is None or volume < 0:
                return False, "Invalid volume in candle"

            return True, None
        except Exception as e:
            handle_error(e, "validate_candle", self.logger)
            return False, str(e)

    def validate_market_state(
        self, state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validates market state data."""
        try:
            required_fields = ["symbol", "price", "volume", "timestamp"]
            if not all(field in state for field in required_fields):
                return False, "Missing required fields in market state"

            price = self.nh.to_decimal(state["price"])
            volume = self.nh.to_decimal(state["volume"])

            if price is None or price <= 0:
                return False, "Invalid price in market state"

            if volume is None or volume < 0:
                return False, "Invalid volume in market state"

            return True, None
        except Exception as e:
            handle_error(e, "validate_market_state", self.logger)
            return False, str(e)
