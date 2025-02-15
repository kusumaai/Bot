#! /usr/bin/env python3
# src/risk/validation.py
"""
Module: src.risk
Provides risk validation.
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

from bot_types.base_types import MarketState, Position, Validatable, ValidationResult
from signals.market_state import prepare_market_state
from utils.error_handler import ValidationError, handle_error
from utils.numeric_handler import NumericHandler

from .limits import RiskLimits


@dataclass
class MarketDataValidation(Validatable):
    """Market data validation with risk limits"""

    risk_limits: RiskLimits
    logger: Optional[logging.Logger] = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self.nh = NumericHandler()

    def validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data meets requirements"""
        try:
            if not data:
                self.logger.warning("Empty market data received")
                return False

            # Validate price and volume; use 'current_price' if 'price' is not provided
            price_val = data.get("price")
            if price_val is None:
                price_val = data.get("current_price", 0)
            price = self.nh.to_decimal(price_val)

            volume = self.nh.to_decimal(data.get("volume", 0))

            if price <= 0 or volume <= 0:
                self.logger.warning(f"Invalid price or volume: {price}, {volume}")
                return False

            # Check emergency stop conditions
            if "drawdown" in data:
                drawdown = self.nh.to_decimal(data["drawdown"])
                if drawdown >= self.risk_limits.emergency_stop_pct:
                    self.logger.warning(
                        f"Emergency stop triggered: drawdown {drawdown} >= {self.risk_limits.emergency_stop_pct}"
                    )
                    return False

            return True

        except Exception as e:
            handle_error(e, "MarketDataValidation.validate_market_data", self.logger)
            return False


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
    """Validate market data freshness and integrity"""
    try:
        # Validate required fields
        required_fields = ["symbol", "price", "volume", "timestamp"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                error_message=f"Missing required fields: {missing_fields}",
            )

        # Validate numeric values
        try:
            price = Decimal(str(data["price"]))
            volume = Decimal(str(data["volume"]))
            if price <= 0 or volume < 0:
                return ValidationResult(
                    is_valid=False, error_message="Invalid price or volume"
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False, error_message="Invalid numeric values"
            )

        # Validate timestamp freshness
        timestamp = float(data.get("timestamp", 0))
        if time.time() - timestamp > 5:  # 5 second staleness check
            return ValidationResult(is_valid=False, error_message="Stale data")

        return ValidationResult(is_valid=True)

    except Exception as e:
        handle_error(e, "validate_market_data")
        return ValidationResult(is_valid=False, error_message=str(e))


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
