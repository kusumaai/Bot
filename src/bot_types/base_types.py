#!/usr/bin/env python3
# bot_types/base_types.py
"""
Base type definitions and shared interfaces
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, TypeVar

from trading.position import Position

# ------------------------------------------------------------------------------
# Helper validation functions to reduce code duplication
# ------------------------------------------------------------------------------


def validate_non_empty_string(value: str, field_name: str) -> Optional[str]:
    """Validate string is not empty"""
    if not value or not isinstance(value, str):
        return f"{field_name} must be a non-empty string"
    return None


def validate_positive_decimal(
    value: Decimal, field_name: str, allow_zero: bool = False
) -> Optional[str]:
    """Validate decimal is positive"""
    if not isinstance(value, Decimal):
        return f"{field_name} must be a Decimal"
    if allow_zero:
        if value < Decimal("0"):
            return f"{field_name} must be non-negative"
    else:
        if value <= Decimal("0"):
            return f"{field_name} must be positive"
    return None


# Shared types and interfaces for the bot types


# Validation types
@dataclass
class ValidationResult:
    """Standard validation result type used across the system"""

    is_valid: bool
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_valid


class Validatable(Protocol):
    """Protocol for objects that can be validated"""

    def validate(self) -> ValidationResult:
        """Validate the object"""
        ...


# Forward references for type hints
Position = TypeVar("Position", bound="Position")
PositionInfo = TypeVar(
    "PositionInfo", bound="Position"
)  # PositionInfo is now just an alias for Position


@dataclass
class MarketState:
    """Current market state information"""

    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: float
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    previous_close: Optional[Decimal] = None
    change: Optional[Decimal] = None
    percentage: Optional[Decimal] = None
    average: Optional[Decimal] = None
    base_volume: Optional[Decimal] = None
    quote_volume: Optional[Decimal] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits(Validatable):
    """Risk management limits and parameters"""

    min_position_size: Decimal
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    max_drawdown: Decimal
    max_daily_loss: Decimal
    emergency_stop_pct: Decimal
    risk_factor: Decimal
    kelly_scaling: Decimal
    max_correlation: Decimal
    max_sector_exposure: Decimal
    max_volatility: Decimal

    # validate the risk limits data
    def validate(self) -> ValidationResult:
        if self.max_position_size > Decimal("0.5"):
            return ValidationResult(
                is_valid=False,
                error_message="max_position_size cannot exceed 0.5 (50%)",
            )
        err = validate_positive_decimal(
            self.min_position_size, "min_position_size", allow_zero=True
        )
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.max_daily_loss > Decimal("0.03"):
            return ValidationResult(
                is_valid=False, error_message="max_daily_loss cannot exceed 0.03 (3%)"
            )
        if self.max_drawdown > Decimal("0.2"):
            return ValidationResult(
                is_valid=False, error_message="max_drawdown cannot exceed 0.2 (20%)"
            )
        if self.emergency_stop_pct > Decimal("0.15"):
            return ValidationResult(
                is_valid=False,
                error_message="emergency_stop_pct cannot exceed 0.15 (15%)",
            )
        if self.risk_factor > Decimal("0.02"):
            return ValidationResult(
                is_valid=False, error_message="risk_factor cannot exceed 0.02 (2%)"
            )
        if self.max_correlation > Decimal("0.7"):
            return ValidationResult(
                is_valid=False, error_message="max_correlation cannot exceed 0.7 (70%)"
            )
        if self.max_sector_exposure > Decimal("0.3"):
            return ValidationResult(
                is_valid=False,
                error_message="max_sector_exposure cannot exceed 0.3 (30%)",
            )
        if self.max_volatility > Decimal("0.4"):
            return ValidationResult(
                is_valid=False, error_message="max_volatility cannot exceed 0.4 (40%)"
            )
        if self.max_leverage > Decimal("3"):
            return ValidationResult(
                is_valid=False, error_message="max_leverage cannot exceed 3x"
            )
        if self.kelly_scaling > Decimal("0.5"):
            return ValidationResult(
                is_valid=False, error_message="kelly_scaling cannot exceed 0.5 (50%)"
            )
        return ValidationResult(is_valid=True)


# dataclass for the trading context
@dataclass
class TradingContext(Validatable):
    """Trading context with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal
    max_favorable_excursion: Decimal
    trailing_stop: Optional[Decimal]
    closed: bool = False
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None

    # validate the trading context data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in [
            "entry_price",
            "current_price",
            "size",
            "stop_loss",
            "take_profit",
            "unrealized_pnl",
            "max_adverse_excursion",
            "max_favorable_excursion",
        ]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        if self.trailing_stop is not None:
            err = validate_positive_decimal(self.trailing_stop, "trailing_stop")
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading signal
@dataclass
class TradingSignal(Validatable):
    """Trading signal with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal
    take_profit: Decimal
    unrealized_pnl: Decimal

    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in [
            "entry_price",
            "current_price",
            "size",
            "stop_loss",
            "take_profit",
            "unrealized_pnl",
        ]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading order
@dataclass
class TradingOrder(Validatable):
    """Trading order with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading order data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading strategy
@dataclass
class TradingStrategy(Validatable):
    """Trading strategy with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading strategy data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading execution
@dataclass
class TradingExecution(Validatable):
    """Trading execution with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading execution data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading result
@dataclass
class TradingResult(Validatable):
    """Trading result with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading result data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading history
@dataclass
class TradingHistory(Validatable):
    """Trading history with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading history data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading analytics
@dataclass
class TradingAnalytics(Validatable):
    """Trading analytics with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading analytics data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading report
@dataclass
class TradingReport(Validatable):
    """Trading report with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading report data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading configuration
@dataclass
class TradingConfiguration(Validatable):
    """Trading configuration with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading configuration data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)


# dataclass for the trading settings
@dataclass
class TradingSettings(Validatable):
    """Trading settings with standardized decimal handling and risk tracking"""

    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    entry_time: float
    stop_loss: Decimal

    # validate the trading settings data
    def validate(self) -> ValidationResult:
        err = validate_non_empty_string(self.symbol, "Symbol")
        if err:
            return ValidationResult(is_valid=False, error_message=err)
        if self.direction not in ["long", "short"]:
            return ValidationResult(
                is_valid=False, error_message="Direction must be 'long' or 'short'"
            )
        for field_name in ["entry_price", "current_price", "size", "stop_loss"]:
            value = getattr(self, field_name)
            err = validate_positive_decimal(value, field_name)
            if err:
                return ValidationResult(is_valid=False, error_message=err)
        return ValidationResult(is_valid=True)
