#! /usr/bin/env python3
# src/utils/exceptions.py
"""
Module: src.utils
Provides custom exceptions.
"""


# trading error class
class TradingError(Exception):
    """Base class for trading system exceptions"""

    pass


# concurrancy error class
class ConcurrentModificationError(Exception):
    """Concurrent Modification exceptions"""

    pass


# order error class
class OrderError(TradingError):
    """Raised when there's an error with an order execution"""

    pass


class OrderStoreError(TradingError):
    """Raised when there's an error with storing an order"""

    pass


class OrderCancelError(TradingError):
    """Raised when there's an error with order cancellation"""

    pass


class InvalidOrderError(OrderError):
    """Raised when order parameters are invalid"""

    pass


class RiskError(TradingError):
    """Raised when risk limits are exceeded"""

    pass


class RiskCalculationError(TradingError):
    """Raised when risk limits are exceeded"""

    pass


class ExchangeError(TradingError):
    """Raised when exchange operations fail"""

    pass


class DatabaseError(TradingError):
    """Raised when database operations fail"""

    pass


class RatchetError(TradingError):
    """Raised when ratchet operations fail"""

    pass


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    pass


class PositionError(Exception):
    """Exception related to position management."""

    pass


class TradingBotError(Exception):
    """Exception related to position management."""

    pass


class PositionUpdateError(Exception):
    """Exception related to position management."""

    pass


class MarketDataValidationError(Exception):
    """Custom exception for market data validation errors."""

    pass


class ExchangeAPIError(Exception):
    """Exception raised for exchange API errors."""

    pass


class PortfolioError(TradingError):
    """Exception raised for portfolio-related errors."""

    pass


class MathError(TradingError):
    """Exception raised for mathematical computation errors."""

    pass


class CircuitBreakerError(TradingError):
    """Exception raised for circuit breaker-related errors."""

    pass


class MarketDataValidationError(TradingError):
    """Exception raised for market data validation failures."""

    pass


class BackTestError(Exception):
    """Exception raised for backtest-related errors."""

    pass


class ValidationError(TradingError):
    """Exception raised for validation-related errors."""

    pass


class RiskManagerError(TradingError):
    """Exception raised for risk manager-related errors."""

    pass


class ExecutionError(TradingError):
    """Exception raised when execution fails."""

    pass


class HealthCheckError(TradingError):
    """Exception raised for health check failures."""

    pass


# Add any other custom exceptions as needed
