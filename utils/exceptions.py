"""Custom exceptions for the trading system"""

class TradingError(Exception):
    """Base class for trading system exceptions"""
    pass

class OrderError(TradingError):
    """Raised when there's an error with order execution"""
    pass

class InvalidOrderError(OrderError):
    """Raised when order parameters are invalid"""
    pass

class RiskError(TradingError):
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

class ValidationError(Exception):
    """Exception raised for validation-related errors."""
    pass

class RiskManagerError(TradingError):
    """Exception raised for risk manager-related errors."""
    pass

# Add any other custom exceptions as needed 