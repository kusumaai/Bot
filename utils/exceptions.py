# Updated with all exceptions. Ensure there are no duplicates.
class TradingBotError(Exception):
    """Base exception for the trading bot."""
    pass

class RatchetError(TradingBotError):
    """Exception raised for ratchet-related errors."""
    pass

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass

class PositionError(Exception):
    """Exception related to position management."""
    pass

class PositionUpdateError(Exception):
    """Exception related to position management."""
    pass
class MarketDataValidationError(Exception):
    """Custom exception for market data validation errors."""
    pass
class InvalidOrderError(TradingBotError):
    """Exception raised for invalid order operations."""
    pass

class ExchangeError(TradingBotError):
    """Exception raised for exchange-related errors."""
    pass

class ExchangeAPIError(Exception):
    """Exception raised for exchange API errors."""
    pass

class PortfolioError(TradingBotError):
    """Exception raised for portfolio-related errors."""
    pass

class MathError(TradingBotError):
    """Exception raised for mathematical computation errors."""
    pass

class CircuitBreakerError(TradingBotError):
    """Exception raised for circuit breaker-related errors."""
    pass

class MarketDataValidationError(TradingBotError):
    """Exception raised for market data validation failures."""
    pass

class DatabaseError(TradingBotError):
    """Exception raised for database-related errors."""
    pass
class BackTestError(Exception):
    """Exception raised for backtest-related errors."""
    pass
class ValidationError(Exception):
    """Exception raised for validation-related errors."""
    pass

class RiskManagerError(TradingBotError):
    """Exception raised for risk manager-related errors."""
    pass

# Add any other custom exceptions as needed 