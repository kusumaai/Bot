#!/usr/bin/env python3
# src/trading/exceptions.py
"""
Module: src.trading
Provides custom trading exceptions.
"""


class ValidationError(Exception):
    """Exception raised when validation fails."""

    pass


class PositionError(Exception):
    pass


class InvalidOrderError(Exception):
    pass


class MarketDataError(Exception):
    pass


class InvalidMarketDataError(Exception):
    pass


class DatabaseError(Exception):
    pass


class RiskManagerError(Exception):
    pass


class RatchetError(Exception):
    pass


class MathError(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


class ExchangeError(Exception):
    pass


class ExchangeAPIError(Exception):
    pass


class PositionUpdateError(Exception):
    pass


class PortfolioError(Exception):
    pass
