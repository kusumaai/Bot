#! /usr/bin/env python3
#src/exchanges/exchange_manager.py
"""
Module: exchanges/exchange_manager.py
Manages exchange connections and operations with proper rate limiting and error handling.
This version consolidates duplicate implementations and provides a consistent API.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from decimal import Decimal
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
from dataclasses import dataclass

from database.database import DatabaseConnection
from utils.error_handler import handle_error_async, ExchangeError
from utils.numeric_handler import NumericHandler
from trading.exceptions import ExchangeAPIError, RateLimitExceeded
from exchanges.rate_limiter import RateLimiter, RateLimit
from exchanges.paper_exchange import PaperExchange
from exchanges.actual_exchange import ActualExchange
from database.queries import DatabaseQueries

@dataclass
class RateLimit:
    """Rate limit configuration"""
    max_calls: int
    period: int  # seconds

@dataclass
class RateLimitConfig:
    max_requests: int
    time_window: int

class ExchangeManager:
    """Manages exchange interactions with proper rate limiting and error handling"""

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = True,
        logger: Optional[logging.Logger] = None,
        db_queries: Optional[Any] = None
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.logger = logger or logging.getLogger(__name__)
        self.db_queries = db_queries
        self._markets = None
        self._last_market_update = None
        self._lock = asyncio.Lock()
        
        if sandbox:
            self.exchange = PaperExchange()
        else:
            # Handle live exchange initialization
            pass

        # Initialize Rate Limiter
        self.rate_limiter = RateLimiter(rate_limits={
            'market': RateLimit(max_calls=100, period=60),
            'trade': RateLimit(max_calls=50, period=60)
        })

        self.numeric_handler = NumericHandler()

        if self.sandbox:
            # Set default paper mode balances and positions
            self.paper_balances = {'USDT': Decimal('10000.0')}
            self.paper_positions = {}

    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        try:
            await self.exchange.load_markets()
            self.logger.info("Exchange initialized successfully")
            return True
        except ExchangeError as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during exchange initialization: {e}")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}") from e

    async def create_order(self, symbol: str, order_type: str, side: str, 
                         amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Creates an order on the exchange with rate limiting"""
        await self.rate_limiter.limit('trade')
        try:
            order = await self.exchange.create_order(symbol, order_type, side, amount, price)
            self.logger.info(f"Created order: {order}")
            return order
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.create_order", self.logger)
            raise ExchangeError(f"Order creation failed: {str(e)}") from e

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a given symbol.
        
        Returns:
            dict: The ticker information.
        """
        try:
            await self.rate_limiter.acquire('market')
            ticker = await self.exchange.fetch_ticker(symbol)
            if self.db_queries:
                # Log ticker information to the database.
                self.db_queries.execute("INSERT INTO tickers (symbol, data) VALUES (?, ?)", (symbol, str(ticker)))
            return ticker
        except Exception as e:
            raise ExchangeError(f"Failed to fetch ticker: {str(e)}") from e

    async def get_markets(self) -> Dict[str, Any]:
        """Fetch markets with caching"""
        await self.rate_limiter.limit('market')
        try:
            if not self._markets or \
               not self._last_market_update or \
               (datetime.utcnow() - self._last_market_update).seconds > 3600:
                self._markets = await self.exchange.fetch_markets()
                self._last_market_update = datetime.utcnow()
            return self._markets
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.get_markets", self.logger)
            raise ExchangeError(f"Failed to fetch markets: {str(e)}") from e

    async def close(self):
        """Close exchange and database connections."""
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
            if self.db_queries:
                await self.db_queries.close()