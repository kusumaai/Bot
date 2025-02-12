from typing import Any, Dict, Optional
import asyncio
from decimal import Decimal
import logging
from datetime import datetime

from utils.numeric_handler import NumericHandler
from utils.exceptions import (
    ExchangeError,
    RateLimitExceeded,
    ExchangeAPIError
)
from exchanges.rate_limiter import RateLimiter, RateLimit
from exchanges.actual_exchange import ActualExchange
from exchanges.paper_exchange import PaperExchange
from utils.error_handler import handle_error_async
from database.queries import DatabaseQueries

class ExchangeManager:
    """Manages exchange interactions with proper rate limiting and error handling"""

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = True,
        logger: Optional[logging.Logger] = None,
        db_queries: Optional[DatabaseQueries] = None
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.logger = logger or logging.getLogger(__name__)
        self.db_queries = db_queries
        self.numeric_handler = NumericHandler()

        # Initialize Rate Limiter
        self.rate_limiter = RateLimiter(rate_limits={
            'market': RateLimit(max_calls=100, period=60),
            'trade': RateLimit(max_calls=50, period=60)
        })

        # Choose Exchange Type
        if self.sandbox:
            self.exchange = PaperExchange({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            })
            self.logger.info("Initialized PaperExchange for sandbox mode.")
        else:
            self.exchange = ActualExchange({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            })
            self.logger.info("Initialized ActualExchange for live trading.")

    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        try:
            await self.exchange.initialize_exchange()
            self.logger.info("Exchange initialized successfully.")
            return True
        except ExchangeError as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during exchange initialization: {e}")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}") from e

    async def get_markets(self):
        await self.rate_limiter.limit('market')
        try:
            self._markets = await self.exchange.fetch_markets()
            self._last_market_update = datetime.utcnow()
            self.logger.info("Fetched exchange markets successfully.")
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.get_markets", self.logger)
            raise ExchangeError("Failed to fetch markets.") from e

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Creates an order on the exchange"""
        await self.rate_limiter.limit('trade')
        try:
            order = await self.exchange.create_order(symbol, order_type, side, amount, price)
            self.logger.info(f"Created order: {order}")
            return order
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.create_order", self.logger)
            return None

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches the status of an order"""
        await self.rate_limiter.limit('trade')
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            self.logger.info(f"Fetched order: {order}")
            return order
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.fetch_order", self.logger)
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels an existing order"""
        await self.rate_limiter.limit('trade')
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.cancel_order", self.logger)
            return False

    # Add other exchange interaction methods as needed

    async def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """Fetches the account balance"""
        await self.rate_limiter.limit('market')
        try:
            balance = await self.exchange.fetch_balance()
            self.logger.info(f"Fetched balance: {balance}")
            return balance
        except Exception as e:
            await handle_error_async(e, "ExchangeManager.fetch_balance", self.logger)
            return None 