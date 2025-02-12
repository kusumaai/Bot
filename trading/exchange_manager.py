from typing import Any, Dict, Optional
import asyncio
from decimal import Decimal
import logging
import time

from utils.numeric_handler import NumericHandler
from trading.exceptions import ExchangeError, RateLimitExceeded, ExchangeAPIError

class ExchangeManager:
    """Manages exchange interactions with proper rate limiting and error handling"""
    
    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self.exchange = None
        self.db_queries = None
        
        # Initialize rate limiter with proper error handling
        self.rate_limiter = RateLimiter({
            'market': RateLimit(20, 60),
            'trade': RateLimit(10, 60),
            'order': RateLimit(50, 60),
            'position': RateLimit(10, 60)
        })
        
        self.is_paper = sandbox
        self.nh = NumericHandler()
        
        try:
            if self.is_paper:
                self.exchange = PaperExchange(api_key, api_secret)
            else:
                self.exchange = ActualExchange(exchange_id, api_key, api_secret)
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {str(e)}")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}")
        
        self._markets: Optional[Dict[str, Any]] = None
        self._last_market_update: Optional[datetime] = None
        
        # Add paper trading flag and methods
        if self.is_paper:
            self.paper_balances = {'USDT': Decimal('10000.0')}  # Default paper balance
            self.paper_positions = {}

    async def close(self):
        """Close exchange connections and clean up resources"""
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
            if self.db_queries:
                await self.db_queries.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def create_order(self, symbol: str, side: str, amount: Decimal, price: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
        try:
            await self.rate_limiter.acquire('order')
            async with self._lock:
                if self.is_paper:
                    # Simulate order creation in paper trading
                    order_id = f"paper_{int(time.time())}"
                    order = {
                        'id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'amount': str(amount),
                        'price': str(price) if price else None,
                        'status': 'OPEN'
                    }
                    self.paper_positions[order_id] = order
                    return order
                else:
                    # Actual exchange order creation
                    order = await self.exchange.create_order(symbol, side, float(amount), float(price) if price else None)
                    return order
        except RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded: {e}")
            return None
        except ExchangeAPIError as e:
            self.logger.error(f"Exchange API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in create_order: {e}")
            return None

    async def close_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            await self.rate_limiter.acquire('order')
            async with self._lock:
                if self.is_paper:
                    # Simulate order closing in paper trading
                    order = self.paper_positions.pop(order_id, None)
                    if order:
                        order['status'] = 'CLOSED'
                        return order
                    return None
                else:
                    # Actual exchange order closing
                    order = await self.exchange.close_order(order_id)
                    return order
        except RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded: {e}")
            return None
        except ExchangeAPIError as e:
            self.logger.error(f"Exchange API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in close_order: {e}")
            return None

    async def get_candles(self, symbol: str, timeframe: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        try:
            await self.rate_limiter.acquire('market')
            async with self._lock:
                candles = await self.exchange.get_candles(symbol, timeframe, limit)
                return candles
        except RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded while fetching candles: {e}")
            return None
        except ExchangeAPIError as e:
            self.logger.error(f"Exchange API error while fetching candles: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in get_candles: {e}")
            return None 