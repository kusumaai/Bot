#!/usr/bin/env python3
"""
Module: exchanges/exchange_manager.py
Manages exchange connections and operations with proper error handling
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import time
from dataclasses import dataclass

from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error, ExchangeError
from database.queries import DatabaseQueries
from utils.numeric_handler import NumericHandler
from trading.exceptions import ExchangeAPIError, RateLimitExceeded
from exchanges.rate_limiter import RateLimiter, RateLimit
from exchanges.paper_exchange import PaperExchange
from exchanges.actual_exchange import ActualExchange

@dataclass
class RateLimit:
    """Rate limit configuration"""
    max_requests: int
    time_window: int  # seconds
    weight: int = 1

class RateLimiter:
    """Manages API rate limiting"""
    
    def __init__(self, limits: Dict[str, RateLimit]):
        self.limits = limits
        self.request_timestamps: Dict[str, List[float]] = {
            endpoint: [] for endpoint in limits.keys()
        }
    
    async def acquire(self, endpoint: str) -> None:
        """
        Acquire permission to make a request
        
        Args:
            endpoint: API endpoint identifier
        """
        try:
            if endpoint not in self.limits:
                return
            
            limit = self.limits[endpoint]
            now = time.time()
            window_start = now - limit.time_window
            
            # Clean old timestamps
            self.request_timestamps[endpoint] = [
                ts for ts in self.request_timestamps[endpoint]
                if ts > window_start
            ]
            
            # Check if we're at the limit
            while len(self.request_timestamps[endpoint]) >= limit.max_requests:
                sleep_time = (
                    self.request_timestamps[endpoint][0] +
                    limit.time_window - now
                )
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.time()
                window_start = now - limit.time_window
                self.request_timestamps[endpoint] = [
                    ts for ts in self.request_timestamps[endpoint]
                    if ts > window_start
                ]
            
            self.request_timestamps[endpoint].append(now)
        except Exception as e:
            logging.error(f"Rate limiter error: {e}")
            await asyncio.sleep(1)  # Fallback delay

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
            
            if not self.is_paper:
                self.db_queries = DatabaseQueries(logger=self.logger)
                
        except (ExchangeError, ValueError) as e:
            self.logger.error(f"Failed to initialize exchange: {str(e)}")
            raise ExchangeError(f"Exchange initialization failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during exchange initialization: {e}")
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
    
    async def get_markets(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Get market information with caching
        
        Args:
            force_update: Force refresh of market data
        """
        now = datetime.utcnow()
        
        if (self._markets is None or
            force_update or
            self._last_market_update is None or
            now - self._last_market_update > timedelta(hours=1)):
            
            try:
                await self.rate_limiter.acquire('market')
                self._markets = await self.exchange.load_markets()
                self._last_market_update = now
            except Exception as e:
                raise ExchangeError(f"Failed to load markets: {str(e)}")
        
        return self._markets
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data"""
        try:
            await self.rate_limiter.acquire('market')
            ticker = await self.exchange.fetch_ticker(symbol)
            
            if self.db_queries:
                # Store ticker data
                await self.db_queries.store_ticker(symbol, ticker)
                
            return ticker
        except Exception as e:
            raise ExchangeError(f"Failed to fetch ticker for {symbol}: {str(e)}")
    
    async def create_order(self, symbol: str, side: str, amount: Decimal, price: Optional[Decimal] = None) -> Optional[Dict[str, Any]]:
        try:
            await self.rate_limiter.acquire('trade')
            async with self._lock:
                order = await self.exchange.create_order(symbol, side, float(amount), float(price) if price else None)
                return order
        except RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded while creating order: {e}")
            return None
        except ExchangeAPIError as e:
            self.logger.error(f"Exchange API error while creating order: {e}")
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
                    order = await self.exchange.close_order(order_id)
                    return order
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
    
    async def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance"""
        try:
            await self.rate_limiter.acquire('position')
            return await self.exchange.fetch_balance()
        except Exception as e:
            raise ExchangeError(f"Failed to fetch balance: {str(e)}")
    
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch open orders"""
        try:
            await self.rate_limiter.acquire('order')
            return await self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch open orders: {str(e)}")

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

    async def _initialize_exchange(self, **kwargs) -> None:
        """Safely initialize exchange with validation"""
        self.is_paper = kwargs['exchange_id'] == 'paper'
        exchange_id = 'binance' if self.is_paper else kwargs['exchange_id']
        
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange {exchange_id} not supported")
            
        exchange_class = getattr(ccxt, exchange_id)
        config = self._build_exchange_config(**kwargs)
        self.exchange = exchange_class(config)
        
    def _build_exchange_config(self, **kwargs) -> dict:
        """Build validated exchange configuration"""
        config = {
            'enableRateLimit': True,
            'timeout': int(kwargs.get('timeout', 30000))
        }
        
        if not self.is_paper and kwargs.get('api_key') and kwargs.get('api_secret'):
            config.update({
                'apiKey': kwargs['api_key'],
                'secret': kwargs['api_secret']
            })
            
        if kwargs.get('sandbox', True) and not self.is_paper:
            config['sandbox'] = True
            
        return config

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            async with self._lock:
                status = await self.exchange.get_order_status(order_id)
                return status
        except ExchangeAPIError as e:
            self.logger.error(f"Exchange API error while fetching order status: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in get_order_status: {e}")
            return None

    async def close(self):
        """Close exchange connections and clean up resources"""
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
            if self.db_queries:
                await self.db_queries.close()
