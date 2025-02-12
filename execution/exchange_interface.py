#!/usr/bin/env python3
"""
Module: execution/exchange_interface.py
Handles all exchange interactions with proper error handling and rate limiting
"""

import time
import asyncio
from decimal import Decimal
import ccxt.async_support as ccxt
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import uuid
import logging

from utils.error_handler import handle_error, handle_error_async, ExchangeError, ValidationError
from database.database import DBConnection, execute_sql
from utils.numeric import NumericHandler
from exchanges.exchange_manager import ExchangeManager
from database.queries import DatabaseQueries
from risk.manager import RiskManager
from risk.validation import MarketDataValidation

class OrderResult:
    """Container for order execution results"""
    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.order_id = order_id
        self.error = error
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class ExchangeInterface:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        self.nh = NumericHandler()
        
        # Initialize caches with TTL
        self.CACHE_TTL = 300  # 5 minutes
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: Dict[str, float] = {}
        
        # Safe config loading with defaults
        self.paper_mode = self.ctx.config.get("paper_mode", True)
        self.rate_limit = int(self.ctx.config.get("rate_limit_per_second", 5))
        self.paper_balance = Decimal(str(self.ctx.config.get("initial_balance", "10000")))
        
        # Initialize exchange
        self.exchange_manager = self._init_exchange_manager()
        self.exchange = None
        
        # Initialize dependent components after exchange setup
        self.risk = RiskManager(ctx)
        self.validator = MarketDataValidation(self.risk.limits, self.logger)
        self.db = DatabaseQueries(ctx, logger=self.logger)

    def _init_exchange_manager(self) -> 'ExchangeManager':
        """Safely initialize exchange manager with validation"""
        try:
            exchange_id = self.ctx.config.get('exchange', 'paper')
            api_key = self.ctx.config.get('exchange_settings', {}).get('api_key')
            api_secret = self.ctx.config.get('exchange_settings', {}).get('api_secret')
            
            if api_key and not self._validate_api_credentials(api_key, api_secret):
                raise ValueError("Invalid API credentials")
                
            return ExchangeManager(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                logger=self.logger
            )
        except Exception as e:
            raise ExchangeError(f"Failed to initialize exchange manager: {str(e)}")
            
    async def _cleanup_cache(self) -> None:
        """Clean expired cache entries"""
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._last_update.items() if now - v > self.CACHE_TTL]
            for k in expired:
                self._ticker_cache.pop(k, None)
                self._last_update.pop(k, None)
                
    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        try:
            self.exchange = self.exchange_manager.exchange
            await self.exchange.load_markets()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            return False

    async def _rate_limit_request(self) -> None:
        """Implement rate limiting for exchange requests"""
        now = time.time()
        if now - self.last_request_time >= 1.0:
            self.request_count = 0
            self.last_request_time = now
        
        if self.request_count >= self.rate_limit:
            await asyncio.sleep(1.0)
            self.request_count = 0
            self.last_request_time = time.time()
        
        self.request_count += 1

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price with proper error handling"""
        if not self.exchange:
            return None

        try:
            await self._rate_limit_request()
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return {
                "symbol": symbol,
                "last": Decimal(str(ticker.get("last", 0))),
                "bid": Decimal(str(ticker.get("bid", 0))),
                "ask": Decimal(str(ticker.get("ask", 0))),
                "volume": Decimal(str(ticker.get("baseVolume", 0))),
                "timestamp": ticker.get("timestamp", 0)
            }

        except Exception as e:
            handle_error(e, "ExchangeInterface.fetch_ticker", logger=self.logger)
            return None

    async def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict]:
        try:
            order_type = "market" if price is None else "limit"
            return await self.exchange_manager.create_order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=Decimal(str(amount)),
            price=Decimal(str(price)) if price else None
        )
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
        return None
                
    async def _validate_order_params(self, 
                                   symbol: str, 
                                   size: Decimal) -> bool:
        """Validate order parameters against exchange rules"""
        try:
            info = await self.fetch_market_info(symbol)
            min_size = self.nh.to_decimal(info['min_size'])
            
            if size < min_size:
                self.ctx.logger.error(f"Size {size} below minimum {min_size}")
                return False
                
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Order validation failed: {e}")
            return False
            
    async def fetch_market_info(self, symbol: str) -> Dict:
        """Fetch and cache market information"""
        if symbol not in self.market_info:
            info = await self.ctx.exchange.fetch_market(symbol)
            self.market_info[symbol] = {
                'min_size': self.nh.to_decimal(info['limits']['amount']['min']),
                'price_precision': info['precision']['price'],
                'size_precision': info['precision']['amount']
            }
        return self.market_info[symbol]
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with verification"""
        try:
            await self.exchange.cancel_order(order_id)
            if order_id in self.order_cache:
                del self.order_cache[order_id]
            return True
        except Exception as e:
            self.ctx.logger.error(f"Order cancellation failed: {e}")
            return False

    async def close_position(self, trade: Dict[str, Any]) -> bool:
        """Close an open position with proper error handling"""
        if not self.exchange:
            return False

        if self.ctx.config.get("paper_mode", False):
            self.logger.info(f"[PAPER] Closing position for {trade['symbol']}")
            return True

        try:
            await self._rate_limit_request()
            await self.exchange.create_market_order(
                trade["symbol"],
                "sell" if trade["side"] == "buy" else "buy",
                float(trade["amount"]),
                None,
                {"reduceOnly": True}
            )
            return True

        except Exception as e:
            handle_error(e, "ExchangeInterface.close_position", logger=self.logger)
            return False

    async def fetch_balance(self) -> Optional[Dict[str, Decimal]]:
        """Fetch account balance with proper error handling"""
        if not self.exchange:
            return None

        try:
            await self._rate_limit_request()
            balance = await self.exchange.fetch_balance()
            
            return {
                currency: Decimal(str(data["free"]))
                for currency, data in balance.items()
                if isinstance(data, dict) and data.get("free", 0) > 0
            }

        except Exception as e:
            handle_error(e, "ExchangeInterface.fetch_balance", logger=self.logger)
            return None

    async def close(self) -> None:
        """Properly close all resources"""
        try:
            if self.exchange:
                await self.exchange.close()
            await self._cleanup_cache()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: str = "market",
        price: Optional[Decimal] = None,
        reduce_only: bool = False
    ) -> OrderResult:
        """
        Execute a trade with full validation and risk checks
        
        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            amount: Order amount
            order_type: Order type (market/limit)
            price: Optional limit price
            reduce_only: Whether order should only reduce position
        """
        try:
            # Validate basic parameters
            if side not in ['buy', 'sell']:
                raise ValidationError(f"Invalid order side: {side}")
            if order_type not in ['market', 'limit']:
                raise ValidationError(f"Invalid order type: {order_type}")
            if order_type == 'limit' and price is None:
                raise ValidationError("Limit orders require a price")
            
            # Get current market data
            ticker = await self.get_ticker(symbol)
            current_price = Decimal(str(ticker['last']))
            
            # Get recent candles for validation
            candles = await self.db.get_recent_candles(symbol, limit=20)
            
            # Perform risk validation for new positions
            if not reduce_only:
                direction = 'long' if side == 'buy' else 'short'
                await self.risk.validate_new_position(
                    symbol=symbol,
                    direction=direction,
                    size=amount,
                    price=current_price,
                    candles=candles
                )
            
            # Prepare order parameters
            order_params = {
                'reduceOnly': reduce_only
            }
            
            # Execute order
            order = await self.exchange.create_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=order_params
            )
            
            # Store order details
            await self._store_trade_details(order, ticker)
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                details=order
            )
            
        except (ValidationError, ExchangeError) as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            return OrderResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in trade execution: {str(e)}")
            return OrderResult(
                success=False,
                error="Internal execution error"
            )
    
    async def get_ticker(
        self,
        symbol: str,
        max_age: int = 5
    ) -> Dict[str, Any]:
        """
        Get ticker with caching
        
        Args:
            symbol: Trading pair symbol
            max_age: Maximum age of cached data in seconds
        """
        now = datetime.utcnow()
        
        # Check cache
        if (symbol in self._ticker_cache and
            symbol in self._last_update and
            now - self._last_update[symbol] < timedelta(seconds=max_age)):
            return self._ticker_cache[symbol]
        
        # Fetch new data
        ticker = await self.exchange.fetch_ticker(symbol)
        self._ticker_cache[symbol] = ticker
        self._last_update[symbol] = now
        
        return ticker
    
    async def close_position(
        self,
        symbol: str,
        position_id: int
    ) -> OrderResult:
        """
        Close an existing position
        
        Args:
            symbol: Trading pair symbol
            position_id: Position database ID
        """
        try:
            # Get position details
            position = await self.db.get_position(position_id)
            if not position:
                raise ValidationError(f"Position {position_id} not found")
            
            if position['status'] != 'active':
                raise ValidationError(f"Position {position_id} is not active")
            
            # Calculate close amount
            close_side = 'sell' if position['direction'] == 'long' else 'buy'
            
            # Execute closing order
            result = await self.execute_trade(
                symbol=symbol,
                side=close_side,
                amount=Decimal(str(position['size'])),
                reduce_only=True
            )
            
            if result.success:
                # Update position status
                await self.db.update_position_status(
                    position_id=position_id,
                    status='closed',
                    metadata={
                        'close_order_id': result.order_id,
                        'close_time': datetime.utcnow().isoformat()
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position closure failed: {str(e)}")
            return OrderResult(
                success=False,
                error=str(e)
            )
    
    async def _store_trade_details(
        self,
        order: Dict[str, Any],
        ticker: Dict[str, Any]
    ) -> None:
        """Store trade execution details"""
        try:
            await self.db.store_trade(
                order_id=order['id'],
                symbol=order['symbol'],
                side=order['side'],
                type=order['type'],
                amount=Decimal(str(order['amount'])),
                price=Decimal(str(order['price'])) if order['price'] else None,
                status=order['status'],
                timestamp=order['timestamp'],
                ticker_data=ticker
            )
        except Exception as e:
            self.logger.error(f"Failed to store trade details: {str(e)}")
            # Don't raise - this is non-critical

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[List]:
        """Fetch OHLCV candles with proper error handling"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")
                
            await self._rate_limit_request()
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return candles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV data: {e}")
            return []