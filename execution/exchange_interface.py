#!/usr/bin/env python3
"""
Module: execution/exchange_interface.py
Handles all exchange interactions with proper error handling and rate limiting
"""

import time
import asyncio
from decimal import Decimal
import ccxt.async_support as ccxt
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import uuid
import logging

from trading.exceptions import DatabaseError
from utils.error_handler import handle_error, handle_error_async, ExchangeError, ValidationError
from database.database import DatabaseQueries, execute_sql
from utils.numeric_handler import NumericHandler
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

class ExchangeInterface:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.exchange_manager = ctx.exchange_manager  # Assuming ctx has exchange_manager
        self.db = ctx.db_queries  # Assuming ctx has db_queries
        self.nh = NumericHandler()
        
        # Initialize caches with TTL
        self.CACHE_TTL = 300  # 5 minutes
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: Dict[str, float] = {}
        
        # Safe config loading with defaults
        self.paper_mode = self.ctx.config.get("paper_mode", True)
        self.rate_limit = int(self.ctx.config.get("rate_limit_per_second", 5))
        self.paper_balance = Decimal(str(self.ctx.config.get("initial_balance", "10000")))
        
        # Initialize exchange manager
        self.exchange_manager = ExchangeManager(
            exchange_id=self.ctx.config.get("exchange_id", "binance"),
            api_key=self.ctx.config.get("api_key"),
            api_secret=self.ctx.config.get("api_secret"),
            sandbox=self.paper_mode,
            logger=self.logger
        )
        self.exchange = self.exchange_manager.exchange
        
        # Initialize dependent components after exchange setup
        self.validator = MarketDataValidation(ctx.risk_manager.risk_limits, self.logger)
        self.db = DatabaseQueries(ctx.config.get("database", {}).get("path", "data/trading.db"), logger=self.logger)

    async def initialize(self) -> bool:
        """Initialize exchange interface"""
        try:
            if not await self.exchange_manager.initialize():
                return False
            if not await self.risk_manager.initialize():
                return False
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange interface: {e}")
            return False

    async def close(self) -> None:
        """Properly close all resources"""
        try:
            await self.exchange_manager.close()
            await self._cleanup_cache()
        except ExchangeError as e:
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
            # Validate trade parameters
            valid, error = self.validator.validate_trade(symbol, side, amount, price)
            if not valid:
                self.logger.error(f"Trade validation failed: {error}")
                return OrderResult(success=False, error=error)
            
            # Risk checks
            allowed, risk_error = self.risk.check_risk(symbol, side, amount, price)
            if not allowed:
                self.logger.error(f"Risk check failed: {risk_error}")
                return OrderResult(success=False, error=risk_error)
            
            # Execute trade
            order = await self.exchange_manager.create_order(symbol, side, amount, price)
            if not order:
                error_msg = "Failed to create order."
                self.logger.error(error_msg)
                return OrderResult(success=False, error=error_msg)
            
            # Store trade details
            ticker = await self.exchange_manager.fetch_ticker(symbol)
            await self._store_trade_details(order, ticker)
            
            return OrderResult(success=True, order_id=order.get('id'), details=order)
        
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return OrderResult(success=False, error=str(e))

    async def _store_trade_details(
        self,
        order: Dict[str, Any],
        ticker: Dict[str, Any]
    ) -> None:
        """Store trade execution details"""
        try:
            trade = {
                'id': order['id'],
                'symbol': order['symbol'],
                'entry_price': Decimal(str(order['price'])) if order['price'] else None,
                'size': Decimal(str(order['amount'])),
                'side': order['side'],
                'strategy': 'default',  # Replace with actual strategy name
                'metadata': {
                    'status': order['status'],
                    'timestamp': order['timestamp'],
                    'ticker': ticker
                }
            }
            success = await self.db.store_trade(trade)
            if not success:
                self.logger.warning(f"Trade details not stored for order {order['id']}")
        except Exception as e:
            self.logger.error(f"Failed to store trade details: {str(e)}")
            # Don't raise - this is non-critical

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[List[Any]]:
        """Fetch OHLCV candles with proper error handling"""
        try:
            if not self.exchange_manager.exchange:
                raise ValueError("Exchange not initialized")
                
            await self.rate_limit_request()
            candles = await self.exchange_manager.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return candles
        except ExchangeError as e:
            self.logger.error(f"Failed to fetch OHLCV data: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_ohlcv: {e}")
            return []

    async def rate_limit_request(self):
        """Handle rate limiting before making a request"""
        await asyncio.sleep(1 / self.rate_limit)

    async def _cleanup_cache(self) -> None:
        """Clean expired cache entries"""
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._last_update.items() if now - v > self.CACHE_TTL]
            for k in expired:
                self._ticker_cache.pop(k, None)
                self._last_update.pop(k, None)

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float) -> Optional[str]:
        try:
            await self.exchange_manager.rate_limiter.limit('trade')
            order = await self.exchange_manager.exchange.create_order(symbol, order_type, side, amount)
            order_id = order.get('id')
            if not order_id:
                raise ValidationError("Order response missing 'id'.")
            await self.db.insert_trade({
                "id": order_id,
                "symbol": symbol,
                "entry_price": float(order.get('price', 0)),
                "size": float(amount),
                "side": side,
                "strategy": "manual",
                "metadata": {}
            })
            return order_id
        except (ExchangeError, ValidationError) as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None

    async def fetch_ticker(self, symbol: str) -> Optional[float]:
        try:
            await self.exchange_manager.rate_limiter.limit('market')
            ticker = await self.exchange_manager.exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_ticker", self.logger)
            return None

    async def close_position(self, symbol: str, amount: Decimal) -> bool:
        try:
            await self.exchange_manager.rate_limiter.limit('trade')
            order = await self.exchange_manager.exchange.create_order(symbol, 'market', 'sell', float(amount))
            await self.db.insert_trade({
                "id": order.get('id'),
                "symbol": symbol,
                "entry_price": float(order.get('price', 0)),
                "size": float(amount),
                "side": 'sell',
                "strategy": "manual",
                "metadata": {}
            })
            return True
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.close_position", self.logger)
            return False

    async def initialize(self) -> bool:
        """Initialize database connection"""
        try:
            # Test database connection
            async with self._lock:
                query = "SELECT 1"
                await self.execute(query)
                self.logger.info("Database connection initialized successfully")
                return True
            
        except DatabaseError as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during database initialization: {e}")
            return False