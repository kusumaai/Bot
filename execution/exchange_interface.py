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
from datetime import datetime
import uuid
import logging

from utils.error_handler import handle_error, handle_error_async
from database.database import DBConnection, execute_sql
from utils.numeric import NumericHandler

class ExchangeInterface:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.exchange: Optional[ccxt.Exchange] = None
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit = ctx.config.get("rate_limit_per_second", 5)
        self.paper_mode = ctx.config.get("paper_mode", True)
        self.paper_balance = Decimal(str(ctx.config.get("initial_balance", "10000")))
        self.paper_positions = {}
        self.nh = NumericHandler()
        self._lock = asyncio.Lock()
        self.order_cache: Dict[str, Dict] = {}
        self.market_info: Dict[str, Dict] = {}
        self.last_prices: Dict[str, Decimal] = {}

    async def initialize(self) -> bool:
        """Initialize exchange connection with credentials"""
        try:
            exchange_id = self.ctx.config.get("exchange", "kucoin")
            credentials = {
                "apiKey": self.ctx.config.get(f"{exchange_id}_api_key", ""),
                "secret": self.ctx.config.get(f"{exchange_id}_secret", ""),
                "password": self.ctx.config.get(f"{exchange_id}_password", ""),
                "enableRateLimit": True
            }

            if not credentials["apiKey"] or not credentials["secret"]:
                self.logger.error(f"Missing API credentials for {exchange_id}")
                return False

            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class(credentials)
            await self.exchange.load_markets()
            self.logger.info(f"Exchange {exchange_id} initialized successfully")
            return True

        except Exception as e:
            handle_error(e, "ExchangeInterface.initialize", logger=self.logger)
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

    async def place_order(self, 
                         symbol: str, 
                         side: str, 
                         size: Decimal, 
                         price: Optional[Decimal] = None) -> Optional[Dict]:
        """Place order with retry logic and validation"""
        async with self._lock:
            try:
                # Validate order parameters
                if not await self._validate_order_params(symbol, size):
                    return None
                    
                # Format order
                order = {
                    'symbol': symbol,
                    'side': side.upper(),
                    'size': self.nh.round_decimal(size, 8),
                    'type': 'MARKET' if price is None else 'LIMIT',
                }
                if price:
                    order['price'] = self.nh.round_decimal(price, 8)
                    
                # Place order with retry
                for attempt in range(3):
                    try:
                        result = await self.ctx.exchange.create_order(**order)
                        self.order_cache[result['id']] = {
                            **result,
                            'timestamp': datetime.utcnow()
                        }
                        return result
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            raise
                        await asyncio.sleep(1)  # Wait before retry
                        
            except Exception as e:
                self.ctx.logger.error(f"Order placement failed: {e}")
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
            await self.ctx.exchange.cancel_order(order_id)
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
        """Properly close exchange connection"""
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Exchange connection closed")
            except Exception as e:
                handle_error(e, "ExchangeInterface.close", logger=self.logger)