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

from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error

class ExchangeManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.exchange_instances: Dict[str, ccxt.Exchange] = {}
        self.last_rate_reset = datetime.now()
        self.request_count = 0
        self.rate_limit = ctx.config.get("rate_limit_per_second", 5)

    async def get_exchange(self, exchange_id: str = None) -> Optional[ccxt.Exchange]:
        """Get or create exchange instance with rate limiting"""
        try:
            if not exchange_id:
                exchange_id = self.ctx.config.get("primary_exchange", "kucoin")

            if exchange_id in self.exchange_instances:
                return self.exchange_instances[exchange_id]

            # Rate limit check
            now = datetime.now()
            if (now - self.last_rate_reset).total_seconds() >= 1.0:
                self.request_count = 0
                self.last_rate_reset = now
            
            if self.request_count >= self.rate_limit:
                await asyncio.sleep(1.0)
                self.request_count = 0
                self.last_rate_reset = datetime.now()

            exchange = await self._create_exchange(exchange_id)
            if not exchange:
                return None

            self.exchange_instances[exchange_id] = exchange
            return exchange

        except Exception as e:
            handle_error(e, context=f"ExchangeManager.get_exchange({exchange_id})", 
                        logger=self.logger)
            return None

    async def _create_exchange(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """Create new exchange instance with credentials"""
        try:
            credentials = {
                "apiKey": self.ctx.config.get(f"{exchange_id}_api_key", ""),
                "secret": self.ctx.config.get(f"{exchange_id}_secret", ""),
                "password": self.ctx.config.get(f"{exchange_id}_password", ""),
                "enableRateLimit": True
            }

            if not credentials["apiKey"] or not credentials["secret"]:
                self.logger.error(f"Missing API credentials for {exchange_id}")
                return None

            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(credentials)
            await exchange.load_markets()
            return exchange

        except Exception as e:
            handle_error(e, context=f"ExchangeManager._create_exchange({exchange_id})", 
                        logger=self.logger)
            return None

    async def fetch_ticker(self, symbol: str, exchange_id: str = None) -> Optional[Dict[str, Any]]:
        """Fetch current ticker with error handling and rate limiting"""
        try:
            exchange = await self.get_exchange(exchange_id)
            if not exchange:
                return None

            self.request_count += 1
            ticker = await exchange.fetch_ticker(symbol)
            if not ticker:
                return None

            return {
                "symbol": symbol,
                "last": Decimal(str(ticker.get("last", 0))),
                "bid": Decimal(str(ticker.get("bid", 0))),
                "ask": Decimal(str(ticker.get("ask", 0))),
                "volume": Decimal(str(ticker.get("baseVolume", 0))),
                "timestamp": ticker.get("timestamp", 0)
            }

        except Exception as e:
            handle_error(e, context=f"ExchangeManager.fetch_ticker({symbol})", 
                        logger=self.logger)
            return None

    async def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        exchange_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """Place order with comprehensive error handling"""
        try:
            exchange = await self.get_exchange(exchange_id)
            if not exchange:
                return None

            self.request_count += 1
            params = {}
            
            if order_type == "market":
                order = await exchange.create_market_order(
                    symbol, side, float(amount), None, params
                )
            else:
                if not price:
                    self.logger.error(f"Price required for limit order: {symbol}")
                    return None
                order = await exchange.create_limit_order(
                    symbol, side, float(amount), float(price), params
                )

            return {
                "id": order.get("id", ""),
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "price": Decimal(str(order.get("price", 0))),
                "amount": Decimal(str(order.get("amount", 0))),
                "filled": Decimal(str(order.get("filled", 0))),
                "status": order.get("status", "unknown"),
                "timestamp": order.get("timestamp", 0)
            }

        except Exception as e:
            handle_error(e, context=f"ExchangeManager.place_order({symbol})", 
                        logger=self.logger)
            return None

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        exchange_id: str = None
    ) -> bool:
        """Cancel order with error handling"""
        try:
            exchange = await self.get_exchange(exchange_id)
            if not exchange:
                return False

            self.request_count += 1
            await exchange.cancel_order(order_id, symbol)
            return True

        except Exception as e:
            handle_error(e, context=f"ExchangeManager.cancel_order({order_id})", 
                        logger=self.logger)
            return False

    async def fetch_balance(self, exchange_id: str = None) -> Optional[Dict[str, Decimal]]:
        """Fetch account balance with error handling"""
        try:
            exchange = await self.get_exchange(exchange_id)
            if not exchange:
                return None

            self.request_count += 1
            balance = await exchange.fetch_balance()
            
            return {
                currency: Decimal(str(data.get("free", 0)))
                for currency, data in balance.get("total", {}).items()
                if data.get("free", 0) > 0
            }

        except Exception as e:
            handle_error(e, context=f"ExchangeManager.fetch_balance", 
                        logger=self.logger)
            return None

    async def close(self):
        """Properly close all exchange connections"""
        for exchange in self.exchange_instances.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing exchange: {str(e)}")
