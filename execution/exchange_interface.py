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

from utils.error_handler import handle_error
from database.database import DBConnection, execute_sql

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

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "limit"
    ) -> Optional[Dict[str, Any]]:
        """Place order with comprehensive error handling"""
        if self.paper_mode:
            # Simulate order execution with slippage
            slippage = self.ctx.config.get("slippage_rate", Decimal("0.001"))
            commission = self.ctx.config.get("commission_rate", Decimal("0.001"))
            
            # Calculate execution price with slippage
            ticker_price = await self.fetch_ticker(symbol)["last"]
            executed_price = ticker_price * (Decimal("1") + (slippage if side == "buy" else -slippage))
            
            # Calculate fees
            fee = amount * executed_price * commission
            
            # Update paper balance
            cost = amount * executed_price + fee
            if side == "buy":
                self.paper_balance -= cost
            else:
                self.paper_balance += cost
            
            # Track paper position
            trade_id = f"paper_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            self.paper_positions[trade_id] = {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "entry_price": executed_price,
                "fee": fee
            }
            
            return {
                "id": trade_id,
                "price": executed_price,
                "amount": amount,
                "cost": cost,
                "fee": fee
            }
        else:
            if not self.exchange:
                return None

            try:
                await self._rate_limit_request()
                params = {}
                
                if order_type == "market":
                    order = await self.exchange.create_market_order(
                        symbol, side, float(amount), None, params
                    )
                else:
                    if not price:
                        self.logger.error("Price required for limit order")
                        return None
                    order = await self.exchange.create_limit_order(
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
                handle_error(e, "ExchangeInterface.place_order", logger=self.logger)
                return None

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