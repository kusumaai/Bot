#!/usr/bin/env python3
"""
Module: execution/exchange_interface.py
Handles all exchange interactions with proper error handling and rate limiting.
This version removes duplicate initialize methods and uses the consolidated ExchangeManager.
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
from utils.error_handler import handle_error_async, ExchangeError, ValidationError
from database.database import DatabaseConnection
from utils.numeric_handler import NumericHandler
from exchanges.exchange_manager import ExchangeManager
from risk.validation import MarketDataValidation

class OrderResult:
    """Container for order execution results."""
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

class ExchangeInterface:
    def __init__(self, exchange_manager, risk_manager, db_queries, logger):
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self.db_queries = db_queries
        self.logger = logger
        #self.initialized = False
        #self.nh = NumericHandler()

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int):
        try:
            candles = await self.exchange_manager.fetch_candles(symbol, timeframe, limit)
            return candles
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_candles", self.logger)
            return None

    async def initialize(self) -> bool:
        """Initialize exchange interface components."""
        try:
            await self.exchange_manager.initialize()
            self.logger.info("ExchangeInterface initialized successfully.")
            return True
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.initialize", self.logger)
            return False

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: str,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Execute a trade with risk validation."""
        try:
            is_valid, error = await self.risk_manager.validate_trade(symbol, side, amount, price)
            if not is_valid:
                return {"success": False, "error": error}
            
            order = await self.exchange_manager.create_order(symbol, side, amount, price, order_type=order_type)
            # Log the trade if needed using the unified database connection
            return {"success": True, "order_id": order['id']}
        except ValidationError as e:
            self.logger.error(f"Invalid order: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.execute_trade", self.logger)
            return {"success": False, "error": "Trade execution failed."}

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float) -> Optional[str]:
        try:
            await self.exchange_manager.rate_limiter.acquire('trade')
            order = await self.exchange_manager.create_order(symbol, side, Decimal(str(amount)), order_type=order_type)
            order_id = order.get('id')
            if not order_id:
                raise ValidationError("Order response missing 'id'.")
            return order_id
        except (ExchangeError, ValidationError) as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.create_order", self.logger)
            return None

    async def fetch_ticker(self, symbol: str) -> Optional[float]:
        try:
            await self.exchange_manager.rate_limiter.acquire('market')
            ticker = await self.exchange_manager.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_ticker", self.logger)
            return None

    async def close_position(self, symbol: str, amount: Decimal) -> bool:
        try:
            await self.exchange_manager.rate_limiter.acquire('trade')
            order = await self.exchange_manager.create_order(symbol, 'sell', amount, order_type="market")
            return True
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.close_position", self.logger)
            return False

    async def close(self) -> None:
        """Properly close all resources."""
        try:
            await self.exchange_manager.close()
        except ExchangeError as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def cancel_trade(self, order_id):
        try:
            result = await self.exchange_manager.exchange.close_order(order_id)
            if result:
                return True
            return False
        except ExchangeError as e:
            self.logger.error(f"Failed to cancel trade {order_id}: {e}")
            return False