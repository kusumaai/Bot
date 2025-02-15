#! /usr/bin/env python3
# src/execution/exchange_interface.py
"""
Module: src.execution
Handles all exchange interactions.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt

from database.database import DatabaseConnection
from exchanges.exchange_manager import ExchangeManager
from risk.validation import MarketDataValidation
from trading.exceptions import DatabaseError
from utils.error_handler import ExchangeError, ValidationError, handle_error_async
from utils.numeric_handler import NumericHandler


class OrderResult:
    """Container for order execution results."""

    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.order_id = order_id
        self.error = error


class ExchangeInterface:
    def __init__(self, ctx):
        self.ctx = ctx
        self.logger = ctx.logger

        # Use preferred exchange from config, default to 'paper'
        exchange_id = ctx.config.get("preferred_exchange", "paper")

        if exchange_id == "paper":
            # Create a dummy exchange manager that supports paper simulation
            class DummyExchangeManager:
                def __init__(self, config, logger, db_queries):
                    self.config = config
                    self.logger = logger
                    self.db_queries = db_queries
                    from unittest.mock import AsyncMock

                    self.exchange = AsyncMock()

                async def initialize(self):
                    return True

                async def close(self):
                    return

                # Dummy implementations for required methods
                async def fetch_candles(self, symbol, timeframe, limit):
                    return []

                async def create_order(
                    self, symbol, side, amount, price=None, order_type="market"
                ):
                    # Return a dummy order with an id
                    return {"id": "dummy_order_id"}

                async def ping(self):
                    return {"ping": "pong"}

                @property
                def rate_limiter(self):
                    # Dummy rate limiter: provide an async acquire that does nothing
                    class DummyRateLimiter:
                        async def acquire(self, _):
                            return

                    return DummyRateLimiter()

            self.exchange_manager = DummyExchangeManager(
                ctx.config, ctx.logger, ctx.db_queries
            )
        else:
            from exchanges.exchange_manager import ExchangeManager

            self.exchange_manager = ExchangeManager(
                exchange_id, sandbox=True, logger=ctx.logger, db_queries=ctx.db_queries
            )

        # Ensure that the exchange_manager has a config attribute
        if not hasattr(self.exchange_manager, "config"):
            self.exchange_manager.config = ctx.config

        self.risk_manager = ctx.risk_manager
        self.db_queries = ctx.db_queries if hasattr(ctx, "db_queries") else None

        # Assign self.exchange to support ping() calls from other components
        self.exchange = self.exchange_manager

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int):
        try:
            candles = await self.exchange_manager.fetch_candles(
                symbol, timeframe, limit
            )
            return candles
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_candles", self.logger)
            return None

    async def initialize(self) -> bool:
        """Initialize exchange interface components."""
        if (
            self.ctx.config.get("paper_mode", False)
            and self.ctx.config.get("preferred_exchange", "paper") == "paper"
        ):
            self.logger.info(
                "Paper mode enabled with 'paper' exchange - skipping real exchange connection initialization for simulated order execution."
            )
            self.initialized = True
            return True

        try:
            await self.exchange_manager.initialize()
            self.logger.info("ExchangeInterface initialized successfully.")
            self.initialized = True
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
        price: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Execute a trade with risk validation."""
        try:
            is_valid, error = await self.risk_manager.validate_trade(
                symbol, side, amount, price
            )
            if not is_valid:
                return {"success": False, "error": error}

            order = await self.exchange_manager.create_order(
                symbol, side, amount, price, order_type=order_type
            )
            # Log the trade if needed using the unified database connection
            return {"success": True, "order_id": order["id"]}
        except ValidationError as e:
            self.logger.error(f"Invalid order: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.execute_trade", self.logger)
            return {"success": False, "error": "Trade execution failed."}

    async def create_order(
        self, symbol: str, order_type: str, side: str, amount: float
    ) -> Optional[str]:
        try:
            await self.exchange_manager.rate_limiter.acquire("trade")
            order = await self.exchange_manager.create_order(
                symbol, side, Decimal(str(amount)), order_type=order_type
            )
            order_id = order.get("id")
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
            await self.exchange_manager.rate_limiter.acquire("market")
            ticker = await self.exchange_manager.fetch_ticker(symbol)
            return ticker.get("last")
        except Exception as e:
            await handle_error_async(e, "ExchangeInterface.fetch_ticker", self.logger)
            return None

    async def get_ticker(self, symbol: str) -> dict:
        await self.exchange_manager.rate_limiter.acquire("market")
        return await self.exchange_manager.exchange.fetch_ticker(symbol)

    async def close_position(self, symbol: str, amount: Decimal) -> bool:
        try:
            await self.exchange_manager.rate_limiter.acquire("trade")
            order = await self.exchange_manager.create_order(
                symbol, "sell", amount, order_type="market"
            )
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
