#! /usr/bin/env python3
# src/trading/market_data.py
"""
Module: src.trading
Provides market data functionality.
"""
import asyncio
import logging
from typing import Any, Dict, List

from risk.validation import MarketDataValidation
from trading.exceptions import InvalidMarketDataError, MarketDataError
from utils.error_handler import handle_error_async
from utils.exceptions import MarketDataValidationError


# market data class that fetches and validates market data
class MarketData:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self.validation = None  # Will be initialized in initialize()

    async def initialize(self) -> bool:
        try:
            if self.initialized:
                return True

            if not self.ctx.risk_manager or not hasattr(
                self.ctx.risk_manager, "risk_limits"
            ):
                self.logger.error("Risk manager must be initialized first")
                return False

            self.validation = MarketDataValidation(
                self.ctx.risk_manager.risk_limits, self.logger
            )
            self.initialized = True
            return True
        except Exception as e:
            await handle_error_async(e, "MarketData.initialize", self.logger)
            return False

    async def get_signals(self) -> List[Dict[str, Any]]:
        try:
            data = await self.fetch_market_data()
            if self.validation.validate_market_data(data):
                # Process data to generate signals
                return []
            else:
                raise MarketDataValidationError("Invalid market data received.")
        except MarketDataValidationError as e:
            await handle_error_async(e, "MarketData.get_signals", self.logger)
            return []
        except Exception as e:
            await handle_error_async(e, "MarketData.get_signals", self.logger)
            return []

    async def fetch_market_data(self) -> Any:
        try:
            if not self.initialized:
                raise MarketDataError("Market data not initialized")

            if not self.ctx.exchange_interface:
                raise MarketDataError("Exchange interface not available")

            # Fetch market data from exchange
            market_data = await self.ctx.exchange_interface.fetch_market_data()

            # Basic validation before returning
            if not market_data or not isinstance(market_data, dict):
                raise InvalidMarketDataError("Invalid market data format received")

            return market_data

        except Exception as e:
            await handle_error_async(e, "MarketData.fetch_market_data", self.logger)
            return {}
        return {}

    async def validate_market_data(self, data: Any) -> bool:
        try:
            if not self.initialized:
                raise MarketDataError("Market data not initialized")

            if not self.validation:
                raise MarketDataError("Validation not initialized")

            return self.validation.validate_market_data(data)
        except Exception as e:
            await handle_error_async(e, "MarketData.validate_market_data", self.logger)
            return False

    async def transform_market_data(self, data: Any) -> Any:
        try:
            if not self.initialized:
                raise MarketDataError("Market data not initialized")

            if not self.ctx.transformer:
                raise MarketDataError("Transformer not initialized")

            return self.ctx.transformer.transform_market_data(data)
        except Exception as e:
            await handle_error_async(e, "MarketData.transform_market_data", self.logger)
            return data
