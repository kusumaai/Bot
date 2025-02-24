#! /usr/bin/env python3
# src/execution/market_data.py
"""
Module: src.execution
Provides market data management.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from execution.exchange_interface import ExchangeInterface
from indicators.indicators_pta import compute_indicators
from indicators.quality_monitor import quality_check
from risk.validation import MarketDataValidation
from signals.market_state import prepare_market_state
from src.bot_types.base_types import MarketState
from src.database.database import DatabaseQueries, execute_sql
from src.utils.market_data_validation import MarketDataValidator
from trading.exceptions import InvalidMarketDataError, MarketDataError
from trading.math import calculate_log_returns, estimate_volatility
from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import MarketDataValidationError
from utils.numeric_handler import NumericHandler

DEFAULT_MIN_TRADE_VALUE = Decimal("10.0")  # 10 USDT minimum by default


class DummyExchange:
    async def fetch_ticker(self, symbol=None):
        return {"last": Decimal("50000")}

    async def fetch_ohlcv(self, symbol, timeframe):
        # Return dummy OHLCV data; each candle as a list: [timestamp, open, high, low, close, volume]
        return []


class MarketData:
    def __init__(self, ctx: Any, exchange, numeric_handler):
        self.ctx = ctx
        self.exchange = exchange
        self.nh = numeric_handler
        self.logger = logging.getLogger(__name__)
        self._data_cache = {}
        self._last_update = {}
        self._cache_timeout = timedelta(minutes=5)
        self._component_lock = asyncio.Lock()
        self.initialized = False
        self.exchange_interface = None  # Will be set in initialize()
        self.validator = None  # Will be set in initialize()
        self._last_emergency_check = time.time()
        self.EMERGENCY_CHECK_INTERVAL = 60  # Check every minute

        # Cache settings
        self.cache = {}
        self.last_update = {}
        self.CACHE_TTL = ctx.config.get("market_data_cache_timeout", 300)
        self.max_cache_size = ctx.config.get("max_cache_size", 1000)

        # Trading limits
        self.min_trade_value = Decimal(
            str(ctx.config.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE))
        )
        self.min_sizes = {}

        self.data_cache = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize market data service"""
        try:
            if self.initialized:
                return True

            if self.ctx.config.get("paper_mode", False):
                self.logger.info(
                    "Paper mode enabled - bypassing detailed MarketData initialization."
                )
                self.validator = MarketDataValidator(
                    logger=self.logger
                )  # Initialize without risk limits
                self.initialized = True
                return True

            if not self.ctx.risk_manager or not hasattr(
                self.ctx.risk_manager, "risk_limits"
            ):
                self.logger.error("Risk manager must be initialized first")
                return False

            self.validator = MarketDataValidator(
                risk_limits=self.ctx.risk_manager.risk_limits, logger=self.logger
            )
            self.initialized = True
            self.logger.info("MarketData initialized successfully.")
            return True
        except Exception as e:
            await handle_error_async(e, "MarketData.initialize", self.logger)
            return False

    async def get_signals(self) -> List[Dict[str, Any]]:
        try:
            data = await self.fetch_market_data()
            is_valid, error_msg = self.validator.validate_market_data(data)
            if is_valid:
                market_state = prepare_market_state(data)
                return []
            else:
                raise MarketDataValidationError(
                    f"Invalid market data received: {error_msg}"
                )
        except MarketDataValidationError as e:
            await handle_error_async(e, "MarketData.get_signals", self.logger)
            return []
        except Exception as e:
            await handle_error_async(e, "MarketData.get_signals", self.logger)
            return []

    async def fetch_market_data(self) -> Any:
        # Implementation to fetch market data
        return {}

    async def load_candles(self, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and process candle data with proper error handling"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{self.ctx.config.get('timeframe', '15m')}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if (
                    datetime.now() - cached_data["timestamp"]
                ).total_seconds() < self.cache_timeout:
                    return cached_data["df"], cached_data["market_data"]

            with DatabaseQueries(self.ctx.db_pool) as conn:
                rows = execute_sql(
                    conn,
                    """
                    SELECT timestamp, open, high, low, close, volume, atr_14 AS ATR_14
                    FROM candles 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    [
                        symbol,
                        self.ctx.config.get("timeframe", "15m"),
                        self.ctx.config.get("max_candles", 1000),
                    ],
                )

                if not rows:
                    self.logger.warning(f"No candle data found for {symbol}")
                    return pd.DataFrame(), {}

                df = pd.DataFrame(rows)
                df["symbol"] = symbol
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Compute indicators and features
                processed_candles = compute_indicators(rows, self.ctx)
                if (
                    isinstance(processed_candles, pd.DataFrame)
                    and not processed_candles.empty
                ):
                    df = pd.concat([df, processed_candles], axis=1)

                # Quality check
                quality_report = quality_check(df, self.ctx)
                if quality_report.get("warnings", []):
                    self.logger.warning(
                        f"Quality issues for {symbol}: {quality_report['warnings']}"
                    )

                # Process market data
                market_data = self._process_market_data(df)
                market_data["candles"] = df.to_dict("records")
                market_data["min_trade_info"] = self._get_min_trade_info(
                    symbol, df["close"].iloc[-1]
                )
                market_data["quality_report"] = quality_report

                # Update cache
                self.data_cache[cache_key] = {
                    "df": df,
                    "market_data": market_data,
                    "timestamp": datetime.now(),
                }

                # Enforce cache size limit
                if len(self.data_cache[cache_key]["df"]) > self.max_cache_size:
                    excess = len(self.data_cache[cache_key]["df"]) - self.max_cache_size
                    self.data_cache[cache_key]["df"] = self.data_cache[cache_key][
                        "df"
                    ].iloc[excess:]
                    self.logger.info(
                        f"Cache size for {symbol} exceeded. Removed oldest {excess} candles."
                    )

                return df, market_data

        except Exception as e:
            handle_error(e, "MarketData.load_candles", logger=self.logger)
            return pd.DataFrame(), {}

    def _process_market_data(self, candles_df: pd.DataFrame) -> Dict[str, Any]:
        """Process market data for signal generation"""
        if candles_df.empty:
            return {}

        try:
            # Calculate returns and state
            prices = candles_df["close"].values
            returns = calculate_log_returns(prices)

            # Get market state
            market_state = prepare_market_state(candles_df.to_dict("records"), self.ctx)

            # Current metrics
            current_price = Decimal(str(prices[-1]))
            volatility = Decimal(str(estimate_volatility(returns[-20:])))

            return {
                "market_state": market_state,
                "current_price": current_price,
                "volatility": volatility,
                "returns": returns.tolist(),
                "timestamp": datetime.now().timestamp(),
            }

        except Exception as e:
            handle_error(e, "MarketData._process_market_data", logger=self.logger)
            return {}

    def _get_min_trade_info(
        self, symbol: str, current_price: float
    ) -> Dict[str, Decimal]:
        """Calculate minimum trade requirements based on exchange rules and current price"""
        try:
            current_price = Decimal(str(current_price))

            # Get symbol-specific minimums if configured
            symbol_mins = self.min_sizes.get(symbol, {})
            min_qty = symbol_mins.get("min_qty", Decimal("0"))
            min_notional = symbol_mins.get("min_notional", self.min_trade_value)

            # Calculate minimum quantity based on current price
            min_qty_by_value = (
                min_notional / current_price if current_price > 0 else Decimal("0")
            )

            # Use the larger of configured min_qty and calculated min_qty
            effective_min_qty = max(min_qty, min_qty_by_value)

            return {
                "min_quantity": effective_min_qty,
                "min_notional": min_notional,
                "current_price": current_price,
                "min_position_value": effective_min_qty * current_price,
            }

        except Exception as e:
            handle_error(e, "MarketData._get_min_trade_info", logger=self.logger)
            return {
                "min_quantity": Decimal("0"),
                "min_notional": self.min_trade_value,
                "current_price": Decimal("0"),
                "min_position_value": Decimal("0"),
            }

    async def load_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        try:
            async with self._lock:
                tasks = []
                for symbol in symbols:
                    if not self._is_cache_valid(symbol):
                        tasks.append(self._fetch_symbol_data(symbol))

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    await self._process_results(symbols, results)

                return {s: self.data_cache[s] for s in symbols if s in self.data_cache}

        except Exception as e:
            self.logger.error(f"Market data load failed: {e}")
            return {}

    async def _fetch_symbol_data(self, symbol: str) -> pd.DataFrame:
        try:
            raw_data = await self.ctx.exchange_interface.fetch_ohlcv(
                symbol, timeframe=self.ctx.config["timeframe"], limit=500
            )
            return self._process_ohlcv(raw_data)
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            raise

    def _process_ohlcv(self, raw_data: List[List]) -> pd.DataFrame:
        df = pd.DataFrame(
            raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].apply(self.nh.to_decimal)
        return df

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        return (
            symbol in self.last_update
            and time.time() - self.last_update[symbol] < self.update_interval
        )

    async def validate_trade_size(self, symbol: str, size: Decimal) -> bool:
        """Validate if trade size meets minimum requirements"""
        try:
            market_info = await self.exchange_interface.fetch_market_info(symbol)
            if not market_info:
                self.logger.error(f"Market info unavailable for {symbol}.")
                return False

            is_valid, error_msg = await self.validator.validate_order_params(
                symbol=symbol,
                side="buy",  # Side doesn't matter for size validation
                amount=size,
                price=None,
            )

            if not is_valid:
                self.logger.error(f"Size validation failed for {symbol}: {error_msg}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Size validation failed for {symbol}: {e}")
            return False

    def get_tradable_symbols(self, balance: Decimal) -> List[str]:
        """Get list of symbols that can be traded with current balance"""
        tradable = []

        for symbol in self.ctx.config.get("market_list", []):
            try:
                with DatabaseQueries(self.ctx.db_pool) as conn:
                    row = execute_sql(
                        conn,
                        """
                        SELECT close 
                        FROM candles 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                        """,
                        [symbol],
                    )

                    if row and row[0]:
                        current_price = Decimal(str(row[0]["close"]))
                        min_info = self._get_min_trade_info(
                            symbol, float(current_price)
                        )

                        if balance >= min_info["min_position_value"]:
                            tradable.append(symbol)

            except Exception as e:
                handle_error(
                    e,
                    f"MarketData.get_tradable_symbols for {symbol}",
                    logger=self.logger,
                )

        return tradable

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear market data cache for specific symbol or all symbols"""
        try:
            if symbol:
                cache_key = f"{symbol}_{self.ctx.config.get('timeframe', '15m')}"
                if cache_key in self.data_cache:
                    del self.data_cache[cache_key]
            else:
                self.data_cache.clear()

        except Exception as e:
            handle_error(e, "MarketData.clear_cache", logger=self.logger)

    async def _process_results(self, symbols: List[str], results: List[pd.DataFrame]):
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                self.data_cache[symbol] = result
                self.last_update[symbol] = time.time()
            else:
                self.logger.error(f"Failed to fetch {symbol}: {result}")

    def _validate_candle(self, candle: Dict[str, Any]) -> bool:
        """Validates a single candle data point."""
        is_valid, error_msg = self.validator.validate_candle(candle)
        if not is_valid:
            self.logger.warning(f"Invalid candle data: {error_msg}")
        return is_valid

    async def _fetch_candles_from_exchange(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[Dict[str, Any]]:
        # Implementation to fetch candles from the exchange API
        # Ensure that the API response is validated
        try:
            candles = await self.ctx.exchange_manager.get_candles(
                symbol, timeframe, limit
            )
            if not isinstance(candles, list):
                raise MarketDataError("Candle data is not a list.")
            return candles
        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {symbol}: {e}")
            raise MarketDataError(f"Failed to fetch candles: {e}")

    async def get_candle(self, symbol: str, index: int) -> Optional[Dict[str, Any]]:
        async with self._lock:
            try:
                if symbol not in self.data_cache:
                    self.logger.warning(f"No cached data for symbol: {symbol}")
                    return None
                if index < 0 or index >= len(self.data_cache[symbol]):
                    self.logger.warning(
                        f"Index {index} out of range for symbol: {symbol}"
                    )
                    return None
                return self.data_cache[symbol][index]
            except Exception as e:
                self.logger.error(
                    f"Failed to get candle for {symbol} at index {index}: {e}"
                )
                return None

    async def clear_cache(self) -> None:
        async with self._lock:
            self.data_cache.clear()
            self.logger.info("Cleared all market data caches.")

    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        try:
            if symbol not in self.cache:
                return None
            data = self.cache[symbol]
            return MarketState(
                symbol=symbol,
                price=data["price"],
                volume=data["volume"],
                timestamp=data["timestamp"],
            )
        except Exception as e:
            handle_error(e, "MarketData.get_market_state")
            return None

    async def validate_market_conditions(self, data: Dict[str, Any]) -> bool:
        """Validate current market conditions against risk limits"""
        try:
            if not self.initialized:
                await self.initialize()
                if not self.initialized:
                    return False

            current_time = time.time()
            if (
                current_time - self._last_emergency_check
                >= self.EMERGENCY_CHECK_INTERVAL
            ):
                self._last_emergency_check = current_time

                # Check for emergency stop conditions
                if "drawdown" in data:
                    drawdown = self.nh.to_decimal(data["drawdown"])
                    if (
                        drawdown
                        >= self.ctx.portfolio_manager.risk_limits.emergency_stop_pct
                    ):
                        self.logger.error(
                            f"Emergency stop condition detected: {drawdown}"
                        )
                        return False

            is_valid, error_msg = self.validator.validate_market_data(data)
            if not is_valid:
                self.logger.error(f"Market data validation failed: {error_msg}")
                return False

            return True

        except Exception as e:
            await handle_error_async(
                e, "MarketData.validate_market_conditions", self.logger
            )
            return False

    async def get_latest_data(self):
        # Return dummy latest market data, e.g., empty dict
        return {}

    async def get_last_update(self, symbol):
        # Return the last update for a symbol; if not available, set it to now
        if symbol in self.last_update:
            return self.last_update[symbol]
        else:
            now = datetime.now()
            self.last_update[symbol] = now
            return now

    def validate_price(self, price):
        # Validate that price is positive and convertible to Decimal
        try:
            if not isinstance(price, Decimal):
                price = Decimal(str(price))
            return price > 0
        except Exception:
            return False

    def normalize_symbol(self, symbol: str) -> str:
        # Normalize symbol to format "XXX/YYY" in uppercase
        symbol = symbol.replace("-", "/")
        parts = symbol.split("/")
        if len(parts) == 2:
            return f"{parts[0].upper()}/{parts[1].upper()}"
        return symbol.upper()

    async def calculate_volatility(self, symbol: str, timeframe: str) -> Decimal:
        # Dummy volatility calculation, e.g., fixed value
        return Decimal("0.05")

    async def is_data_fresh(self, symbol: str = "BTC/USDT") -> bool:
        """Check if market data is fresh."""
        if symbol not in self._last_update:
            return False

        return datetime.now() - self._last_update[symbol] < self._cache_timeout

    async def get_from_cache(self, symbol: str) -> Optional[Dict]:
        """Get market data from cache if not expired."""
        if symbol not in self._data_cache:
            return None

        cached_data = self._data_cache[symbol]
        if datetime.now() - cached_data["timestamp"] > self._cache_timeout:
            del self._data_cache[symbol]
            return None

        return cached_data

    async def store_in_cache(self, symbol: str, data: Dict):
        """Store market data in cache."""
        self._data_cache[symbol] = data
        self._last_update[symbol] = data["timestamp"]

    async def get_current_price(self, symbol: str) -> Decimal:
        """Fetch current price for the given symbol using exchange ticker."""
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    async def get_candles(self, symbol: str, timeframe: str, limit: int) -> list:
        """Fetch OHLCV candles for the given symbol and timeframe."""
        data = await self.exchange.fetch_ohlcv(symbol, timeframe)
        return data[:limit]

    async def analyze_volume(self, symbol: str, timeframe: str) -> Decimal:
        """Analyze volume by computing the average volume over fetched candles."""
        candles = await self.get_candles(symbol, timeframe, limit=100)
        if not candles:
            return Decimal("0")
        total_volume = sum(candle[5] for candle in candles)
        avg_volume = total_volume / len(candles)
        return Decimal(str(avg_volume))

    async def validate_price(self, price) -> bool:
        """Asynchronously validate that the given price is a positive Decimal."""
        try:
            if not isinstance(price, Decimal):
                price = Decimal(str(price))
            return price > 0
        except Exception:
            return False

    async def analyze_volume(self, candles: List[List]) -> Dict[str, Decimal]:
        """Analyze volume data from candles."""
        try:
            total_volume = sum(Decimal(str(candle[5])) for candle in candles)
            avg_volume = total_volume / len(candles)
            return {"total_volume": total_volume, "avg_volume": avg_volume}
        except Exception as e:
            self.logger.error(f"Failed to analyze volume: {e}")
            raise MarketDataError(f"Volume analysis failed: {str(e)}")


if __name__ == "__main__":
    import asyncio
    import json
    import logging
    import os
    from dataclasses import dataclass

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Load config.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "config", "config.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        config = {}

    @dataclass
    class Context:
        logger: logging.Logger
        config: dict
        db_pool: str

    # Create context with actual config
    ctx = Context(
        logger=logger,
        config=config,
        db_pool=os.path.join(project_root, "data", "candles.db"),
    )

    async def main():
        market_data = MarketData(ctx)

        data = await market_data.load_market_data(
            ctx.config.get("market_list", ["BTC/USDT"])
        )

        if data:
            for symbol, df in data.items():
                print(f"\nSymbol: {symbol}")
                print(
                    "Min Trade Info:",
                    market_data._get_min_trade_info(
                        symbol, float(df["close"].iloc[-1])
                    ),
                )
                print("Current Price:", df["close"].iloc[-1])

        tradable = market_data.get_tradable_symbols(Decimal("100.0"))  # with 100 USDT
        print("\nTradable symbols with 100 USDT:", tradable)

    asyncio.run(main())
