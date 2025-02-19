import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from src.utils.exceptions import MarketDataError
from src.utils.numeric_handler import NumericHandler

logger = logging.getLogger(__name__)


class MarketData:
    """Manages market data operations."""

    def __init__(
        self, ctx: Any, exchange: Any, numeric_handler: Optional[NumericHandler] = None
    ):
        """
        Initialize market data manager.

        Args:
            ctx: Trading context
            exchange: Exchange interface
            numeric_handler: Optional numeric handler instance
        """
        self.logger = logging.getLogger(__name__)
        self.ctx = ctx
        self.exchange = exchange
        self.nh = numeric_handler or NumericHandler()
        self._data_cache: Dict[str, Dict] = {}
        self._last_update: Dict[str, datetime] = {}
        self._cache_timeout = 300  # 5 minutes
        self._component_lock = asyncio.Lock()

    async def get_market_price(self, symbol: str) -> Decimal:
        """
        Get current market price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current market price

        Raises:
            MarketDataError: If price cannot be retrieved
        """
        try:
            data = await self.get_market_data(symbol)
            return self.nh.convert_to_decimal(str(data["price"]))
        except Exception as e:
            raise MarketDataError(f"Failed to get market price for {symbol}: {e}")

    async def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Market data dictionary

        Raises:
            MarketDataError: If data cannot be retrieved
        """
        try:
            # Check cache first
            if symbol in self._data_cache:
                cache_age = datetime.now() - self._last_update[symbol]
                if cache_age.seconds < self._cache_timeout:
                    return self._data_cache[symbol]

            # Fetch fresh data
            data = await self._fetch_market_data(symbol)

            # Validate and cache data
            if not self._validate_market_data(data):
                raise MarketDataError(f"Invalid market data for {symbol}")

            self._data_cache[symbol] = data
            self._last_update[symbol] = datetime.now()
            return data

        except Exception as e:
            raise MarketDataError(f"Failed to get market data for {symbol}: {e}")

    async def _fetch_market_data(self, symbol: str) -> Dict:
        """
        Fetch market data from exchange.

        Args:
            symbol: Trading symbol

        Returns:
            Market data dictionary

        Raises:
            MarketDataError: If data cannot be fetched
        """
        try:
            # Add exchange API call here
            # For now return dummy data
            return {
                "symbol": symbol,
                "price": "50000",
                "volume": "1000000",
                "timestamp": datetime.now(),
            }
        except Exception as e:
            raise MarketDataError(f"Failed to fetch market data for {symbol}: {e}")

    def _validate_market_data(self, data: Dict) -> bool:
        """
        Validate market data.

        Args:
            data: Market data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["symbol", "price", "volume", "timestamp"]
        return all(field in data for field in required_fields)

    async def get_last_update(self, symbol: str) -> Optional[datetime]:
        """
        Get last update time for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Last update time or None if not available
        """
        return self._last_update.get(symbol)

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(symbol, str):
            return False
        parts = symbol.split("/")
        return len(parts) == 2 and all(p.isalnum() for p in parts)

    def validate_candle(self, candle: Dict) -> bool:
        """
        Validate candle data.

        Args:
            candle: Candle data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(field in candle for field in required_fields):
            return False

        try:
            # Validate numeric fields
            for field in ["open", "high", "low", "close", "volume"]:
                value = self.nh.convert_to_decimal(str(candle[field]))
                if value <= 0:
                    return False

            # Validate high/low relationship
            high = self.nh.convert_to_decimal(str(candle["high"]))
            low = self.nh.convert_to_decimal(str(candle["low"]))
            if high < low:
                return False

            return True

        except Exception:
            return False

    async def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear market data cache.

        Args:
            symbol: Optional symbol to clear, if None clears all
        """
        async with self._component_lock:
            if symbol:
                self._data_cache.pop(symbol, None)
                self._last_update.pop(symbol, None)
            else:
                self._data_cache.clear()
                self._last_update.clear()
