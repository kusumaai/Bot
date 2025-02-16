#! /usr/bin/env python3
# src/exchanges/exchange_cache.py
"""
Module: src.exchanges.exchange_cache
Provides a thread-safe exchange instance caching mechanism.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from src.exchanges.exchange_manager import ExchangeManager
from src.utils.error_handler import handle_error_async

logger = logging.getLogger(__name__)


class ExchangeCache:
    """Thread-safe singleton cache for exchange instances."""

    def __init__(self):
        self._instances: Dict[str, ExchangeManager] = {}
        self._lock = asyncio.Lock()
        self._initialization_locks: Dict[str, asyncio.Lock] = {}
        self.logger = logger

    async def get_exchange(
        self, exchange_id: str, ctx: Any
    ) -> Optional[ExchangeManager]:
        """
        Get or create an exchange instance with proper synchronization.

        Args:
            exchange_id: Unique identifier for the exchange
            ctx: Context object containing exchange configuration

        Returns:
            ExchangeManager instance or None if initialization fails
        """
        try:
            # Fast path - check if instance exists
            if exchange_id in self._instances:
                return self._instances[exchange_id]

            # Get or create initialization lock for this exchange_id
            async with self._lock:
                if exchange_id not in self._initialization_locks:
                    self._initialization_locks[exchange_id] = asyncio.Lock()
                init_lock = self._initialization_locks[exchange_id]

            # Ensure only one initialization per exchange_id
            async with init_lock:
                # Double-check pattern
                if exchange_id in self._instances:
                    return self._instances[exchange_id]

                # Create and initialize new instance
                exchange = ExchangeManager(
                    exchange_id=exchange_id,
                    api_key=getattr(ctx, "api_key", None),
                    api_secret=getattr(ctx, "api_secret", None),
                    sandbox=getattr(ctx, "sandbox", True),
                    logger=getattr(ctx, "logger", None),
                    db_queries=getattr(ctx, "db_queries", None),
                )

                if await exchange.initialize():
                    async with self._lock:
                        self._instances[exchange_id] = exchange
                    return exchange
                else:
                    self.logger.error(f"Failed to initialize exchange {exchange_id}")
                    return None

        except Exception as e:
            await handle_error_async(e, "ExchangeCache.get_exchange", self.logger)
            return None

    async def remove_exchange(self, exchange_id: str) -> None:
        """
        Safely remove an exchange instance from the cache.

        Args:
            exchange_id: Unique identifier for the exchange to remove
        """
        try:
            async with self._lock:
                if exchange_id in self._instances:
                    exchange = self._instances[exchange_id]
                    await exchange.close()
                    del self._instances[exchange_id]
                    if exchange_id in self._initialization_locks:
                        del self._initialization_locks[exchange_id]
        except Exception as e:
            await handle_error_async(e, "ExchangeCache.remove_exchange", self.logger)

    async def clear(self) -> None:
        """Safely clear all cached exchange instances."""
        try:
            async with self._lock:
                for exchange_id, exchange in self._instances.items():
                    try:
                        await exchange.close()
                    except Exception as e:
                        self.logger.error(f"Error closing exchange {exchange_id}: {e}")
                self._instances.clear()
                self._initialization_locks.clear()
        except Exception as e:
            await handle_error_async(e, "ExchangeCache.clear", self.logger)


# Thread-safe singleton instance
_exchange_cache: Optional[ExchangeCache] = None
_cache_lock = asyncio.Lock()


async def get_exchange_instance(
    exchange_id: str, ctx: Any
) -> Optional[ExchangeManager]:
    """
    Thread-safe function to get an exchange instance from the cache.

    Args:
        exchange_id: Unique identifier for the exchange
        ctx: Context object containing exchange configuration

    Returns:
        ExchangeManager instance or None if initialization fails
    """
    global _exchange_cache

    try:
        async with _cache_lock:
            if _exchange_cache is None:
                _exchange_cache = ExchangeCache()

        return await _exchange_cache.get_exchange(exchange_id, ctx)

    except Exception as e:
        await handle_error_async(e, "get_exchange_instance", logger)
        return None
