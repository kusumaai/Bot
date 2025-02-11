#!/usr/bin/env python3
"""
Module: ingestion/exchange_data.py
Handles exchange data ingestion with proper error handling and rate limiting
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from utils.error_handler import handle_error

# Global cache for exchange instances with timeout management
class ExchangeCache:
    def __init__(self):
        self.instances: Dict[str, "ExchangeInstance"] = {}
        self.last_access: Dict[str, float] = {}
        self.cache_timeout = 3600  # 1 hour default

    def get(self, eid: str) -> Optional["ExchangeInstance"]:
        if eid not in self.instances:
            return None
        
        last_access = self.last_access.get(eid, 0)
        if time.time() - last_access > self.cache_timeout:
            del self.instances[eid]
            return None
            
        self.last_access[eid] = time.time()
        return self.instances[eid]

    def set(self, eid: str, instance: "ExchangeInstance") -> None:
        self.instances[eid] = instance
        self.last_access[eid] = time.time()

_exchange_cache = ExchangeCache()

class ExchangeInstance:
    """Represents an exchange instance with rate limiting and error tracking"""
    def __init__(self, id: str, instance: Any) -> None:
        self.id = id
        self.instance = instance
        self.last_request = 0
        self.request_count = 0
        self.error_count = 0
        self.last_error = None

    async def rate_limit(self) -> None:
        """Implement rate limiting"""
        now = time.time()
        if now - self.last_request >= 1.0:
            self.request_count = 0
            self.last_request = now
        
        if self.request_count >= 20:  # Default rate limit
            await asyncio.sleep(1.0)
            self.request_count = 0
            self.last_request = time.time()
        
        self.request_count += 1

async def get_exchange_instance(eid: str, ctx: Any) -> ExchangeInstance:
    """Get or create an exchange instance with proper caching and validation"""
    cached = _exchange_cache.get(eid)
    if cached:
        return cached

    exchange_cfg = ctx.config.get("exchange_settings", {})
    default_timeout = exchange_cfg.get("timeout", 30000)
    default_enable_rate_limit = exchange_cfg.get("enableRateLimit", True)
    
    ex_args = {
        "enableRateLimit": default_enable_rate_limit,
        "timeout": default_timeout
    }

    # Set credentials based on environment variables
    eid_lower = eid.lower()
    credentials = {
        "kucoin": {
            "apiKey": "KUCOIN_API_KEY",
            "secret": "KUCOIN_SECRET",
            "password": "KUCOIN_PASSWORD"
        },
        "bybit": {
            "apiKey": "BYBIT_API_KEY",
            "secret": "BYBIT_SECRET"
        }
    }.get(eid_lower, {})

    for arg_name, env_key in credentials.items():
        if env_key in ctx.env:
            ex_args[arg_name] = ctx.env[env_key]

    try:
        exchange_class = getattr(ccxt, eid_lower)
    except AttributeError as e:
        handle_error(e, f"get_exchange_instance: Unknown exchange '{eid}'", logger=ctx.logger)
        raise

    try:
        instance = exchange_class(ex_args)
        await instance.load_markets()
        
        exch = ExchangeInstance(eid, instance)
        _exchange_cache.set(eid, exch)
        return exch

    except Exception as e:
        handle_error(e, f"get_exchange_instance: Failed to load markets for '{eid}'", logger=ctx.logger)
        raise

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int, ctx: Any) -> List[List[Any]]:
    """Fetch OHLCV data with proper error handling and validation"""
    exchanges_config = ctx.config.get("exchanges", [])
    if not exchanges_config:
        ctx.logger.error("No exchanges configured. Cannot fetch OHLCV data.")
        return []

    primary = exchanges_config[0]
    max_retries = ctx.config.get("exchange_retries", 1)
    retry_delay = ctx.config.get("retry_delay", 1.0)

    for attempt in range(1, max_retries + 1):
        try:
            exch_instance = await get_exchange_instance(primary, ctx)
            await exch_instance.rate_limit()
            
            data = await exch_instance.instance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Validate timestamps
            if data:
                timestamps = [candle[0] for candle in data]
                expected_interval = {
                    '15m': 15 * 60 * 1000,
                    '1h': 60 * 60 * 1000,
                    '4h': 4 * 60 * 60 * 1000,
                    '1d': 24 * 60 * 60 * 1000
                }.get(timeframe)
                
                if expected_interval:
                    gaps = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                    large_gaps = [g for g in gaps if g > expected_interval * 2]
                    if large_gaps:
                        ctx.logger.warning(f"Found {len(large_gaps)} large time gaps in {symbol} data")
            
            return data

        except Exception as e:
            handle_error(e, f"fetch_ohlcv for {symbol}, attempt {attempt}", logger=ctx.logger)
            if attempt == max_retries:
                ctx.logger.error("Max retries reached. Returning empty OHLCV data.")
                return []
            await asyncio.sleep(retry_delay * attempt)  # Exponential backoff

    return []
