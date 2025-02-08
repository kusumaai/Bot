#!/usr/bin/env python3
"""
Module: ingestion/exchange_data.py
"""

import asyncio
from typing import Any, Dict, List
import ccxt.async_support as ccxt
from utils.error_handler import handle_error

# Global cache for exchange instances
_exchange_cache: Dict[str, "ExchangeInstance"] = {}


class ExchangeInstance:
    """
    Represents an exchange instance with its identifier and the actual CCXT exchange instance.
    """
    def __init__(self, id: str, instance: Any) -> None:
        self.id = id
        self.instance = instance


async def get_exchange_instance(eid: str, ctx: Any) -> ExchangeInstance:
    """
    Retrieve an exchange instance for the given exchange identifier, using the global cache.
    It references credentials from ctx.env, and also looks up config overrides 
    such as timeout and rate limiting from ctx.config.

    Args:
        eid (str): The exchange identifier (e.g., 'kucoin', 'bybit').
        ctx (Any): Global context containing configuration, environment variables, and logger.
    
    Returns:
        ExchangeInstance: An object containing the exchange ID and the CCXT exchange instance.
                         Raises an exception if the exchange is unknown or fails to load.
    """
    if eid in _exchange_cache:
        return _exchange_cache[eid]

    # Retrieve exchange config from context if any
    exchange_cfg = ctx.config.get("exchange_settings", {})
    default_timeout = exchange_cfg.get("timeout", 30000)
    default_enable_rate_limit = exchange_cfg.get("enableRateLimit", True)

    ex_args = {
        "enableRateLimit": default_enable_rate_limit,
        "timeout": default_timeout
    }

    # Attempt to set API credentials if present in ctx.env
    # This is purely optional and exchange-specific.
    eid_lower = eid.lower()
    if eid_lower == "kucoin":
        ex_args["apiKey"] = ctx.env.get("KUCOIN_API_KEY", "")
        ex_args["secret"] = ctx.env.get("KUCOIN_SECRET", "")
        if "KUCOIN_PASSWORD" in ctx.env:
            ex_args["password"] = ctx.env["KUCOIN_PASSWORD"]
    elif eid_lower == "bybit":
        ex_args["apiKey"] = ctx.env.get("BYBIT_API_KEY", "")
        ex_args["secret"] = ctx.env.get("BYBIT_SECRET", "")
    else:
        # If it's an exchange that doesn't need secrets or wasn't recognized, 
        # we won't attempt to set credentials here.
        pass

    try:
        # Dynamically obtain the exchange class from ccxt based on the identifier.
        exchange_class = getattr(ccxt, eid_lower)
    except AttributeError as e:
        handle_error(
            e,
            context=f"get_exchange_instance: Unknown exchange '{eid}'",
            logger=ctx.logger
        )
        raise

    try:
        instance = exchange_class(ex_args)
        await instance.load_markets()
    except Exception as e:
        handle_error(
            e,
            context=f"get_exchange_instance: Failed to load markets for '{eid}'",
            logger=ctx.logger
        )
        raise

    exch = ExchangeInstance(eid, instance)
    _exchange_cache[eid] = exch
    return exch


async def fetch_ohlcv(symbol: str, timeframe: str, limit: int, ctx: Any) -> List[List[Any]]:
    """
    Fetch OHLCV (candlestick) data for a given symbol from the primary exchange.
    Respects any retry configuration in `ctx.config.get("exchange_retries", 1)`.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe for the candles (e.g., '1h').
        limit (int): The number of candlesticks to fetch.
        ctx (Any): Global context containing configuration, environment variables, and logger.
    
    Returns:
        List[List[Any]]: A list of candlestick data; each candle is represented as a list.
                         Returns an empty list if an error occurs or if no exchange is configured.
    """
    exchanges_config = ctx.config.get("exchanges", [])
    if not exchanges_config:
        ctx.logger.error("No exchanges configured. Cannot fetch OHLCV data.")
        return []

    primary = exchanges_config[0]
    max_retries = ctx.config.get("exchange_retries", 1)

    for attempt in range(1, max_retries + 1):
        try:
            exch_instance = await get_exchange_instance(primary, ctx)
            data = await exch_instance.instance.fetch_ohlcv(symbol, timeframe, limit=limit)
            return data
        except Exception as e:
            handle_error(
                e,
                context=f"fetch_ohlcv for symbol {symbol}, attempt {attempt}",
                logger=ctx.logger
            )
            if attempt == max_retries:
                ctx.logger.error("Max retries reached. Returning empty OHLCV data.")
                return []
            else:
                # Retry after a small sleep if desired
                await asyncio.sleep(1.0)
    return []
