#!/usr/bin/env python3
"""
Module: exchanges/exchange_manager.py
"""
import logging
import ccxt.async_support as ccxt  # if using async ccxt
# or import ccxt if synchronous

async def select_exchange(symbol: str, ctx) -> object:
    """
    Returns an instantiated exchange object, 
    configured with API keys from ctx.config, 
    ignoring ctx.env entirely.
    """
    exchange_name = ctx.config.get("primary_exchange", "kucoin")

    # If you want different logic for different symbols, do it here
    if exchange_name.lower() == "kucoin":
        return await _get_kucoin_exchange(ctx)
    # elif exchange_name.lower() == "bybit":
    #     return await _get_bybit_exchange(ctx)
    # etc.

    ctx.logger.error(f"select_exchange => unknown exchange '{exchange_name}'")
    return None


async def _get_kucoin_exchange(ctx):
    """
    Creates a ccxt kucoin instance using config-based credentials,
    ignoring any 'env' references.
    """
    api_key     = ctx.config.get("kucoin_api_key", "")
    secret      = ctx.config.get("kucoin_secret", "")
    password    = ctx.config.get("kucoin_password", "")

    if not api_key or not secret or not password:
        ctx.logger.warning("KuCoin API credentials missing or empty in config.")

    exchange = ccxt.kucoin({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True
    })

    # load markets if needed
    await exchange.load_markets()
    return exchange


async def fetch_ticker(symbol: str, ctx) -> float:
    """
    Fetch the last price (ticker) for 'symbol' 
    from whichever exchange is configured.
    Return -1 if there's an error, 
    so the bot can skip the trade.
    """
    try:
        exch = await select_exchange(symbol, ctx)
        if not exch:
            return -1
        ticker_data = await exch.fetch_ticker(symbol)
        if not ticker_data:
            return -1
        last_price = ticker_data.get("last") or ticker_data.get("close") or -1
        return last_price
    except Exception as e:
        ctx.logger.error(f"Error in ExchangeManager.fetch_ticker for {symbol}: {e}")
        return -1


async def place_order(exchange, symbol, side, amount, price, ctx):
    """
    Basic example of placing a limit order 
    with ccxt (async). Adjust as needed.
    """
    try:
        params = {}
        # For KuCoin spot: might need "type": "market" or "limit"
        # or other param if you want reduce_only, etc.
        order_type = ctx.config.get("order_type", "limit")

        if order_type == "market":
            # place a market order
            order = await exchange.create_market_order(symbol, side, amount, None, params)
        else:
            # place a limit order
            order = await exchange.create_limit_order(symbol, side, amount, price, params)

        return {
            "id": order.get("id", ""),
            "symbol": symbol,
            "direction": side,
            "price": price,
            "amount": amount,
            "timestamp": order.get("timestamp")
        }
    except Exception as e:
        ctx.logger.error(f"Error in ExchangeManager.place_order => {symbol}, side={side}, e={e}")
        return None


async def close_order(exchange, trade_id: str, price: float, ctx):
    """
    Example. If you must cancel + place an opposite side 
    or something else, do it here.
    """
    # There's no universal "close_order" in spot 
    # unless you're doing margin or futures with ccxt.
    # For demonstration:
    ctx.logger.info(f"Close order => trade_id={trade_id}, price={price}")
    # do real logic if on futures, e.g. create opposite order to close short/long
    return
