#!/usr/bin/env python3
"""
Module: execution/exchange_interface.py
Handles all exchange interactions
"""

import time
import ccxt.async_support as ccxt
from typing import Dict, Any, Optional
from utils.error_handler import handle_error

async def create_exchange(ctx) -> Optional[ccxt.Exchange]:
    """Initialize exchange connection"""
    key = ctx.config.get("kucoin_api_key", "")
    sec = ctx.config.get("kucoin_secret", "")
    pwd = ctx.config.get("kucoin_password", "")

    exchange = ccxt.kucoin({
        "apiKey": key,
        "secret": sec,
        "password": pwd,
        "enableRateLimit": True
    })
    
    try:
        await exchange.load_markets()
        ctx.logger.info("Exchange loaded successfully")
        return exchange
    except Exception as e:
        handle_error(e, "ExchangeInterface.create_exchange", logger=ctx.logger)
        return None

async def fetch_ticker(symbol: str, ctx) -> float:
    """Fetch current price"""
    if not ctx.exchange:
        return -1
    try:
        tdata = await ctx.exchange.fetch_ticker(symbol)
        return tdata.get("last") or tdata.get("close") or -1
    except Exception as e:
        handle_error(e, "ExchangeInterface.fetch_ticker", logger=ctx.logger)
        return -1

async def place_order(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    ctx
) -> Optional[Dict[str, Any]]:
    """Place order with position sizing"""
    if ctx.config.get("paper_mode", False):
        ctx.logger.info(f"[PAPER] {side} {amount} {symbol} @ {price}")
        return {
            "id": f"paper-{time.time()}",
            "symbol": symbol,
            "direction": side,
            "price": price,
            "amount": amount,
            "timestamp": time.time()
        }

    if not ctx.exchange:
        return None

    try:
        order_type = ctx.config.get("order_type", "limit")
        params = {}
        
        if order_type == "market":
            order = await ctx.exchange.create_market_order(
                symbol, side, amount, None, params
            )
        else:
            order = await ctx.exchange.create_limit_order(
                symbol, side, amount, price, params
            )

        if not order:
            return None

        return {
            "id": order.get("id", ""),
            "symbol": symbol,
            "direction": side,
            "price": price,
            "amount": amount,
            "timestamp": order.get("timestamp")
        }
    except Exception as e:
        handle_error(e, "ExchangeInterface.place_order", logger=ctx.logger)
        return None

async def close_position(trade: Dict[str, Any], ctx) -> bool:
    """Close an open position"""
    if ctx.config.get("paper_mode", False):
        return True
        
    try:
        await ctx.exchange.create_market_order(
            trade["symbol"],
            "sell" if trade["direction"] == "long" else "buy",
            trade["position_size"]
        )
        return True
    except Exception as e:
        handle_error(e, "ExchangeInterface.close_position", logger=ctx.logger)
        return False