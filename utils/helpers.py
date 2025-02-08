#!/usr/bin/env python3
"""
Module: utils/helpers.py
"""

import numpy as np
from typing import Any
from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error


def vol_estimate(symbol: str, ctx: Any) -> float:
    """
    Estimate the volatility for a given symbol using historical candle data.

    This function retrieves the most recent N candle records (default 100)
    for the symbol from the 'candles' table. It then computes the logarithmic
    returns of the closing prices and returns the standard deviation of
    these returns as the volatility estimate.

    Args:
        symbol (str): The trading pair symbol (e.g., "BTC/USDT").
        ctx (Any): Global context containing configuration (including the DB path),
                   a logger, and other settings.

    Returns:
        float: The estimated volatility. Returns 0.0 if insufficient data or error.
    """
    try:
        limit = ctx.config.get("vol_estimate_limit", 100)
        with DBConnection(ctx.db_pool) as conn:
            rows = execute_sql(
                conn,
                """
                SELECT close
                FROM candles
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [symbol, limit]
            )

        if len(rows) < 2:
            return 0.0

        prices = [r["close"] for r in rows if r["close"] is not None]
        if len(prices) < 2:
            return 0.0

        prices.reverse()
        arr = np.array(prices, dtype=float)
        log_returns = np.diff(np.log(arr))
        return float(np.std(log_returns))

    except Exception as e:
        handle_error(e, context="Helpers.vol_estimate", logger=ctx.logger)
        return 0.0
