#!/usr/bin/env python3
"""
Module: execution/market_data.py
Handles loading and processing of market data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error
from indicators.indicators_pta import compute_indicators
from indicators.quality_monitor import quality_check
from signals.ga_synergy import prepare_market_state
from trading.math import calculate_log_returns, estimate_volatility

async def load_candles(symbol: str, ctx) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and process candle data with proper return handling"""
    try:
        with DBConnection(ctx.db_pool) as conn:
            rows = execute_sql(
                conn,
                """
                SELECT timestamp, open, high, low, close, volume, atr_14 AS ATR_14
                FROM candles 
                WHERE symbol = ?
                ORDER BY timestamp ASC
                """,
                [symbol]
            )
            
            if not rows:
                return pd.DataFrame(), {}

            df = pd.DataFrame(rows)
            df["symbol"] = symbol
            
            # Compute indicators and features
            processed_candles = compute_indicators(rows, ctx)
            if isinstance(processed_candles, pd.DataFrame) and not processed_candles.empty:
                df = pd.concat([df, processed_candles], axis=1)
            
            # Quality check
            quality_report = quality_check(df, ctx)
            
            # Process market data
            market_data = process_market_data(df, ctx)
            market_data["candles"] = df.to_dict("records")
            
            return df, market_data

    except Exception as e:
        handle_error(e, "MarketData.load_candles", logger=ctx.logger)
        return pd.DataFrame(), {}

def process_market_data(candles_df: pd.DataFrame, ctx) -> Dict[str, Any]:
    """Process market data for signal generation"""
    if candles_df.empty:
        return {}

    # Calculate returns and state
    prices = candles_df["close"].values
    returns = calculate_log_returns(prices)
    
    # Get market state
    market_state = prepare_market_state(
        candles_df.to_dict("records")
    )

    # Current metrics
    current_price = prices[-1]
    volatility = estimate_volatility(returns[-20:])
    
    return {
        "market_state": market_state,
        "current_price": current_price,
        "volatility": volatility,
        "returns": returns
    }

async def load_market_data(symbols: List[str], ctx) -> Tuple[Dict[str, Dict[str, Any]], List[pd.DataFrame]]:
    """Load market data for multiple symbols"""
    market_data = {}
    dataframes = []
    
    for symbol in symbols:
        df, market_info = await load_candles(symbol, ctx)
        if not df.empty:
            market_data[symbol] = market_info
            dataframes.append(df)
            
    return market_data, dataframes