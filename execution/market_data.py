#!/usr/bin/env python3
"""
Module: execution/market_data.py
Handles loading and processing of market data with proper error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from decimal import Decimal
from datetime import datetime, timedelta

from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error
from indicators.indicators_pta import compute_indicators
from indicators.quality_monitor import quality_check
from signals.ga_synergy import prepare_market_state
from trading.math import calculate_log_returns, estimate_volatility

DEFAULT_MIN_TRADE_VALUE = Decimal('10.0')  # 10 USDT minimum by default

class MarketData:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.min_trade_value = Decimal(str(ctx.config.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE)))
        self.min_sizes = {
            symbol: {
                "min_qty": Decimal(str(data.get("min_qty", "0"))),
                "min_notional": Decimal(str(data.get("min_notional", self.min_trade_value)))
            }
            for symbol, data in ctx.config.get("min_trade_sizes", {}).items()
        }
        self.cache = {}
        self.cache_timeout = ctx.config.get("market_data_cache_timeout", 60)  # seconds

    async def load_candles(self, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and process candle data with proper error handling"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{self.ctx.config.get('timeframe', '15m')}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data["timestamp"]).total_seconds() < self.cache_timeout:
                    return cached_data["df"], cached_data["market_data"]

            with DBConnection(self.ctx.db_pool) as conn:
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
                        self.ctx.config.get("max_candles", 1000)
                    ]
                )
                
                if not rows:
                    self.logger.warning(f"No candle data found for {symbol}")
                    return pd.DataFrame(), {}

                df = pd.DataFrame(rows)
                df["symbol"] = symbol
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                
                # Compute indicators and features
                processed_candles = compute_indicators(rows, self.ctx)
                if isinstance(processed_candles, pd.DataFrame) and not processed_candles.empty:
                    df = pd.concat([df, processed_candles], axis=1)
                
                # Quality check
                quality_report = quality_check(df, self.ctx)
                if quality_report.get("warnings", []):
                    self.logger.warning(f"Quality issues for {symbol}: {quality_report['warnings']}")
                
                # Process market data
                market_data = self._process_market_data(df)
                market_data["candles"] = df.to_dict("records")
                market_data["min_trade_info"] = self._get_min_trade_info(symbol, df["close"].iloc[-1])
                market_data["quality_report"] = quality_report
                
                # Update cache
                self.cache[cache_key] = {
                    "df": df,
                    "market_data": market_data,
                    "timestamp": datetime.now()
                }
                
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
                "timestamp": datetime.now().timestamp()
            }

        except Exception as e:
            handle_error(e, "MarketData._process_market_data", logger=self.logger)
            return {}

    def _get_min_trade_info(self, symbol: str, current_price: float) -> Dict[str, Decimal]:
        """Calculate minimum trade requirements based on exchange rules and current price"""
        try:
            current_price = Decimal(str(current_price))
            
            # Get symbol-specific minimums if configured
            symbol_mins = self.min_sizes.get(symbol, {})
            min_qty = symbol_mins.get("min_qty", Decimal('0'))
            min_notional = symbol_mins.get("min_notional", self.min_trade_value)
            
            # Calculate minimum quantity based on current price
            min_qty_by_value = min_notional / current_price if current_price > 0 else Decimal('0')
            
            # Use the larger of configured min_qty and calculated min_qty
            effective_min_qty = max(min_qty, min_qty_by_value)
            
            return {
                "min_quantity": effective_min_qty,
                "min_notional": min_notional,
                "current_price": current_price,
                "min_position_value": effective_min_qty * current_price
            }

        except Exception as e:
            handle_error(e, "MarketData._get_min_trade_info", logger=self.logger)
            return {
                "min_quantity": Decimal('0'),
                "min_notional": self.min_trade_value,
                "current_price": Decimal('0'),
                "min_position_value": Decimal('0')
            }

    def validate_trade_size(self, symbol: str, quantity: Decimal, price: Decimal) -> bool:
        """Validate if trade size meets minimum requirements"""
        try:
            if quantity <= 0 or price <= 0:
                return False
                
            trade_value = quantity * price
            min_info = self._get_min_trade_info(symbol, float(price))
            
            return (
                quantity >= min_info["min_quantity"] and
                trade_value >= min_info["min_notional"]
            )

        except Exception as e:
            handle_error(e, "MarketData.validate_trade_size", logger=self.logger)
            return False

    async def load_market_data(
        self,
        symbols: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[pd.DataFrame]]:
        """Load market data for multiple symbols"""
        market_data = {}
        dataframes = []
        
        for symbol in symbols:
            try:
                df, market_info = await self.load_candles(symbol)
                if not df.empty:
                    market_data[symbol] = market_info
                    dataframes.append(df)
                
            except Exception as e:
                handle_error(e, f"MarketData.load_market_data for {symbol}", logger=self.logger)
                
        return market_data, dataframes

    def get_tradable_symbols(self, balance: Decimal) -> List[str]:
        """Get list of symbols that can be traded with current balance"""
        tradable = []
        
        for symbol in self.ctx.config.get("market_list", []):
            try:
                with DBConnection(self.ctx.db_pool) as conn:
                    row = execute_sql(
                        conn,
                        """
                        SELECT close 
                        FROM candles 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                        """,
                        [symbol]
                    )
                    
                    if row and row[0]:
                        current_price = Decimal(str(row[0]["close"]))
                        min_info = self._get_min_trade_info(symbol, float(current_price))
                        
                        if balance >= min_info["min_position_value"]:
                            tradable.append(symbol)
                            
            except Exception as e:
                handle_error(e, f"MarketData.get_tradable_symbols for {symbol}", logger=self.logger)
                
        return tradable

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear market data cache for specific symbol or all symbols"""
        try:
            if symbol:
                cache_key = f"{symbol}_{self.ctx.config.get('timeframe', '15m')}"
                if cache_key in self.cache:
                    del self.cache[cache_key]
            else:
                self.cache.clear()
                
        except Exception as e:
            handle_error(e, "MarketData.clear_cache", logger=self.logger)

if __name__ == "__main__":
    import asyncio
    import logging
    import json
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
        with open(config_path, 'r') as f:
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
        db_pool=os.path.join(project_root, "data", "candles.db")
    )

    async def main():
        market_data = MarketData(ctx)
        
        data, dfs = await market_data.load_market_data(ctx.config.get("market_list", ["BTC/USDT"]))
        
        if data:
            for symbol, info in data.items():
                print(f"\nSymbol: {symbol}")
                print("Min Trade Info:", info["min_trade_info"])
                print("Current Price:", info["current_price"])
                print("Volatility:", info["volatility"])
        
        tradable = market_data.get_tradable_symbols(Decimal('100.0'))  # with 100 USDT
        print("\nTradable symbols with 100 USDT:", tradable)

    asyncio.run(main())