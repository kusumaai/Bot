#!/usr/bin/env python3
"""
Module: execution/market_data.py
Handles loading and processing of market data with proper trade size validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error
from indicators.indicators_pta import compute_indicators
from indicators.quality_monitor import quality_check
from signals.ga_synergy import prepare_market_state
from trading.math import calculate_log_returns, estimate_volatility

DEFAULT_MIN_TRADE_VALUE = 10.0  # 10 USDT minimum by default

class MarketData:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.min_trade_value = ctx.config.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE)
        self.min_sizes = ctx.config.get("min_trade_sizes", {})

    async def load_candles(self, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and process candle data with proper return handling"""
        try:
            with DBConnection(self.ctx.db_pool) as conn:
                rows = execute_sql(
                    conn,
                    """
                    SELECT timestamp, open, high, low, close, volume, atr_14 AS ATR_14
                    FROM candles 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                    """,
                    [symbol, self.ctx.config.get("timeframe", "15m")]
                )
                
                if not rows:
                    return pd.DataFrame(), {}

                df = pd.DataFrame(rows)
                df["symbol"] = symbol
                
                # Compute indicators and features
                processed_candles = compute_indicators(rows, self.ctx)
                if isinstance(processed_candles, pd.DataFrame) and not processed_candles.empty:
                    df = pd.concat([df, processed_candles], axis=1)
                
                # Quality check
                quality_report = quality_check(df, self.ctx)
                
                # Process market data
                market_data = self._process_market_data(df)
                market_data["candles"] = df.to_dict("records")
                market_data["min_trade_info"] = self._get_min_trade_info(symbol, df["close"].iloc[-1])
                
                return df, market_data

        except Exception as e:
            handle_error(e, "MarketData.load_candles", logger=self.ctx.logger)
            return pd.DataFrame(), {}

    def _process_market_data(self, candles_df: pd.DataFrame) -> Dict[str, Any]:
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

    def _get_min_trade_info(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Calculate minimum trade requirements based on exchange rules and current price"""
        # Get symbol-specific minimums if configured
        symbol_mins = self.min_sizes.get(symbol, {})
        min_qty = symbol_mins.get("min_qty", 0.0)
        min_notional = symbol_mins.get("min_notional", self.min_trade_value)
        
        # Calculate minimum quantity based on current price
        min_qty_by_value = min_notional / current_price if current_price > 0 else 0.0
        
        # Use the larger of configured min_qty and calculated min_qty
        effective_min_qty = max(min_qty, min_qty_by_value)
        
        return {
            "min_quantity": effective_min_qty,
            "min_notional": min_notional,
            "current_price": current_price,
            "min_position_value": effective_min_qty * current_price
        }

    def validate_trade_size(self, symbol: str, quantity: float, price: float) -> bool:
        """Validate if trade size meets minimum requirements"""
        if quantity <= 0 or price <= 0:
            return False
            
        trade_value = quantity * price
        min_info = self._get_min_trade_info(symbol, price)
        
        return (
            quantity >= min_info["min_quantity"] and
            trade_value >= min_info["min_notional"]
        )

    async def load_market_data(
        self,
        symbols: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[pd.DataFrame]]:
        """Load market data for multiple symbols"""
        market_data = {}
        dataframes = []
        
        for symbol in symbols:
            df, market_info = await self.load_candles(symbol)
            if not df.empty:
                market_data[symbol] = market_info
                dataframes.append(df)
                
        return market_data, dataframes

    def get_tradable_symbols(self, balance: float) -> List[str]:
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
                        current_price = row[0]["close"]
                        min_info = self._get_min_trade_info(symbol, current_price)
                        
                        if balance >= min_info["min_position_value"]:
                            tradable.append(symbol)
                            
            except Exception as e:
                handle_error(e, "MarketData.get_tradable_symbols", logger=self.ctx.logger)
                
        return tradable

if __name__ == "__main__":
    # Example usage
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    class DummyContext:
        def __init__(self):
            self.logger = logging.getLogger()
            self.config = {
                "timeframe": "15m",
                "market_list": ["BTC/USDT", "ETH/USDT"],
                "min_trade_value": 10.0,
                "min_trade_sizes": {
                    "BTC/USDT": {
                        "min_qty": 0.0001,
                        "min_notional": 10.0
                    },
                    "ETH/USDT": {
                        "min_qty": 0.001,
                        "min_notional": 10.0
                    }
                }
            }
            self.db_pool = "data/candles.db"
    
    async def main():
        ctx = DummyContext()
        market_data = MarketData(ctx)
        
        data, dfs = await market_data.load_market_data(["BTC/USDT"])
        
        if data:
            for symbol, info in data.items():
                print(f"\nSymbol: {symbol}")
                print("Min Trade Info:", info["min_trade_info"])
                print("Current Price:", info["current_price"])
                print("Volatility:", info["volatility"])
        
        tradable = market_data.get_tradable_symbols(100.0)  # with 100 USDT
        print("\nTradable symbols with 100 USDT:", tradable)
    
    asyncio.run(main())