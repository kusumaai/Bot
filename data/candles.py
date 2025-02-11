#!/usr/bin/env python3
"""
Tool: data/candles.py
Database management for OHLCV candle data with proper error handling
"""

import os
import sys
import time
import sqlite3
import ccxt
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

from utils.logger import setup_logger
from utils.error_handler import handle_error, ValidationError
from database.queries import DatabaseQueries

# Initialize logger at module level
logger = setup_logger(name="CandleManager", level="INFO")

def get_stable_coin_markets(exchange: ccxt.Exchange, base_coins: List[str], 
                           stable_coins: List[str]) -> List[str]:
    """Get valid trading pairs for specified base and stable coins"""
    markets = exchange.fetch_markets()
    selected = []
    for market in markets:
        symbol = market.get('symbol')
        if symbol and '/' in symbol:
            base, quote = symbol.split('/')
            if base in base_coins and quote in stable_coins:
                selected.append(symbol)
    return selected

def fetch_and_save_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    cursor: sqlite3.Cursor,
    limit: int = 100,
    period: int = 14
) -> int:
    """Fetch and save candle data with ATR calculation"""
    state = {"prev_close": None, "atr": None, "tr_list": []}
    total_inserted = 0

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=since, 
                limit=limit
            )
            
            if not candles:
                break

            for candle in candles:
                ts, open_val, high, low, close, volume = candle
                
                # Calculate ATR
                if state["prev_close"] is None:
                    atr = 0.0
                    state["prev_close"] = close
                else:
                    tr = max(
                        high - low,
                        abs(high - state["prev_close"]),
                        abs(low - state["prev_close"])
                    )
                    if len(state["tr_list"]) < period:
                        state["tr_list"].append(tr)
                        atr = sum(state["tr_list"]) / len(state["tr_list"]) if len(state["tr_list"]) == period else 0.0
                    else:
                        atr = (state["atr"] * (period - 1) + tr) / period
                        state["atr"] = atr
                    state["prev_close"] = close

                dt_str = datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO candles (
                        symbol, timeframe, timestamp, open, high, low, close, volume,
                        datetime, atr_14, exchange
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timeframe, ts, open_val, high, low, close, volume, 
                      dt_str, atr, exchange.id))
                total_inserted += 1

            logger.info(f"Saved {len(candles)} candles for {symbol} at {timeframe}")
            since = candles[-1][0] + 1
            
            if len(candles) < limit:
                break
                
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            handle_error(e, f"fetch_and_save_candles for {symbol}", logger=logger)
            time.sleep(exchange.rateLimit / 1000)  # Error backoff
            break

    return total_inserted

def main():
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        base_coins = ['BTC', 'ETH']
        stable_coins = ['USDT', 'USDC', 'BUSD', 'USDP', 'DAI']
        timeframes = ['15m', '1h', '4h', '1d', '1w']
        
        symbols = get_stable_coin_markets(exchange, base_coins, stable_coins)
        logger.info(f"Selected markets: {symbols}")

        since = exchange.parse8601('2020-01-01T00:00:00Z')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "candles.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    atr_14 REAL,
                    exchange TEXT,
                    PRIMARY KEY(symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()

            for symbol in symbols:
                for tf in timeframes:
                    logger.info(f"\nFetching candles for {symbol} at {tf} from {since}")
                    total = fetch_and_save_candles(
                        exchange, symbol, tf, since, cursor, limit=1000, period=14
                    )
                    conn.commit()
                    logger.info(f"Total candles saved for {symbol} at {tf}: {total}")

        logger.info("All candles saved successfully.")

    except Exception as e:
        handle_error(e, "main", logger=logger)
        sys.exit(1)

class CandleProcessor:
    def __init__(self, db_queries: DatabaseQueries, logger: logging.Logger):
        self.db = db_queries
        self.logger = logger
        
    async def process_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]],
        validate: bool = True
    ) -> pd.DataFrame:
        try:
            df = pd.DataFrame(candles)
            if validate:
                self._validate_candle_data(df)
            
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            df = self._calculate_indicators(df)
            
            await self._store_processed_candles(df)
            return df
            
        except Exception as e:
            self.logger.error(f"Candle processing failed: {str(e)}")
            raise ValidationError(f"Failed to process candles: {str(e)}")
    
    def _validate_candle_data(self, df: pd.DataFrame) -> None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
            
        if df.isnull().any().any():
            raise ValidationError("Found missing values in candle data")
            
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_prices.any():
            raise ValidationError(f"Found {invalid_prices.sum()} invalid price levels")
            
        if (df['volume'] <= 0).any():
            raise ValidationError("Found non-positive volumes")
            
        if not df['timestamp'].is_monotonic_increasing:
            raise ValidationError("Timestamps are not monotonically increasing")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        df['returns'] = df['close'].pct_change()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        return df
    
    async def _store_processed_candles(self, df: pd.DataFrame) -> None:
        candles = df.to_dict('records')
        await self.db.store_processed_candles(candles)

if __name__ == '__main__':
    main()
