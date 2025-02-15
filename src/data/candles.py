#! /usr/bin/env python3
#src/data/candles.py
"""
Tool: data/candles.py
Database management for OHLCV candle data with proper error handling
"""
import os
import sys
import time
import sqlite3
import ccxt.async_support as ccxt
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
import asyncio
import logging

from utils.logger import setup_logging
from utils.error_handler import handle_error, ValidationError
from database.queries import DatabaseQueries
from database.database import DatabaseConnection

# Initialize logger at module level
logger = setup_logging(name="CandleManager", level="INFO")

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

async def fetch_and_save_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    cursor: sqlite3.Cursor,
    limit: int = 1000,
    period: int = 14
) -> int:
    """Fetch OHLCV data and save to the database"""
    try:
        candles = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        total = 0
        for candle in candles:
            timestamp, open_, high, low, close, volume = candle
            datetime_str = datetime.utcfromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
            atr_14 = calculate_atr(candles, period=period)
            cursor.execute('''
                INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume, datetime, atr_14, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timeframe,
                timestamp,
                open_,
                high,
                low,
                close,
                volume,
                datetime_str,
                atr_14,
                exchange.name
            ))
            total += 1
        return total

    except ccxt.NetworkError as e:
        logger.error(f"Network error while fetching candles for {symbol}: {e}")
        return 0
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while fetching candles for {symbol}: {e}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error while fetching candles for {symbol}: {e}")
        return 0

def calculate_atr(candles: List[List[Any]], period: int = 14) -> Optional[float]:
    """Calculate the Average True Range (ATR) for the given candles"""
    try:
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close_prev'] = df['close'].shift(1)
        df['tr'] = df[['high', 'low', 'close_prev']].apply(
            lambda row: max(row['high'] - row['low'], 
                           abs(row['high'] - row['close_prev']) if pd.notnull(row['close_prev']) else 0,
                           abs(row['low'] - row['close_prev']) if pd.notnull(row['close_prev']) else 0),
            axis=1
        )
        atr = df['tr'].rolling(window=period).mean().iloc[-1]
        return atr if not np.isnan(atr) else None
    except Exception as e:
        logger.error(f"Failed to calculate ATR: {e}")
        return None

class CandleProcessor:
    def __init__(self, db_queries: DatabaseQueries, logger: logging.Logger):
        self.db_queries = db_queries
        self.logger = logger

    async def process_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: List[List[Any]]
    ) -> pd.DataFrame:
        """Process raw candle data into a DataFrame"""
        try:
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Failed to process candles for {symbol}: {e}")
            raise ValidationError(f"Failed to process candles for {symbol}: {e}") from e

class CandleManager:
    def __init__(self, db_connection: DatabaseConnection, logger: logging.Logger):
        self.db = db_connection
        self.logger = logger
        self.processor = CandleProcessor(self.db, self.logger)

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch and process candles from database"""
        try:
            raw_candles = await self.db.fetch_candles(symbol, timeframe, limit)
            if raw_candles:
                return await self.processor.process_candles(symbol, timeframe, raw_candles)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return pd.DataFrame()

async def main():
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

        db_connection = DatabaseQueries(db_path, logger=logger)

        async with db_connection.db_connection.get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute('''
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
            await conn.commit()

            for symbol in symbols:
                for tf in timeframes:
                    logger.info(f"\nFetching candles for {symbol} at {tf} from {since}")
                    total = await fetch_and_save_candles(
                        exchange, symbol, tf, since, cursor, limit=1000, period=14
                    )
                    await conn.commit()
                    logger.info(f"Total candles saved for {symbol} at {tf}: {total}")

        logger.info("All candles saved successfully.")

    except Exception as e:
        handle_error(e, "main", logger=logger)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
