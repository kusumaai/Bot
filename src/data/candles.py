#! /usr/bin/env python3
# src/data/candles.py
"""
Tool: data/candles.py
Database management for OHLCV candle data with proper error handling
"""
import asyncio
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from database.database import DatabaseConnection
from database.queries import DatabaseQueries
from utils.error_handler import ValidationError, handle_error
from utils.logger import setup_logging
from utils.numeric_handler import NumericHandler

# Initialize logger at module level
logger = setup_logging(name="CandleManager", log_level="INFO")


def get_stable_coin_markets(
    exchange: ccxt.Exchange, base_coins: List[str], stable_coins: List[str]
) -> List[str]:
    """Get valid trading pairs for specified base and stable coins"""
    markets = exchange.fetch_markets()
    selected = []
    for market in markets:
        symbol = market.get("symbol")
        if symbol and "/" in symbol:
            base, quote = symbol.split("/")
            if base in base_coins and quote in stable_coins:
                selected.append(symbol)
    return selected


@dataclass
class CandleData:
    """Validated candle data structure"""

    timestamp: int
    datetime: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    atr_14: Optional[Decimal] = None

    @classmethod
    def validate_and_create(
        cls, raw_candle: List[Any], period: int = 14
    ) -> "CandleData":
        """
        Validate and create a CandleData instance from raw data.
        Raises ValidationError if data is invalid.
        """
        if len(raw_candle) < 6:
            raise ValidationError("Candle data missing required fields")

        timestamp, open_, high, low, close, volume = raw_candle

        # Validate timestamp
        if not isinstance(timestamp, (int, float)) or timestamp <= 0:
            raise ValidationError(f"Invalid timestamp: {timestamp}")

        # Convert and validate all numeric values
        try:
            open_dec = Decimal(str(open_)).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
            high_dec = Decimal(str(high)).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
            low_dec = Decimal(str(low)).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
            close_dec = Decimal(str(close)).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
            volume_dec = Decimal(str(volume)).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, TypeError, ValueError) as e:
            raise ValidationError(f"Invalid numeric value in candle data: {e}")

        # Validate price relationships
        if not (low_dec <= open_dec <= high_dec and low_dec <= close_dec <= high_dec):
            raise ValidationError(
                f"Invalid price relationships: low={low_dec}, high={high_dec}, open={open_dec}, close={close_dec}"
            )

        # Validate volume
        if volume_dec < 0:
            raise ValidationError(f"Invalid negative volume: {volume_dec}")

        # Create datetime string
        datetime_str = datetime.utcfromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return cls(
            timestamp=timestamp,
            datetime=datetime_str,
            open=open_dec,
            high=high_dec,
            low=low_dec,
            close=close_dec,
            volume=volume_dec,
        )


class CandleBatch:
    """Manages a batch of candles with validation and error tracking"""

    def __init__(self, symbol: str, timeframe: str, exchange_name: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_name = exchange_name
        self.valid_candles: List[CandleData] = []
        self.errors: List[Tuple[int, str]] = []  # (index, error_message)
        self.total_processed = 0

    def add_candle(self, index: int, raw_candle: List[Any], period: int = 14) -> None:
        """Add a candle to the batch with validation"""
        self.total_processed += 1
        try:
            candle = CandleData.validate_and_create(raw_candle, period)
            self.valid_candles.append(candle)
        except ValidationError as e:
            self.errors.append((index, str(e)))

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return len(self.valid_candles) / self.total_processed


async def fetch_and_save_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    cursor: sqlite3.Cursor,
    limit: int = 1000,
    period: int = 14,
    error_threshold: float = 0.05,  # 5% error threshold
) -> Tuple[int, List[str]]:
    """
    Fetch OHLCV data and save to the database with strict validation and error handling.
    Returns tuple of (number of successful saves, list of error messages).
    """
    try:
        # Fetch candles
        raw_candles = await exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit
        )

        # Process candles in a batch
        batch = CandleBatch(symbol, timeframe, exchange.name)
        for i, candle in enumerate(raw_candles):
            batch.add_candle(i, candle, period)

        # Check error threshold
        if batch.has_errors:
            error_rate = 1 - batch.success_rate
            error_messages = [
                f"Error at index {idx}: {msg}" for idx, msg in batch.errors
            ]

            if error_rate > error_threshold:
                logger.error(
                    f"Error rate {error_rate:.2%} exceeds threshold {error_threshold:.2%} "
                    f"for {symbol} {timeframe}. Errors: {error_messages}"
                )
                return 0, error_messages
            else:
                logger.warning(
                    f"Non-critical errors occurred while processing {symbol} {timeframe}. "
                    f"Error rate: {error_rate:.2%}. Errors: {error_messages}"
                )

        # Begin transaction
        await cursor.execute("BEGIN TRANSACTION")

        try:
            # Save valid candles
            for candle in batch.valid_candles:
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO candles (
                        symbol, timeframe, timestamp, open, high, low, close, 
                        volume, datetime, atr_14, exchange
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        timeframe,
                        candle.timestamp,
                        str(candle.open),
                        str(candle.high),
                        str(candle.low),
                        str(candle.close),
                        str(candle.volume),
                        candle.datetime,
                        str(candle.atr_14) if candle.atr_14 else None,
                        exchange.name,
                    ),
                )

            # Commit transaction
            await cursor.execute("COMMIT")
            logger.info(
                f"Successfully saved {len(batch.valid_candles)} candles for {symbol} {timeframe}"
            )
            return len(batch.valid_candles), batch.errors

        except Exception as e:
            # Rollback on any error
            await cursor.execute("ROLLBACK")
            logger.error(f"Transaction failed, rolling back: {e}")
            return 0, [str(e)]

    except ccxt.NetworkError as e:
        logger.error(f"Network error while fetching candles for {symbol}: {e}")
        return 0, [str(e)]
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while fetching candles for {symbol}: {e}")
        return 0, [str(e)]
    except Exception as e:
        logger.error(f"Unexpected error while processing candles for {symbol}: {e}")
        return 0, [str(e)]


def calculate_atr(candles: List[List[Any]], period: int = 14) -> Optional[Decimal]:
    """
    Calculate Average True Range with strict data validation.

    Args:
        candles: List of OHLCV candles
        period: ATR period (default 14)

    Returns:
        Decimal ATR value or None if calculation fails validation

    Raises:
        ValidationError: If data fails validation checks
    """
    try:
        # Validate input data
        if not candles or len(candles) < period + 1:
            logger.error(
                f"Insufficient candles for ATR calculation. Need at least {period + 1}, got {len(candles)}"
            )
            return None

        # Validate period
        if period < 1:
            raise ValidationError(f"Invalid ATR period: {period}")

        tr_values = []
        invalid_candles = 0
        max_invalid_threshold = min(
            int(period * 0.1), 2
        )  # Max 10% invalid or 2 candles

        # Calculate True Range values with validation
        for i in range(1, len(candles)):
            try:
                current = candles[i]
                previous = candles[i - 1]

                # Validate candle structure
                if len(current) < 5 or len(previous) < 5:
                    raise ValidationError(f"Invalid candle structure at index {i}")

                # Convert and validate prices
                high = Decimal(str(current[2]))
                low = Decimal(str(current[3]))
                close = Decimal(str(current[4]))
                prev_close = Decimal(str(previous[4]))

                # Price sanity checks
                if not all(x > 0 for x in [high, low, close, prev_close]):
                    raise ValidationError(f"Non-positive price values at index {i}")

                if low > high:
                    raise ValidationError(f"Low price exceeds high price at index {i}")

                if close < low or close > high:
                    raise ValidationError(f"Close price outside HL range at index {i}")

                # Calculate TR with proper decimal handling
                tr0 = abs(high - low)
                tr1 = abs(high - prev_close)
                tr2 = abs(low - prev_close)

                tr = max(tr0, tr1, tr2)

                # Validate TR value
                if tr <= 0:
                    raise ValidationError(f"Invalid TR value at index {i}: {tr}")

                tr_values.append(tr)

            except (InvalidOperation, TypeError, ValueError, ValidationError) as e:
                logger.warning(f"Candle validation failed at index {i}: {str(e)}")
                invalid_candles += 1

                # Check if too many invalid candles
                if invalid_candles > max_invalid_threshold:
                    logger.error(
                        "Too many invalid candles for reliable ATR calculation",
                        extra={
                            "invalid_count": invalid_candles,
                            "threshold": max_invalid_threshold,
                            "period": period,
                        },
                    )
                    return None

                # Skip this candle instead of using 0
                continue

        # Ensure we have enough valid TR values
        if len(tr_values) < period:
            logger.error(
                f"Insufficient valid TR values for ATR calculation",
                extra={"required": period, "available": len(tr_values)},
            )
            return None

        # Calculate ATR using Wilder's smoothing
        atr = Decimal(sum(tr_values[:period])) / Decimal(period)

        # Validate final ATR
        if atr <= 0:
            logger.error("ATR calculation resulted in non-positive value")
            return None

        # Log calculation metadata
        logger.debug(
            "ATR calculation completed",
            extra={
                "period": period,
                "valid_values": len(tr_values),
                "invalid_candles": invalid_candles,
                "atr_value": str(atr),
            },
        )

        return atr.quantize(Decimal("0.00000001"))

    except Exception as e:
        logger.error(f"ATR calculation failed: {str(e)}", exc_info=True)
        return None


def validate_candle_data(candle: List[Any], index: int) -> ValidationResult:
    """
    Validate a single candle's data integrity.

    Args:
        candle: Single OHLCV candle data
        index: Candle index for error reporting

    Returns:
        ValidationResult with validation status and any error messages
    """
    try:
        if len(candle) < 6:
            return ValidationResult(False, f"Insufficient candle data at index {index}")

        timestamp, open_price, high, low, close, volume = candle[:6]

        # Validate timestamp
        if not isinstance(timestamp, (int, float)) or timestamp <= 0:
            return ValidationResult(False, f"Invalid timestamp at index {index}")

        # Convert and validate prices
        try:
            prices = [
                Decimal(str(x)).quantize(Decimal("0.00000001"))
                for x in [open_price, high, low, close]
            ]
        except (InvalidOperation, TypeError):
            return ValidationResult(False, f"Invalid price values at index {index}")

        # Price relationship validation
        if not (
            prices[2] <= prices[0] <= prices[1]  # low <= open <= high
            and prices[2] <= prices[3] <= prices[1]
        ):  # low <= close <= high
            return ValidationResult(
                False, f"Invalid price relationships at index {index}"
            )

        # Volume validation
        try:
            volume = Decimal(str(volume))
            if volume < 0:
                return ValidationResult(False, f"Negative volume at index {index}")
        except (InvalidOperation, TypeError):
            return ValidationResult(False, f"Invalid volume at index {index}")

        return ValidationResult(True)

    except Exception as e:
        return ValidationResult(
            False, f"Candle validation failed at index {index}: {str(e)}"
        )


def calculate_tr(
    current_candle: List[Any], previous_candle: List[Any], index: int
) -> Optional[Decimal]:
    """
    Calculate True Range with validation.

    Args:
        current_candle: Current period candle data
        previous_candle: Previous period candle data
        index: Candle index for error reporting

    Returns:
        Decimal TR value or None if calculation fails validation
    """
    try:
        # Validate both candles
        current_valid = validate_candle_data(current_candle, index)
        previous_valid = validate_candle_data(previous_candle, index - 1)

        if not (current_valid.is_valid and previous_valid.is_valid):
            logger.warning(
                f"Candle validation failed for TR calculation at index {index}",
                extra={
                    "current_error": current_valid.error_message,
                    "previous_error": previous_valid.error_message,
                },
            )
            return None

        # Calculate TR with validated data
        high = Decimal(str(current_candle[2]))
        low = Decimal(str(current_candle[3]))
        prev_close = Decimal(str(previous_candle[4]))

        tr = max(abs(high - low), abs(high - prev_close), abs(low - prev_close))

        return tr.quantize(Decimal("0.00000001"))

    except Exception as e:
        logger.error(f"TR calculation failed at index {index}: {str(e)}")
        return None


class CandleProcessor:
    """Process and validate candle data with error recovery"""

    def __init__(self, db_connection: DatabaseConnection, logger: logging.Logger):
        self.db = db_connection
        self.logger = logger
        self._lock = asyncio.Lock()
        self.nh = NumericHandler()
        self.error_threshold = 10
        self.retry_delay = 1.0
        self.max_retries = 3
        self._error_count = 0
        self._last_error = None
        self._processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "last_success": None,
            "error_rate": 0.0,
        }

    async def process_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: List[List[Any]],
        retry_count: int = 0,
    ) -> pd.DataFrame:
        """Process raw candle data into a DataFrame with error recovery"""
        try:
            if not candles:
                self.logger.warning(f"No candles provided for {symbol}")
                return pd.DataFrame()

            async with self._lock:
                # Create DataFrame with proper column names
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                # Convert timestamp to datetime
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("datetime", inplace=True)

                # Validate data before conversion
                if not await self._validate_raw_data(df, symbol):
                    self._handle_processing_error(
                        ValidationError(f"Invalid raw candle data for {symbol}"), symbol
                    )
                    if retry_count < self.max_retries:
                        self.logger.warning(
                            f"Retrying candle processing for {symbol} (attempt {retry_count + 1})"
                        )
                        await asyncio.sleep(self.retry_delay)
                        return await self.process_candles(
                            symbol, timeframe, candles, retry_count + 1
                        )
                    raise ValidationError(f"Invalid raw candle data for {symbol}")

                # Convert price and volume columns to Decimal with validation
                numeric_columns = ["open", "high", "low", "close", "volume"]
                for col in numeric_columns:
                    try:
                        df[col] = df[col].apply(self._safe_decimal_convert)
                    except Exception as e:
                        self.logger.error(f"Error converting {col} to Decimal: {e}")
                        if retry_count < self.max_retries:
                            await asyncio.sleep(self.retry_delay)
                            return await self.process_candles(
                                symbol, timeframe, candles, retry_count + 1
                            )
                        raise ValidationError(
                            f"Failed to convert {col} values to Decimal"
                        ) from e

                # Validate converted data
                if not await self._validate_candle_data(df, symbol):
                    self._handle_processing_error(
                        ValidationError(f"Invalid candle data for {symbol}"), symbol
                    )
                    if retry_count < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        return await self.process_candles(
                            symbol, timeframe, candles, retry_count + 1
                        )
                    raise ValidationError(f"Invalid candle data for {symbol}")

                # Calculate additional metrics
                df = await self._calculate_metrics(df)

                # Update processing stats
                self._update_processing_stats(len(df), 0)

                return df

        except Exception as e:
            self._handle_processing_error(e, symbol)
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self.process_candles(
                    symbol, timeframe, candles, retry_count + 1
                )
            raise ValidationError(f"Failed to process candles for {symbol}: {e}") from e

    async def _validate_raw_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate raw data before conversion"""
        try:
            # Check for missing values
            if df.isnull().any().any():
                missing_info = df.isnull().sum()
                self.logger.error(
                    f"Found missing values in {symbol} raw data: {missing_info}"
                )
                return False

            # Check timestamp sequence
            if not df.index.is_monotonic_increasing:
                self.logger.error(f"Timestamps not in sequence for {symbol}")
                return False

            # Check for string values that can't be converted to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                try:
                    pd.to_numeric(df[col], errors="raise")
                except Exception as e:
                    self.logger.error(
                        f"Non-numeric values found in {symbol} {col} column: {e}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating raw data for {symbol}: {e}")
            return False

    def _safe_decimal_convert(self, value: Any) -> Decimal:
        """Safely convert value to Decimal with proper error handling"""
        try:
            if pd.isna(value):
                return Decimal("0")

            # Handle different input types
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                # Remove any whitespace and validate
                value = value.strip()
                if not value:
                    return Decimal("0")
                return Decimal(value)
            elif isinstance(value, Decimal):
                return value
            else:
                self.logger.warning(f"Unexpected type for value {value}: {type(value)}")
                return Decimal(str(float(value)))

        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"Error converting value {value} to Decimal: {e}")
            return Decimal("0")
        except Exception as e:
            self.logger.error(f"Unexpected error converting {value} to Decimal: {e}")
            return Decimal("0")

    async def _validate_candle_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate candle data integrity with detailed checks"""
        try:
            # Verify positive values
            for col in ["open", "high", "low", "close", "volume"]:
                if (df[col] <= 0).any():
                    negative_rows = df[df[col] <= 0].index.tolist()
                    self.logger.error(
                        f"Found non-positive {col} values in {symbol} at: {negative_rows}"
                    )
                    return False

            # Verify price relationships
            invalid_prices = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )
            if invalid_prices.any():
                invalid_rows = df[invalid_prices].index.tolist()
                self.logger.error(
                    f"Found invalid price relationships in {symbol} at: {invalid_rows}"
                )
                return False

            # Check for extreme values
            for col in ["open", "high", "low", "close"]:
                mean = df[col].astype(float).mean()
                std = df[col].astype(float).std()
                extreme_values = df[abs(df[col].astype(float) - mean) > 3 * std]
                if not extreme_values.empty:
                    self.logger.warning(
                        f"Found extreme {col} values in {symbol} at: {extreme_values.index.tolist()}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Error validating {symbol} candle data: {e}")
            return False

    async def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional metrics with error handling"""
        try:
            # Calculate returns
            df["returns"] = df["close"].astype(float).pct_change()

            # Calculate true range
            df["tr"] = df.apply(lambda x: self._safe_tr_calculation(x), axis=1)

            # Calculate ATR with error handling
            try:
                df["atr_14"] = (
                    df["tr"]
                    .rolling(window=14, min_periods=1)
                    .mean()
                    .apply(lambda x: self.nh.to_decimal(str(x)))
                )
            except Exception as e:
                self.logger.error(f"Error calculating ATR: {e}")
                df["atr_14"] = pd.Series([Decimal("0")] * len(df))

            # Calculate volume metrics
            try:
                df["volume_ma"] = (
                    df["volume"]
                    .astype(float)
                    .rolling(window=20, min_periods=1)
                    .mean()
                    .apply(lambda x: self.nh.to_decimal(str(x)))
                )
                df["volume_std"] = (
                    df["volume"]
                    .astype(float)
                    .rolling(window=20, min_periods=1)
                    .std()
                    .apply(lambda x: self.nh.to_decimal(str(x)))
                )
            except Exception as e:
                self.logger.error(f"Error calculating volume metrics: {e}")
                df["volume_ma"] = pd.Series([Decimal("0")] * len(df))
                df["volume_std"] = pd.Series([Decimal("0")] * len(df))

            return df

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return df

    def _safe_tr_calculation(self, row: pd.Series) -> Decimal:
        """Safely calculate True Range"""
        try:
            high = self.nh.to_decimal(str(row["high"]))
            low = self.nh.to_decimal(str(row["low"]))
            prev_close = self.nh.to_decimal(
                str(row["close"].shift(1))
                if not pd.isna(row["close"].shift(1))
                else "0"
            )

            if prev_close == 0:
                return high - low

            tr0 = high - low
            tr1 = abs(high - prev_close)
            tr2 = abs(low - prev_close)

            return max(tr0, tr1, tr2)
        except Exception as e:
            self.logger.error(f"Error calculating TR: {e}")
            return Decimal("0")

    def _handle_processing_error(self, error: Exception, symbol: str):
        """Handle processing errors and update stats"""
        self._error_count += 1
        self._last_error = error
        self._update_processing_stats(0, 1)

        if self._error_count >= self.error_threshold:
            self.logger.error(
                f"Error threshold exceeded for {symbol}: {self._error_count} errors"
            )
            raise ValidationError(f"Too many errors processing {symbol}: {error}")

    def _update_processing_stats(self, processed: int, errors: int):
        """Update processing statistics"""
        self._processing_stats["total_processed"] += processed
        self._processing_stats["total_errors"] += errors
        if processed > 0:
            self._processing_stats["last_success"] = datetime.now()

        total = (
            self._processing_stats["total_processed"]
            + self._processing_stats["total_errors"]
        )
        self._processing_stats["error_rate"] = (
            self._processing_stats["total_errors"] / total if total > 0 else 0.0
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self._processing_stats.copy()

    async def reset_error_count(self):
        """Reset error counter and stats"""
        async with self._lock:
            self._error_count = 0
            self._last_error = None
            self._processing_stats = {
                "total_processed": 0,
                "total_errors": 0,
                "last_success": None,
                "error_rate": 0.0,
            }


class CandleManager:
    """Manage candle data operations"""

    def __init__(self, db_connection: DatabaseConnection, logger: logging.Logger):
        self.db = db_connection
        self.logger = logger
        self.processor = CandleProcessor(self.db, self.logger)
        self._lock = asyncio.Lock()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes

    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch and process candles with caching"""
        cache_key = f"{symbol}_{timeframe}_{limit}"

        try:
            # Check cache first
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cached_data

            # Fetch from database
            raw_candles = await self.db.fetch_candles(symbol, timeframe, limit)
            if not raw_candles:
                self.logger.warning(f"No candles found for {symbol} {timeframe}")
                return pd.DataFrame()

            # Process candles
            df = await self.processor.process_candles(symbol, timeframe, raw_candles)

            # Update cache
            self.cache[cache_key] = (time.time(), df)

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return pd.DataFrame()

    async def update_candles(
        self, symbol: str, timeframe: str, new_candles: List[List[Any]]
    ) -> bool:
        """Update candle data in database"""
        try:
            async with self._lock:
                # Process new candles
                df = await self.processor.process_candles(
                    symbol, timeframe, new_candles
                )
                if df.empty:
                    return False

                # Store in database
                success = await self.db.store_candles(symbol, timeframe, df)
                if success:
                    # Clear cache
                    cache_keys = [
                        k
                        for k in self.cache.keys()
                        if k.startswith(f"{symbol}_{timeframe}")
                    ]
                    for key in cache_keys:
                        self.cache.pop(key, None)

                return success

        except Exception as e:
            self.logger.error(f"Failed to update candles for {symbol}: {e}")
            return False

    def clear_cache(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ):
        """Clear candle cache"""
        if symbol and timeframe:
            # Clear specific symbol/timeframe
            cache_keys = [
                k for k in self.cache.keys() if k.startswith(f"{symbol}_{timeframe}")
            ]
        elif symbol:
            # Clear all timeframes for symbol
            cache_keys = [k for k in self.cache.keys() if k.startswith(f"{symbol}_")]
        else:
            # Clear all cache
            cache_keys = list(self.cache.keys())

        for key in cache_keys:
            self.cache.pop(key, None)


async def main():
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        base_coins = ["BTC", "ETH"]
        stable_coins = ["USDT", "USDC", "BUSD", "USDP", "DAI"]
        timeframes = ["15m", "1h", "4h", "1d", "1w"]

        symbols = get_stable_coin_markets(exchange, base_coins, stable_coins)
        logger.info(f"Selected markets: {symbols}")

        since = exchange.parse8601("2020-01-01T00:00:00Z")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "candles.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        db_connection = DatabaseQueries(db_path, logger=logger)

        async with db_connection.db_connection.get_connection() as conn:
            cursor = await conn.cursor()

            # Drop existing table if exists to ensure clean schema
            await cursor.execute("DROP TABLE IF EXISTS candles")

            # Create table with proper DECIMAL types and constraints
            await cursor.execute(
                """
                CREATE TABLE candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open TEXT NOT NULL,  -- Stored as text to preserve Decimal precision
                    high TEXT NOT NULL,
                    low TEXT NOT NULL,
                    close TEXT NOT NULL,
                    volume TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    atr_14 TEXT,
                    returns TEXT,
                    tr TEXT,
                    volume_ma TEXT,
                    volume_std TEXT,
                    exchange TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """
            )

            # Create indices for performance
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_lookup 
                ON candles(symbol, timeframe, timestamp DESC)
            """
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_time 
                ON candles(timestamp DESC)
            """
            )

            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_symbol 
                ON candles(symbol)
            """
            )

            # Create trigger to update updated_at
            await cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_candles_timestamp 
                AFTER UPDATE ON candles
                BEGIN
                    UPDATE candles SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """
            )

            await conn.commit()

            for symbol in symbols:
                for tf in timeframes:
                    logger.info(f"\nFetching candles for {symbol} at {tf} from {since}")
                    total, errors = await fetch_and_save_candles(
                        exchange, symbol, tf, since, cursor, limit=1000, period=14
                    )
                    await conn.commit()
                    logger.info(f"Total candles saved for {symbol} at {tf}: {total}")
                    if errors:
                        logger.error(f"Errors occurred: {', '.join(errors)}")

        logger.info("All candles saved successfully.")

    except Exception as e:
        handle_error(e, "main", logger=logger)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
