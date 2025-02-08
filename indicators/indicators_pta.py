#!/usr/bin/env python3
"""
Module: indicators/indicators_pta.py
"""

import pandas as pd
import pandas_ta as ta

from utils.error_handler import handle_error

def compute_indicators(candles: list, ctx: object) -> pd.DataFrame:
    """
    Compute technical indicators for the provided candlestick data.
    If there are 7 columns, assumes the 7th is an existing ATR_14 to drop.
    
    Args:
        candles (list): Each element is [timestamp, open, high, low, close, volume, (optional)ATR_14].
        ctx (object): Context with config and logger (ctx.config["indicators"] can override default periods).
    
    Returns:
        pd.DataFrame: DataFrame with original columns plus new indicator columns.
                     If insufficient rows or error, returns an empty DataFrame.
    """
    try:
        # Drop extra ATR column if present.
        if candles and len(candles[0]) == 7:
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "ATR_14"])
            df.drop(columns=["ATR_14"], inplace=True)
        else:
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Retrieve indicator settings from config or use defaults.
        ind_cfg    = ctx.config.get("indicators", {})
        ema_periods = ind_cfg.get("ema_periods", [8, 21, 55, 233])
        rsi_period  = ind_cfg.get("rsi_period", 14)
        macd_slow   = ind_cfg.get("macd_slow", 26)
        macd_fast   = ind_cfg.get("macd_fast", 12)
        macd_signal = ind_cfg.get("macd_signal", 9)
        atr_period  = ind_cfg.get("atr_period", 14)
        bb_length   = ind_cfg.get("bb_length", 20)
        bb_std      = ind_cfg.get("bb_std", 2)

        # Check if enough rows exist to compute requested indicators.
        min_rows_needed = max(ema_periods + [rsi_period, macd_slow, atr_period, bb_length])
        if len(df) < min_rows_needed:
            ctx.logger.warning("Not enough rows in candle data to compute all requested indicators.")
            return pd.DataFrame()

        # Convert timestamp to datetime in UTC, sort by time.
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Compute EMAs.
        for period in ema_periods:
            df[f"EMA_{period}"] = ta.ema(df["close"], length=period)

        # Compute RSI.
        df[f"RSI_{rsi_period}"] = ta.rsi(df["close"], length=rsi_period)

        # Compute MACD.
        macd_df = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if macd_df is not None and not macd_df.empty:
            df["MACD"]  = macd_df[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
            df["MACDs"] = macd_df[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        else:
            df["MACD"]  = pd.Series([None]*len(df))
            df["MACDs"] = pd.Series([None]*len(df))
            ctx.logger.warning("MACD indicator could not be computed; partial data or other issue.")

        # Compute ATR.
        df[f"ATR_{atr_period}"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)

        # Compute Bollinger Bands.
        bb = ta.bbands(df["close"], length=bb_length, std=bb_std)
        if bb is not None and not bb.empty:
            bbl_key = f"BBL_{bb_length}_{float(bb_std)}"
            bbm_key = f"BBM_{bb_length}_{float(bb_std)}"
            bbu_key = f"BBU_{bb_length}_{float(bb_std)}"
            df["BBL"] = bb[bbl_key]
            df["BBM"] = bb[bbm_key]
            df["BBU"] = bb[bbu_key]
        else:
            df["BBL"] = pd.Series([None]*len(df))
            df["BBM"] = pd.Series([None]*len(df))
            df["BBU"] = pd.Series([None]*len(df))
            ctx.logger.warning("Bollinger Bands could not be computed; partial data or other issue.")

        # Optional check for missing columns after calculations.
        expected_cols = [
            "timestamp", "open", "high", "low", "close", "volume", "datetime",
            *(f"EMA_{p}" for p in ema_periods),
            f"RSI_{rsi_period}", "MACD", "MACDs", f"ATR_{atr_period}",
            "BBL", "BBM", "BBU"
        ]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            ctx.logger.warning(f"Some indicator columns are missing: {missing_cols}")

        return df

    except Exception as e:
        handle_error(e, context="Indicators.compute_indicators", logger=ctx.logger)
        return pd.DataFrame()
