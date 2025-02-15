#! /usr/bin/env python3
# src/indicators/indicators_pta.py
"""
Module: src.indicators
Provides technical indicator calculations
"""
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from utils.error_handler import handle_error


def compute_indicators(candles: list, ctx: object) -> pd.DataFrame:
    """
    Compute technical indicators for the provided candlestick data.

    Args:
        candles (list): Each element is [timestamp, open, high, low, close, volume, (optional)ATR_14].
        ctx (object): Context with config and logger.

    Returns:
        pd.DataFrame: DataFrame with original columns plus new indicator columns.
                     If insufficient rows or error, returns an empty DataFrame.
    """
    try:
        # Drop extra ATR column if present
        if candles and len(candles[0]) == 7:
            df = pd.DataFrame(
                candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ATR_14",
                ],
            )
            df.drop(columns=["ATR_14"], inplace=True)
        else:
            df = pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Get indicator settings from config
        ind_cfg = ctx.config.get("indicators", {})
        ema_periods = ind_cfg.get("ema_periods", [8, 21, 55, 233])
        rsi_period = ind_cfg.get("rsi_period", 14)
        macd_slow = ind_cfg.get("macd_slow", 26)
        macd_fast = ind_cfg.get("macd_fast", 12)
        macd_signal = ind_cfg.get("macd_signal", 9)
        atr_period = ind_cfg.get("atr_period", 14)
        bb_length = ind_cfg.get("bb_length", 20)
        bb_std = ind_cfg.get("bb_std", 2)
        stoch_k = ind_cfg.get("stoch_k", 14)
        stoch_d = ind_cfg.get("stoch_d", 3)
        stoch_smooth = ind_cfg.get("stoch_smooth", 3)

        # Check minimum data requirements
        min_rows_needed = max(
            ema_periods + [rsi_period, macd_slow, atr_period, bb_length, stoch_k]
        )
        if len(df) < min_rows_needed:
            ctx.logger.warning(
                f"Insufficient data for indicators. Need {min_rows_needed} rows, got {len(df)}"
            )
            return pd.DataFrame()

        # Convert timestamp and sort
        df["datetime"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True, errors="coerce"
        )
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Calculate EMAs
        for period in ema_periods:
            df[f"EMA_{period}"] = ta.ema(df["close"], length=period)

        # Calculate RSI
        df[f"RSI_{rsi_period}"] = ta.rsi(df["close"], length=rsi_period)

        # Calculate Stochastic Oscillator
        stoch = ta.stoch(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            k=stoch_k,
            d=stoch_d,
            smooth_k=stoch_smooth,
        )
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch[f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth}"]
            df["STOCH_D"] = stoch[f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth}"]
        else:
            df["STOCH_K"] = pd.Series([None] * len(df))
            df["STOCH_D"] = pd.Series([None] * len(df))
            ctx.logger.warning("Stochastic oscillator could not be computed")

        # Calculate MACD
        macd_df = ta.macd(
            df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        if macd_df is not None and not macd_df.empty:
            df["MACD"] = macd_df[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
            df["MACDs"] = macd_df[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        else:
            df["MACD"] = pd.Series([None] * len(df))
            df["MACDs"] = pd.Series([None] * len(df))
            ctx.logger.warning("MACD could not be computed")

        # Calculate ATR
        df[f"ATR_{atr_period}"] = ta.atr(
            df["high"], df["low"], df["close"], length=atr_period
        )

        # Calculate Bollinger Bands
        bb = ta.bbands(df["close"], length=bb_length, std=bb_std)
        if bb is not None and not bb.empty:
            df["BBL"] = bb[f"BBL_{bb_length}_{float(bb_std)}"]
            df["BBM"] = bb[f"BBM_{bb_length}_{float(bb_std)}"]
            df["BBU"] = bb[f"BBU_{bb_length}_{float(bb_std)}"]
        else:
            df["BBL"] = pd.Series([None] * len(df))
            df["BBM"] = pd.Series([None] * len(df))
            df["BBU"] = pd.Series([None] * len(df))
            ctx.logger.warning("Bollinger Bands could not be computed")

        # Validate output
        expected_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "datetime",
            *(f"EMA_{p}" for p in ema_periods),
            f"RSI_{rsi_period}",
            "MACD",
            "MACDs",
            f"ATR_{atr_period}",
            "BBL",
            "BBM",
            "BBU",
            "STOCH_K",
            "STOCH_D",
        ]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            ctx.logger.warning(f"Missing indicator columns: {missing_cols}")

        return df

    except Exception as e:
        handle_error(e, context="Indicators.compute_indicators", logger=ctx.logger)
        return pd.DataFrame()
