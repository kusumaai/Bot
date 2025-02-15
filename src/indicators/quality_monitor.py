#! /usr/bin/env python3
#src/indicators/quality_monitor.py
"""
Module: src.indicators
Provides data quality monitoring.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from utils.error_handler import handle_error
#quality check class that evaluates the quality of the provided DataFrame with smart handling of indicators
def quality_check(df: pd.DataFrame, ctx: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of the provided DataFrame with smart handling of indicators.
    
    Args:
        df (pd.DataFrame): DataFrame containing market or indicator data
        ctx (object, optional): Global context with configuration and logger
        
    Returns:
        dict: Quality report containing:
            - missing_ratios: Dict of column names to missing value ratios
            - warnings: List of warning messages
            - long_period_stats: Stats for long-period indicators
    """
    logger = ctx.logger if (ctx and hasattr(ctx, "logger")) else logging.getLogger(__name__)

    # Get thresholds from config or use defaults
    config = getattr(ctx, "config", {}) if ctx else {}
    thresholds = {
        "normal": config.get("missing_threshold", 0.3),
        "long_period": config.get("long_period_threshold", 0.5),
        "critical": config.get("critical_threshold", 0.8)
    }

    try:
        report = {
            "missing_ratios": {},
            "warnings": [],
            "long_period_stats": {}
        }

        # Define special indicator groups
        long_period_indicators = [
            'EMA_89', 'EMA_144', 'EMA_233',
            'SMA_200', 'SMA_100'
        ]
        critical_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'timestamp', 'datetime'
        ]

        # Check each column
        for col in df.columns:
            total = len(df[col])
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / total if total > 0 else 0
            report["missing_ratios"][col] = missing_ratio

            # Determine appropriate threshold
            if col in critical_columns:
                threshold = thresholds["critical"]
                if missing_ratio > threshold:
                    msg = f"Critical column '{col}' missing ratio {missing_ratio:.2f} exceeds {threshold}"
                    report["warnings"].append(msg)
                    logger.warning(msg)

            elif col in long_period_indicators:
                threshold = thresholds["long_period"]
                if missing_ratio > threshold:
                    msg = f"Long-period indicator '{col}' missing ratio {missing_ratio:.2f}"
                    logger.debug(msg)
                    report["long_period_stats"][col] = missing_ratio

            else:
                threshold = thresholds["normal"]
                if missing_ratio > threshold:
                    msg = f"Column '{col}' missing ratio {missing_ratio:.2f} exceeds {threshold}"
                    logger.debug(msg)

        # Additional quality checks
        if not df.empty:
            # Check for timestamp continuity
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                gaps = timestamps.diff().dropna()
                expected_interval = pd.Timedelta(minutes=15)  # Assuming 15m timeframe
                large_gaps = gaps[gaps > expected_interval * 2]
                
                if not large_gaps.empty:
                    msg = f"Found {len(large_gaps)} timestamp gaps larger than {expected_interval * 2}"
                    report["warnings"].append(msg)
                    logger.warning(msg)

            # Check for price anomalies
            if all(col in df.columns for col in ['high', 'low', 'close']):
                invalid_prices = (
                    (df['high'] < df['low']) | 
                    (df['close'] > df['high']) | 
                    (df['close'] < df['low'])
                ).sum()
                
                if invalid_prices > 0:
                    msg = f"Found {invalid_prices} rows with invalid price relationships"
                    report["warnings"].append(msg)
                    logger.warning(msg)

        return report

    except Exception as e:
        handle_error(e, context="quality_check", logger=logger)
        return {
            "missing_ratios": {},
            "warnings": [f"Error in quality check: {str(e)}"],
            "long_period_stats": {}
        }
