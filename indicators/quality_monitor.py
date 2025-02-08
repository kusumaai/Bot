#!/usr/bin/env python3
"""
Module: indicators/quality_monitor.py
"""

import pandas as pd
import logging
from utils.error_handler import handle_error


def quality_check(df: pd.DataFrame, ctx=None) -> dict:
    """
    Evaluate the quality of the provided DataFrame.
    Now with smarter handling of expected missing values for long-period indicators.
    
    Args:
        df (pd.DataFrame): DataFrame containing market or indicator data
        ctx (object, optional): Global context that may contain configuration and logger
        
    Returns:
        dict: Column names to missing value ratios
    """
    # Determine which logger to use
    logger = ctx.logger if (ctx and hasattr(ctx, "logger")) else logging.getLogger(__name__)

    # Get threshold from config or use default
    missing_threshold = 0.3  # Increased default threshold
    if ctx and hasattr(ctx, "config"):
        missing_threshold = ctx.config.get("missing_threshold", 0.3)

    try:
        report = {}
        
        # Define columns that are allowed to have higher missing ratios
        long_period_indicators = ['EMA_89', 'EMA_144', 'EMA_233']
        
        for col in df.columns:
            total = len(df[col])
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / total if total > 0 else 0
            report[col] = missing_ratio

            # Only warn if:
            # 1. Not a long-period indicator and exceeds normal threshold
            # 2. Long-period indicator and exceeds higher threshold (0.5)
            if col in long_period_indicators:
                if missing_ratio > 0.5:  # Higher threshold for long-period indicators
                    logger.debug(f"Long-period indicator {col} missing ratio: {missing_ratio:.2f}")
            else:
                if missing_ratio > missing_threshold:
                    logger.debug(f"Column '{col}' missing ratio {missing_ratio:.2f}")

        return report

    except Exception as e:
        handle_error(e, context="quality_check", logger=logger)
        return {}
    """
    Evaluate the quality of the provided DataFrame by computing the missing value ratio for each column.
    If a global missing_threshold is specified in ctx.config, columns exceeding that threshold
    will be logged at warning level.

    Args:
        df (pd.DataFrame): DataFrame containing market or indicator data.
        ctx (object, optional): Global context that may contain configuration and a logger.
                                If None, falls back to a default logger.

    Returns:
        dict: A dictionary where keys are column names and values are the fraction
              of missing entries (0 to 1).
    """
    # Determine which logger to use
    logger = ctx.logger if (ctx and hasattr(ctx, "logger")) else logging.getLogger(__name__)

    # Determine missing_threshold from config or use default 1.0 (meaning no warnings unless 100% missing)
    missing_threshold = 1.0
    if ctx and hasattr(ctx, "config"):
        missing_threshold = ctx.config.get("missing_threshold", 1.0)

    try:
        report = {}
        for col in df.columns:
            total = len(df[col])
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / total if total > 0 else 0
            report[col] = missing_ratio

            # Log a warning if the ratio exceeds the threshold
            if missing_ratio > missing_threshold and missing_threshold < 1.0:
                logger.warning(
                    f"Column '{col}' missing ratio {missing_ratio:.2f} exceeds configured threshold {missing_threshold}."
                )

        return report

    except Exception as e:
        handle_error(e, context="quality_check", logger=logger)
        # Reraise to make debugging easier if needed
        raise
