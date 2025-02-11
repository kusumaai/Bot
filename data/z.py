#!/usr/bin/env python3
"""
Module: data/z.py
Data sanity checks and cleaning for candle data
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from decimal import Decimal
import logging
from datetime import datetime

from utils.logger import setup_logger
from utils.error_handler import handle_error

# Initialize logger at module level
logger = setup_logger(name="DataSanity", level="INFO")

def load_data(db_path: str) -> pd.DataFrame:
    """Load candle data from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT * FROM candles 
            WHERE datetime >= date('now', '-30 days')
        """, conn)
        conn.close()
        return df
    except Exception as e:
        handle_error(e, "load_data", logger=logger)
        return pd.DataFrame()

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate candle entries"""
    before = len(df)
    df_clean = df.drop_duplicates(subset=['symbol', 'timeframe', 'timestamp'])
    removed = before - len(df_clean)
    logger.info(f"Removed {removed} duplicate entries")
    return df_clean, removed

def drop_missing_key_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows with missing critical values"""
    before = len(df)
    df_clean = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    removed = before - len(df_clean)
    logger.info(f"Removed {removed} rows with missing key values")
    return df_clean, removed

def check_timestamp_consistency(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Check for consistent time intervals between candles"""
    results = {}
    for (symbol, timeframe), group in df.groupby(['symbol', 'timeframe']):
        sorted_grp = group.sort_values('timestamp')
        diffs = sorted_grp['timestamp'].diff().dropna()
        median_diff = diffs.median()
        results[(symbol, timeframe)] = median_diff
        
        # Check for gaps
        expected_diff = {
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }.get(timeframe, median_diff)
        
        large_gaps = diffs[diffs > expected_diff * 2]
        if not large_gaps.empty:
            logger.warning(
                f"{symbol} {timeframe}: Found {len(large_gaps)} large time gaps"
            )
            
    return results

def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3) -> int:
    """Detect outliers using Z-score method"""
    try:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0:
            return 0
        z_scores = (df[col] - mean_val) / std_val
        outliers = int((z_scores.abs() > threshold).sum())
        logger.debug(f"Found {outliers} Z-score outliers in {col}")
        return outliers
    except Exception as e:
        handle_error(e, "detect_outliers_zscore", logger=logger)
        return 0

def detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> int:
    """Detect outliers using IQR method"""
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = int(df[(df[col] < lower) | (df[col] > upper)].shape[0])
        logger.debug(f"Found {outliers} IQR outliers in {col}")
        return outliers
    except Exception as e:
        handle_error(e, "detect_outliers_iqr", logger=logger)
        return 0

def logical_consistency_checks(df: pd.DataFrame) -> int:
    """Check for logical price relationships"""
    inconsistent = df[
        (df['high'] < df['open']) | (df['high'] < df['close']) |
        (df['low'] > df['open']) | (df['low'] > df['close']) |
        (df['high'] < df['low'])  | (df['volume'] < 0)
    ]
    if len(inconsistent) > 0:
        logger.warning(f"Found {len(inconsistent)} rows with inconsistent prices")
    return len(inconsistent)

def save_cleaned_data(db_path: str, df_cleaned: pd.DataFrame) -> None:
    """Save cleaned data to database"""
    try:
        conn = sqlite3.connect(db_path)
        df_cleaned.to_sql('candles_cleaned', conn, if_exists='replace', index=False)
        logger.info("Cleaned data saved to table 'candles_cleaned'")
        conn.close()
    except Exception as e:
        handle_error(e, "save_cleaned_data", logger=logger)

def write_report(report_path: str, report_lines: List[str]) -> None:
    """Write data quality report to file"""
    try:
        with open(report_path, 'w') as f:
            for line in report_lines:
                f.write(line + "\n")
        logger.info(f"Report written to {report_path}")
    except Exception as e:
        handle_error(e, "write_report", logger=logger)

def main():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "candles.db")
        report_path = os.path.join(base_dir, "data_sanity_report.txt")

        report_lines = [
            f"Data Sanity Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Database: {db_path}",
            "-" * 50
        ]

        logger.info(f"Loading data from {db_path}")
        df = load_data(db_path)
        
        if df.empty:
            logger.error("No data loaded from database")
            return
            
        report_lines.append(f"Total rows loaded: {len(df)}")

        df, dup_count = remove_duplicates(df)
        report_lines.append(f"Duplicate rows removed: {dup_count}")

        df, missing_dropped = drop_missing_key_values(df)
        report_lines.append(f"Rows dropped due to missing key values: {missing_dropped}")

        ts_consistency = check_timestamp_consistency(df)
        report_lines.append("\nTimestamp Consistency:")
        for (symbol, tf), median_diff in ts_consistency.items():
            report_lines.append(f"{symbol} {tf}: Median interval = {median_diff/1000:.0f}s")

        key_cols = ['open', 'high', 'low', 'close', 'volume']
        desc_stats = df[key_cols].describe()
        report_lines.append("\nDescriptive Statistics:")
        report_lines.append(desc_stats.to_string())

        report_lines.append("\nOutlier Detection:")
        for col in key_cols:
            z_count = detect_outliers_zscore(df, col)
            iqr_count = detect_outliers_iqr(df, col)
            report_lines.append(f"\n{col}:")
            report_lines.append(f"  Z-score outliers (>3σ): {z_count}")
            report_lines.append(f"  IQR outliers (1.5×IQR): {iqr_count}")

        inconsistent_count = logical_consistency_checks(df)
        report_lines.append(f"\nInconsistent price relationships: {inconsistent_count}")

        write_report(report_path, report_lines)
        save_cleaned_data(db_path, df)
        logger.info("Data sanity checks complete")

    except Exception as e:
        handle_error(e, "main", logger=logger)
        raise

if __name__ == "__main__":
    main()
