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

def remove_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
    """Remove duplicate rows from DataFrame"""
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    return df, removed

def drop_missing_key_values(df: pd.DataFrame) -> (pd.DataFrame, int):
    """Drop rows with missing key values"""
    key_values = ['open', 'high', 'low', 'close', 'volume']
    initial_count = len(df)
    df = df.dropna(subset=key_values)
    removed = initial_count - len(df)
    return df, removed

def check_timestamp_consistency(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Check timestamp consistency for each symbol (stub implementation)"""
    result = {}
    if 'symbol' in df.columns and 'datetime' in df.columns:
        for symbol, group in df.groupby('symbol'):
            timestamps = pd.to_datetime(group['datetime'])
            if len(timestamps) > 1:
                diffs = timestamps.diff().dropna().dt.total_seconds()
                median_diff = diffs.median()
                result[(symbol, 'default')] = median_diff * 1000  # convert seconds to milliseconds
    return result

def detect_outliers_zscore(df: pd.DataFrame, col: str) -> int:
    """Detect outliers using Z-score method"""
    try:
        from scipy import stats
        if col not in df.columns:
            return 0
        z_scores = stats.zscore(df[col])
        return int(sum(abs(z_scores) > 3))
    except Exception as e:
        logger.error(f"Error in detect_outliers_zscore: {e}")
        return 0

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> int:
    """Detect outliers using IQR method"""
    try:
        if col not in df.columns:
            return 0
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0])
    except Exception as e:
        logger.error(f"Error in detect_outliers_iqr: {e}")
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
        report_lines.append(f"\nLogical inconsistencies found: {inconsistent_count}")

        write_report(report_path, report_lines)
        save_cleaned_data(db_path, df)
        logger.info("Data sanity checks complete")

    except Exception as e:
        handle_error(e, "main", logger=logger)
        raise

if __name__ == "__main__":
    main()
