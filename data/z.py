#!/usr/bin/env python3
"""
Module: data/z.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM candles", conn)
    conn.close()
    return df

def remove_duplicates(df):
    before = len(df)
    df_clean = df.drop_duplicates(subset=['symbol', 'timeframe', 'timestamp'])
    return df_clean, (before - len(df_clean))

def drop_missing_key_values(df):
    before = len(df)
    df_clean = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    return df_clean, (before - len(df_clean))

def check_timestamp_consistency(df):
    results = {}
    for (symbol, timeframe), group in df.groupby(['symbol', 'timeframe']):
        sorted_grp = group.sort_values('timestamp')
        diffs = sorted_grp['timestamp'].diff().dropna()
        results[(symbol, timeframe)] = diffs.median()
    return results

def detect_outliers_zscore(df, col, threshold=3):
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val == 0:
        return 0
    z_scores = (df[col] - mean_val) / std_val
    return int((z_scores.abs() > threshold).sum())

def detect_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    iqr = Q3 - Q1
    lower = Q1 - factor * iqr
    upper = Q3 + factor * iqr
    return int(df[(df[col] < lower) | (df[col] > upper)].shape[0])

def logical_consistency_checks(df):
    inconsistent = df[
        (df['high'] < df['open']) | (df['high'] < df['close']) |
        (df['low']  > df['open']) | (df['low']  > df['close'])
    ]
    return len(inconsistent)

def save_cleaned_data(db_path, df_cleaned):
    conn = sqlite3.connect(db_path)
    try:
        df_cleaned.to_sql('candles_cleaned', conn, if_exists='replace', index=False)
        print("Cleaned data saved to table 'candles_cleaned'.")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
    finally:
        conn.close()

def write_report(report_path, report_lines):
    with open(report_path, 'w') as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"Report written to {report_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "candles.db")
    report_path = os.path.join(base_dir, "data_sanity_report.txt")

    report_lines = [f"Data Sanity Report for database: {db_path}"]
    print(f"Loading data from {db_path} ...")

    df = load_data(db_path)
    report_lines.append(f"Total rows loaded: {len(df)}")

    df, dup_count = remove_duplicates(df)
    report_lines.append(f"Duplicate rows removed: {dup_count}")

    df, missing_dropped = drop_missing_key_values(df)
    report_lines.append(f"Rows dropped due to missing key values: {missing_dropped}")

    ts_consistency = check_timestamp_consistency(df)
    for (symbol, tf), median_diff in ts_consistency.items():
        report_lines.append(f"{symbol} {tf}: Median timestamp diff = {median_diff}")

    key_cols = ['open', 'high', 'low', 'close', 'volume']
    desc_stats = df[key_cols].describe().to_string()
    report_lines.append("\nDescriptive Statistics for key columns:")
    report_lines.append(desc_stats)

    for c in key_cols:
        z_count = detect_outliers_zscore(df, c, threshold=3)
        iqr_count = detect_outliers_iqr(df, c, factor=1.5)
        report_lines.append(f"\nColumn '{c}':")
        report_lines.append(f"  Outliers (z-score > 3): {z_count}")
        report_lines.append(f"  Outliers (IQR method): {iqr_count}")

    inconsistent_count = logical_consistency_checks(df)
    report_lines.append(f"\nInconsistent rows (price relations): {inconsistent_count}")

    write_report(report_path, report_lines)
    save_cleaned_data(db_path, df)
    print("Data sanity checks complete.")

if __name__ == "__main__":
    main()
