#!/usr/bin/env python3
"""
Module: models/train.py
"""

import os
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.error_handler import handle_error
from database.database import DBConnection
# from database.database import execute_sql, execute_sql_one  # Not used here


def merge_trade_and_candle_data(ctx) -> pd.DataFrame:
    """
    Merge 'closed' trades with candle data on time:
      For each closed trade, find the candle (same symbol) that
      is <= trade's entry_time (most recent). Use merge_asof.
    """
    try:
        with DBConnection(ctx.db_pool) as conn:
            trades_df = pd.read_sql_query(
                "SELECT * FROM trades WHERE close_reason='closed'",
                conn
            )
            candles_df = pd.read_sql_query("SELECT * FROM candles", conn)

        if trades_df.empty:
            ctx.logger.warning("No closed trades found.")
            return pd.DataFrame()
        if candles_df.empty:
            ctx.logger.warning("No candle data found.")
            return pd.DataFrame()

        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")
        candles_df["datetime"]   = pd.to_datetime(candles_df["datetime"], errors="coerce")
        trades_df  = trades_df.sort_values("entry_time")
        candles_df = candles_df.sort_values("datetime")

        merged_df = pd.merge_asof(
            trades_df,
            candles_df,
            left_on="entry_time",
            right_on="datetime",
            by="symbol",
            direction="backward",
            suffixes=("_trade", "")
        )
        merged_df = merged_df.dropna(subset=["datetime"])

        ctx.logger.info(
            f"Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} cols."
        )
        return merged_df
    except Exception as e:
        handle_error(e, context="Models.merge_trade_and_candle_data", logger=ctx.logger)
        return pd.DataFrame()


def extract_features_and_labels(merged: pd.DataFrame, feature_columns: list):
    """
    Pull feature cols from candle portion; label from 'result' (1 if > 0, else 0).
    """
    try:
        if merged.empty:
            return pd.DataFrame(), []

        for col in feature_columns:
            if col not in merged.columns:
                merged[col] = 0

        X = merged[feature_columns].copy()
        if "result" not in merged.columns:
            raise ValueError("No 'result' column in merged data.")
        y = merged["result"].apply(lambda r: 1 if r > 0 else 0).tolist()
        return X, y
    except Exception as e:
        handle_error(e, context="Models.extract_features_and_labels", logger=ctx.logger)
        return pd.DataFrame(), []


def reshape_feature_vector(X: pd.DataFrame) -> np.ndarray:
    return X.values


def ensure_directory_exists(filepath: str) -> None:
    dir_ = os.path.dirname(filepath)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)


def save_model(model, filepath: str) -> None:
    ensure_directory_exists(filepath)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def write_json(data, filepath: str) -> None:
    ensure_directory_exists(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f)


def train_models(ctx) -> None:
    """
    Orchestrates the data merge, feature extraction, train–test split, training,
    and saving of both models and columns.
    """
    try:
        merged_data = merge_trade_and_candle_data(ctx)
        if merged_data.empty:
            ctx.logger.info("No merged data; training aborted.")
            return

        # Feature columns
        default_feats = [
            "open", "high", "low", "close", "volume",
            "EMA_8", "EMA_21", "EMA_55",
            "RSI_14", "MACD", "MACDs",
            "ATR_14"
        ]
        feature_columns = ctx.config.get("model_training", {}).get("feature_columns", default_feats)
        X, y = extract_features_and_labels(merged_data, feature_columns)
        if X.empty or not y:
            ctx.logger.error("No usable training data after feature extraction.")
            return

        X_arr = reshape_feature_vector(X)
        ctx.logger.info(f"Train data shape: {X_arr.shape}, labels: {len(y)}")

        # Split
        test_sz = ctx.config.get("model_training", {}).get("test_size", 0.2)
        X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y, test_size=test_sz, random_state=42)

        # RF hyperparams
        rf_params = ctx.config.get("model_training", {}).get("rf_params", {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        })

        # Train RF
        rf_model = None
        try:
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_tr, y_tr)
            pred_rf = rf_model.predict(X_te)
            acc_rf = accuracy_score(y_te, pred_rf)
            ctx.logger.info(f"Random Forest accuracy: {acc_rf:.4f}")
        except Exception as e:
            handle_error(e, context="Models.train_models: RF", logger=ctx.logger)
            rf_model = None

        # XGB if enabled
        xgb_model = None
        use_xgb = ctx.config.get("model_training", {}).get("use_xgb", False)
        if XGBOOST_AVAILABLE and use_xgb:
            try:
                xgb_params = ctx.config.get("model_training", {}).get("xgb_params", {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss"
                })
                xgb_model = XGBClassifier(**xgb_params)
                xgb_model.fit(X_tr, y_tr)
                pred_xgb = xgb_model.predict(X_te)
                acc_xgb = accuracy_score(y_te, pred_xgb)
                ctx.logger.info(f"XGBoost accuracy: {acc_xgb:.4f}")
            except Exception as e:
                handle_error(e, context="Models.train_models: XGB", logger=ctx.logger)
                xgb_model = None
        else:
            ctx.logger.info("XGB not used or not installed.")

        # Filepaths
        mpaths    = ctx.config.get("model_paths", {})
        rf_path   = mpaths.get("rf_model", os.path.join("models", "trained_rf.pkl"))
        xgb_path  = mpaths.get("xgb_model", os.path.join("models", "trained_xgb.pkl"))
        cols_path = mpaths.get("trained_columns", os.path.join("models", "trained_columns.json"))

        # Save
        if rf_model:
            save_model(rf_model, rf_path)
            ctx.logger.info(f"RF model saved: {rf_path}")
        else:
            ctx.logger.error("RF model training failed; no save.")

        if xgb_model:
            save_model(xgb_model, xgb_path)
            ctx.logger.info(f"XGB model saved: {xgb_path}")
        else:
            ctx.logger.info("No XGB model to save.")

        write_json(feature_columns, cols_path)
        ctx.logger.info(f"Feature columns saved: {cols_path}")
        ctx.logger.info("Training complete.")
    except Exception as e:
        handle_error(e, context="Models.train_models", logger=ctx.logger)


if __name__ == "__main__":
    # Demo usage
    import logging
    logging.basicConfig(level=logging.INFO)

    class DummyCtx:
        pass

    ctx = DummyCtx()
    ctx.logger = logging.getLogger("TradingBot")

    ctx.config = {
        "db_pool": "data/candles.db",
        "model_training": {
            "feature_columns": [
                "open", "high", "low", "close", "volume",
                "EMA_8", "EMA_21", "EMA_55",
                "RSI_14", "MACD", "MACDs",
                "ATR_14"
            ],
            "test_size": 0.2,
            "rf_params": {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            },
            "use_xgb": False
        },
        "model_paths": {
            "rf_model": "models/trained_rf.pkl",
            "xgb_model": "models/trained_xgb.pkl",
            "trained_columns": "models/trained_columns.json"
        }
    }

    train_models(ctx)
