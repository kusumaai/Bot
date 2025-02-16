#! /usr/bin/env python3
# src/models/train.py
"""
Module: src.models
Provides model training
"""
import json
import os
import pickle
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Optional XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from database.database import DBConnection
from indicators.quality_monitor import quality_check
from utils.error_handler import handle_error


class ModelTrainer:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.default_features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "EMA_8",
            "EMA_21",
            "EMA_55",
            "RSI_14",
            "MACD",
            "MACDs",
            "ATR_14",
            "STOCH_K",
            "STOCH_D",
        ]

    def merge_trade_and_candle_data(self) -> pd.DataFrame:
        """Merge trade and candle data with proper validation"""
        try:
            with DBConnection(self.ctx.db_pool) as conn:
                trades_df = pd.read_sql_query(
                    """
                    SELECT * FROM trades 
                    WHERE close_reason = 'closed' 
                    AND ABS(result) > 0
                    """,
                    conn,
                )
                candles_df = pd.read_sql_query(
                    "SELECT * FROM candles ORDER BY timestamp", conn
                )

            if trades_df.empty:
                self.logger.warning("No closed trades found")
                return pd.DataFrame()
            if candles_df.empty:
                self.logger.warning("No candle data found")
                return pd.DataFrame()

            # Convert timestamps
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            candles_df["datetime"] = pd.to_datetime(candles_df["timestamp"], unit="ms")

            # Sort data
            trades_df = trades_df.sort_values("entry_time")
            candles_df = candles_df.sort_values("datetime")

            # Merge with proper validation
            merged_df = pd.merge_asof(
                trades_df,
                candles_df,
                left_on="entry_time",
                right_on="datetime",
                by="symbol",
                direction="backward",
                suffixes=("_trade", ""),
            )

            # Validate merge results
            if merged_df.empty:
                self.logger.warning("Merge resulted in empty DataFrame")
                return pd.DataFrame()

            # Check data quality
            quality_report = quality_check(merged_df, self.ctx)
            if quality_report["warnings"]:
                for warning in quality_report["warnings"]:
                    self.logger.warning(f"Data quality issue: {warning}")
                if len(quality_report["warnings"]) > 3:
                    self.logger.error(
                        "Excessive data quality issues detected. Aborting merge."
                    )
                    raise ValueError("Critical data quality issues in merged data.")

            self.logger.info(
                f"Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns"
            )
            return merged_df

        except Exception as e:
            handle_error(
                e, "ModelTrainer.merge_trade_and_candle_data", logger=self.logger
            )
            return pd.DataFrame()

    def extract_features_and_labels(
        self, merged: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Extract features and labels with validation"""
        try:
            if merged.empty:
                return pd.DataFrame(), []

            features = feature_columns or self.default_features

            # Validate feature columns
            missing_cols = [col for col in features if col not in merged.columns]
            if missing_cols:
                self.logger.warning(f"Missing feature columns: {missing_cols}")
                for col in missing_cols:
                    merged[col] = 0

            X = merged[features].copy()

            if "result" not in merged.columns:
                raise ValueError("No 'result' column in merged data")

            y = (merged["result"] > 0).astype(int).tolist()

            # Validate features
            if X.isnull().any().any():
                self.logger.warning("Features contain null values, filling with 0")
                X = X.fillna(0)

            return X, y

        except Exception as e:
            handle_error(
                e, "ModelTrainer.extract_features_and_labels", logger=self.logger
            )
            return pd.DataFrame(), []

    def train_models(self) -> bool:
        try:
            self.logger.info("Starting model training...")
            merged = self.merge_trade_and_candle_data()
            if merged.empty:
                self.logger.error("No data available for training")
                return False
            X, y = self.extract_features_and_labels(merged)
            if X.empty or not y:
                self.logger.error("Insufficient data for training")
                return False

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if XGBOOST_AVAILABLE:
                from xgboost import XGBClassifier

                model = XGBClassifier()
            else:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(n_estimators=100)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, zero_division=0)
            rec = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            self.logger.info(
                f"Model training completed. Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}"
            )

            import json
            import os
            import pickle

            model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
            os.makedirs(model_dir, exist_ok=True)
            from datetime import datetime

            model_file = os.path.join(
                model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            with open(
                os.path.join(os.path.dirname(__file__), "trained_columns.json"), "w"
            ) as f:
                json.dump(X.columns.tolist(), f)

            return True
        except Exception as e:
            handle_error(e, "ModelTrainer.train_models", logger=self.logger)
            return False


def main():
    """CLI entry point for model training"""
    import logging
    from dataclasses import dataclass

    @dataclass
    class DummyContext:
        logger: Any
        config: Dict[str, Any]
        db_pool: str

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("ModelTraining")

    # Create dummy context
    ctx = DummyContext(
        logger=logger,
        config={
            "model_training": {
                "feature_columns": [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "EMA_8",
                    "EMA_21",
                    "EMA_55",
                    "RSI_14",
                    "MACD",
                    "MACDs",
                    "ATR_14",
                    "STOCH_K",
                    "STOCH_D",
                ],
                "test_size": 0.2,
                "use_xgb": XGBOOST_AVAILABLE,
                "rf_params": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            },
            "model_paths": {
                "rf_model": "models/trained_rf.pkl",
                "xgb_model": "models/trained_xgb.pkl",
                "trained_columns": "models/trained_columns.json",
            },
        },
        db_pool="data/trading.db",
    )

    trainer = ModelTrainer(ctx)
    success = trainer.train_models()
    if not success:
        logger.error("Model training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
