#!/usr/bin/env python3
"""
Module: models/train.py
Handles ML model training with proper error handling and validation
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from decimal import Decimal
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.error_handler import handle_error
from database.database import DBConnection
from indicators.quality_monitor import quality_check

class ModelTrainer:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.default_features = [
            "open", "high", "low", "close", "volume",
            "EMA_8", "EMA_21", "EMA_55",
            "RSI_14", "MACD", "MACDs",
            "ATR_14", "STOCH_K", "STOCH_D"
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
                    conn
                )
                candles_df = pd.read_sql_query(
                    "SELECT * FROM candles ORDER BY timestamp",
                    conn
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
                suffixes=("_trade", "")
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

            self.logger.info(
                f"Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns"
            )
            return merged_df

        except Exception as e:
            handle_error(e, "ModelTrainer.merge_trade_and_candle_data", logger=self.logger)
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
            handle_error(e, "ModelTrainer.extract_features_and_labels", logger=self.logger)
            return pd.DataFrame(), []

    def train_models(self) -> bool:
        """Train ML models with proper error handling and validation"""
        try:
            # Get training configuration
            train_cfg = self.ctx.config.get("model_training", {})
            feature_columns = train_cfg.get("feature_columns", self.default_features)
            test_size = train_cfg.get("test_size", 0.2)
            use_xgb = train_cfg.get("use_xgb", False) and XGBOOST_AVAILABLE

            # Get model paths
            model_paths = self.ctx.config.get("model_paths", {})
            rf_path = model_paths.get("rf_model", "models/trained_rf.pkl")
            xgb_path = model_paths.get("xgb_model", "models/trained_xgb.pkl")
            cols_path = model_paths.get("trained_columns", "models/trained_columns.json")

            # Prepare data
            merged_data = self.merge_trade_and_candle_data()
            if merged_data.empty:
                self.logger.error("No training data available")
                return False

            X, y = self.extract_features_and_labels(merged_data, feature_columns)
            if X.empty or not y:
                self.logger.error("Feature extraction failed")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train and evaluate Random Forest
            rf_params = train_cfg.get("rf_params", {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            })
            
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            
            y_pred_rf = rf_model.predict(X_test)
            metrics_rf = {
                "accuracy": accuracy_score(y_test, y_pred_rf),
                "precision": precision_score(y_test, y_pred_rf),
                "recall": recall_score(y_test, y_pred_rf),
                "f1": f1_score(y_test, y_pred_rf)
            }
            self.logger.info(f"Random Forest metrics: {metrics_rf}")

            # Train and evaluate XGBoost if enabled
            if use_xgb:
                xgb_params = train_cfg.get("xgb_params", {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss"
                })
                
                xgb_model = XGBClassifier(**xgb_params)
                xgb_model.fit(X_train, y_train)
                
                y_pred_xgb = xgb_model.predict(X_test)
                metrics_xgb = {
                    "accuracy": accuracy_score(y_test, y_pred_xgb),
                    "precision": precision_score(y_test, y_pred_xgb),
                    "recall": recall_score(y_test, y_pred_xgb),
                    "f1": f1_score(y_test, y_pred_xgb)
                }
                self.logger.info(f"XGBoost metrics: {metrics_xgb}")

            # Save models and columns
            os.makedirs(os.path.dirname(rf_path), exist_ok=True)
            with open(rf_path, "wb") as f:
                pickle.dump(rf_model, f)
            self.logger.info(f"Saved Random Forest model: {rf_path}")

            if use_xgb:
                with open(xgb_path, "wb") as f:
                    pickle.dump(xgb_model, f)
                self.logger.info(f"Saved XGBoost model: {xgb_path}")

            with open(cols_path, "w") as f:
                json.dump(feature_columns, f)
            self.logger.info(f"Saved feature columns: {cols_path}")

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
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("ModelTraining")

    # Create dummy context
    ctx = DummyContext(
        logger=logger,
        config={
            "model_training": {
                "feature_columns": [
                    "open", "high", "low", "close", "volume",
                    "EMA_8", "EMA_21", "EMA_55",
                    "RSI_14", "MACD", "MACDs",
                    "ATR_14", "STOCH_K", "STOCH_D"
                ],
                "test_size": 0.2,
                "use_xgb": XGBOOST_AVAILABLE,
                "rf_params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42
                }
            },
            "model_paths": {
                "rf_model": "models/trained_rf.pkl",
                "xgb_model": "models/trained_xgb.pkl",
                "trained_columns": "models/trained_columns.json"
            }
        },
        db_pool="data/trading.db"
    )

    trainer = ModelTrainer(ctx)
    success = trainer.train_models()
    if not success:
        logger.error("Model training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
