#!/usr/bin/env python3
"""
Module: models/ml_signal.py
Handles ML model loading and signal generation with proper error handling
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time

from utils.error_handler import handle_error

@dataclass
class MLModels:
    rf_model: Any
    xgb_model: Optional[Any]
    trained_columns: List[str]

class MLSignalGenerator:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.models: Optional[MLModels] = None
        self.cache_timeout = ctx.config.get("model_cache_timeout", 3600)  # 1 hour
        self.last_load_time = 0

    def load_models(self) -> Optional[MLModels]:
        """Load trained ML models with caching and validation"""
        try:
            current_time = time.time()
            if self.models and (current_time - self.last_load_time) < self.cache_timeout:
                return self.models

            model_paths = self.ctx.config.get("model_paths", {})
            rf_path = model_paths.get("rf_model", os.path.join("models", "trained_rf.pkl"))
            xgb_path = model_paths.get("xgb_model", os.path.join("models", "trained_xgb.pkl"))
            cols_path = model_paths.get("trained_columns", os.path.join("models", "trained_columns.json"))

            # Load RF (required)
            try:
                with open(rf_path, "rb") as f:
                    rf_model = pickle.load(f)
            except Exception as e:
                handle_error(e, f"MLSignal.load_models: RF from {rf_path}", logger=self.logger)
                return None

            # Load XGB (optional)
            xgb_model = None
            if os.path.exists(xgb_path):
                try:
                    with open(xgb_path, "rb") as f:
                        xgb_model = pickle.load(f)
                except Exception as e:
                    handle_error(e, f"MLSignal.load_models: XGB from {xgb_path}", logger=self.logger)

            # Load trained columns (required)
            try:
                with open(cols_path, "r") as f:
                    trained_columns = json.load(f)
                if not trained_columns:
                    raise ValueError("Empty trained columns list")
            except Exception as e:
                handle_error(e, f"MLSignal.load_models: columns from {cols_path}", logger=self.logger)
                return None

            self.models = MLModels(rf_model, xgb_model, trained_columns)
            self.last_load_time = current_time
            return self.models

        except Exception as e:
            handle_error(e, "MLSignal.load_models", logger=self.logger)
            return None

    def generate_signals(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate ML-based signals with proper validation and error handling"""
        signals = []
        
        try:
            # Load/refresh models
            models = self.load_models()
            if not models:
                self.logger.error("Failed to load ML models")
                return signals

            # Validate input data
            if features_df.empty:
                self.logger.warning("Empty features DataFrame")
                return signals

            # Get configuration
            ml_cfg = self.ctx.config.get("ml_signals", {})
            ml_long_th = ml_cfg.get("long_threshold", 0.6)
            ml_short_th = ml_cfg.get("short_threshold", 0.4)
            allow_shorts = ml_cfg.get("allow_shorts", False)
            weights = ml_cfg.get("model_weights", {"rf": 0.6, "xgb": 0.4})

            # Process each symbol
            for symbol, group in features_df.groupby("symbol"):
                try:
                    latest = group.iloc[-1]
                    
                    # Build feature vector
                    fv = self._build_feature_vector(latest, models.trained_columns, symbol)
                    if fv is None:
                        continue

                    # Get predictions
                    probability = self._get_ensemble_prediction(
                        fv, models.rf_model, models.xgb_model, weights
                    )
                    if probability is None:
                        continue

                    # Generate signals based on thresholds
                    signal = self._create_signal(
                        symbol, probability, latest["close"],
                        ml_long_th, ml_short_th, allow_shorts
                    )
                    if signal:
                        signals.append(signal)

                except Exception as e:
                    handle_error(e, f"MLSignal.generate_signals: {symbol}", logger=self.logger)

        except Exception as e:
            handle_error(e, "MLSignal.generate_signals", logger=self.logger)

        return signals

    def _build_feature_vector(
        self, latest: pd.Series, trained_columns: List[str], symbol: str
    ) -> Optional[np.ndarray]:
        """Build feature vector with validation"""
        try:
            fv = []
            missing_features = []
            
            for col in trained_columns:
                if col in latest:
                    fv.append(latest[col])
                else:
                    missing_features.append(col)
                    fv.append(0)

            if missing_features:
                self.logger.warning(
                    f"Missing features for {symbol}: {missing_features}"
                )

            return np.array(fv).reshape(1, -1)

        except Exception as e:
            handle_error(e, f"MLSignal._build_feature_vector: {symbol}", logger=self.logger)
            return None

    def _get_ensemble_prediction(
        self, fv: np.ndarray, rf_model: Any, xgb_model: Optional[Any], weights: Dict[str, float]
    ) -> Optional[float]:
        """Get ensemble prediction with proper error handling"""
        try:
            prob_rf = rf_model.predict_proba(fv)[0][1]
            
            if xgb_model:
                try:
                    prob_xgb = xgb_model.predict_proba(fv)[0][1]
                    return (
                        prob_rf * weights["rf"] + 
                        prob_xgb * weights["xgb"]
                    )
                except Exception as e:
                    handle_error(e, "MLSignal._get_ensemble_prediction: XGB", logger=self.logger)
            
            return prob_rf

        except Exception as e:
            handle_error(e, "MLSignal._get_ensemble_prediction: RF", logger=self.logger)
            return None

    def _create_signal(
        self, symbol: str, probability: float, price: float,
        long_th: float, short_th: float, allow_shorts: bool
    ) -> Optional[Dict[str, Any]]:
        """Create signal dictionary with validation"""
        try:
            if probability >= long_th:
                return {
                    "symbol": symbol,
                    "direction": "long",
                    "probability": probability,
                    "entry_price": price,
                    "exchange": self.ctx.config.get("exchanges", ["unknown"])[0]
                }
            elif allow_shorts and probability <= short_th:
                return {
                    "symbol": symbol,
                    "direction": "short",
                    "probability": probability,
                    "entry_price": price,
                    "exchange": self.ctx.config.get("exchanges", ["unknown"])[0]
                }
            return None

        except Exception as e:
            handle_error(e, f"MLSignal._create_signal: {symbol}", logger=self.logger)
            return None
