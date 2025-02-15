#! /usr/bin/env python3
# src/signals/ml_signals.py
"""
Module: src.signals
Provides machine learning signal generation.
"""
# import required modules
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from database.queries import DatabaseQueries
from signals.base_types import BaseSignal
from signals.trading_types import SignalMetadata
from utils.error_handler import handle_error_async
from utils.exceptions import ValidationError
from utils.logger import get_logger


# ML signal class that defines the ML signal for the machine learning signal generator
@dataclass
class MLSignal(BaseSignal):
    """ML-specific signal extension"""

    probability: Decimal = Decimal("0")
    confidence: float = 0.0
    expiry: Optional[datetime] = None


# ML signal generator class that defines the ML signal generator for the machine learning signal generator
class MLSignalGenerator:
    def __init__(
        self, db_queries: DatabaseQueries, logger: Optional[logging.Logger] = None
    ):
        self.db = db_queries
        self.logger = logger or get_logger(__name__)

    async def generate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        timeframe: str,
        lookback_periods: int = 100,
    ) -> Optional[Dict[str, Any]]:
        """Generate ML-based trading signal for a symbol"""
        try:
            # Feature engineering
            features = self._prepare_features(data)

            # Generate prediction
            prediction = await self._generate_prediction(features)

            if prediction and prediction["probability"] >= self.min_probability:
                signal = MLSignal(
                    symbol=symbol,
                    direction=prediction["direction"],
                    probability=Decimal(str(prediction["probability"])),
                    timestamp=datetime.utcnow(),
                    metadata={
                        "timeframe": timeframe,
                        "features": features.to_dict(),
                        "model_version": "1.0",
                    },
                )

                return signal.metadata

            return None

        except Exception as e:
            await handle_error_async(
                e, "MLSignalGenerator.generate_signal", self.logger
            )
            return None

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            self.logger.info(f"Preparing features from data with shape: {data.shape}")
            # Placeholder: perform feature engineering as needed
            return data
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise

    # generate the prediction using the ML model
    async def _generate_prediction(
        self, features: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Generate prediction using trained models from ModelTrainer"""
        import json
        import os
        import pickle
        from statistics import mean

        models_dir = os.path.join("src", "models", "saved_models")
        columns_file = os.path.join("src", "models", "trained_columns.json")

        if not os.path.exists(models_dir):
            self.logger.error(f"Models directory {models_dir} does not exist.")
            return None

        if not os.path.exists(columns_file):
            self.logger.error(f"Trained columns file {columns_file} is missing.")
            return None

        try:
            with open(columns_file, "r") as f:
                expected_columns = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading trained columns: {e}", exc_info=True)
            return None

        try:
            # Reorder features based on expected columns and take the last row for prediction
            features_processed = features[expected_columns].tail(1)
        except Exception as e:
            self.logger.error(f"Error reordering features: {e}", exc_info=True)
            return None

        model_files = [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith(".pkl")
        ]
        if not model_files:
            self.logger.error("No trained model files found in the models directory.")
            return None

        predictions = []
        for model_file in model_files:
            try:
                with open(model_file, "rb") as mf:
                    model = pickle.load(mf)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features_processed)[0]
                    pred_prob = proba[1]  # probability for class 1
                else:
                    pred = model.predict(features_processed)[0]
                    pred_prob = 1.0 if pred == 1 else 0.0
                predictions.append(pred_prob)
                self.logger.info(
                    f"Model {model_file} predicted probability: {pred_prob:.3f}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error during inference from model {model_file}: {e}",
                    exc_info=True,
                )

        if not predictions:
            self.logger.error("No predictions were generated from any models.")
            return None

        avg_prob = mean(predictions)
        self.logger.info(f"Averaged prediction probability: {avg_prob:.3f}")

        if avg_prob < self.min_probability:
            self.logger.info(
                f"Average probability {avg_prob:.3f} is below min threshold {self.min_probability}, no signal generated."
            )
            return None

        direction = "long" if avg_prob >= 0.5 else "short"
        return {"direction": direction, "probability": avg_prob}

    # get the minimum probability threshold for the ML signal
    @property
    def min_probability(self) -> float:
        """Minimum probability threshold"""
        return 0.7


async def generate_ml_signals(ctx, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates machine learning-based trading signals.

    :param ctx: Trading context
    :param data: Input data for signal generation
    :return: Generated signals
    """
    if not data.get("trend") or not data.get("strength"):
        raise ValidationError("Missing required fields: trend and strength")
    # validate the trend for the machine learning signal generator
    trend = data["trend"].lower()
    if trend not in ["bullish", "bearish"]:
        raise ValidationError(f"Invalid trend: {trend}")
    # generate the signals for the machine learning signal generator
    signals = {
        "symbol": data["symbol"],
        "action": "buy" if trend == "bullish" else "sell",
        "strength": Decimal(str(data["strength"])),
        "timestamp": datetime.utcnow(),
        "metadata": {"source": "ml_model", "version": "1.0", "trend": trend},
    }
    # return the signals for the machine learning signal generator to the trading context
    return signals
