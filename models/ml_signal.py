#!/usr/bin/env python3
"""
Module: models/ml_signal.py
Handles ML model loading and signal generation with proper error handling
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

from utils.error_handler import handle_error, ValidationError, ModelError
from database.queries import DatabaseQueries
from utils.numeric import NumericHandler

@dataclass
class MLModels:
    rf_model: Any 
    xgb_model: Optional[Any]
    trained_columns: List[str]

class MLSignalGenerator:
    def __init__(
        self,
        db_queries: DatabaseQueries,
        logger: logging.Logger,
        model_path: Optional[str] = None
    ):
        self.db = db_queries
        self.logger = logger
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.nh = NumericHandler()
        
        self._model = None
        self._feature_names: List[str] = []
        self._last_validation: Optional[datetime] = None
        
    async def _load_model(self) -> None:
        try:
            if not self.model_path:
                raise ModelError("Model path not specified")
                
            self._model = joblib.load(self.model_path)
            
            metadata_path = self.model_path.replace('.joblib', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self._feature_names = metadata.get('features', [])
            if not self._feature_names:
                raise ModelError("No feature names found in model metadata")
                
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def _prepare_features(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma_cross'] = (df['close'].rolling(20).mean() > df['close'].rolling(50).mean()).astype(int)
        
        # Ensure required features present
        missing_features = set(self._feature_names) - set(df.columns)
        if missing_features:
            raise ValidationError(f"Missing features: {missing_features}")
        
        return df[self._feature_names].dropna()
    
    def _validate_features(self, features: pd.DataFrame) -> None:
        if features.empty:
            raise ValidationError("No valid features generated")
            
        if features.isnull().any().any():
            raise ValidationError("Features contain missing values")
            
        if np.isinf(features.values).any():
            raise ValidationError("Features contain infinite values")
            
        for col in features.columns:
            col_stats = features[col].describe()
            if col_stats['std'] == 0:
                raise ValidationError(f"Feature {col} has zero variance")
    
    def _generate_prediction(self, features: pd.DataFrame) -> float:
        scaled_features = self.scaler.fit_transform(features)
        
        try:
            prediction = self._model.predict_proba(scaled_features)[-1]
            return prediction[1]  # Probability of positive class
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def _create_signal(
        self,
        prediction: float,
        features: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        strong_threshold = self.nh.to_decimal('0.7')
        weak_threshold = self.nh.to_decimal('0.5')
        
        prediction = self.nh.to_decimal(str(prediction))
        
        if prediction > strong_threshold:
            direction = 'long'
            strength = prediction
        elif prediction < (1 - strong_threshold):
            direction = 'short'
            strength = 1 - prediction
        elif prediction > weak_threshold:
            direction = 'long'
            strength = (prediction - weak_threshold) / (strong_threshold - weak_threshold)
        else:
            direction = 'short'
            strength = (weak_threshold - prediction) / weak_threshold
        
        return {
            'direction': direction,
            'strength': float(strength),
            'entry_price': current_price,
            'prediction': float(prediction),
            'features': features.iloc[-1].to_dict(),
            'timestamp': time.time()
        }

async def generate_ml_signals(data: pd.DataFrame, ctx: Any) -> List[Dict[str, Any]]:
    """Generate ML trading signals from data"""
    try:
        signal_generator = MLSignalGenerator(
            ctx.db_queries,
            ctx.logger,
            ctx.config.get('model_paths', {}).get('rf_model')
        )
        signals = []
        
        for symbol in ctx.config.get('market_list', []):
            symbol_data = data[data['symbol'] == symbol].copy()
            if not symbol_data.empty:
                signal = await signal_generator.generate_signal(
                    symbol=symbol,
                    timeframe=ctx.config.get('timeframe', '15m'),
                    lookback_periods=ctx.config.get('lookback_periods', 100)
                )
                if signal:
                    signal['symbol'] = symbol
                    signal['exchange'] = ctx.config.get('exchange', 'unknown')
                    signals.append(signal)
                    
        return signals
        
    except Exception as e:
        ctx.logger.error(f"ML signal generation failed: {str(e)}")
        return []