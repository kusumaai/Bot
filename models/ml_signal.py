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
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import logging

from utils.error_handler import handle_error, ValidationError, ModelError
from database.queries import DatabaseQueries

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
        
        self._model = None
        self._feature_names: List[str] = []
        self._last_validation: Optional[datetime] = None
        
    async def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        try:
            # Load and validate model
            if not self._model:
                await self._load_model()
            
            # Get and prepare data
            candles = await self.db.get_recent_candles(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback_periods
            )
            
            features = self._prepare_features(candles)
            
            # Validate features
            self._validate_features(features)
            
            # Generate prediction
            prediction = self._generate_prediction(features)
            
            # Create signal
            signal = self._create_signal(prediction, features)
            
            # Store signal
            await self.db.store_signal(
                symbol=symbol,
                signal_type='ml_model',
                direction=signal['direction'],
                strength=signal['strength'],
                metadata={
                    'features': signal['features'],
                    'prediction': float(prediction),
                    'model_version': self._get_model_version()
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"ML signal generation failed: {str(e)}")
            raise
    
    async def _load_model(self) -> None:
        try:
            if not self.model_path:
                raise ModelError("Model path not specified")
                
            self._model = joblib.load(self.model_path)
            
            # Load model metadata
            metadata_path = self.model_path.replace('.joblib', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self._feature_names = metadata.get('features', [])
            if not self._feature_names:
                raise ModelError("No feature names found in model metadata")
                
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def _prepare_features(
        self,
        candles: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        
        # Calculate technical features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['sma_cross'] = (
            df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
        ).astype(int)
        
        # Ensure all required features are present
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
            
        # Validate feature ranges
        for col in features.columns:
            col_stats = features[col].describe()
            if col_stats['std'] == 0:
                raise ValidationError(f"Feature {col} has zero variance")
    
    def _generate_prediction(self, features: pd.DataFrame) -> float:
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Generate prediction
        try:
            prediction = self._model.predict_proba(scaled_features)[-1]
            return prediction[1]  # Probability of positive class
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def _create_signal(
        self,
        prediction: float,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        # Define signal thresholds
        strong_threshold = 0.7
        weak_threshold = 0.5
        
        # Determine signal direction and strength
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
            'features': features.iloc[-1].to_dict()
        }
    
    def _get_model_version(self) -> str:
        if not self.model_path:
            return 'unknown'
        return self.model_path.split('/')[-1].replace('.joblib', '')
