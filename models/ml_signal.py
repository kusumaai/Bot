#!/usr/bin/env python3
"""
Module: models/ml_signal.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from utils.error_handler import handle_error


def load_models(ctx: object):
    """
    Load trained ML models (RF, XGB) and the feature columns from configured paths or defaults.

    Returns:
        (rf_model, xgb_model, trained_columns)
    """
    rf_model       = None
    xgb_model      = None
    trained_columns = []

    model_paths = ctx.config.get("model_paths", {})
    rf_path     = model_paths.get("rf_model", os.path.join("models", "trained_rf.pkl"))
    xgb_path    = model_paths.get("xgb_model", os.path.join("models", "trained_xgb.pkl"))
    cols_path   = model_paths.get("trained_columns", os.path.join("models", "trained_columns.json"))

    # Load RF
    try:
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    except Exception as e:
        handle_error(e, context=f"Models.load_models: RF from {rf_path}", logger=ctx.logger)
        rf_model = None

    # Load XGB if available
    if os.path.exists(xgb_path):
        try:
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)
        except Exception as e:
            handle_error(e, context=f"Models.load_models: XGB from {xgb_path}", logger=ctx.logger)
            xgb_model = None

    # Load trained columns
    try:
        with open(cols_path, "r") as f:
            trained_columns = json.load(f)
    except Exception as e:
        handle_error(e, context=f"Models.load_models: columns from {cols_path}", logger=ctx.logger)
        trained_columns = []

    return rf_model, xgb_model, trained_columns


def weighted_average(values: list, weights: list) -> float:
    """
    Compute the weighted average of values given their corresponding weights.
    Returns 0.0 if total weight is zero.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def generate_ml_signals(features_df: pd.DataFrame, ctx: object) -> list:
    """
    Generate ML-based signals from the given features DataFrame, grouped by 'symbol'.

    Steps:
      1. Load models & columns
      2. For each symbol, get last row => build feature vector
      3. Predict probability w/ RF; ensemble w/ XGB if present
      4. Compare to thresholds => create signals (long/short)
    """
    signals = []
    try:
        rf_model, xgb_model, trained_columns = load_models(ctx)
    except Exception as e:
        handle_error(e, context="Models.generate_ml_signals: load_models", logger=ctx.logger)
        return signals

    # Check if we have any model loaded
    if not rf_model and not xgb_model:
        ctx.logger.error("No ML models (RF/XGB) loaded; skipping ML signals.")
        return signals

    # Check if columns are available
    if not trained_columns:
        ctx.logger.error("No trained columns found; skipping ML signals.")
        return signals

    # Check if we have features to process
    if features_df.empty:
        ctx.logger.warning("Empty features DataFrame; cannot generate ML signals.")
        return signals

    # Thresholds from config
    ml_long_th    = ctx.config.get("ml_long_threshold", 0.6)
    ml_short_th   = ctx.config.get("ml_short_threshold", 0.4)
    allow_shorts  = ctx.config.get("allow_shorts", False)

    # Ensemble weighting
    weights_cfg   = ctx.config.get("model_weights", {})
    rf_weight     = weights_cfg.get("rf", 0.6)
    xgb_weight    = weights_cfg.get("xgb", 0.4)

    # Group by symbol
    grouped = features_df.groupby("symbol")

    for symbol, group in grouped:
        try:
            latest = group.iloc[-1]

            # Build feature vector
            fv = []
            for col in trained_columns:
                if col in latest:
                    fv.append(latest[col])
                else:
                    ctx.logger.warning(f"Missing feature '{col}' for symbol {symbol}; using 0.")
                    fv.append(0)
            fv = np.array(fv).reshape(1, -1)

            # Predict w/ RF
            prob_rf = 0.5
            if rf_model:
                try:
                    prob_rf = rf_model.predict_proba(fv)[0][1]
                except Exception as e:
                    handle_error(e, context=f"ML_signal: RF predict {symbol}", logger=ctx.logger)

            # Predict w/ XGB if present
            prob_xgb = None
            if xgb_model:
                try:
                    prob_xgb = xgb_model.predict_proba(fv)[0][1]
                except Exception as e:
                    handle_error(e, context=f"ML_signal: XGB predict {symbol}", logger=ctx.logger)
                    prob_xgb = None

            # Ensemble if XGB
            probability = prob_rf
            if prob_xgb is not None:
                probability = weighted_average([prob_rf, prob_xgb], [rf_weight, xgb_weight])

            # Compare to thresholds
            if probability >= ml_long_th:
                signals.append({
                    "symbol":      symbol,
                    "direction":   "long",
                    "probability": probability,
                    "entry_price": latest["close"],
                    "exchange":    ctx.config["exchanges"][0] if ctx.config.get("exchanges") else "unknown"
                })
            elif allow_shorts and probability <= ml_short_th:
                signals.append({
                    "symbol":      symbol,
                    "direction":   "short",
                    "probability": probability,
                    "entry_price": latest["close"],
                    "exchange":    ctx.config["exchanges"][0] if ctx.config.get("exchanges") else "unknown"
                })

        except Exception as e:
            handle_error(e, context=f"ML_signal: processing symbol {symbol}", logger=ctx.logger)

    return signals
