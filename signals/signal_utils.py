#!/usr/bin/env python3
"""
signals/signal_utils.py - Signal combination and utility functions
"""
from typing import List, Dict, Any
from utils.error_handler import handle_error

def combine_signals(
    ml_signals: List[Dict[str, Any]],
    ga_signals: List[Dict[str, Any]],
    ctx: Any
) -> List[Dict[str, Any]]:
    """
    Combine and reconcile trading signals from ML and GA modules.
    
    Priority Logic:
    1. If signals agree -> Use average of probabilities
    2. If signals conflict -> Use signal with higher confidence if above threshold
    3. If only one signal exists -> Use it directly
    """
    try:
        ml_prefer_threshold = ctx.config.get("ml_prefer_threshold", 0.5)
        ga_fitness_scale = ctx.config.get("ga_fitness_scale", 1.0)

        final_signals = []
        ml_map = {sig["symbol"]: sig for sig in ml_signals}
        ga_map = {sig["symbol"]: sig for sig in ga_signals}
        all_symbols = set(ml_map.keys()).union(ga_map.keys())

        for symbol in all_symbols:
            signal = None
            
            # Both signals exist
            if symbol in ml_map and symbol in ga_map:
                ml_signal = ml_map[symbol]
                ga_signal = ga_map[symbol]
                
                # Scale GA probability
                ga_prob = ga_signal.get("probability", 0.0) * ga_fitness_scale

                # Signals agree
                if ml_signal["direction"] == ga_signal["direction"]:
                    # Average probabilities
                    combined_prob = (ml_signal.get("probability", 0.0) + ga_prob) / 2.0
                    signal = {
                        "symbol": symbol,
                        "direction": ml_signal["direction"],
                        "probability": combined_prob,
                        "expected_value": (
                            ml_signal.get("expected_value", 0.0) + 
                            ga_signal.get("expected_value", 0.0)
                        ) / 2.0,
                        "entry_price": ml_signal["entry_price"],
                        "stop_loss": ml_signal.get("stop_loss") or ga_signal.get("stop_loss"),
                        "take_profit": ml_signal.get("take_profit") or ga_signal.get("take_profit"),
                        "kelly_fraction": (
                            ml_signal.get("kelly_fraction", 0.0) + 
                            ga_signal.get("kelly_fraction", 0.0)
                        ) / 2.0,
                        "exchange": ml_signal["exchange"]
                    }
                
                # Signals conflict
                else:
                    # Use ML if probability exceeds threshold
                    if ml_signal.get("probability", 0.0) >= ml_prefer_threshold:
                        signal = ml_signal
                    else:
                        signal = {**ga_signal, "probability": ga_prob}
            
            # Only ML signal exists
            elif symbol in ml_map:
                signal = ml_map[symbol]
            
            # Only GA signal exists
            else:
                ga_signal = ga_map[symbol]
                signal = {
                    **ga_signal,
                    "probability": ga_signal.get("probability", 0.0) * ga_fitness_scale
                }

            if signal:
                final_signals.append(signal)

        return final_signals

    except Exception as e:
        handle_error(e, "signal_utils.combine_signals", logger=ctx.logger)
        return []

def validate_signal(signal: Dict[str, Any], ctx: Any) -> bool:
    """
    Validate trading signal meets minimum requirements.
    """
    try:
        if not signal:
            return False
            
        required_fields = [
            "symbol", "direction", "probability", 
            "entry_price", "exchange"
        ]
        
        # Check required fields exist
        if not all(field in signal for field in required_fields):
            return False
            
        # Validate direction
        if signal["direction"] not in ["long", "short"]:
            return False
            
        # Check probability thresholds
        min_prob = ctx.config.get("min_signal_probability", 0.0)
        if signal["probability"] < min_prob:
            return False
            
        # Validate numeric values
        if signal["entry_price"] <= 0:
            return False
            
        return True
        
    except Exception as e:
        handle_error(e, "signal_utils.validate_signal", logger=ctx.logger)
        return False