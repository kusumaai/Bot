#! /usr/bin/env python3
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import time
import logging

from utils.error_handler import handle_error
from risk.validation import validate_market_data, validate_risk_parameters

def validate_signal(signal: Dict[str, Any], ctx: Any) -> bool:
    """Validate trading signal meets minimum requirements and risk parameters."""
    try:
        # Basic validation
        if not signal:
            return False
            
        # Required fields check
        required_fields = [
            "symbol", "direction", "probability", 
            "entry_price", "exchange"
        ]
        
        if not all(field in signal for field in required_fields):
            ctx.logger.debug(f"Signal missing required fields: {signal.get('symbol', 'unknown')}")
            return False
            
        # Direction validation
        if signal["direction"] not in ["long", "short"]:
            ctx.logger.debug(f"Invalid signal direction: {signal.get('direction')}")
            return False
            
        return True
        
    except Exception as e:
        handle_error(e, "validate_signal", logger=ctx.logger)
        return False

def analyze_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze trading signal and return analysis result"""
    return {
        "status": True,
        "metadata": signal.get("metadata", {})
    }

def combine_signals(
    ml_signals: List[Dict[str, Any]],
    ga_signals: List[Dict[str, Any]],
    ctx: Any
) -> List[Dict[str, Any]]:
    """
    Combine and reconcile trading signals from ML and GA modules.
    
    Priority Logic:
    1. If signals agree -> Use weighted average based on confidence
    2. If signals conflict -> Use signal with higher confidence if above threshold
    3. If only one signal exists -> Use it with validation
    """
    try:
        # Get configuration parameters with proper decimal handling
        ml_prefer_threshold = Decimal(str(ctx.config.get("ml_prefer_threshold", "0.6")))
        ga_fitness_scale = Decimal(str(ctx.config.get("ga_fitness_scale", "1.0")))
        min_probability = Decimal(str(ctx.config.get("min_signal_probability", "0.55")))

        # Initialize containers
        final_signals = []
        
        # Pre-validate signals and create lookup maps
        ml_map = {}
        ga_map = {}
        
        for sig in ml_signals:
            if validate_signal(sig, ctx):
                ml_map[sig["symbol"]] = sig
                
        for sig in ga_signals:
            if validate_signal(sig, ctx):
                ga_map[sig["symbol"]] = sig
                
        all_symbols = set(ml_map.keys()).union(ga_map.keys())
        
        ctx.logger.debug(f"Processing signals for {len(all_symbols)} symbols")

        # Process each symbol
        for symbol in all_symbols:
            try:
                signal = None
                
                # Case 1: Both signals exist
                if symbol in ml_map and symbol in ga_map:
                    ml_signal = ml_map[symbol]
                    ga_signal = ga_map[symbol]
                    
                    # Scale GA probability
                    ga_prob = Decimal(str(ga_signal.get("probability", 0))) * ga_fitness_scale

                    # Signals agree on direction
                    if ml_signal["direction"] == ga_signal["direction"]:
                        ml_weight = Decimal(str(ml_signal.get("probability", 0)))
                        combined_prob = (ml_weight + ga_prob) / Decimal("2.0")
                        
                        if combined_prob >= min_probability:
                            signal = {
                                "symbol": symbol,
                                "direction": ml_signal["direction"],
                                "probability": float(combined_prob),
                                "expected_value": float((Decimal(str(ml_signal.get("expected_value", 0))) + 
                                                 Decimal(str(ga_signal.get("expected_value", 0)))) / 2.0),
                                "entry_price": float(ml_signal["entry_price"]),
                                "stop_loss": float(ml_signal.get("stop_loss") or ga_signal.get("stop_loss", 0)),
                                "take_profit": float(ml_signal.get("take_profit") or ga_signal.get("take_profit", 0)),
                                "kelly_fraction": float((Decimal(str(ml_signal.get("kelly_fraction", 0))) + 
                                                 Decimal(str(ga_signal.get("kelly_fraction", 0)))) / 2.0),
                                "exchange": ml_signal["exchange"],
                                "timestamp": time.time()
                            }
                    
                    # Signals conflict
                    else:
                        ml_prob = Decimal(str(ml_signal.get("probability", 0)))
                        if ml_prob >= ml_prefer_threshold:
                            signal = dict(ml_signal)
                            signal["timestamp"] = time.time()
                        elif ga_prob >= min_probability:
                            signal = dict(ga_signal)
                            signal["probability"] = float(ga_prob)
                            signal["timestamp"] = time.time()
                
                # Case 2: Only ML signal exists
                elif symbol in ml_map:
                    ml_signal = ml_map[symbol]
                    if Decimal(str(ml_signal.get("probability", 0))) >= min_probability:
                        signal = dict(ml_signal)
                        signal["timestamp"] = time.time()
                
                # Case 3: Only GA signal exists
                elif symbol in ga_map:
                    ga_signal = ga_map[symbol]
                    ga_prob = Decimal(str(ga_signal.get("probability", 0))) * ga_fitness_scale
                    if ga_prob >= min_probability:
                        signal = dict(ga_signal)
                        signal["probability"] = float(ga_prob)
                        signal["timestamp"] = time.time()

                # Validate and add final signal
                if signal and validate_signal(signal, ctx):
                    final_signals.append(signal)

            except Exception as e:
                handle_error(e, f"signal_utils.combine_signals.symbol_loop: {symbol}", logger=ctx.logger)
                continue

        ctx.logger.info(f"Generated {len(final_signals)} combined signals")
        return final_signals

    except Exception as e:
        handle_error(e, "signal_utils.combine_signals", logger=ctx.logger)
        return []