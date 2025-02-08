#!/usr/bin/env python3
"""
validation_tests.py - Comprehensive validation suite for trading bot
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

from database.database import DBConnection
from models.ml_signal import generate_ml_signals
from signals.ga_synergy import generate_ga_signals
from trading.math import (
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_expected_value
)
from execution.position_manager import calculate_position_params
from trading.ratchet import RatchetManager

class ValidationContext:
    """Test context that mimics production environment"""
    def __init__(self):
        self.logger = logging.getLogger("Validation")
        self.config = {
            "timeframe": "15m",
            "emergency_stop_pct": -3,
            "ratchet_thresholds": [2, 4, 6],
            "ratchet_lock_ins": [1, 2, 3],
            "kelly_scaling": 0.5,
            "initial_balance": 10000,
            "risk_factor": 0.1
        }
        self.db_pool = "data/candles.db"

async def validate_signal_generation(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate ML and GA signal generation"""
    results = {"ml_signals": 0, "ga_signals": 0, "conflicts": 0}
    
    with DBConnection(ctx.db_pool) as conn:
        # Get recent candles
        df = pd.read_sql_query(
            """
            SELECT * FROM candles 
            WHERE datetime >= datetime('now', '-1 day')
            ORDER BY timestamp DESC
            """,
            conn
        )
        
        if df.empty:
            return results
            
        # Generate signals
        ml_signals = generate_ml_signals(df, ctx)
        ga_signals = generate_ga_signals(df.to_dict('records'), [], ctx)
        
        results["ml_signals"] = len(ml_signals)
        results["ga_signals"] = len(ga_signals)
        
        # Check for conflicts
        ml_dict = {s["symbol"]: s["direction"] for s in ml_signals}
        ga_dict = {s["symbol"]: s["direction"] for s in ga_signals}
        
        for symbol in set(ml_dict.keys()) & set(ga_dict.keys()):
            if ml_dict[symbol] != ga_dict[symbol]:
                results["conflicts"] += 1
                
    return results

async def validate_position_sizing(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate position sizing calculations"""
    results = {"positions": [], "warnings": []}
    
    # Test scenarios
    test_cases = [
        {
            "balance": 10000,
            "price": 50000,
            "volatility": 0.02,
            "probability": 0.6,
            "expected_return": 0.03
        },
        {
            "balance": 5000,
            "price": 1800,
            "volatility": 0.015,
            "probability": 0.55,
            "expected_return": 0.02
        }
    ]
    
    for case in test_cases:
        ev, win_target, loss_target = calculate_expected_value(
            case["price"],
            case["expected_return"],
            case["probability"],
            ctx.config["emergency_stop_pct"] / 100,
            0.001  # Transaction cost
        )
        
        kelly = calculate_kelly_fraction(
            case["probability"],
            win_target,
            loss_target
        )
        
        position_size = calculate_position_size(
            case["balance"],
            kelly * ctx.config["kelly_scaling"],
            case["price"],
            case["volatility"],
            ctx.config["risk_factor"]
        )
        
        position_value = position_size * case["price"]
        leverage = position_value / case["balance"]
        
        results["positions"].append({
            "position_size": position_size,
            "position_value": position_value,
            "leverage": leverage,
            "kelly": kelly
        })
        
        if leverage > 2:
            results["warnings"].append(
                f"High leverage warning: {leverage:.2f}x"
            )
            
    return results

async def validate_ratchet_system(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate ratchet stop loss system"""
    results = {"stops": [], "warnings": []}
    
    ratchet = RatchetManager(ctx)
    
    # Test price scenarios
    scenarios = [
        # Steady uptrend
        {"name": "uptrend", "prices": [100, 102, 104, 106, 108]},
        # Volatile
        {"name": "volatile", "prices": [100, 104, 101, 105, 102]},
        # Downtrend
        {"name": "downtrend", "prices": [100, 98, 96, 94, 92]}
    ]
    
    for scenario in scenarios:
        trade_id = f"test_{scenario['name']}"
        entry_price = scenario["prices"][0]
        
        ratchet.initialize_trade(trade_id, entry_price)
        stops = []
        
        for price in scenario["prices"][1:]:
            result = ratchet.update_price(trade_id, price)
            if result:
                new_stop, reason = result
                stops.append({
                    "price": price,
                    "new_stop": new_stop,
                    "reason": reason
                })
                
        results["stops"].append({
            "scenario": scenario["name"],
            "stops": stops
        })
        
        metrics = ratchet.get_trade_metrics(trade_id)
        if metrics["max_drawdown_pct"] < ctx.config["emergency_stop_pct"]:
            results["warnings"].append(
                f"Emergency stop triggered in {scenario['name']}"
            )
            
    return results

async def run_validation_suite() -> None:
    """Run all validation tests"""
    ctx = ValidationContext()
    logging.basicConfig(level=logging.INFO)
    
    print("\nStarting Trading Bot Validation Suite...")
    
    # Signal Generation
    print("\nValidating Signal Generation...")
    signal_results = await validate_signal_generation(ctx)
    print(f"ML Signals: {signal_results['ml_signals']}")
    print(f"GA Signals: {signal_results['ga_signals']}")
    print(f"Signal Conflicts: {signal_results['conflicts']}")
    
    # Position Sizing
    print("\nValidating Position Sizing...")
    position_results = await validate_position_sizing(ctx)
    for i, pos in enumerate(position_results["positions"]):
        print(f"\nPosition {i+1}:")
        print(f"Size: {pos['position_size']:.4f}")
        print(f"Value: ${pos['position_value']:.2f}")
        print(f"Leverage: {pos['leverage']:.2f}x")
        print(f"Kelly Fraction: {pos['kelly']:.2f}")
    
    if position_results["warnings"]:
        print("\nPosition Warnings:")
        for warning in position_results["warnings"]:
            print(f"- {warning}")
    
    # Ratchet System
    print("\nValidating Ratchet System...")
    ratchet_results = await validate_ratchet_system(ctx)
    for scenario in ratchet_results["stops"]:
        print(f"\nScenario: {scenario['scenario']}")
        for stop in scenario["stops"]:
            print(f"Price: {stop['price']}, Stop: {stop['new_stop']}, Reason: {stop['reason']}")
    
    if ratchet_results["warnings"]:
        print("\nRatchet Warnings:")
        for warning in ratchet_results["warnings"]:
            print(f"- {warning}")

if __name__ == "__main__":
    asyncio.run(run_validation_suite())