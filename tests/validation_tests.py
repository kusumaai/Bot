#!/usr/bin/env python3
"""
validation_tests.py - Comprehensive validation suite for trading bot
"""

import asyncio
import logging
import pandas as pd
from decimal import Decimal
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
from risk.limits import RiskLimits
from risk.manager import RiskManager
from risk.position import Position
from risk.validation import MarketDataValidation
from utils.error_handler import handle_error

class ValidationContext:
    """Test context that mimics production environment"""
    def __init__(self):
        self.logger = logging.getLogger("Validation")
        self.config = {
            "timeframe": "15m",
            "emergency_stop_pct": Decimal("-3"),
            "ratchet_thresholds": [2, 4, 6],
            "ratchet_lock_ins": [1, 2, 3],
            "kelly_scaling": Decimal("0.5"),
            "initial_balance": Decimal("10000"),
            "risk_factor": Decimal("0.1"),
            "max_position_size": Decimal("0.1"),
            "min_position_size": Decimal("0.01"),
            "max_positions": 3,
            "max_leverage": Decimal("2.0"),
            "max_correlation": Decimal("0.7"),
            "max_drawdown": Decimal("0.1"),
            "max_daily_loss": Decimal("0.03"),
            "market_list": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        }
        self.risk_limits = RiskLimits.from_config(self.config)
        self.risk_manager = RiskManager(self)
        self.db_pool = "data/test.db"

async def validate_signal_generation(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate ML and GA signal generation"""
    results = {"ml_signals": 0, "ga_signals": 0, "conflicts": 0, "warnings": []}
    
    try:
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
                results["warnings"].append("No recent candle data found")
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
                    results["warnings"].append(
                        f"Signal conflict for {symbol}: ML={ml_dict[symbol]}, GA={ga_dict[symbol]}"
                    )
                    
    except Exception as e:
        handle_error(e, "validation_tests.validate_signal_generation", logger=ctx.logger)
        results["warnings"].append(f"Error during signal validation: {str(e)}")
        
    return results

async def validate_position_sizing(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate position sizing calculations"""
    results = {"positions": [], "warnings": []}
    
    try:
        test_cases = [
            {
                "balance": Decimal("10000"),
                "price": Decimal("50000"),
                "volatility": Decimal("0.02"),
                "probability": Decimal("0.6"),
                "expected_return": Decimal("0.03")
            },
            {
                "balance": Decimal("5000"),
                "price": Decimal("1800"),
                "volatility": Decimal("0.015"),
                "probability": Decimal("0.55"),
                "expected_return": Decimal("0.02")
            }
        ]
        
        for case in test_cases:
            kelly = calculate_kelly_fraction(
                case["probability"],
                case["expected_return"],
                ctx.config["risk_factor"]
            )
            
            position_size = calculate_position_size(
                case["balance"],
                kelly,
                case["price"],
                case["volatility"]
            )
            
            position_value = position_size * case["price"]
            leverage = position_value / case["balance"]
            
            position = {
                "position_size": float(position_size),
                "position_value": float(position_value),
                "leverage": float(leverage),
                "kelly": float(kelly)
            }
            
            results["positions"].append(position)
            
            if leverage > ctx.config["max_leverage"]:
                results["warnings"].append(
                    f"High leverage warning: {float(leverage):.2f}x"
                )
                
    except Exception as e:
        handle_error(e, "validation_tests.validate_position_sizing", logger=ctx.logger)
        results["warnings"].append(f"Error during position sizing validation: {str(e)}")
        
    return results

async def validate_risk_limits(ctx: ValidationContext) -> Dict[str, Any]:
    """Validate risk limit enforcement"""
    results = {"tests": [], "warnings": []}
    
    try:
        # Test position count limit
        positions = [
            Position("BTC/USDT", "long", Decimal("0.1"), Decimal("50000")),
            Position("ETH/USDT", "long", Decimal("1"), Decimal("3000")),
            Position("SOL/USDT", "long", Decimal("10"), Decimal("100"))
        ]
        
        for pos in positions:
            is_valid, reason = ctx.risk_manager.validate_new_position(
                pos.symbol,
                pos.size,
                pos.entry_price
            )
            results["tests"].append({
                "type": "position_limit",
                "symbol": pos.symbol,
                "valid": is_valid,
                "reason": reason
            })
            
        # Test correlation limit
        correlation_test = ctx.risk_manager.validate_correlation(
            "WETH/USDT",  # Should be highly correlated with ETH
            "long"
        )
        results["tests"].append({
            "type": "correlation",
            "valid": correlation_test[0],
            "reason": correlation_test[1]
        })
        
        # Test drawdown limit
        ctx.risk_manager.portfolio.peak_value = Decimal("10000")
        ctx.risk_manager.portfolio.current_value = Decimal("9000")
        drawdown_test = ctx.risk_manager.validate_drawdown()
        results["tests"].append({
            "type": "drawdown",
            "valid": drawdown_test[0],
            "reason": drawdown_test[1]
        })
        
    except Exception as e:
        handle_error(e, "validation_tests.validate_risk_limits", logger=ctx.logger)
        results["warnings"].append(f"Error during risk limit validation: {str(e)}")
        
    return results

async def run_validation_suite() -> None:
    """Run all validation tests"""
    ctx = ValidationContext()
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("\nStarting Trading Bot Validation Suite...")
        
        # Signal Generation
        print("\nValidating Signal Generation...")
        signal_results = await validate_signal_generation(ctx)
        print(f"ML Signals: {signal_results['ml_signals']}")
        print(f"GA Signals: {signal_results['ga_signals']}")
        print(f"Signal Conflicts: {signal_results['conflicts']}")
        if signal_results["warnings"]:
            print("\nSignal Warnings:")
            for warning in signal_results["warnings"]:
                print(f"- {warning}")
        
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
        
        # Risk Limits
        print("\nValidating Risk Limits...")
        risk_results = await validate_risk_limits(ctx)
        for test in risk_results["tests"]:
            print(f"\n{test['type'].title()} Test:")
            print(f"Valid: {test['valid']}")
            print(f"Reason: {test['reason']}")
        
        if risk_results["warnings"]:
            print("\nRisk Warnings:")
            for warning in risk_results["warnings"]:
                print(f"- {warning}")
                
    except Exception as e:
        handle_error(e, "validation_tests.run_validation_suite", logger=ctx.logger)
        print(f"\nError during validation suite: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_validation_suite())