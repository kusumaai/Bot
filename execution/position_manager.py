#!/usr/bin/env python3
"""
Module: execution/position_manager.py
Handles position sizing and management
"""

import datetime
from typing import Dict, Any
from accounting.accounting import get_free_balance
from trading.math import (
    predict_next_return,
    calculate_trend_probability,
    calculate_expected_value,
    calculate_kelly_fraction,
    calculate_position_size
)

def calculate_position_params(
    signal: Dict[str, Any],
    market_data: Dict[str, Any],
    ctx: Any
) -> Dict[str, Any]:
    """Calculate position parameters using trading math"""
    current_price = market_data["current_price"]
    volatility = market_data["volatility"]
    
    # Predict return and probability
    predicted_return = predict_next_return(
        market_data["returns"][-1],
        market_data["market_state"].ar1_coef
    )
    
    trend_prob = calculate_trend_probability(
        predicted_return,
        market_data["market_state"].ema_short,
        market_data["market_state"].ema_long
    )
    
    # Calculate expected value and targets
    ev, win_target, loss_target = calculate_expected_value(
        current_price,
        predicted_return,
        trend_prob,
        ctx.config.get("stop_loss_pct", 0.02),
        ctx.config.get("transaction_cost", 0.001)
    )
    
    # Kelly position sizing
    kelly_frac = calculate_kelly_fraction(
        trend_prob, win_target, loss_target
    )
    
    free_balance = get_free_balance(ctx.config["exchanges"][0], ctx)
    position_size = calculate_position_size(
        free_balance,
        kelly_frac,
        current_price,
        volatility,
        ctx.config.get("risk_factor", 0.1)
    )
    
    return {
        "position_size": position_size,
        "kelly_fraction": kelly_frac,
        "expected_value": ev,
        "win_target": win_target,
        "loss_target": loss_target
    }

def should_close_position(
    trade: Dict[str, Any],
    current_price: float,
    market_data: Dict[str, Any],
    ctx: Any
) -> bool:
    """Evaluate position close conditions"""
    pct_change = (
        (current_price - trade["entry_price"]) / trade["entry_price"]
        if trade["direction"] == "long"
        else (trade["entry_price"] - current_price) / trade["entry_price"]
    )
    
    age = (
        datetime.datetime.utcnow() - 
        datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
    ).total_seconds()
    
    return (
        pct_change <= ctx.config.get("emergency_stop_pct", -3) or
        (trade["direction"] == "long" and current_price <= trade["sl"]) or
        (trade["direction"] == "long" and current_price >= trade.get("tp", float("inf"))) or
        (trade["direction"] == "short" and current_price >= trade["sl"]) or
        (trade["direction"] == "short" and current_price <= trade.get("tp", 0)) or
        age >= ctx.config.get("max_hold_hours", 24) * 3600
    )