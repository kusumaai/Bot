#!/usr/bin/env python3
"""
trading/math.py - Core trading mathematics and relationships
"""
import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate logarithmic returns series"""
    return np.log(prices[1:] / prices[:-1])

def estimate_ar1_coefficient(returns: np.ndarray) -> float:
    """Estimate AR(1) coefficient for return prediction"""
    if len(returns) < 2:
        return 0.0
    return np.corrcoef(returns[:-1], returns[1:])[0,1]

def predict_next_return(current_return: float, ar1_coef: float) -> float:
    """Predict next period return using AR(1) model"""
    return ar1_coef * current_return

def estimate_volatility(returns: np.ndarray, min_vol: float = 0.001) -> float:
    """
    Estimate forward volatility with minimum threshold
    """
    if len(returns) < 2:
        return min_vol
    vol = np.std(returns)
    return max(vol, min_vol)

def calculate_trend_probability(
    predicted_return: float,
    ema_short: float,
    ema_long: float,
    base_prob: float = 0.4
) -> float:
    """
    Calculate probability of upward move based on trend and predicted return
    Base probability is 0.4, increases to 0.6 if trend confirms return
    """
    trend_up = ema_short > ema_long
    return_up = predicted_return > 0
    
    if trend_up and return_up:
        return base_prob + 0.2
    return base_prob

def calculate_expected_value(
    current_price: float,
    predicted_return: float,
    up_prob: float,
    stop_loss_pct: float,
    transaction_cost: float
) -> Tuple[float, float, float]:
    """
    Calculate trade expected value and profit/loss targets
    Returns: (expected_value, win_target, loss_target)
    """
    # Calculate profit target from predicted return
    win_target = current_price * (np.exp(predicted_return) - 1)
    
    # Calculate stop loss level
    loss_target = current_price * stop_loss_pct
    
    # Expected value calculation
    ev = (
        up_prob * (win_target - transaction_cost) - 
        (1 - up_prob) * (loss_target + transaction_cost)
    )
    
    return ev, win_target, loss_target

def calculate_kelly_fraction(
    up_prob: float,
    win_target: float,
    loss_target: float
) -> float:
    """Calculate optimal Kelly fraction for position sizing"""
    if win_target <= 0 or loss_target >= 0:
        return 0.0
        
    ratio = abs(loss_target / win_target)
    kelly = up_prob - ((1 - up_prob) * ratio)
    
    # Cap kelly fraction
    return max(0.0, min(kelly, 1.0))

def calculate_position_size(
    account_balance: float,
    kelly_fraction: float,
    current_price: float,
    volatility: float,
    risk_factor: float = 0.1
) -> float:
    """
    Calculate position size using:
    - Kelly fraction
    - Account balance
    - Price volatility
    - Risk factor
    """
    if volatility <= 0 or current_price <= 0:
        return 0.0
        
    base_size = (account_balance * risk_factor) / (current_price * volatility)
    return base_size * kelly_fraction