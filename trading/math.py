#!/usr/bin/env python3
"""
trading/math.py - Core trading mathematics and relationships
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from decimal import Decimal, InvalidOperation
from utils.error_handler import handle_error
from utils.numeric_handler import NumericHandler
from trading.exceptions import MathError

def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate logarithmic returns series"""
    try:
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])
    except Exception as e:
        handle_error(e, "math.calculate_log_returns")
        return np.array([])

def estimate_ar1_coefficient(returns: np.ndarray) -> float:
    """Estimate AR(1) coefficient for return prediction"""
    try:
        if len(returns) < 2:
            return 0.0
        return float(np.corrcoef(returns[:-1], returns[1:])[0,1])
    except Exception as e:
        handle_error(e, "math.estimate_ar1_coefficient")
        return 0.0

def predict_next_return(current_return: float, ar1_coef: float) -> float:
    """Predict next period return using AR(1) model"""
    try:
        return ar1_coef * current_return
    except Exception as e:
        handle_error(e, "math.predict_next_return")
        return 0.0

def estimate_volatility(returns: np.ndarray, min_vol: float = 0.001) -> float:
    """
    Estimate forward volatility with minimum threshold
    """
    try:
        if len(returns) < 2:
            return min_vol
        vol = float(np.std(returns))
        return max(vol, min_vol)
    except Exception as e:
        handle_error(e, "math.estimate_volatility")
        return min_vol

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
    try:
        trend_up = ema_short > ema_long
        return_up = predicted_return > 0
        
        if trend_up and return_up:
            return min(base_prob + 0.2, 1.0)
        return max(base_prob, 0.0)
    except Exception as e:
        handle_error(e, "math.calculate_trend_probability")
        return base_prob

def calculate_expected_value(
    current_price: Decimal,
    predicted_return: Decimal,
    up_prob: Decimal,
    stop_loss_pct: Decimal,
    transaction_cost: Decimal
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate trade expected value and profit/loss targets
    Returns: (expected_value, win_target, loss_target)
    """
    try:
        # Calculate profit target from predicted return
        win_target = current_price * (Decimal(str(np.exp(float(predicted_return)))) - Decimal("1"))
        
        # Calculate stop loss level
        loss_target = current_price * stop_loss_pct
        
        # Expected value calculation
        ev = (
            up_prob * (win_target - transaction_cost) - 
            (Decimal("1") - up_prob) * (loss_target + transaction_cost)
        )
        
        return ev, win_target, loss_target
    except Exception as e:
        handle_error(e, "math.calculate_expected_value")
        return Decimal("0"), Decimal("0"), Decimal("0")

def calculate_kelly_fraction(
    up_prob: Decimal,
    win_target: Decimal,
    loss_target: Decimal
) -> Decimal:
    """Calculate optimal Kelly fraction for position sizing"""
    try:
        if win_target <= Decimal("0") or loss_target >= Decimal("0"):
            return Decimal("0")
            
        ratio = abs(loss_target / win_target)
        kelly = up_prob - ((Decimal("1") - up_prob) * ratio)
        
        # Cap kelly fraction
        return max(Decimal("0"), min(kelly, Decimal("1")))
    except Exception as e:
        handle_error(e, "math.calculate_kelly_fraction")
        return Decimal("0")

def calculate_position_size(
    account_balance: Decimal,
    kelly_fraction: Decimal,
    current_price: Decimal,
    volatility: Decimal,
    risk_factor: Decimal = Decimal("0.1")
) -> Decimal:
    """
    Calculate position size using:
    - Kelly fraction
    - Account balance
    - Price volatility
    - Risk factor
    """
    try:
        if volatility <= Decimal("0") or current_price <= Decimal("0"):
            return Decimal("0")
            
        base_size = (account_balance * risk_factor) / (current_price * volatility)
        return base_size * kelly_fraction
    except Exception as e:
        handle_error(e, "math.calculate_position_size")
        return Decimal("0")

class MathHandler:
    def __init__(self):
        self.nh = NumericHandler()
        self.logger = logging.getLogger(__name__)

    def calculate_kelly_fraction(self, win_prob: Decimal, win_loss_ratio: Decimal) -> Decimal:
        try:
            return self.nh.safe_divide(win_prob - (Decimal('1') - win_prob) / win_loss_ratio, Decimal('1'))
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            raise MathError(f"Error calculating Kelly fraction: {e}")

    def calculate_position_size(self, account_size: Decimal, risk_per_trade: Decimal, stop_loss: Decimal) -> Decimal:
        try:
            return self.nh.safe_divide(account_size * risk_per_trade, stop_loss)
        except (InvalidOperation, DivisionByZero) as e:
            self.logger.error(f"Error calculating position size: {e}")
            raise MathError(f"Error calculating position size: {e}")

    def calculate_expected_value(self, win_prob: Decimal, win_amount: Decimal, loss_amount: Decimal) -> Decimal:
        try:
            return (win_prob * win_amount) - ((Decimal('1') - win_prob) * loss_amount)
        except InvalidOperation as e:
            self.logger.error(f"Error calculating expected value: {e}")
            raise MathError(f"Error calculating expected value: {e}")