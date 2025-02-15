#! /usr/bin/env python3
#src/trading/math.py
"""
Module: src.trading
Provides mathematical functions for trading.
"""
import numpy as np
import asyncio
import logging
from decimal import Decimal, InvalidOperation, DivisionByZero
from typing import Tuple, Dict

from utils.error_handler import handle_error, handle_error_async
from utils.exceptions import MathError

#calculate the log returns of the prices
def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate logarithmic returns series."""
    try:
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])
    except Exception as e:
        asyncio.create_task(handle_error_async(e, "math.calculate_log_returns", logging.getLogger(__name__)))
        return np.array([])
#estimate the ar1 coefficient for return prediction
def estimate_ar1_coefficient(returns: np.ndarray) -> float:
    """Estimate AR(1) coefficient for return prediction."""
    try:
        if len(returns) < 2:
            return 0.0
        return float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    except Exception as e:
        handle_error(e, "math.estimate_ar1_coefficient")
        return 0.0
#predict the next period return using the ar1 model
def predict_next_return(current_return: float, ar1_coef: float) -> float:
    """Predict next period return using AR(1) model."""
    try:
        return ar1_coef * current_return
    except Exception as e:
        handle_error(e, "math.predict_next_return")
        return 0.0
#estimate the volatility of the returns
def estimate_volatility(returns: np.ndarray, min_vol: float = 0.001) -> float:
    """
    Estimate forward volatility with a minimum threshold.
    """
    try:
        if len(returns) < 2:
            return min_vol
        vol = float(np.std(returns))
        return max(vol, min_vol)
    except Exception as e:
        handle_error(e, "math.estimate_volatility")
        return min_vol
#calculate the probability of an upward move based on the trend and predicted return
def calculate_trend_probability(
    predicted_return: float,
    ema_short: float,
    ema_long: float,
    base_prob: float = 0.4
) -> float:
    """
    Calculate the probability of an upward move based on trend and predicted return.
    Base probability is 0.4, increasing to 0.6 if both trend and return are positive.
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
#calculate the expected value of a trade
def calculate_expected_value(
    current_price: Decimal,
    predicted_return: Decimal,
    up_prob: Decimal,
    stop_loss_pct: Decimal,
    transaction_cost: Decimal
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate trade expected value and profit/loss targets.
    Returns a tuple: (expected_value, win_target, loss_target).
    """
    try:
        win_target = current_price * (Decimal(str(np.exp(float(predicted_return)))) - Decimal("1"))
        loss_target = current_price * stop_loss_pct
        ev = (
            up_prob * (win_target - transaction_cost) -
            (Decimal("1") - up_prob) * (loss_target + transaction_cost)
        )
        return ev, win_target, loss_target
    except Exception as e:
        handle_error(e, "math.calculate_expected_value")
        return Decimal("0"), Decimal("0"), Decimal("0")
#calculate the kelly fraction for position sizing
def calculate_kelly_fraction(
    up_prob: Decimal,
    win_target: Decimal,
    loss_target: Decimal
) -> Decimal:
    """Calculate the optimal Kelly fraction for position sizing."""
    try:
        if win_target <= Decimal("0") or loss_target >= Decimal("0"):
            return Decimal("0")
        ratio = abs(loss_target / win_target)
        kelly = up_prob - ((Decimal("1") - up_prob) * ratio)
        return max(Decimal("0"), min(kelly, Decimal("1")))
    except Exception as e:
        handle_error(e, "math.calculate_kelly_fraction")
        return Decimal("0")
#calculate the position size based on the kelly criterion
def calculate_position_size(
    account_balance: Decimal,
    kelly_fraction: Decimal,
    current_price: Decimal,
    volatility: Decimal,
    risk_factor: Decimal = Decimal("0.1")
) -> Decimal:
    """
    Calculate the position size using the Kelly criterion, account balance, price volatility, and a risk factor.
    """
    try:
        if volatility <= Decimal("0") or current_price <= Decimal("0"):
            return Decimal("0")
        base_size = (account_balance * risk_factor) / (current_price * volatility)
        return base_size * kelly_fraction
    except Exception as e:
        handle_error(e, "math.calculate_position_size")
        return Decimal("0")
#safe divide function that safely divides two decimal numbers and returns 0 on division errors
def safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Safely divide two Decimal numbers, returning 0 on division errors."""
    try:
        return numerator / denominator
    except DivisionByZero:
        return Decimal('0')
    except InvalidOperation:
        return Decimal('0')
#object-oriented math handler class that provides alternative methods for trading math calculations
class MathHandler:
    """
    MathHandler encapsulates alternative methods for trading math calculations.
    Use the free functions for vectorized computations and this class for object-oriented handling.
    """
    def safe_divide(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        return safe_divide(numerator, denominator)

    def calculate_expected_value(self, probability: Decimal, win_amount: Decimal, loss_amount: Decimal) -> Decimal:
        """
        Alternative expected value calculation.
        Formula: (win_amount - loss_amount) * probability - loss_amount * (1 - probability)
        """
        try:
            return (win_amount - loss_amount) * probability - loss_amount * (Decimal('1') - probability)
        except (InvalidOperation, Exception) as e:
            raise MathError(f"Error calculating expected value: {e}")
    #calculate the kelly fraction for position sizing
    def calculate_kelly_fraction(self, win_amount: Decimal, loss_amount: Decimal) -> Decimal:
        """
        Alternative Kelly fraction calculation.
        """
        try:
            if win_amount <= Decimal("0") or loss_amount >= Decimal("0"):
                return Decimal("0")
            ratio = abs(loss_amount / win_amount)
            kelly = win_amount - ((Decimal("1") - win_amount) * ratio)
            return max(Decimal("0"), min(kelly, Decimal("1")))
        except (InvalidOperation, Exception) as e:
            raise MathError(f"Error calculating Kelly fraction: {e}")
    #calculate the position size based on the kelly criterion
    def calculate_position_size(self, account_balance: Decimal, kelly_fraction: Decimal, current_price: Decimal, volatility: Decimal) -> Decimal:
        """
        Alternative position size calculation using the Kelly criterion.
        """
        try:
            if volatility <= Decimal("0") or current_price <= Decimal("0"):
                return Decimal("0")
            base_size = (account_balance * Decimal("0.1")) / (current_price * volatility)
            return base_size * kelly_fraction
        except (InvalidOperation, Exception) as e:
            raise MathError(f"Error calculating position size: {e}")
    #calculate the correlation between the position and the market
    def calculate_position_correlation(self, symbol: str, correlations: Dict[str, Decimal]) -> Decimal:
        """
        Retrieve the correlation for a given symbol from a dictionary.
        """
        try:
            return correlations[symbol]
        except KeyError as e:
            raise e
