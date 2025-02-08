#!/usr/bin/env python3
"""
signals/evaluation.py - Strategy evaluation and metrics calculation
"""
import numpy as np
from typing import Dict, Any, List, Optional
from utils.error_handler import handle_error
from trading.math import (
    predict_next_return,
    calculate_trend_probability,
    calculate_expected_value,
    calculate_kelly_fraction
)
from .trading_types import TradeMetrics, TradingRule, SimulationResult, MarketState

def evaluate_condition(condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Evaluate single trading condition with error handling"""
    try:
        if not condition or not data:
            return False
            
        indicator = condition.get("indicator")
        operator = condition.get("op")
        reference = condition.get("ref")
        
        if not indicator or not operator or reference is None:
            return False
            
        value = data.get(indicator)
        if value is None:
            return False
            
        ref_value = data.get(reference, reference) if isinstance(reference, str) else reference
        if ref_value is None:
            return False
            
        if operator == ">":
            return float(value) > float(ref_value)
        elif operator == ">=":
            return float(value) >= float(ref_value)
        elif operator == "<":
            return float(value) < float(ref_value)
        elif operator == "<=":
            return float(value) <= float(ref_value)
            
        return False
        
    except Exception:
        return False

def evaluate_rule(rule: TradingRule, data: Dict[str, Any], market_state: MarketState) -> str:
    """Evaluate trading rule and return signal direction"""
    try:
        if not rule or not data or not market_state:
            return ""
            
        # Check buy conditions
        buy_signal = all(
            evaluate_condition(cond, data)
            for cond in rule.buy_conditions
        )
        
        # Check sell conditions
        sell_signal = all(
            evaluate_condition(cond, data)
            for cond in rule.sell_conditions
        )
        
        if not (buy_signal or sell_signal):
            return ""
            
        # Validate with prediction
        predicted_return = predict_next_return(
            market_state.current_return,
            market_state.ar1_coef
        )
        
        if buy_signal and predicted_return > 0:
            return "long"
        elif sell_signal and predicted_return < 0:
            return "short"
            
        return ""
        
    except Exception as e:
        handle_error(e, "evaluation.evaluate_rule", logger=None)
        return ""

def calculate_metrics(trades: List[Dict[str, Any]]) -> TradeMetrics:
    """Calculate comprehensive trading metrics"""
    if not trades:
        return TradeMetrics()
    
    returns = [t.get('return', 0) for t in trades]
    kelly_fracs = [t.get('kelly_fraction', 0) for t in trades]
    position_sizes = [t.get('position_size', 0) for t in trades]
    
    winning_trades = sum(1 for r in returns if r > 0)
    
    return TradeMetrics(
        win_rate=winning_trades / len(trades),
        profit_factor=_calculate_profit_factor(returns),
        sharpe_ratio=_calculate_sharpe_ratio(returns),
        max_drawdown=_calculate_max_drawdown(returns),
        exposure_ratio=_calculate_exposure_ratio(trades),
        avg_trade_return=np.mean(returns),
        total_trades=len(trades),
        consecutive_losses=_calculate_max_consecutive_losses(returns),
        kelly_fraction=np.mean(kelly_fracs) if kelly_fracs else 0,
        avg_leverage=np.mean(position_sizes) if position_sizes else 0
    )

def _calculate_profit_factor(returns: List[float]) -> float:
    """Calculate ratio of gross profits to gross losses"""
    if not returns:
        return 0.0
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    return gross_profit / gross_loss if gross_loss > 0 else 0.0

def _calculate_sharpe_ratio(returns: List[float]) -> float:
    """Calculate annualized Sharpe ratio"""
    if not returns:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

def _calculate_max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown percentage"""
    if not returns:
        return 0.0
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return float(np.max(drawdowns))

def _calculate_exposure_ratio(trades: List[Dict[str, Any]]) -> float:
    """Calculate ratio of time in market"""
    if not trades:
        return 0.0
    total_bars = sum(t.get('hold_periods', 0) for t in trades)
    return total_bars / len(trades)

def _calculate_max_consecutive_losses(returns: List[float]) -> int:
    """Calculate maximum consecutive losing trades"""
    current = maximum = 0
    for r in returns:
        if r < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum

def calculate_fitness(metrics: TradeMetrics, ctx: Any) -> float:
    """Calculate overall strategy fitness score"""
    # Minimum requirements
    if metrics.total_trades < ctx.config.get("min_trades", 20):
        return 0.0

    # Component weights
    weights = {
        "sharpe": 0.3,
        "kelly": 0.2,
        "drawdown": 0.2,
        "win_rate": 0.15,
        "profit_factor": 0.15
    }
    
    # Calculate components
    components = {
        "sharpe": min(1.0, metrics.sharpe_ratio / 3.0),
        "kelly": metrics.kelly_fraction,
        "drawdown": 1.0 - metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": min(1.0, metrics.profit_factor / 3.0)
    }
    
    # Combined score
    fitness = sum(weights[k] * components[k] for k in weights)
    
    # Reality checks
    if metrics.profit_factor > 3.0:
        fitness *= 0.5  # Penalize unrealistic performance
        
    return min(fitness, ctx.config.get("max_fitness", 10.0))

def simulate_rule(
    rule: TradingRule,
    candles: List[Dict[str, Any]],
    market_state: MarketState,
    ctx: Any
) -> SimulationResult:
    """Simulate trading rule on historical data"""
    if not rule or not candles or not market_state:
        return SimulationResult(0.0, TradeMetrics(), [], ["Invalid input"])
        
    try:
        position = None
        trades = []
        warnings = []
        
        for i in range(1, len(candles)):
            candle = candles[i]
            
            if position is None:
                # Check for entry
                signal = evaluate_rule(rule, candle, market_state)
                
                if signal:
                    # Calculate signal parameters
                    predicted_return = predict_next_return(
                        market_state.current_return,
                        market_state.ar1_coef
                    )
                    
                    trend_prob = calculate_trend_probability(
                        predicted_return,
                        candle.get("EMA_8", 0),
                        candle.get("EMA_21", 0)
                    )
                    
                    ev, win_target, loss_target = calculate_expected_value(
                        candle["close"],
                        predicted_return,
                        trend_prob,
                        ctx.config.get("stop_loss_pct", 0.02),
                        ctx.config.get("transaction_cost", 0.001)
                    )
                    
                    if ev > 0:
                        kelly_frac = calculate_kelly_fraction(
                            trend_prob,
                            win_target,
                            loss_target
                        )
                        
                        position = signal
                        trades.append({
                            "entry_price": candle["close"],
                            "position": position,
                            "expected_value": ev,
                            "kelly_fraction": kelly_frac,
                            "probability": trend_prob,
                            "entry_time": i
                        })
            
            else:
                # Check for exit
                pnl = (
                    (candle["close"] - trades[-1]["entry_price"]) / trades[-1]["entry_price"]
                    if position == "long"
                    else (trades[-1]["entry_price"] - candle["close"]) / trades[-1]["entry_price"]
                )
                
                should_exit = (
                    pnl <= -ctx.config.get("stop_loss_pct", 0.02) or
                    pnl >= ctx.config.get("take_profit_pct", 0.03) or
                    (i - trades[-1]["entry_time"]) >= ctx.config.get("max_hold_bars", 48)
                )
                
                if should_exit:
                    trades[-1].update({
                        "exit_price": candle["close"],
                        "return": pnl,
                        "hold_periods": i - trades[-1]["entry_time"]
                    })
                    position = None

        # Calculate final metrics
        metrics = calculate_metrics(trades)
        fitness = calculate_fitness(metrics, ctx)
        
        return SimulationResult(fitness, metrics, trades, warnings)

    except Exception as e:
        handle_error(e, "evaluation.simulate_rule", logger=ctx.logger)
        return SimulationResult(0.0, TradeMetrics(), [], [str(e)])