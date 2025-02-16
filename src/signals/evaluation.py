#! /usr/bin/env python3
# src/signals/evaluation.py
"""
Module: src.signals
Provides signal evaluation and metrics calculation.
"""
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.math import (
    calculate_expected_value,
    calculate_kelly_fraction,
    calculate_trend_probability,
    predict_next_return,
)
from utils.error_handler import handle_error
from utils.numeric_handler import NumericHandler

from .trading_types import MarketState, SimulationResult, TradeMetrics, TradingRule


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

        ref_value = (
            data.get(reference, reference) if isinstance(reference, str) else reference
        )
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

    except Exception as e:
        handle_error(e, "evaluation.evaluate_condition", logger=None)
        return False


def evaluate_rule(
    rule: TradingRule, data: Dict[str, Any], market_state: MarketState
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Evaluate trading rule and return signal direction with metadata"""
    try:
        if not rule or not data or not market_state:
            return "", None

        # Check buy conditions
        buy_signal = all(evaluate_condition(cond, data) for cond in rule.buy_conditions)

        # Check sell conditions only if shorts are enabled
        sell_signal = False
        if market_state.ctx.config.get("ga_settings", {}).get("allow_shorts", False):
            sell_signal = all(
                evaluate_condition(cond, data) for cond in rule.sell_conditions
            )

        if not (buy_signal or sell_signal):
            return "", None

        # Validate with prediction
        predicted_return = predict_next_return(
            market_state.current_return, market_state.ar1_coef
        )

        # Calculate signal probability and expected value
        probability = calculate_trend_probability(
            predicted_return, market_state.volatility
        )

        ev = calculate_expected_value(
            probability, predicted_return, market_state.volatility
        )

        # Calculate Kelly fraction for position sizing
        kelly = calculate_kelly_fraction(probability, predicted_return)

        signal_metadata = {
            "probability": probability,
            "expected_value": ev,
            "kelly_fraction": kelly,
            "predicted_return": predicted_return,
        }

        if buy_signal and predicted_return > 0:
            return "long", signal_metadata
        elif sell_signal and predicted_return < 0:
            return "short", signal_metadata

        return "", None

    except Exception as e:
        handle_error(e, "evaluation.evaluate_rule", logger=None)
        return "", None


def calculate_metrics(trades: List[Dict[str, Any]]) -> TradeMetrics:
    """Calculate comprehensive trading metrics"""
    try:
        if not trades:
            return TradeMetrics()

        returns = [Decimal(str(t.get("return", 0))) for t in trades]
        kelly_fracs = [Decimal(str(t.get("kelly_fraction", 0))) for t in trades]
        position_sizes = [Decimal(str(t.get("position_size", 0))) for t in trades]

        winning_trades = sum(1 for r in returns if r > 0)

        return TradeMetrics(
            win_rate=Decimal(str(winning_trades)) / Decimal(str(len(trades))),
            profit_factor=_calculate_profit_factor(returns),
            sharpe_ratio=_calculate_sharpe_ratio(returns),
            max_drawdown=_calculate_max_drawdown(returns),
            exposure_ratio=_calculate_exposure_ratio(trades),
            avg_trade_return=Decimal(str(np.mean([float(r) for r in returns]))),
            total_trades=len(trades),
            consecutive_losses=_calculate_max_consecutive_losses(returns),
            kelly_fraction=(
                Decimal(str(np.mean([float(k) for k in kelly_fracs])))
                if kelly_fracs
                else Decimal(0)
            ),
            avg_leverage=(
                Decimal(str(np.mean([float(p) for p in position_sizes])))
                if position_sizes
                else Decimal(0)
            ),
        )

    except Exception as e:
        handle_error(e, "evaluation.calculate_metrics", logger=None)
        return TradeMetrics()


def _calculate_profit_factor(returns: List[Decimal]) -> Decimal:
    """Calculate profit factor with proper decimal handling"""
    try:
        wins = sum((r for r in returns if r > 0), Decimal(0))
        losses = abs(sum((r for r in returns if r < 0), Decimal(0)))
        return wins / losses if losses != 0 else Decimal(0)
    except Exception:
        return Decimal(0)


def _calculate_sharpe_ratio(returns: List[Decimal]) -> Decimal:
    """Calculate Sharpe ratio with proper decimal handling"""
    try:
        if not returns:
            return Decimal(0)
        returns_float = [float(r) for r in returns]
        return Decimal(
            str(
                np.mean(returns_float) / np.std(returns_float)
                if np.std(returns_float) != 0
                else 0
            )
        )
    except Exception:
        return Decimal(0)


def _calculate_max_drawdown(returns: List[Decimal]) -> Decimal:
    """Calculate maximum drawdown with proper decimal handling"""
    try:
        cumulative = np.cumsum([float(r) for r in returns])
        max_dd = 0
        peak = cumulative[0]

        for value in cumulative[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)

        return Decimal(str(max_dd))
    except Exception:
        return Decimal(0)


def _calculate_exposure_ratio(trades: List[Dict[str, Any]]) -> Decimal:
    """Calculate market exposure ratio"""
    try:
        total_duration = sum(
            float(t.get("exit_time", 0)) - float(t.get("entry_time", 0)) for t in trades
        )
        if not trades:
            return Decimal(0)
        total_time = float(trades[-1].get("exit_time", 0)) - float(
            trades[0].get("entry_time", 0)
        )
        return Decimal(str(total_duration / total_time if total_time > 0 else 0))
    except Exception:
        return Decimal(0)


def _calculate_max_consecutive_losses(returns: List[Decimal]) -> int:
    """Calculate maximum consecutive losses"""
    try:
        max_losses = current_losses = 0
        for r in returns:
            if r < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        return max_losses
    except Exception:
        return 0


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
        "profit_factor": 0.15,
    }

    # Calculate components
    components = {
        "sharpe": min(1.0, metrics.sharpe_ratio / 3.0),
        "kelly": metrics.kelly_fraction,
        "drawdown": 1.0 - metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": min(1.0, metrics.profit_factor / 3.0),
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
    ctx: Any,
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
                signal, metadata = evaluate_rule(rule, candle, market_state)

                if signal:
                    # Calculate signal parameters
                    predicted_return = metadata["predicted_return"]

                    trend_prob = metadata["probability"]

                    ev, win_target, loss_target = calculate_expected_value(
                        candle["close"],
                        predicted_return,
                        trend_prob,
                        ctx.config.get("stop_loss_pct", 0.02),
                        ctx.config.get("transaction_cost", 0.001),
                    )

                    if ev > 0:
                        kelly_frac = metadata["kelly_fraction"]

                        position = signal
                        trades.append(
                            {
                                "entry_price": candle["close"],
                                "position": position,
                                "expected_value": ev,
                                "kelly_fraction": kelly_frac,
                                "probability": trend_prob,
                                "entry_time": i,
                            }
                        )

            else:
                # Check for exit
                pnl = (
                    (candle["close"] - trades[-1]["entry_price"])
                    / trades[-1]["entry_price"]
                    if position == "long"
                    else (trades[-1]["entry_price"] - candle["close"])
                    / trades[-1]["entry_price"]
                )

                should_exit = (
                    pnl <= -ctx.config.get("stop_loss_pct", 0.02)
                    or pnl >= ctx.config.get("take_profit_pct", 0.03)
                    or (i - trades[-1]["entry_time"])
                    >= ctx.config.get("max_hold_bars", 48)
                )

                if should_exit:
                    trades[-1].update(
                        {
                            "exit_price": candle["close"],
                            "return": pnl,
                            "hold_periods": i - trades[-1]["entry_time"],
                        }
                    )
                    position = None

        # Calculate final metrics
        metrics = calculate_metrics(trades)
        fitness = calculate_fitness(metrics, ctx)

        return SimulationResult(fitness, metrics, trades, warnings)

    except Exception as e:
        handle_error(e, "evaluation.simulate_rule", logger=ctx.logger)
        return SimulationResult(0.0, TradeMetrics(), [], [str(e)])


class SignalEvaluator:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.nh = NumericHandler()
        self.performance_cache: Dict[str, Dict] = {}

    async def evaluate_signal(
        self, signal: Dict, market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Evaluate trading signal with comprehensive metrics"""
        try:
            # Calculate core metrics
            win_rate = self._calculate_win_rate(signal["strategy"], market_data)
            sharpe = self._calculate_sharpe_ratio(market_data["returns"])
            expected_value = self._calculate_expected_value(
                signal["probability"], signal["target_price"], signal["current_price"]
            )

            # Validate signal strength
            if not self._validate_signal_metrics(win_rate, sharpe, expected_value):
                return None

            evaluation = {
                "timestamp": datetime.utcnow(),
                "win_rate": self.nh.round_decimal(win_rate, 4),
                "sharpe_ratio": self.nh.round_decimal(sharpe, 4),
                "expected_value": self.nh.round_decimal(expected_value, 4),
                "confidence_score": self._calculate_confidence(signal),
            }

            # Cache evaluation results
            self.performance_cache[signal["id"]] = evaluation
            return evaluation

        except Exception as e:
            self.ctx.logger.error(f"Signal evaluation failed: {e}")
            return None

    def _calculate_win_rate(self, strategy: str, data: pd.DataFrame) -> Decimal:
        """Calculate historical win rate for strategy"""
        try:
            trades = data[data["strategy"] == strategy]
            if len(trades) < 30:  # Minimum sample size
                return Decimal("0")

            wins = len(trades[trades["pnl"] > 0])
            return self.nh.to_decimal(wins / len(trades))

        except Exception:
            return Decimal("0")

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> Decimal:
        """Calculate Sharpe ratio from returns"""
        try:
            if len(returns) < 30:
                return Decimal("0")

            excess_returns = returns - risk_free_rate / 252  # Daily adjustment
            sharpe = np.sqrt(252) * (excess_returns.mean() / returns.std())
            return self.nh.to_decimal(sharpe)

        except Exception:
            return Decimal("0")

    def _calculate_expected_value(
        self, probability: float, target: Decimal, current: Decimal
    ) -> Decimal:
        """Calculate expected value of trade"""
        try:
            prob = self.nh.to_decimal(probability)
            target_return = (target - current) / current
            return prob * target_return

        except Exception:
            return Decimal("0")

    def _validate_signal_metrics(
        self, win_rate: Decimal, sharpe: Decimal, ev: Decimal
    ) -> bool:
        """Validate if signal meets minimum criteria"""
        return (
            win_rate >= Decimal("0.5")
            and sharpe >= Decimal("1.0")
            and ev > Decimal("0")
        )

    def _calculate_confidence(self, signal: Dict) -> Decimal:
        """Calculate overall confidence score"""
        try:
            # Weighted average of multiple factors
            weights = {
                "win_rate": Decimal("0.4"),
                "sharpe": Decimal("0.3"),
                "volume": Decimal("0.3"),
            }

            scores = {
                "win_rate": self.nh.to_decimal(signal.get("win_rate", 0)),
                "sharpe": self.nh.to_decimal(signal.get("sharpe", 0)),
                "volume": self.nh.to_decimal(signal.get("volume_score", 0)),
            }

            confidence = sum(weights[k] * scores[k] for k in weights)

            return self.nh.round_decimal(confidence, 4)

        except Exception:
            return Decimal("0")
