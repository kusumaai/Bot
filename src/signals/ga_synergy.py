#! /usr/bin/env python3
# src/signals/ga_synergy_new.py
"""
Enhanced Genetic Algorithm Trading System
Combines sophisticated risk management, adaptive parameters, and comprehensive performance tracking.
Now with ML prediction stub and simulation fixes.
"""

import copy
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


# Simulated ML prediction function (replace with real ML integration)
def predict_next_return(market_data: pd.DataFrame) -> float:
    """Stub for ML prediction of next return."""
    # Simulate a predicted return based on last close change
    if len(market_data) < 2:
        return 0.0
    last_change = (
        market_data["close"].iloc[-1] - market_data["close"].iloc[-2]
    ) / market_data["close"].iloc[-2]
    return last_change * 0.5  # Dampened prediction


# Math helper functions (assumed from trading.math)
def calculate_expected_value(signal: Dict[str, Any], price: float) -> Decimal:
    """Calculate expected value of a trade."""
    prob = signal.get("probability", 0.5)
    win = (signal["take_profit"] - price) * prob
    loss = (price - signal["stop_loss"]) * (1 - prob)
    return Decimal(str(win - loss))


def calculate_kelly_fraction(signal: Dict[str, Any]) -> Decimal:
    """Calculate Kelly fraction for position sizing."""
    prob = signal.get("probability", 0.5)
    win = signal["take_profit"] / signal["price"] - 1
    loss = 1 - signal["stop_loss"] / signal["price"]
    if loss == 0:
        return Decimal("0")
    edge = prob * win - (1 - prob) * loss
    return Decimal(str(edge / (win * loss))) if win * loss != 0 else Decimal("0")


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str = ""


@dataclass
class GeneticRule:
    """Enhanced genetic algorithm trading rule with parameter management and validation"""

    rule_id: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    fitness_score: Decimal = Decimal("0")
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_rate: Decimal = Decimal("0.1")
    crossover_rate: Decimal = Decimal("0.8")
    last_update: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        try:
            if not self.rule_id:
                return ValidationResult(
                    is_valid=False, error_message="Rule ID cannot be empty"
                )
            for rate_name, rate_value in [
                ("mutation_rate", self.mutation_rate),
                ("crossover_rate", self.crossover_rate),
            ]:
                if not 0 <= float(rate_value) <= 1:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"{rate_name} must be between 0 and 1",
                    )
            param_ranges = {
                "ema_fast": (5, 20),
                "ema_slow": (21, 60),
                "ma_exit": (100, 250),
                "tp_ratio": (0.01, 0.05),
                "sl_ratio": (0.005, 0.02),
                "atr_multiplier": (0.5, 3.0),
                "position_size": (0.1, 1.0),
                "trailing_stop_pct": (0.5, 3.0),
                "emergency_stop_pct": (1.0, 5.0),
                "ratchet_threshold_1": (0.5, 2.0),
                "ratchet_threshold_2": (1.5, 3.0),
                "ratchet_threshold_3": (2.5, 4.0),
                "ratchet_lockin_1": (0.2, 1.0),
                "ratchet_lockin_2": (0.8, 1.5),
                "ratchet_lockin_3": (1.2, 2.0),
            }
            for param, (min_val, max_val) in param_ranges.items():
                if param in self.parameters:
                    value = float(self.parameters[param])
                    if not min_val <= value <= max_val:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Parameter {param} must be between {min_val} and {max_val}",
                        )
            return ValidationResult(is_valid=True)
        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Validation failed: {str(e)}"
            )


@dataclass
class GASignal:
    """Enhanced trading signal with comprehensive metadata"""

    symbol: str
    action: str
    price: Decimal
    quantity: Decimal
    timestamp: float
    rule_id: str
    expected_value: Decimal
    kelly_fraction: Decimal
    predicted_return: float
    generation: int
    fitness_score: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    meta: Dict[str, Any] = field(default_factory=dict)


class TechnicalAnalysis:
    """Efficient technical analysis with result caching"""

    def __init__(self):
        self._cache = {}

    def calculate_indicators(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> pd.DataFrame:
        cache_key = (data.index[-1], tuple(sorted(parameters.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]
        indicators = pd.DataFrame(index=data.index)
        close = data["close"]
        for period in [8, 21, 55, 89, 200]:
            alpha = 2.0 / (period + 1.0)
            indicators[f"ema_{period}"] = close.ewm(alpha=alpha, adjust=False).mean()
        high, low = data["high"], data["low"]
        tr = pd.DataFrame(
            np.vstack(
                [high - low, abs(high - close.shift()), abs(low - close.shift())]
            ).max(axis=0),
            index=close.index,
        )
        indicators["atr"] = tr.rolling(14).mean()
        indicators["volume_sma"] = data["volume"].rolling(20).mean()
        indicators["volume_ratio"] = data["volume"] / indicators["volume_sma"]
        self._cache[cache_key] = indicators
        return indicators


def evaluate_rule_conditions(
    rule: GeneticRule,
    data: pd.DataFrame,
    parameters: Dict[str, Any],
    market_data: pd.DataFrame,
    commission_rate: float = 0.001,
) -> Optional[GASignal]:
    try:
        if data.empty or len(data) < 2:
            return None
        tech_analysis = TechnicalAnalysis()
        indicators = tech_analysis.calculate_indicators(data, parameters)
        last_close = Decimal(str(market_data["close"].iloc[-1]))
        atr = Decimal(str(indicators["atr"].iloc[-1]))
        ema_fast = indicators["ema_8"]
        ema_slow = indicators["ema_55"]
        ema_trend = indicators["ema_200"]
        volume_ratio = indicators["volume_ratio"].iloc[-1]

        signal = None
        if parameters.get("use_volume_filter", True) and volume_ratio < 0.5:
            logger.debug(
                f"Rule {rule.rule_id} filtered out: volume_ratio={volume_ratio}"
            )
            return None

        if (
            ema_fast.iloc[-1] > ema_slow.iloc[-1]
            and ema_fast.iloc[-2] <= ema_slow.iloc[-2]
        ):
            if ema_fast.iloc[-1] > ema_trend.iloc[-1]:
                signal = {
                    "action": "buy",
                    "stop_loss": last_close
                    - (atr * Decimal(str(parameters.get("atr_multiplier", 2)))),
                    "take_profit": last_close
                    + (
                        atr
                        * Decimal(str(parameters.get("atr_multiplier", 2)))
                        * Decimal("1.5")
                    ),
                    "probability": 0.65,
                }
        elif (
            ema_fast.iloc[-1] < ema_slow.iloc[-1]
            and ema_fast.iloc[-2] >= ema_slow.iloc[-2]
        ):
            if ema_fast.iloc[-1] < ema_trend.iloc[-1]:
                signal = {
                    "action": "sell",
                    "stop_loss": last_close
                    + (atr * Decimal(str(parameters.get("atr_multiplier", 2)))),
                    "take_profit": last_close
                    - (
                        atr
                        * Decimal(str(parameters.get("atr_multiplier", 2)))
                        * Decimal("1.5")
                    ),
                    "probability": 0.65,
                }

        if signal:
            position_size = calculate_position_size(
                capital=Decimal("10000"),
                price=float(last_close),
                atr=float(atr),
                risk_per_trade=parameters.get("risk_per_trade", Decimal("0.01")),
                parameters=parameters,
            )
            ratchet_params = {
                "thresholds": parameters.get("ratchet_thresholds", [1.0, 2.0, 3.0]),
                "lock_ins": parameters.get("ratchet_lock_ins", [0.5, 1.0, 1.5]),
                "trailing_pct": parameters.get("trailing_stop_pct", 1.5),
                "emergency_stop_pct": parameters.get("emergency_stop_pct", 2.0),
            }
            return GASignal(
                symbol=rule.symbol,
                action=signal["action"],
                price=last_close,
                quantity=position_size,
                timestamp=time.time(),
                rule_id=rule.rule_id,
                expected_value=calculate_expected_value(signal, float(last_close)),
                kelly_fraction=calculate_kelly_fraction(signal),
                predicted_return=predict_next_return(market_data),
                generation=rule.generation,
                fitness_score=rule.fitness_score,
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                meta={
                    "ratchet_params": ratchet_params,
                    "atr": float(atr),
                    "volume_ratio": float(volume_ratio),
                    "commission_rate": commission_rate,
                },
            )
        return None
    except Exception as e:
        logger.error(f"Error evaluating rule conditions: {str(e)}")
        return None


def calculate_position_size(
    capital: Decimal,
    price: float,
    atr: float,
    risk_per_trade: Decimal,
    parameters: Dict[str, Any],
) -> Decimal:
    risk_amount = capital * risk_per_trade
    atr_multiplier = Decimal(str(parameters.get("atr_multiplier", 2)))
    stop_distance = Decimal(str(atr)) * atr_multiplier
    if stop_distance == 0:
        return Decimal("0")
    position_size = (risk_amount / stop_distance).quantize(Decimal("0.00001"))
    if "kelly_fraction" in parameters:
        kelly_size = capital * Decimal(str(parameters["kelly_fraction"]))
        position_size = min(position_size, kelly_size)
    return position_size


class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.metrics = {}

    def add_trade(self, trade: Dict[str, Any]):
        self.trades.append(trade)
        self._update_metrics()

    def _update_metrics(self):
        if not self.trades:
            return
        returns = [t["return"] for t in self.trades]
        self.metrics = {
            "total_trades": len(self.trades),
            "win_rate": sum(1 for r in returns if r > 0) / len(returns),
            "avg_win": (
                np.mean([r for r in returns if r > 0])
                if any(r > 0 for r in returns)
                else 0
            ),
            "avg_loss": (
                np.mean([r for r in returns if r < 0])
                if any(r < 0 for r in returns)
                else 0
            ),
            "profit_factor": self._calculate_profit_factor(returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "max_drawdown": self._calculate_max_drawdown(),
        }

    def _calculate_profit_factor(self, returns: List[float]) -> float:
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        return (
            np.mean(returns) / np.std(returns)
            if returns and np.std(returns) != 0
            else 0
        )

    def _calculate_max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0


def generate_ga_signals(
    market_data: pd.DataFrame,
    population: List[GeneticRule],
    commission_rate: float = 0.001,
) -> List[GASignal]:
    signals = []
    tech_analysis = TechnicalAnalysis()
    for rule in population:
        try:
            indicators = tech_analysis.calculate_indicators(
                market_data, rule.parameters
            )
            signal = evaluate_rule_conditions(
                rule, indicators, rule.parameters, market_data, commission_rate
            )
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.error(f"Error generating signals for rule {rule.rule_id}: {e}")
    return signals


def crossover_rules(parent1: GeneticRule, parent2: GeneticRule) -> GeneticRule:
    child_params = {}
    param_ranges = {
        "ema_fast": (5, 20),
        "ema_slow": (21, 60),
        "ma_exit": (100, 250),
        "tp_ratio": (0.01, 0.05),
        "sl_ratio": (0.005, 0.02),
        "atr_multiplier": (0.5, 3.0),
        "position_size": (0.1, 1.0),
        "trailing_stop_pct": (0.5, 3.0),
        "emergency_stop_pct": (1.0, 5.0),
    }
    for key in set(parent1.parameters) | set(parent2.parameters):
        if isinstance(parent1.parameters.get(key), bool):
            child_params[key] = random.choice(
                [parent1.parameters.get(key), parent2.parameters.get(key)]
            )
        elif isinstance(parent1.parameters.get(key), (int, float)):
            val1 = parent1.parameters.get(key, 0)
            val2 = parent2.parameters.get(key, 0)
            if key in param_ranges:
                min_val, max_val = param_ranges[key]
                blend_ratio = random.random()
                child_params[key] = min(
                    max_val, max(min_val, val1 * blend_ratio + val2 * (1 - blend_ratio))
                )
                if isinstance(val1, int):
                    child_params[key] = int(round(child_params[key]))
            else:
                child_params[key] = random.choice([val1, val2])
    return GeneticRule(
        rule_id=str(uuid.uuid4()),
        symbol=parent1.symbol,
        timeframe=parent1.timeframe,
        parameters=child_params,
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_ids=[parent1.rule_id, parent2.rule_id],
    )


def mutate_rule(rule: GeneticRule, mutation_rate: float = 0.1) -> GeneticRule:
    mutated_rule = copy.deepcopy(rule)
    param_ranges = {
        "ema_fast": (5, 20),
        "ema_slow": (21, 60),
        "ma_exit": (100, 250),
        "tp_ratio": (0.01, 0.05),
        "sl_ratio": (0.005, 0.02),
        "atr_multiplier": (0.5, 3.0),
        "position_size": (0.1, 1.0),
        "trailing_stop_pct": (0.5, 3.0),
        "emergency_stop_pct": (1.0, 5.0),
        "ratchet_threshold_1": (0.5, 2.0),
        "ratchet_threshold_2": (1.5, 3.0),
        "ratchet_threshold_3": (2.5, 4.0),
        "ratchet_lockin_1": (0.2, 1.0),
        "ratchet_lockin_2": (0.8, 1.5),
        "ratchet_lockin_3": (1.2, 2.0),
    }
    for key, value in mutated_rule.parameters.items():
        if random.random() < mutation_rate:
            if isinstance(value, bool):
                mutated_rule.parameters[key] = not value
            elif isinstance(value, (int, float)) and key in param_ranges:
                min_val, max_val = param_ranges[key]
                mutation_size = (max_val - min_val) * 0.1
                new_value = value + random.uniform(-mutation_size, mutation_size)
                mutated_rule.parameters[key] = max(min_val, min(max_val, new_value))
                if isinstance(value, int):
                    mutated_rule.parameters[key] = int(
                        round(mutated_rule.parameters[key])
                    )
    mutated_rule.mutation_rate = Decimal(str(mutation_rate))
    mutated_rule.last_update = time.time()
    return mutated_rule


def simulate_trades(rule: GeneticRule, market_data: pd.DataFrame) -> List[float]:
    """Simulate trades to evaluate rule performance."""
    signals = generate_ga_signals(market_data, [rule])
    returns = []
    equity = Decimal("10000")
    for signal in signals:
        entry = signal.price
        exit = signal.take_profit if signal.action == "buy" else signal.stop_loss
        trade_return = (exit - entry) / entry * signal.quantity
        returns.append(float(trade_return))
        equity += trade_return * equity
    return returns


def evolve_population(
    population: List[GeneticRule],
    market_data: pd.DataFrame,
    selection_size: int,
    mutation_rate: float,
    crossover_rate: float,
) -> List[GeneticRule]:
    performance_tracker = PerformanceTracker()
    for rule in population:
        trade_returns = simulate_trades(rule, market_data)
        for r in trade_returns:
            performance_tracker.add_trade({"return": r, "timestamp": time.time()})
        fitness = (
            performance_tracker.metrics["sharpe_ratio"] * 0.4
            + performance_tracker.metrics["profit_factor"] * 0.3
            + performance_tracker.metrics["win_rate"] * 0.3
        )
        rule.fitness_score = Decimal(str(fitness or 0))

    if len(population) > 10:
        avg_fitness = sum(float(rule.fitness_score) for rule in population) / len(
            population
        )
        best_fitness = max(float(rule.fitness_score) for rule in population)
        logger.info(f"Avg Fitness: {avg_fitness}, Best Fitness: {best_fitness}")
        if best_fitness - avg_fitness < 0.1:
            mutation_rate = min(mutation_rate * 1.5, 0.4)
        else:
            mutation_rate = max(0.1, mutation_rate * 0.9)

    selected = sorted(population, key=lambda x: float(x.fitness_score), reverse=True)[
        :selection_size
    ]
    new_population = selected.copy()
    while len(new_population) < len(population):
        if random.random() < crossover_rate and len(selected) >= 2:
            parents = random.sample(selected, 2)
            child = crossover_rules(parents[0], parents[1])
            child = mutate_rule(child, mutation_rate)
        else:
            parent = random.choice(selected)
            child = mutate_rule(parent, mutation_rate)
        new_population.append(child)
    return new_population


def create_initial_population(
    size: int, symbol: str, timeframe: str
) -> List[GeneticRule]:
    population = []
    param_ranges = {
        "ema_fast": (5, 20),
        "ema_slow": (21, 60),
        "ma_exit": (100, 250),
        "tp_ratio": (0.01, 0.1),
        "sl_ratio": (0.005, 0.02),
        "atr_multiplier": (1.0, 5.0),
        "position_size": (0.1, 1.0),
        "volume_ma_period": (10, 30),
        "trailing_stop_pct": (0.5, 3.0),
        "emergency_stop_pct": (1.0, 5.0),
        "ratchet_threshold_1": (0.5, 2.0),
        "ratchet_threshold_2": (1.5, 3.0),
        "ratchet_threshold_3": (2.5, 4.0),
        "ratchet_lockin_1": (0.2, 1.0),
        "ratchet_lockin_2": (0.8, 1.5),
        "ratchet_lockin_3": (1.2, 2.0),
    }
    seed_params = {
        "ema_fast": 8,
        "ema_slow": 55,
        "ma_exit": 200,
        "tp_ratio": 0.02,
        "sl_ratio": 0.01,
        "atr_multiplier": 2.0,
        "position_size": 0.5,
        "use_volume_filter": True,
        "volume_ma_period": 20,
        "base_probability": 0.6,
        "trailing_stop_pct": 1.5,
        "emergency_stop_pct": 2.0,
        "ratchet_thresholds": [1.0, 2.0, 3.0],
        "ratchet_lock_ins": [0.5, 1.0, 1.5],
    }
    population.append(
        GeneticRule(
            rule_id=str(uuid.uuid4()),
            symbol=symbol,
            timeframe=timeframe,
            parameters=seed_params,
        )
    )
    for _ in range(size - 1):
        params = {}
        for key, (min_val, max_val) in param_ranges.items():
            params[key] = random.uniform(min_val, max_val)
            if key in ["ema_fast", "ema_slow", "ma_exit", "volume_ma_period"]:
                params[key] = int(params[key])
        params["use_volume_filter"] = random.choice([True, False])
        params["base_probability"] = random.uniform(0.5, 0.7)
        params["ratchet_thresholds"] = [
            params[f"ratchet_threshold_{i}"] for i in range(1, 4)
        ]
        params["ratchet_lock_ins"] = [
            params[f"ratchet_lockin_{i}"] for i in range(1, 4)
        ]
        population.append(
            GeneticRule(
                rule_id=str(uuid.uuid4()),
                symbol=symbol,
                timeframe=timeframe,
                parameters=params,
            )
        )
    return population


def main():
    """Test the GA system with dummy data."""
    # Dummy market data
    market_data = pd.DataFrame(
        {
            "close": np.random.normal(100, 5, 500).cumsum(),
            "high": np.random.normal(101, 5, 500).cumsum(),
            "low": np.random.normal(99, 5, 500).cumsum(),
            "volume": np.random.randint(100, 1000, 500),
        },
        index=pd.date_range("2023-01-01", periods=500, freq="1h"),
    )

    population = create_initial_population(20, "BTCUSD", "1h")
    for gen in range(10):
        population = evolve_population(population, market_data, 10, 0.1, 0.8)
        best_rule = max(population, key=lambda x: float(x.fitness_score))
        signals = generate_ga_signals(market_data, [best_rule])
        logger.info(
            f"Gen {gen}: Best Fitness = {best_rule.fitness_score}, Signals = {len(signals)}"
        )


if __name__ == "__main__":
    main()
