#! /usr/bin/env python3
# src/signals/ga_synergy.py
"""
Module: src.signals
Provides genetic algorithm synergy.
"""
import time
import uuid

# import required modules
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bot_types.base_types import Validatable, ValidationResult
from indicators.indicators_pta import compute_indicators
from trading.math import (
    calculate_expected_value,
    calculate_kelly_fraction,
    predict_next_return,
)
from utils.error_handler import handle_error
from utils.exceptions import InvalidOrderError


# genetic algorithm trading rule class that defines the genetic algorithm trading rule
@dataclass
class GeneticRule(Validatable):
    """Genetic algorithm trading rule"""

    # Required fields (no defaults)
    rule_id: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]

    # Optional fields (with defaults)
    fitness_score: Decimal = Decimal("0")
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_rate: Decimal = Decimal("0.1")
    crossover_rate: Decimal = Decimal("0.8")
    last_update: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # validate the genetic rule parameters
    def validate(self) -> ValidationResult:
        """Validate genetic rule parameters"""
        try:
            if not self.rule_id:
                return ValidationResult(
                    is_valid=False, error_message="Rule ID cannot be empty"
                )

            if self.mutation_rate < 0 or self.mutation_rate > 1:
                return ValidationResult(
                    is_valid=False,
                    error_message="Mutation rate must be between 0 and 1",
                )

            if self.crossover_rate < 0 or self.crossover_rate > 1:
                return ValidationResult(
                    is_valid=False,
                    error_message="Crossover rate must be between 0 and 1",
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Genetic rule validation failed: {str(e)}",
            )


# genetic algorithm signal class that defines the genetic algorithm signal
@dataclass
class GASignal:
    symbol: str
    action: str
    price: Decimal
    quantity: Decimal


# generate the genetic algorithm signals
def generate_ga_signals(data: dict) -> GASignal:
    """Generate trading signals using genetic algorithm rules"""
    action = data.get("action")
    if action not in ["buy", "sell"]:
        raise InvalidOrderError(f"Invalid action: {action}")
    return GASignal(
        symbol=data["symbol"],
        action=action,
        price=Decimal(data["price"]),
        quantity=Decimal(data["quantity"]),
    )


def generate_ga_signals(
    market_data: pd.DataFrame, population: List[GeneticRule]
) -> List[Dict[str, Any]]:
    """Generate trading signals using genetic algorithm rules"""
    try:
        signals = []

        for rule in population:
            # Compute technical indicators
            indicators = compute_indicators(market_data, rule.parameters)

            # Generate signal based on rule conditions
            signal = evaluate_rule_conditions(
                rule=rule, data=indicators, parameters=rule.parameters
            )

            if signal:
                # Calculate signal metrics
                expected_value = calculate_expected_value(
                    price=Decimal(str(market_data["close"].iloc[-1])),
                    probability=signal["probability"],
                    target=signal["take_profit"],
                    stop=signal["stop_loss"],
                )

                kelly = calculate_kelly_fraction(
                    win_prob=signal["probability"],
                    win_loss_ratio=abs(
                        (signal["take_profit"] - market_data["close"].iloc[-1])
                        / (signal["stop_loss"] - market_data["close"].iloc[-1])
                    ),
                )

                predicted_return = predict_next_return(
                    data=market_data, lookback=rule.parameters.get("lookback", 20)
                )

                # Enhance signal metadata
                signal.update(
                    {
                        "rule_id": rule.rule_id,
                        "expected_value": expected_value,
                        "kelly_fraction": kelly,
                        "predicted_return": predicted_return,
                        "generation": rule.generation,
                        "fitness_score": rule.fitness_score,
                        "timestamp": datetime.now().timestamp(),
                    }
                )

                signals.append(signal)

        return signals

    except Exception as e:
        handle_error(e, "generate_ga_signals")
        return []


# evaluate the trading rule conditions
def evaluate_rule_conditions(
    rule: GeneticRule, data: pd.DataFrame, parameters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Evaluate trading rule conditions"""
    try:
        # Example decision rule: compare the last close price to a threshold parameter
        last_close = data["close"].iloc[-1]
        threshold = parameters.get("threshold", 100)

        if last_close > threshold:
            # Generate a 'buy' signal
            return {
                "action": "buy",
                "take_profit": last_close * 1.02,
                "stop_loss": last_close * 0.98,
                "probability": 0.75,
            }
        else:
            # Generate a 'sell' signal
            return {
                "action": "sell",
                "take_profit": last_close * 0.98,
                "stop_loss": last_close * 1.02,
                "probability": 0.65,
            }
    except Exception as e:
        handle_error(e, "evaluate_rule_conditions")
        return None


# create the initial population of trading rules for the genetic algorithm to optimize
def create_initial_population(
    size: int, symbol: str, timeframe: str
) -> List[GeneticRule]:
    """Create initial population of trading rules"""
    try:
        population = []
        # create the initial population of trading rules for the genetic algorithm to optimize the trading rules for the genetic algorithm in the trading context
        for _ in range(size):
            rule = GeneticRule(
                rule_id=str(uuid.uuid4()),
                symbol=symbol,
                timeframe=timeframe,
                parameters={
                    "direction": np.random.choice(["long", "short"]),
                    "rsi_period": np.random.randint(10, 30),
                    "rsi_oversold": np.random.randint(20, 40),
                    "rsi_overbought": np.random.randint(60, 80),
                    "macd_fast": np.random.randint(8, 20),
                    "macd_slow": np.random.randint(21, 40),
                    "macd_signal": np.random.randint(5, 15),
                    "volume_threshold": np.random.uniform(1.2, 2.0),
                    "lookback": np.random.randint(10, 50),
                    "threshold": np.random.uniform(90, 110),
                },
            )
            population.append(rule)
        # return the initial population of trading rules for the genetic algorithm to optimize the trading rules for the genetic algorithm in the trading context
        return population

    except Exception as e:
        handle_error(e, "create_initial_population")
        return []


###############################
# GA Operators Implementation #
###############################

import copy
import random


def crossover_rules(parent1: GeneticRule, parent2: GeneticRule) -> GeneticRule:
    """
    Perform crossover between two GeneticRule parents to produce a child rule.
    For each parameter, randomly select from one of the parents.
    """
    child_params = {}
    keys = set(parent1.parameters.keys()).union(set(parent2.parameters.keys()))
    for key in keys:
        # Randomly select parameter value from parent1 or parent2
        if random.random() < 0.5:
            child_params[key] = parent1.parameters.get(key, parent2.parameters.get(key))
        else:
            child_params[key] = parent2.parameters.get(key, parent1.parameters.get(key))

    child = GeneticRule(
        rule_id=str(uuid.uuid4()),
        symbol=parent1.symbol,
        timeframe=parent1.timeframe,
        parameters=child_params,
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_ids=[parent1.rule_id, parent2.rule_id],
    )
    return child


def mutate_rule(rule: GeneticRule, mutation_rate: float = 0.1) -> GeneticRule:
    """
    Mutate a GeneticRule's parameters according to the mutation_rate.
    For numeric parameters, apply a small random perturbation.
    """
    mutated_rule = copy.deepcopy(rule)
    mutated_params = {}
    for key, value in mutated_rule.parameters.items():
        if isinstance(value, (int, float)):
            if random.random() < mutation_rate:
                # Apply a perturbation of up to ±10%
                perturbation = value * random.uniform(-0.1, 0.1)
                mutated_params[key] = value + perturbation
            else:
                mutated_params[key] = value
        else:
            # Non-numeric parameters remain unchanged
            mutated_params[key] = value
    mutated_rule.parameters = mutated_params
    mutated_rule.mutation_rate = Decimal(str(mutation_rate))
    mutated_rule.last_update = time.time()
    return mutated_rule


def selection_operator(
    population: List[GeneticRule], selection_size: int
) -> List[GeneticRule]:
    """
    Select the top rules from the population based on fitness_score.
    """
    sorted_population = sorted(
        population, key=lambda rule: float(rule.fitness_score), reverse=True
    )
    return sorted_population[:selection_size]


def evolve_population(
    population: List[GeneticRule], selection_size: int, mutation_rate: float
) -> List[GeneticRule]:
    """
    Evolve the population by selecting top performers and generating new offspring via crossover and mutation.

    Steps:
      1. Select top rules using the selection_operator.
      2. Generate offspring until the population size is restored using crossover and mutation.
    """
    selected = selection_operator(population, selection_size)
    new_population = selected.copy()
    while len(new_population) < len(population):
        parents = random.sample(selected, 2)
        child = crossover_rules(parents[0], parents[1])
        child = mutate_rule(child, mutation_rate)
        new_population.append(child)
    return new_population

    # End of GA Operators implementation

    mutated_rule.parameters = mutated_params
