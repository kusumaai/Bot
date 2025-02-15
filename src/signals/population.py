#! /usr/bin/env python3
#src/signals/population.py
"""
Module: src.signals
Provides population management with enhanced risk controls.
"""
import random
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import logging
import time

from utils.error_handler import handle_error
from .trading_types import TradingRule
from risk.validation import validate_risk_parameters

def create_baseline_rule() -> TradingRule:
    """Create baseline strategy with proven technical conditions"""
    try:
        return TradingRule(
            buy_conditions=[
                # Primary signal: EMA8 crosses above EMA55
                {
                    "indicator": "EMA_8",
                    "op": ">",
                    "ref": "EMA_55"
                },
                # Trend confirmation: EMA21 above EMA233
                {
                    "indicator": "EMA_21",
                    "op": ">",
                    "ref": "EMA_233"
                },
                # Momentum confirmation
                {
                    "indicator": "RSI_14",
                    "op": "<",
                    "ref": 60
                },
                # Volatility filter
                {
                    "indicator": "BB_WIDTH",
                    "op": ">",
                    "ref": 0.015
                }
            ],
            sell_conditions=[
                # Primary signal: EMA8 crosses below EMA55
                {
                    "indicator": "EMA_8",
                    "op": "<",
                    "ref": "EMA_55"
                },
                # Trend confirmation: EMA21 below EMA233
                {
                    "indicator": "EMA_21",
                    "op": "<",
                    "ref": "EMA_233"
                },
                # Momentum confirmation
                {
                    "indicator": "RSI_14",
                    "op": ">",
                    "ref": 40
                },
                # Volatility filter
                {
                    "indicator": "BB_WIDTH",
                    "op": ">",
                    "ref": 0.015
                }
            ]
        )
    except Exception as e:
        handle_error(e, "population.create_baseline_rule", logger=None)
        return TradingRule(buy_conditions=[], sell_conditions=[])

def generate_random_rule(indicators: List[str], ctx: Any) -> Optional[TradingRule]:
    """Generate random trading strategy with risk validation"""
    try:
        # Get GA settings
        ga_settings = ctx.config.get("ga_settings", {})
        buy_n = ga_settings.get("buy_conditions_count", 3)
        sell_n = ga_settings.get("sell_conditions_count", 3)
        
        if buy_n < 1 or sell_n < 1:
            return None
            
        operators = [">", ">=", "<", "<="]
        
        # Generate buy conditions
        buy_conditions = []
        for _ in range(buy_n):
            indicator = random.choice(indicators)
            op = random.choice(operators)
            
            # Generate reference with proper risk limits
            if indicator in ["RSI_14", "STOCH_K", "STOCH_D"]:
                ref = round(random.uniform(20, 80), 2)
            elif "EMA" in indicator or "SMA" in indicator:
                ref = random.choice([i for i in indicators if "EMA" in i or "SMA" in i])
            else:
                ref = round(random.uniform(0, 100), 2)
                
            buy_conditions.append({
                "indicator": indicator,
                "op": op,
                "ref": ref
            })
        
        # Generate sell conditions
        sell_conditions = []
        if ctx.config.get("allow_shorts", True):
            for _ in range(sell_n):
                indicator = random.choice(indicators)
                op = random.choice(operators)
                
                if indicator in ["RSI_14", "STOCH_K", "STOCH_D"]:
                    ref = round(random.uniform(20, 80), 2)
                elif "EMA" in indicator or "SMA" in indicator:
                    ref = random.choice([i for i in indicators if "EMA" in i or "SMA" in i])
                else:
                    ref = round(random.uniform(0, 100), 2)
                    
                sell_conditions.append({
                    "indicator": indicator,
                    "op": op,
                    "ref": ref
                })
        
        return TradingRule(
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions
        )
        
    except Exception as e:
        handle_error(e, "population.generate_random_rule", logger=ctx.logger)
        return None

def initialize_population(ctx: Any) -> List[TradingRule]:
    """Initialize population with baseline and random rules"""
    try:
        # Get GA settings
        ga_settings = ctx.config.get("ga_settings", {})
        population_size = ga_settings.get("population_size", 100)
        
        # Get available indicators
        indicators = ga_settings.get("available_indicators", [
            "close", "open", "high", "low",
            "EMA_8", "EMA_21", "EMA_55", "EMA_89", "EMA_144", "EMA_233",
            "RSI_14", "STOCH_K", "STOCH_D",
            "ATR_14", "BB_WIDTH", "BB_UPPER", "BB_LOWER",
            "MACD", "MACD_SIGNAL", "MACD_HIST",
            "candle_range_pct", "candle_body_pct", "volume_ma_ratio"
        ])
        
        # Start with baseline
        population = [create_baseline_rule()]
        
        # Add random rules
        attempts = 0
        max_attempts = population_size * 2
        
        while len(population) < population_size and attempts < max_attempts:
            rule = generate_random_rule(indicators, ctx)
            if rule:
                population.append(rule)
            attempts += 1
        
        ctx.logger.info(f"Initialized population with {len(population)} members "
                       f"after {attempts} attempts")
        return population
        
    except Exception as e:
        handle_error(e, "population.initialize_population", logger=ctx.logger)
        return [create_baseline_rule()]

def tournament_select(population: List[TradingRule], tournament_size: int = 3) -> TradingRule:
    """Select single parent using tournament selection"""
    try:
        if not population:
            return create_baseline_rule()
        
        contestants = random.sample(population, min(tournament_size, len(population)))
        return max(contestants, key=lambda x: float(x.fitness) if hasattr(x, 'fitness') else 0)
        
    except Exception as e:
        handle_error(e, "population.tournament_select", logger=None)
        return create_baseline_rule()

def crossover(parent_a: TradingRule, parent_b: TradingRule) -> TradingRule:
    """Single point crossover with validation"""
    try:
        if not parent_a or not parent_b:
            return create_baseline_rule()
        
        # Crossover buy conditions
        buy_split = len(parent_a.buy_conditions) // 2
        child_buy = parent_a.buy_conditions[:buy_split] + parent_b.buy_conditions[buy_split:]
        
        # Crossover sell conditions
        sell_split = len(parent_a.sell_conditions) // 2
        child_sell = parent_a.sell_conditions[:sell_split] + parent_b.sell_conditions[sell_split:]
        
        return TradingRule(
            buy_conditions=child_buy,
            sell_conditions=child_sell
        )
        
    except Exception as e:
        handle_error(e, "population.crossover", logger=None)
        return create_baseline_rule()

def mutate(rule: TradingRule, ctx: Any) -> TradingRule:
    """Apply mutation with risk validation"""
    try:
        if not rule:
            return create_baseline_rule()
            
        ga_settings = ctx.config.get("ga_settings", {})
        mutation_rate = ga_settings.get("mutation_rate", 0.1)
        
        indicators = ga_settings.get("available_indicators", [
            "EMA_8", "EMA_21", "EMA_55", "RSI_14", "MACD", "ATR_14"
        ])
        
        operators = [">", ">=", "<", "<="]
        
        mutated_buy = [dict(c) for c in rule.buy_conditions]
        mutated_sell = [dict(c) for c in rule.sell_conditions]
        
        for conditions in [mutated_buy, mutated_sell]:
            for condition in conditions:
                if random.random() < mutation_rate:
                    mutation_type = random.choice(["operator", "indicator", "reference"])
                    
                    if mutation_type == "operator":
                        condition["op"] = random.choice(operators)
                    elif mutation_type == "indicator":
                        condition["indicator"] = random.choice(indicators)
                    else:
                        if condition["indicator"] in ["RSI_14", "STOCH_K", "STOCH_D"]:
                            condition["ref"] = round(random.uniform(20, 80), 2)
                        elif "EMA" in condition["indicator"]:
                            condition["ref"] = random.choice(
                                [i for i in indicators if "EMA" in i]
                            )
                        else:
                            condition["ref"] = round(random.uniform(0, 100), 2)
        
        return TradingRule(
            buy_conditions=mutated_buy,
            sell_conditions=mutated_sell
        )
        
    except Exception as e:
        handle_error(e, "population.mutate", logger=ctx.logger)
        return create_baseline_rule()

def evolve_population(
    population: List[TradingRule],
    ctx: Any
) -> List[TradingRule]:
    """Create new generation through selection, crossover and mutation"""
    if not population:
        return [create_baseline_rule()]
    
    config = ctx.config.get("ga_settings", {})
    if isinstance(config, dict) and "ga_settings" in config:
        config = config["ga_settings"]
        
    population_size = len(population)
    elitism_count = int(population_size * config.get("elitism_ratio", 0.1))
    crossover_rate = config.get("crossover_rate", 0.7)
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Keep elite rules
    new_population = population[:elitism_count]
    
    # Get available indicators
    indicators = config.get("available_indicators", [
        "close", "EMA_8", "EMA_21", "EMA_55", "RSI_14", "MACD", "ATR_14"
    ])
    
    # Generate rest of population
    while len(new_population) < population_size:
        if random.random() < crossover_rate:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            child = crossover(parent_a, parent_b)
        else:
            child = tournament_select(population)
        
        child = mutate(child, ctx)
        new_population.append(child)
    
    return new_population