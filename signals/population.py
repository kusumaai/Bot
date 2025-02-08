#!/usr/bin/env python3
"""
signals/population.py - Population management and genetic operations
"""
import random
from typing import List, Dict, Any, Optional
from utils.error_handler import handle_error
from .types import TradingRule

def create_baseline_rule() -> TradingRule:
    """Create baseline trading strategy using EMA crossover with RSI"""
    return TradingRule(
        buy_conditions=[
            {
                "indicator": "EMA_8",
                "op": ">",
                "ref": "EMA_21"
            },
            {
                "indicator": "RSI_14",
                "op": ">",
                "ref": 40
            }
        ],
        sell_conditions=[
            {
                "indicator": "EMA_8",
                "op": "<",
                "ref": "EMA_21"
            },
            {
                "indicator": "RSI_14",
                "op": "<",
                "ref": 60
            }
        ]
    )

def generate_random_rule(indicators: List[str], ctx: Any) -> Optional[TradingRule]:
    """Generate random trading strategy"""
    try:
        config = ctx.config.get("ga_settings", {})
        buy_n = config.get("buy_conditions_count", 3)
        sell_n = config.get("sell_conditions_count", 3)
        
        if buy_n < 1 or sell_n < 1:
            return None
            
        operators = [">", ">=", "<", "<="]
        
        buy_conditions = []
        sell_conditions = []
        
        # Generate buy conditions
        for _ in range(buy_n):
            indicator = random.choice(indicators)
            op = random.choice(operators)
            ref = random.choice(
                indicators + [round(random.uniform(0, 100), 2) for _ in range(3)]
            )
            
            buy_conditions.append({
                "indicator": indicator,
                "op": op,
                "ref": ref
            })
        
        # Generate sell conditions if allowed
        if ctx.config.get("allow_shorts", True):
            for _ in range(sell_n):
                indicator = random.choice(indicators)
                op = random.choice(operators)
                ref = random.choice(
                    indicators + [round(random.uniform(0, 100), 2) for _ in range(3)]
                )
                
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
    config = ctx.config.get("ga_settings", {})
    population_size = config.get("population_size", 50)
    
    indicators = config.get("available_indicators", [
        "close", "EMA_8", "EMA_21", "EMA_55", "RSI_14", "MACD", "ATR_14"
    ])
    
    # Start with baseline
    population = [create_baseline_rule()]
    
    # Add random rules
    while len(population) < population_size:
        rule = generate_random_rule(indicators, ctx)
        if rule:
            population.append(rule)
    
    return population

def tournament_select(population: List[TradingRule], tournament_size: int = 3) -> TradingRule:
    """Select single parent using tournament selection"""
    if not population:
        return create_baseline_rule()
    
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda x: x.fitness)

def crossover(parent_a: TradingRule, parent_b: TradingRule) -> TradingRule:
    """Single point crossover for trading rules"""
    if not parent_a or not parent_b:
        return create_baseline_rule()
    
    buy_split = len(parent_a.buy_conditions) // 2
    sell_split = len(parent_a.sell_conditions) // 2
    
    child_buy = parent_a.buy_conditions[:buy_split] + parent_b.buy_conditions[buy_split:]
    child_sell = parent_a.sell_conditions[:sell_split] + parent_b.sell_conditions[sell_split:]
    
    return TradingRule(
        buy_conditions=child_buy,
        sell_conditions=child_sell
    )

def mutate(rule: TradingRule, indicators: List[str], ctx: Any) -> TradingRule:
    """Apply mutation to trading rule"""
    if not rule:
        return create_baseline_rule()
        
    config = ctx.config.get("ga_settings", {})
    mutation_rate = config.get("mutation_rate", 0.1)
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
                    condition["ref"] = random.choice(
                        indicators + [round(random.uniform(0, 100), 2)]
                    )
    
    return TradingRule(
        buy_conditions=mutated_buy,
        sell_conditions=mutated_sell
    )

def evolve_population(
    population: List[TradingRule],
    ctx: Any
) -> List[TradingRule]:
    """Create new generation through selection, crossover and mutation"""
    if not population:
        return [create_baseline_rule()]
    
    config = ctx.config.get("ga_settings", {})
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
        
        child = mutate(child, indicators, ctx)
        new_population.append(child)
    
    return new_population