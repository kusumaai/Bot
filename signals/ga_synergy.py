#!/usr/bin/env python3
"""
Module: signals/ga_synergy.py
Genetic algorithm for trading rule optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from decimal import Decimal
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime

from utils.error_handler import handle_error
from bot_types.base_types import ValidationResult, Validatable
from indicators.indicators_pta import compute_indicators
from trading.math import (
    calculate_expected_value,
    calculate_kelly_fraction,
    predict_next_return
)
from utils.exceptions import InvalidOrderError

@dataclass
class GeneticRule(Validatable):
    """Genetic algorithm trading rule"""
    # Required fields (no defaults)
    rule_id: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]
    
    # Optional fields (with defaults)
    fitness_score: Decimal = Decimal('0')
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_rate: Decimal = Decimal('0.1')
    crossover_rate: Decimal = Decimal('0.8')
    last_update: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate genetic rule parameters"""
        try:
            if not self.rule_id:
                return ValidationResult(
                    is_valid=False,
                    error_message="Rule ID cannot be empty"
                )
            
            if self.mutation_rate < 0 or self.mutation_rate > 1:
                return ValidationResult(
                    is_valid=False,
                    error_message="Mutation rate must be between 0 and 1"
                )
                
            if self.crossover_rate < 0 or self.crossover_rate > 1:
                return ValidationResult(
                    is_valid=False,
                    error_message="Crossover rate must be between 0 and 1"
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Genetic rule validation failed: {str(e)}"
            )

@dataclass
class GASignal:
    symbol: str
    action: str
    price: Decimal
    quantity: Decimal

def generate_ga_signals(data: dict) -> GASignal:
    action = data.get("action")
    if action not in ["buy", "sell"]:
        raise InvalidOrderError(f"Invalid action: {action}")
    return GASignal(
        symbol=data["symbol"],
        action=action,
        price=Decimal(data["price"]),
        quantity=Decimal(data["quantity"])
    )

def generate_ga_signals(market_data: pd.DataFrame, population: List[GeneticRule]) -> List[Dict[str, Any]]:
    """Generate trading signals using genetic algorithm rules"""
    try:
        signals = []
        
        for rule in population:
            # Compute technical indicators
            indicators = compute_indicators(market_data, rule.parameters)
            
            # Generate signal based on rule conditions
            signal = evaluate_rule_conditions(
                rule=rule,
                data=indicators,
                parameters=rule.parameters
            )
            
            if signal:
                # Calculate signal metrics
                expected_value = calculate_expected_value(
                    price=Decimal(str(market_data['close'].iloc[-1])),
                    probability=signal['probability'],
                    target=signal['take_profit'],
                    stop=signal['stop_loss']
                )
                
                kelly = calculate_kelly_fraction(
                    win_prob=signal['probability'],
                    win_loss_ratio=abs(
                        (signal['take_profit'] - market_data['close'].iloc[-1]) /
                        (signal['stop_loss'] - market_data['close'].iloc[-1])
                    )
                )
                
                predicted_return = predict_next_return(
                    data=market_data,
                    lookback=rule.parameters.get('lookback', 20)
                )
                
                # Enhance signal metadata
                signal.update({
                    'rule_id': rule.rule_id,
                    'expected_value': expected_value,
                    'kelly_fraction': kelly,
                    'predicted_return': predicted_return,
                    'generation': rule.generation,
                    'fitness_score': rule.fitness_score,
                    'timestamp': datetime.now().timestamp()
                })
                
                signals.append(signal)
        
        return signals

    except Exception as e:
        handle_error(e, "generate_ga_signals")
        return []

def evaluate_rule_conditions(
    rule: GeneticRule,
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Evaluate trading rule conditions"""
    try:
        # Get latest values
        current_price = data['close'].iloc[-1]
        
        # Check entry conditions based on parameters
        conditions = []
        
        # Technical indicator conditions
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            conditions.append(
                rsi < parameters.get('rsi_oversold', 30) if parameters.get('direction') == 'long'
                else rsi > parameters.get('rsi_overbought', 70)
            )
            
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            conditions.append(
                data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]
                if parameters.get('direction') == 'long'
                else data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]
            )
        
        # Volume confirmation
        if 'volume_sma' in data.columns:
            conditions.append(
                data['volume'].iloc[-1] > data['volume_sma'].iloc[-1] * 
                parameters.get('volume_threshold', 1.5)
            )
        
        # Generate signal if all conditions met
        if all(conditions):
            atr = data['atr'].iloc[-1] if 'atr' in data.columns else current_price * 0.02
            
            return {
                'symbol': rule.symbol,
                'direction': parameters.get('direction', 'long'),
                'probability': Decimal('0.6'),  # Base probability
                'stop_loss': current_price * (1 - atr * 2),
                'take_profit': current_price * (1 + atr * 3),
                'metadata': {
                    'timeframe': rule.timeframe,
                    'indicators': {k: data[k].iloc[-1] for k in data.columns 
                                 if k not in ['open', 'high', 'low', 'close', 'volume']}
                }
            }
            
        return None

    except Exception as e:
        handle_error(e, "evaluate_rule_conditions")
        return None

def create_initial_population(
    size: int,
    symbol: str,
    timeframe: str
) -> List[GeneticRule]:
    """Create initial population of trading rules"""
    try:
        population = []
        
        for _ in range(size):
            rule = GeneticRule(
                rule_id=str(uuid.uuid4()),
                symbol=symbol,
                timeframe=timeframe,
                parameters={
                    'direction': np.random.choice(['long', 'short']),
                    'rsi_period': np.random.randint(10, 30),
                    'rsi_oversold': np.random.randint(20, 40),
                    'rsi_overbought': np.random.randint(60, 80),
                    'macd_fast': np.random.randint(8, 20),
                    'macd_slow': np.random.randint(21, 40),
                    'macd_signal': np.random.randint(5, 15),
                    'volume_threshold': np.random.uniform(1.2, 2.0),
                    'lookback': np.random.randint(10, 50)
                }
            )
            population.append(rule)
            
        return population

    except Exception as e:
        handle_error(e, "create_initial_population")
        return []
