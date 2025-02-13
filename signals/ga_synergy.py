#!/usr/bin/env python3
"""
signals/ga_synergy.py - Genetic Algorithm Trading Strategy Orchestrator
Enhanced version with optimized settings and proper error handling
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from decimal import Decimal
import random
import time
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.exceptions import InvalidOrderError
from utils import logger
from utils.error_handler import handle_error, handle_error_async, ValidationError
from database.database import DatabaseQueries, execute_sql
from signals.trading_types import MarketState, TradingRule
from signals.storage import store_rule, load_best_rule
from signals.population import (
    initialize_population,
    tournament_select,
    crossover,
    mutate
)
from signals.evaluation import (
    evaluate_rule,
    calculate_metrics,
    TradeMetrics
)
from database.queries import DatabaseQueries
from signals.base_types import BaseSignal
from signals.market_state import prepare_market_state

@dataclass
class GAParameters:
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    
    def validate(self) -> None:
        if not 0 < self.population_size <= 1000:
            raise ValidationError("Population size must be between 1 and 1000")
        if not 0 < self.generations <= 200:
            raise ValidationError("Generations must be between 1 and 200")
        if not 0 <= self.mutation_rate <= 1:
            raise ValidationError("Mutation rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValidationError("Crossover rate must be between 0 and 1")
        if self.elite_size >= self.population_size:
            raise ValidationError("Elite size must be less than population size")

@dataclass
class GASignal(BaseSignal):
    """GA-specific signal extension"""
    # Required fields (no defaults)
    symbol: str
    direction: str
    strength: Decimal
    timestamp: datetime
    rule_id: str
    
    # Optional fields (with defaults)
    confidence: float = 0.0
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class GASynergySignal:
    def __init__(
        self,
        db_queries: DatabaseQueries,
        logger: logging.Logger,
        params: Optional[GAParameters] = None
    ):
        self.db = db_queries
        self.logger = logger
        self.params = params or GAParameters()
        self.params.validate()
        
        self._best_individual: Optional[Dict[str, Any]] = None
        self._population: Optional[List[Dict[str, Any]]] = None
        
    async def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        try:
            # Get historical data
            candles = await self.db.get_recent_candles(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback_periods
            )
            
            if len(candles) < lookback_periods:
                raise ValidationError(
                    f"Insufficient data: got {len(candles)}, need {lookback_periods}"
                )
            
            # Prepare data
            df = self._prepare_data(candles)
            
            # Generate and optimize signals
            population = self._initialize_population()
            
            for generation in range(self.params.generations):
                fitness_scores = self._evaluate_population(population, df)
                population = self._evolve_population(population, fitness_scores)
                
                best_idx = np.argmax(fitness_scores)
                self._best_individual = population[best_idx]
            
            # Generate final signal
            signal = self._generate_final_signal(df, self._best_individual)
            
            # Store signal
            await self._store_signal(symbol, signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            raise
    
    def _prepare_data(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        return df.dropna()
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        population = []
        for _ in range(self.params.population_size):
            individual = {
                'sma_weight': np.random.uniform(0, 1),
                'rsi_weight': np.random.uniform(0, 1),
                'vol_weight': np.random.uniform(0, 1),
                'threshold': np.random.uniform(0.3, 0.7)
            }
            population.append(individual)
        return population
    
    def _evaluate_population(
        self,
        population: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> np.ndarray:
        fitness_scores = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            signals = self._apply_strategy(df, individual)
            fitness_scores[i] = self._calculate_fitness(df, signals)
            
        return fitness_scores
    
    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(fitness_scores)[-self.params.elite_size:]
        new_population.extend([population[i] for i in elite_indices])
        
        while len(new_population) < self.params.population_size:
            if np.random.random() < self.params.crossover_rate:
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                child = self._crossover(parent1, parent2)
            else:
                child = self._tournament_select(population, fitness_scores).copy()
            
            if np.random.random() < self.params.mutation_rate:
                child = self._mutate(child)
                
            new_population.append(child)
            
        return new_population
    
    async def _store_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        await self.db.store_signal(
            symbol=symbol,
            signal_type='ga_synergy',
            direction=signal['direction'],
            strength=signal['strength'],
            metadata={
                'indicators': signal['indicators'],
                'parameters': signal['parameters']
            }
        )

def prepare_market_state(candles: List[Dict[str, Any]], ctx: Any) -> Optional[MarketState]:
    """Create market state with proper validation"""
    try:
        if not candles or len(candles) < 55:  # Minimum required for indicators
            ctx.logger.warning("Insufficient candle data for market state")
            return None
            
        prices = np.array([c["close"] for c in candles if c["close"] > 0])
        if len(prices) < 55:
            return None
            
        returns = np.log(prices[1:] / prices[:-1])
        latest = candles[-1]
        
        # Enhanced market state with more indicators
        market_state = MarketState(
            returns=returns,
            ar1_coef=np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0,
            current_return=returns[-1] if len(returns) > 0 else 0,
            volatility=np.std(returns[-20:]) if len(returns) >= 20 else 0,
            last_price=Decimal(str(prices[-1])),
            ema_short=Decimal(str(latest.get("EMA_8", 0))),
            ema_long=Decimal(str(latest.get("EMA_21", 0))),
            atr=Decimal(str(latest.get("ATR_14", 0))),
            rsi=Decimal(str(latest.get("RSI_14", 50))),
            bb_width=Decimal(str(latest.get("BB_WIDTH", 0))),
            ctx=ctx
        )
        
        ctx.logger.debug(f"Market state prepared: volatility={market_state.volatility:.4f}, "
                        f"ar1_coef={market_state.ar1_coef:.4f}")
        
        return market_state
        
    except Exception as e:
        handle_error(e, "ga_synergy.prepare_market_state", logger=ctx.logger)
        return None

def evolve_population(population: List[TradingRule], ctx: Any) -> List[TradingRule]:
    """Enhanced evolution with better convergence"""
    if not population:
        ctx.logger.warning("Empty population for evolution")
        return []
        
    try:
        ga_settings = ctx.config.get("ga_settings", {})
        population_size = len(population)
        elite_count = max(1, int(population_size * 0.1))  # Keep at least 1 elite
        
        # Sort by fitness with proper decimal handling
        population.sort(key=lambda x: float(x.fitness) if hasattr(x, 'fitness') else 0, reverse=True)
        
        # Keep elite rules
        new_population = population[:elite_count]
        
        # Dynamic mutation rate based on population diversity
        fitness_values = [float(p.fitness) for p in population if hasattr(p, 'fitness')]
        fitness_std = np.std(fitness_values) if fitness_values else 0
        mutation_rate = min(0.3, max(0.1, fitness_std))
        
        # Generate rest of population
        while len(new_population) < population_size:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            
            if random.random() < ga_settings.get("crossover_rate", 0.8):
                child = crossover(parent_a, parent_b)
            else:
                child = random.choice([parent_a, parent_b])
                
            if random.random() < mutation_rate:
                child = mutate(child, ctx)
                
            new_population.append(child)
            
        ctx.logger.info(f"Evolution complete. Best fitness: {population[0].fitness if population else 0}")
        return new_population
        
    except Exception as e:
        handle_error(e, "ga_synergy.evolve_population", logger=ctx.logger)
        return population[:population_size] if population else []

async def generate_ga_signals(data: Dict[str, Any]) -> GASignal:
    if data.get("action") not in ["buy", "sell"]:
        raise InvalidOrderError(f"Invalid action: {data.get('action')}")
    return GASignal(
        symbol=data.get("symbol"),
        action=data.get("action"),
        price=Decimal(str(data.get("price"))),
        quantity=Decimal(str(data.get("quantity")))
    )

async def evaluate_population(population: List[TradingRule], candle: pd.Series) -> List[float]:
    """Evaluate the fitness of each rule in the population"""
    fitness_scores = []
    for rule in population:
        metrics = await evaluate_rule(rule, candle)
        fitness = calculate_fitness(metrics)
        fitness_scores.append(fitness)
    return fitness_scores

def calculate_fitness(metrics: TradeMetrics) -> float:
    """Calculate fitness score based on trade metrics"""
    # Example fitness calculation: Sharpe Ratio
    if metrics.volatility == 0:
        return 0
    return float(metrics.return_rate / metrics.volatility)

def select_next_generation(
    population: List[TradingRule], 
    fitness_scores: List[float],
    elite_size: int = 2
) -> Tuple[List[TradingRule], List[float]]:
    """Select the next generation of rules based on fitness scores"""
    sorted_population = [rule for _, rule in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]
    sorted_fitness = sorted(fitness_scores, reverse=True)
    
    # Elitism: carry forward the top performers
    next_generation = sorted_population[:elite_size]
    next_fitness = sorted_fitness[:elite_size]
    
    # Select the rest based on fitness (e.g., roulette wheel selection)
    remaining = len(population) - elite_size
    selected_indices = random.choices(range(len(population)), weights=sorted_fitness, k=remaining)
    next_generation.extend([sorted_population[i] for i in selected_indices])
    next_fitness.extend([sorted_fitness[i] for i in selected_indices])
    
    return next_generation, next_fitness

def apply_rule_to_candle(rule: TradingRule, candle: pd.Series) -> Optional[Dict[str, Any]]:
    """Apply a trading rule to the current candle to generate a signal"""
    # Example implementation based on rule conditions
    try:
        if rule.evaluate(candle):
            direction = 'buy' if rule.direction == 'long' else 'sell'
            signal = {
                "symbol": candle["symbol"],
                "direction": direction,
                "probability": float(rule.probability),
                "potential_profit": float(rule.potential_profit),
                "potential_loss": float(rule.potential_loss),
                "price": float(candle["close"])
            }
            return signal
        return None
    except Exception as e:
        handle_error(e, "apply_rule_to_candle", logger=logger)
        return None

async def run_ga_optimization(ctx: Any) -> None:
    """Run GA optimization process with enhanced data handling"""
    try:
        ctx.logger.info("Initializing GA population...")
        population = initialize_population(ctx)
        ctx.logger.info(f"Population initialized with {len(population)} members")
        
        # Enhanced SQL query for richer dataset
        sql_query = """
            SELECT c.*, 
                   (c.high - c.low) / c.low * 100 as candle_range_pct,
                   (c.close - c.open) / c.open * 100 as candle_body_pct,
                   COALESCE(v.volume_ma_ratio, 1) as volume_ma_ratio
            FROM candles c
            LEFT JOIN volume_metrics v ON c.id = v.candle_id
            WHERE c.timeframe = ?
            AND c.datetime >= datetime('now', '-30 day')
            AND c.symbol IN ('BTC/USDT', 'ETH/USDT')
            ORDER BY c.timestamp ASC
            LIMIT 10000
        """
        
        while True:
            try:
                ctx.logger.info("Fetching dataset...")
                with DatabaseQueries(ctx.db_pool) as conn:
                    rows = execute_sql(
                        conn,
                        sql_query,
                        [ctx.config.get("timeframe", "15m")]
                    )
                    
                if not rows:
                    ctx.logger.warning("No candle data found")
                    await asyncio.sleep(60)
                    continue
                    
                ctx.logger.info(f"Processing {len(rows)} candles...")
                candles = [dict(row) for row in rows]
                
                signals = [generate_ga_signals(candle) for candle in candles]
                if signals:
                    ctx.logger.info(f"Generated {len(signals)} signals")
                else:
                    ctx.logger.info("No signals met criteria this iteration")
                
            except Exception as e:
                handle_error(e, "ga_synergy.run_ga_optimization loop", logger=ctx.logger)
                
            # Dynamic sleep interval based on market activity
            interval = ctx.config.get("ga_interval", 300)  # Default 5 minutes
            if signals:  # Reduce interval if signals were generated
                interval = max(60, interval // 2)
            ctx.logger.info(f"Sleeping for {interval} seconds...")
            await asyncio.sleep(interval)
            
    except Exception as e:
        handle_error(e, "ga_synergy.run_ga_optimization", logger=ctx.logger)

if __name__ == "__main__":
    # Setup for standalone testing
    logging.basicConfig(level=logging.INFO)
    
    class Context:
        def __init__(self):
            self.logger = logging.getLogger("GeneticAlgorithm")
            self.config = {
                "timeframe": "15m",
                "ga_interval": 300,
                "exchanges": ["kucoin"],
                "ga_settings": {
                    "population_size": 100,
                    "mutation_rate": 0.2,
                    "crossover_rate": 0.8,
                    "tournament_size": 3,
                    "elite_count": 5,
                    "min_fitness": 0.05,
                    "available_indicators": [
                        "close", "open", "high", "low",
                        "EMA_8", "EMA_21", "EMA_55",
                        "RSI_14", "MACD", "MACDs",
                        "ATR_14", "BBL", "BBM", "BBU",
                        "candle_range_pct", "candle_body_pct"
                    ]
                }
            }
            self.db_pool = "data/candles.db"
    
    ctx = Context()
    
    try:
        asyncio.run(run_ga_optimization(ctx))
    except KeyboardInterrupt:
        print("\nStopping GA optimization...")