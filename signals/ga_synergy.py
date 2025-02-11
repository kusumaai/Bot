#!/usr/bin/env python3
"""
signals/ga_synergy.py - Genetic Algorithm Trading Strategy Orchestrator
Enhanced version with optimized settings and proper error handling
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from decimal import Decimal
import random
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.error_handler import handle_error
from database.database import DBConnection, execute_sql
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

def generate_ga_signals(
    candles: List[Dict[str, Any]],
    population: List[TradingRule],
    ctx: Any
) -> List[Dict[str, Any]]:
    """Generate trading signals using enhanced genetic algorithm"""
    if not candles or not population:
        ctx.logger.warning("Missing candles or population for GA signal generation")
        return []
        
    try:
        # Get GA settings
        ga_settings = ctx.config.get("ga_settings", {})
        min_fitness = Decimal(str(ga_settings.get("min_fitness", "0.05")))
        
        ctx.logger.info(f"Processing {len(candles)} candles with population size {len(population)}")
        
        # Prepare market state with enhanced metrics
        market_state = prepare_market_state(candles, ctx)
        if not market_state:
            ctx.logger.warning("Could not prepare market state")
            return []
            
        # Evolve population
        ctx.logger.info("Evolving population...")
        evolved = evolve_population(population, ctx)
        if not evolved:
            ctx.logger.warning("Population evolution returned no results")
            return []
            
        # Get best performing rule
        best_rule = evolved[0]
        if not best_rule:
            ctx.logger.warning("No best rule found after evolution")
            return []
            
        # Add fitness logging
        ctx.logger.info(f"Best rule fitness: {best_rule.fitness}")
        
        # Store if fitness exceeds threshold
        if best_rule.fitness > min_fitness:
            ctx.logger.info(f"Storing rule with fitness: {best_rule.fitness}")
            store_rule(best_rule, ctx)
        
        # Generate signal from best rule
        last_candle = candles[-1]
        direction, signal_meta = evaluate_rule(best_rule, last_candle, market_state)
        
        if not direction or not signal_meta:
            ctx.logger.info("Rule evaluation produced no signal")
            return []
            
        # Enhanced signal metrics
        signal = {
            "symbol": last_candle.get("symbol", ""),
            "direction": direction,
            "probability": float(signal_meta["probability"]),
            "expected_value": float(signal_meta["expected_value"]),
            "kelly_fraction": float(signal_meta["kelly_fraction"]),
            "entry_price": float(last_candle["close"]),
            "atr": float(last_candle.get("ATR_14", 0)),
            "volatility": float(market_state.volatility),
            "timestamp": time.time(),
            "exchange": ctx.config["exchanges"][0] if ctx.config.get("exchanges") else "unknown"
        }
        
        ctx.logger.info(f"Signal details: {signal}")
        return [signal]
        
    except Exception as e:
        handle_error(e, "ga_synergy.generate_ga_signals", logger=ctx.logger)
        return []

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
                with DBConnection(ctx.db_pool) as conn:
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
                
                signals = generate_ga_signals(candles, population, ctx)
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
                "exchanges": ["binance"],
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