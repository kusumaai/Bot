#!/usr/bin/env python3
"""
signals/ga_synergy.py - Genetic Algorithm Trading Strategy Orchestrator
Enhanced version with optimized settings for complex strategy evolution
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any

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
    evolve_population
)
from signals.evaluation import (
    evaluate_rule,
    simulate_rule
)

def prepare_market_state(candles: List[Dict[str, Any]], ctx: Any) -> MarketState:
    """Create market state from candle data with enhanced metrics"""
    try:
        import numpy as np
        
        if not candles or len(candles) < 2:
            ctx.logger.warning("Insufficient candle data for market state")
            return None
            
        prices = np.array([c["close"] for c in candles])
        returns = np.log(prices[1:] / prices[:-1])
        
        # Get the most recent candle for current state
        latest = candles[-1]
        
        # Calculate enhanced market state metrics
        market_state = MarketState(
            returns=returns,
            ar1_coef=np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0,
            current_return=returns[-1] if len(returns) > 0 else 0,
            volatility=np.std(returns[-20:]) if len(returns) >= 20 else 0,
            last_price=prices[-1],
            ema_short=latest.get("EMA_8", 0),
            ema_long=latest.get("EMA_21", 0)
        )
        
        ctx.logger.debug(f"Market state prepared: volatility={market_state.volatility:.4f}, " 
                        f"ar1_coef={market_state.ar1_coef:.4f}")
        
        return market_state
        
    except Exception as e:
        handle_error(e, "ga_synergy.prepare_market_state", logger=ctx.logger)
        return None

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
        if isinstance(ga_settings, dict) and "ga_settings" in ga_settings:
            ga_settings = ga_settings["ga_settings"]
            
        min_fitness = ga_settings.get("min_fitness", 0.05)
        
        ctx.logger.info(f"Processing {len(candles)} candles with population size {len(population)} (min_fitness: {min_fitness})")
        
        # Prepare market state with enhanced metrics
        market_state = prepare_market_state(candles, ctx)
        if not market_state:
            ctx.logger.warning("Could not prepare market state")
            return []
            
        # Evolve population with increased complexity
        ctx.logger.info(f"Evolving population of size {len(population)}...")
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
        
        # Adjust min fitness threshold to be more permissive initially
        min_fitness = ctx.config.get("ga_settings", {}).get("min_fitness", 0.05)  # Lowered from 0.1
        
        # Store if fitness exceeds threshold
        if best_rule.fitness > min_fitness:
            ctx.logger.info(f"Storing rule with fitness: {best_rule.fitness}")
            store_rule(best_rule, ctx)
        else:
            ctx.logger.info(f"Rule fitness {best_rule.fitness} below threshold {min_fitness}")
        
        # Generate signal from best rule
        last_candle = candles[-1]
        signal = evaluate_rule(best_rule, last_candle, market_state)
        
        if not signal:
            ctx.logger.info("Rule evaluation produced no signal")
            return []
            
        # Run detailed simulation for signal parameters
        ctx.logger.info("Running comprehensive simulation for signal parameters...")
        sim_result = simulate_rule(best_rule, candles, market_state, ctx)
        if not sim_result or not sim_result.trades:
            ctx.logger.info("Simulation produced no valid trades")
            return []
            
        latest_trade = sim_result.trades[-1]
        
        # Log signal details
        ctx.logger.info(f"Generated signal: {signal} with probability {latest_trade.get('probability', 0):.3f}")
        
        # Enhanced signal validation logging
        signal_metrics = {
            "symbol": last_candle.get("symbol", ""),
            "direction": signal,
            "probability": latest_trade.get("probability", 0),
            "expected_value": latest_trade.get("expected_value", 0),
            "entry_price": last_candle["close"],
            "stop_loss": latest_trade.get("stop_loss", 0),
            "take_profit": latest_trade.get("take_profit", 0),
            "kelly_fraction": latest_trade.get("kelly_fraction", 0),
            "exchange": ctx.config["exchanges"][0] if ctx.config.get("exchanges") else "unknown"
        }
        
        ctx.logger.info(f"Signal details: {signal_metrics}")
        
        return [signal_metrics]
        
    except Exception as e:
        handle_error(e, "ga_synergy.generate_ga_signals", logger=ctx.logger)
        return []

async def run_ga_optimization(ctx: Any) -> None:
    """Run GA optimization process"""
    try:
        ctx.logger.info("Initializing GA population...")
        population = initialize_population(ctx)
        ctx.logger.info(f"Population initialized successfully with {len(population)} members")
        
        # Enhanced SQL query for richer dataset
        sql_query = """
            SELECT c.*, 
                   (c.high - c.low) / c.low * 100 as candle_range_pct,
                   (c.close - c.open) / c.open * 100 as candle_body_pct
            FROM candles c
            WHERE c.timeframe = ?
            AND c.datetime >= datetime('now', '-720 day')
            AND c.symbol IN ('BTC/USDT', 'ETH/USDT')
            ORDER BY c.timestamp DESC
            LIMIT 1000000
        """
        
        while True:
            ctx.logger.info("Fetching dataset...")
            with DBConnection(ctx.db_pool) as conn:
                rows = execute_sql(
                    conn,
                    sql_query,
                    [ctx.config.get("timeframe", "1h")]
                )
                
                if rows:
                    ctx.logger.info(f"Processing {len(rows)} candles metrics...")
                    candles = [dict(row) for row in rows]
                    signals = generate_ga_signals(candles, population, ctx)
                    if signals:
                        ctx.logger.info(f"Generated signals with parameters: {signals}")
                    else:
                        ctx.logger.info("No signals met criteria this iteration")
                else:
                    ctx.logger.warning("No candle data found")
                
            # Dynamic sleep interval based on market activity
            interval = ctx.config.get("ga_interval", 30)
            ctx.logger.info(f"Sleeping for {interval} seconds...")
            await asyncio.sleep(interval)
            
    except Exception as e:
        handle_error(e, "ga_synergy.run_ga_optimization", logger=ctx.logger)

if __name__ == "__main__":
    # Setup for standalone testing with enhanced parameters
    logging.basicConfig(level=logging.INFO)
    
    class Context:
        def __init__(self):
            self.logger = logging.getLogger("GeneticAlgorithm")
            self.config = {
                "timeframe": "1h",
                "ga_interval": 30,
                "exchanges": ["binance"],
                "ga_settings": {
                    # Enhanced population settings
                    "population_size": 200,
                    "mutation_rate": 0.25,
                    "crossover_rate": 0.85,
                    "elitism_ratio": 0.15,
                    
                    # Enhanced strategy complexity
                    "buy_conditions_count": 6,
                    "sell_conditions_count": 6,
                    "min_conditions": 3,
                    "max_conditions": 8,
                    
                    # Tournament selection settings
                    "tournament_size": 8,
                    
                    # Minimum fitness threshold for storing rules
                    "min_fitness": 0.05,
                    
                    # Comprehensive indicator set
                    "available_indicators": [
                        # Price data
                        "close", "open", "high", "low",
                        # Moving averages
                        "EMA_8", "EMA_21", "EMA_55", "EMA_89", "EMA_144", "EMA_233",
                        # Momentum
                        "RSI_14", "MACD", "MACDs",
                        # Volatility
                        "ATR_14", "BBL", "BBM", "BBU",
                        # Derived values
                        "candle_range_pct", "candle_body_pct"
                    ],
                    
                    # Enhanced evolution settings
                    "allow_nested_conditions": True,
                    "mutation_types": ["operator", "indicator", "reference", "threshold"]
                },
                "database": {
                    "path": os.path.join(project_root, "data", "candles.db")
                }
            }
            self.db_pool = self.config["database"]["path"]
    
    ctx = Context()
    
    try:
        asyncio.run(run_ga_optimization(ctx))
    except KeyboardInterrupt:
        print("\nStopping GA optimization...")