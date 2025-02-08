#!/usr/bin/env python3
"""
signals/ga_synergy.py - Genetic Algorithm Trading Strategy Orchestrator
"""
import os
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
from typing import List, Dict, Any
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
    """Create market state from candle data"""
    try:
        import numpy as np
        
        if not candles or len(candles) < 2:
            return None
            
        prices = np.array([c["close"] for c in candles])
        returns = np.log(prices[1:] / prices[:-1])
        
        latest = candles[-1]
        
        return MarketState(
            returns=returns,
            ar1_coef=np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0,
            current_return=returns[-1] if len(returns) > 0 else 0,
            volatility=np.std(returns[-20:]) if len(returns) >= 20 else 0,
            last_price=prices[-1],
            ema_short=latest.get("EMA_8", 0),
            ema_long=latest.get("EMA_21", 0)
        )
        
    except Exception as e:
        handle_error(e, "ga_synergy.prepare_market_state", logger=ctx.logger)
        return None
    
    """Create market state from candle data"""
    try:
        import numpy as np
        
        if not candles or len(candles) < 2:
            return None
            
        prices = np.array([c["close"] for c in candles])
        returns = np.log(prices[1:] / prices[:-1])
        
        latest = candles[-1]
        
        return MarketState(
            returns=returns,
            ar1_coef=np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0,
            current_return=returns[-1] if len(returns) > 0 else 0,
            volatility=np.std(returns[-20:]) if len(returns) >= 20 else 0,
            last_price=prices[-1],
            ema_short=latest.get("EMA_8", 0),
            ema_long=latest.get("EMA_21", 0)
        )
        
    except Exception as e:
        handle_error(e, "ga_synergy.prepare_market_state", logger=None)
        return None

def generate_ga_signals(
    candles: List[Dict[str, Any]],
    population: List[TradingRule],
    ctx: Any
) -> List[Dict[str, Any]]:
    """Generate trading signals using genetic algorithm"""
    if not candles or not population:
        return []
        
    try:
        # Get market state
        market_state = prepare_market_state(candles, ctx)  # Pass ctx here
        if not market_state:
            return []
            
        # Evolve population
        evolved = evolve_population(population, ctx)
        if not evolved:
            return []
            
        # Get best rule
        best_rule = evolved[0]
        if not best_rule:
            return []
            
        # Store if valid
        if best_rule.fitness > 0:
            store_rule(best_rule, ctx)
        
        # Generate signal
        last_candle = candles[-1]
        signal = evaluate_rule(best_rule, last_candle, market_state)
        
        if not signal:
            return []
            
        # Simulate for parameters
        sim_result = simulate_rule(best_rule, candles, market_state, ctx)
        if not sim_result or not sim_result.trades:
            return []
            
        latest_trade = sim_result.trades[-1]
        
        return [{
            "symbol": last_candle.get("symbol", ""),
            "direction": signal,
            "probability": latest_trade.get("probability", 0),
            "expected_value": latest_trade.get("expected_value", 0),
            "entry_price": last_candle["close"],
            "stop_loss": latest_trade.get("stop_loss", 0),
            "take_profit": latest_trade.get("take_profit", 0),
            "kelly_fraction": latest_trade.get("kelly_fraction", 0),
            "exchange": ctx.config["exchanges"][0] if ctx.config.get("exchanges") else "unknown"
        }]
        
    except Exception as e:
        handle_error(e, "ga_synergy.generate_ga_signals", logger=ctx.logger)
        return []

async def run_ga_optimization(ctx: Any) -> None:
    """Run GA optimization process"""
    try:
        population = initialize_population(ctx)
        
        while True:
            with DBConnection(ctx.db_pool) as conn:
                rows = execute_sql(
                    conn,
                    """
                    SELECT * FROM candles 
                    WHERE timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT 1000
                    """,
                    [ctx.config.get("timeframe", "1h")]
                )
                
                if rows:
                    # Convert sqlite rows to dictionaries
                    candles = [dict(row) for row in rows]
                    signals = generate_ga_signals(candles, population, ctx)
                    if signals:
                        ctx.logger.info(f"Generated signals: {signals}")
                
            await asyncio.sleep(ctx.config.get("ga_interval", 300))
            
    except Exception as e:
        handle_error(e, "ga_synergy.run_ga_optimization", logger=ctx.logger)

if __name__ == "__main__":
    # Setup for standalone testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    class Context:
        def __init__(self):
            self.logger = logging.getLogger("GeneticAlgorithm")
            self.config = {
                "timeframe": "1h",
                "ga_interval": 300,
                "exchanges": ["binance"],
                "ga_settings": {
                    "population_size": 50,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.1,
                    "buy_conditions_count": 3,
                    "sell_conditions_count": 3,
                    "available_indicators": [
                        "close", "EMA_8", "EMA_21", "EMA_55", 
                        "RSI_14", "MACD", "ATR_14"
                    ]
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
        print("\nStopping GA optimization")