#!/usr/bin/env python3
"""
Module: execution/bot.py
Main trading bot orchestrator using modular components
"""

import asyncio
import time
import json
import os
import logging
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

# Import core components
from execution.market_data import MarketData
from execution.position_manager import calculate_position_params, should_close_position
from execution.exchange_interface import create_exchange, fetch_ticker, place_order, close_position
from trading.ratchet import RatchetManager

# Import signal components
from models.ml_signal import generate_ml_signals
from signals.ga_synergy import generate_ga_signals
from signals.population import initialize_population
from signals.signal_utils import combine_signals, validate_signal

# Import accounting components
from accounting.accounting import (
    validate_account,
    record_new_trade,
    fetch_open_trades,
    update_trade_result,
    update_daily_performance,
    log_performance_summary
)

from utils.error_handler import handle_error

@dataclass
class TradingContext:
    """Trading context container"""
    logger: logging.Logger
    config: Dict[str, Any]
    exchange: Any = None
    db_pool: str = ""
    rf_model: Any = None
    xgb_model: Any = None
    ratchet_manager: Any = None
    market_data: Any = None

def load_config() -> dict:
    """Load configuration file"""
    config_path = os.path.join(project_root, "config", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def safe_load_ml_models(ctx: TradingContext) -> None:
    """Load ML models safely"""
    try:
        from models.ml_signal import load_models
        rf_model, xgb_model, _ = load_models(ctx)
        ctx.rf_model, ctx.xgb_model = rf_model, xgb_model
        if rf_model or xgb_model:
            ctx.logger.info("ML models loaded successfully")
    except Exception as e:
        handle_error(e, "Bot.safe_load_ml_models", logger=ctx.logger)

async def process_signals(
    signals: List[Dict[str, Any]],
    market_data: Dict[str, Dict[str, Any]],
    ctx: TradingContext
) -> None:
    """Process and execute trading signals"""
    for signal in signals:
        if signal["symbol"] not in market_data:
            continue
            
        try:
            if not validate_account(signal, ctx):
                continue
                
            current_price = await fetch_ticker(signal["symbol"], ctx)
            if current_price <= 0:
                continue
            
            params = calculate_position_params(
                signal,
                market_data[signal["symbol"]],
                ctx
            )
            
            if params["position_size"] <= 0:
                continue
            
            order = await place_order(
                signal["symbol"],
                signal["direction"],
                params["position_size"],
                current_price,
                ctx
            )
            
            if order:
                # Initialize ratchet tracking for new trade
                ctx.ratchet_manager.initialize_trade(
                    order["id"],
                    signal["entry_price"]
                )
                
                record_new_trade(
                    order,
                    signal,
                    params["expected_value"],
                    params["kelly_fraction"],
                    params["position_size"],
                    ctx,
                    trade_source="paper" if ctx.config.get("paper_mode", False) else "real"
                )
                
                ctx.logger.info(f"New trade opened: {order['id']} - {signal['symbol']}")
        
        except Exception as e:
            handle_error(e, "Bot.process_signals", logger=ctx.logger)

async def manage_positions(
    market_data: Dict[str, Dict[str, Any]],
    ctx: TradingContext
) -> None:
    """Manage open positions"""
    try:
        open_positions = fetch_open_trades(ctx)
        for trade in open_positions:
            if trade["symbol"] not in market_data:
                continue
                
            current_price = await fetch_ticker(trade["symbol"], ctx)
            if current_price <= 0:
                continue
            
            # Update ratchet system and check for close conditions
            if should_close_position(
                trade,
                current_price,
                market_data[trade["symbol"]],
                ctx
            ):
                if await close_position(trade, ctx):
                    pnl = (
                        (current_price - trade["entry_price"]) * trade["position_size"]
                        if trade["direction"] == "long"
                        else (trade["entry_price"] - current_price) * trade["position_size"]
                    )
                    
                    # Clean up ratchet tracking
                    ctx.ratchet_manager.remove_trade(trade["id"])
                    update_trade_result(trade["id"], pnl, ctx)
                    
                    ctx.logger.info(
                        f"Closed trade {trade['id']} with PNL: {pnl:.2f}"
                    )
    
    except Exception as e:
        handle_error(e, "Bot.manage_positions", logger=ctx.logger)

async def main_loop(ctx: TradingContext) -> None:
    """Main trading loop"""
    ctx.logger.info("Starting main trading loop")
    
    # Initialize GA population
    population = initialize_population(ctx)
    
    while True:
        try:
            start_time = time.time()
            
            # Load market data
            market_data, dataframes = await ctx.market_data.load_market_data(
                ctx.config["market_list"]
            )
            
            # Generate signals
            ml_signals = []
            ga_signals = []
            
            if dataframes:
                # ML signals from combined features
                features_df = pd.concat(dataframes, ignore_index=True)
                if not features_df.empty:
                    ml_signals = generate_ml_signals(features_df, ctx)
                
                # GA signals per symbol
                for symbol, data in market_data.items():
                    if data and "market_state" in data:
                        signals = generate_ga_signals(
                            data["candles"],
                            population,
                            ctx
                        )
                        ga_signals.extend(signals)
            
            # Combine and validate signals
            final_signals = combine_signals(ml_signals, ga_signals, ctx)
            valid_signals = [sig for sig in final_signals if validate_signal(sig, ctx)]
            
            # Process valid signals
            await process_signals(valid_signals, market_data, ctx)
            
            # Manage positions
            await manage_positions(market_data, ctx)
            
            # Update metrics
            update_daily_performance(ctx)
            log_performance_summary(ctx)
            
            # Sleep until next iteration
            elapsed = time.time() - start_time
            wait = max(0, ctx.config.get("execution_interval", 60) - elapsed)
            await asyncio.sleep(wait)
        
        except Exception as e:
            handle_error(e, "Bot.main_loop", logger=ctx.logger)
            await asyncio.sleep(5)

async def initialize_and_run() -> None:
    """Initialize and run the trading bot"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TradingBot")
    
    # Create context
    ctx = TradingContext(
        logger=logger,
        config=load_config(),
        db_pool=None,
        exchange=None
    )
    
    # Initialize components
    ctx.db_pool = ctx.config.get("database", {}).get("path", "data/candles.db")
    ctx.ratchet_manager = RatchetManager(ctx)
    ctx.market_data = MarketData(ctx)
    
    safe_load_ml_models(ctx)
    ctx.exchange = await create_exchange(ctx)
    
    try:
        await main_loop(ctx)
    finally:
        if ctx.exchange:
            await ctx.exchange.close()

def main() -> None:
    """Entry point"""
    try:
        asyncio.run(initialize_and_run())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")

if __name__ == "__main__":
    main()