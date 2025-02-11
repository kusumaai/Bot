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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core components
from execution.market_data import MarketData
from execution.position_manager import calculate_position_params, should_close_position
from execution.exchange_interface import create_exchange, fetch_ticker, place_order, close_position
from trading.ratchet import RatchetManager

# Signal components
from models.ml_signal import generate_ml_signals, load_models
from signals.ga_synergy import generate_ga_signals
from signals.population import initialize_population
from signals.signal_utils import combine_signals, validate_signal

# Accounting components
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
    db_pool: str
    exchange: Optional[Any] = None
    rf_model: Optional[Any] = None
    xgb_model: Optional[Any] = None
    ratchet_manager: Optional[Any] = None
    market_data: Optional[Any] = None

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
                ctx.logger.debug(f"Account validation failed for {signal['symbol']}")
                continue
                
            current_price = await fetch_ticker(signal["symbol"], ctx)
            if current_price <= 0:
                ctx.logger.warning(f"Invalid price for {signal['symbol']}: {current_price}")
                continue
            
            params = calculate_position_params(signal, market_data[signal["symbol"]], ctx)
            if params["position_size"] <= 0:
                ctx.logger.debug(f"Invalid position size for {signal['symbol']}")
                continue
            
            order = await place_order(
                signal["symbol"],
                signal["direction"],
                params["position_size"],
                current_price,
                ctx
            )
            
            if order:
                # Initialize ratchet tracking
                ctx.ratchet_manager.initialize_trade(order["id"], signal["entry_price"])
                
                # Record the trade
                record_new_trade(
                    order,
                    signal,
                    params["expected_value"],
                    params["kelly_fraction"],
                    params["position_size"],
                    ctx,
                    "paper" if ctx.config.get("paper_mode", False) else "real"
                )
                
                ctx.logger.info(
                    f"New trade opened: {order['id']} - {signal['symbol']} "
                    f"Direction: {signal['direction']} Size: {params['position_size']:.4f}"
                )
        
        except Exception as e:
            handle_error(e, f"Bot.process_signals for {signal.get('symbol', 'unknown')}", logger=ctx.logger)

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
            
            if should_close_position(trade, current_price, market_data[trade["symbol"]], ctx):
                if await close_position(trade, ctx):
                    pnl = calculate_pnl(trade, current_price)
                    
                    # Clean up trade
                    ctx.ratchet_manager.remove_trade(trade["id"])
                    update_trade_result(trade["id"], pnl, ctx)
                    
                    ctx.logger.info(
                        f"Closed trade {trade['id']} ({trade['symbol']}) "
                        f"PNL: {pnl:.2f} Direction: {trade['direction']}"
                    )
    
    except Exception as e:
        handle_error(e, "Bot.manage_positions", logger=ctx.logger)

def calculate_pnl(trade: Dict[str, Any], current_price: float) -> float:
    """Calculate PnL for a trade"""
    multiplier = 1 if trade["direction"] == "long" else -1
    price_diff = current_price - trade["entry_price"]
    return multiplier * price_diff * trade["position_size"]

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
            ml_signals, ga_signals = [], []
            
            if dataframes:
                # ML signals from combined features
                features_df = pd.concat(dataframes, ignore_index=True)
                if not features_df.empty:
                    ml_signals = generate_ml_signals(features_df, ctx)
                
                # GA signals per symbol
                for symbol, data in market_data.items():
                    if data and "market_state" in data:
                        symbol_signals = generate_ga_signals(
                            data["candles"],
                            population,
                            ctx
                        )
                        ga_signals.extend(symbol_signals)
            
            # Combine and validate signals
            final_signals = combine_signals(ml_signals, ga_signals, ctx)
            valid_signals = [sig for sig in final_signals if validate_signal(sig, ctx)]
            
            if valid_signals:
                ctx.logger.info(f"Processing {len(valid_signals)} valid signals")
                await process_signals(valid_signals, market_data, ctx)
            
            # Manage existing positions
            await manage_positions(market_data, ctx)
            
            # Update metrics
            update_daily_performance(ctx)
            log_performance_summary(ctx)
            
            # Sleep until next iteration
            elapsed = time.time() - start_time
            wait = max(0, ctx.config.get("execution_interval", 60) - elapsed)
            
            ctx.logger.debug(f"Loop completed in {elapsed:.2f}s, waiting {wait:.2f}s")
            await asyncio.sleep(wait)
        
        except Exception as e:
            handle_error(e, "Bot.main_loop", logger=ctx.logger)
            await asyncio.sleep(5)

async def initialize_and_run() -> None:
    """Initialize and run the trading bot"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("TradingBot")
        
        # Load config
        config = load_config()
        
        # Create context
        ctx = TradingContext(
            logger=logger,
            config=config,
            db_pool=config.get("database", {}).get("path", "data/candles.db")
        )
        
        # Initialize components
        ctx.ratchet_manager = RatchetManager(ctx)
        ctx.market_data = MarketData(ctx)
        ctx.exchange = await create_exchange(ctx)
        
        safe_load_ml_models(ctx)
        
        # Start main loop
        await main_loop(ctx)
        
    except Exception as e:
        logger.error(f"Bot initialization failed: {e}")
        raise
        
    finally:
        if hasattr(ctx, 'exchange') and ctx.exchange:
            await ctx.exchange.close()

def main() -> None:
    """Entry point"""
    try:
        asyncio.run(initialize_and_run())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()