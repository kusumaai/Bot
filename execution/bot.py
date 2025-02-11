#!/usr/bin/env python3
"""
Module: execution/bot.py
Production‑ready main bot orchestrator.
"""

import asyncio
import time
import json
import os
import logging
import sys
import signal
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from decimal import Decimal

# Core trading components
from risk.manager import RiskManager
from risk.position import Position
from risk.portfolio import PortfolioManager
from risk.limits import RiskLimits

# Market data and signals
from market.data import MarketData
from models.signals import MLSignal
from signals.ga_synergy import generate_ga_signals
from signals.utils import combine_signals, validate_signal

# Exchange and execution
from exchange.client import ExchangeClient
from exchange.orders import OrderManager

# System components
from utils.health_monitor import HealthMonitor
from utils.circuit_breaker import CircuitBreaker
from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TradingContext:
    config: dict
    logger: logging.Logger
    db_pool: str
    running: bool = True
    exchange: Any = None
    market_data: Optional[MarketData] = None
    risk_manager: Optional[RiskManager] = None
    ratchet_manager: Optional[RatchetManager] = None
    ml_models: Optional[Dict[str, Any]] = None

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def safe_load_ml_models(ctx: TradingContext) -> None:
    try:
        ctx.logger.info("Loading ML models …")
        ctx.ml_models = load_models(ctx.config.get("model_path"), ctx.config.get("scaler_path"))
        ctx.logger.info("ML models loaded.")
    except Exception as e:
        ctx.logger.error(f"Error loading ML models: {e}")
        ctx.ml_models = None

async def main_loop(ctx: TradingContext) -> None:
    while ctx.running:
        try:
            symbols: List[str] = ctx.config.get("market_list", [])
            market_data_dict, _ = await ctx.market_data.load_market_data(symbols)
            if not market_data_dict:
                ctx.logger.warning("No market data available.")
                await asyncio.sleep(5)
                continue

            all_signals = []
            for symbol, mdata in market_data_dict.items():
                ml_signal = generate_ml_signals(mdata, ctx.ml_models)
                ga_signal = generate_ga_signals(mdata)
                signal_combined = combine_signals([ml_signal, ga_signal])
                if validate_signal(signal_combined):
                    signal_combined["symbol"] = symbol
                    all_signals.append(signal_combined)
                    ctx.logger.info(f"Signal generated for {symbol}: {signal_combined}")

            for signal_data in all_signals:
                symbol = signal_data["symbol"]
                mdata = market_data_dict[symbol]
                params = calculate_position_params(signal_data, mdata, ctx)
                position_size = params.get("position_size", 0)
                if position_size > 0:
                    order = await place_order(symbol, "buy", position_size, mdata["current_price"], ctx)
                    if order:
                        ctx.logger.info(f"Order placed for {symbol}: {order}")
                        record_new_trade(order, ctx)
                        stop_loss = params.get("stop_loss", mdata["current_price"] * 0.98)
                        take_profit = params.get("take_profit", mdata["current_price"] * 1.03)
                        added = ctx.risk_manager.add_position(symbol, "long", position_size, mdata["current_price"], stop_loss, take_profit)
                        if not added:
                            ctx.logger.error(f"Failed to add position for {symbol}")
                    else:
                        ctx.logger.error(f"Order placement failed for {symbol}")
                else:
                    ctx.logger.info(f"Position parameters invalid for signal on {symbol}.")

            current_positions = list(ctx.risk_manager.positions.keys())
            for symbol in current_positions:
                current_price = await fetch_ticker(symbol, ctx)
                if current_price <= 0:
                    continue
                exit_decision = ctx.risk_manager.update_position(symbol, current_price)
                if exit_decision and exit_decision.get("action") == "close":
                    reason = exit_decision.get("reason")
                    if await close_position(ctx.risk_manager.positions[symbol], ctx):
                        trade_record = ctx.risk_manager.close_position(symbol, current_price, reason)
                        if trade_record:
                            update_trade_result(trade_record, ctx)
                            ctx.logger.info(f"Closed position for {symbol}. Reason: {reason}")
                        else:
                            ctx.logger.error(f"Failed to log closed position for {symbol}")
                    else:
                        ctx.logger.error(f"Close order failed for {symbol}")

            asyncio.create_task(ctx.ratchet_manager.monitor_trades(ctx.exchange))
            update_daily_performance(ctx)
            log_performance_summary(ctx)
            await asyncio.sleep(ctx.config.get("trading_loop_interval", 5))
        except Exception as e:
            ctx.logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(5)

async def shutdown_bot(ctx: TradingContext) -> None:
    ctx.logger.info("Shutting down …")
    try:
        if ctx.exchange:
            await ctx.exchange.close()
        ctx.logger.info("Exchange connection closed.")
    except Exception as e:
        ctx.logger.error(f"Error during shutdown: {e}")

async def initialize_and_run() -> None:
    ctx: Optional[TradingContext] = None
    try:
        logger = logging.getLogger("TradingBot")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        config = load_config()
        db_pool = config.get("db_path", "data/trading.db")
        ctx = TradingContext(config=config, logger=logger, db_pool=db_pool)
        ctx.running = True

        ctx.market_data = MarketData(ctx)
        ctx.exchange = await create_exchange(ctx)
        ctx.risk_manager = RiskManager(ctx)
        ctx.ratchet_manager = RatchetManager(ctx)
        safe_load_ml_models(ctx)

        if not validate_account({}, ctx):
            logger.warning("Account validation failed on startup.")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: setattr(ctx, "running", False))

        await main_loop(ctx)
    except Exception as e:
        if ctx and ctx.logger:
            ctx.logger.error(f"Bot initialization failed: {e}")
        else:
            print(f"Bot initialization failed: {e}")
        raise
    finally:
        if ctx:
            await shutdown_bot(ctx)

def main() -> None:
    try:
        asyncio.run(initialize_and_run())
    except KeyboardInterrupt:
        print("\nShutting down gracefully…")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()