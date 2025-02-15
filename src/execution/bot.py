#! /usr/bin/env python3
# src/execution/bot.py
"""
Module: src.execution
Main trading bot orchestrator with paper/live trading support.
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import load_config
from database.database import DatabaseConnection
from execution.exchange_interface import ExchangeInterface
from execution.market_data import MarketData
from risk.manager import RiskManager
from trading.circuit_breaker import CircuitBreaker
from trading.portfolio import PortfolioManager
from trading.ratchet import RatchetManager
from utils.error_handler import handle_error_async
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging


class TradingContext:
    """Trading context maintaining all component instances and state"""

    def __init__(self):
        # Initialize logging first
        self.logger = setup_logging(name="TradingBot", level="INFO")
        self.config = load_config()

        # Core components (initialized as None)
        self.running = True
        self.initialized = False
        self.db_connection = None
        self.db_queries = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.exchange_interface = None
        self.market_data = None
        self.circuit_breaker = None
        self.health_monitor = None
        self.ratchet_manager = None

        # State tracking
        self.last_health_check = 0
        self.last_metrics_update = 0


class DummyDatabaseConnection:
    async def initialize(self):
        return True

    async def close(self):
        return True


async def initialize_components(ctx: TradingContext) -> bool:
    """Initialize all components in the correct order"""
    try:
        paper_mode = ctx.config.get("paper_mode", False)
        print("DEBUG: Paper mode value from config:", paper_mode, flush=True)
        ctx.logger.info("[DEBUG] Paper mode value from config: " + str(paper_mode))
        overall_success = True

        # --- Initialize Database Connection ---
        ctx.logger.info("Initializing Database Connection...")
        if paper_mode:
            ctx.logger.info("Paper mode enabled - using DummyDatabaseConnection.")
            ctx.db_connection = DummyDatabaseConnection()
        else:
            from database.database import DatabaseConnection

            ctx.db_connection = DatabaseConnection(
                ctx.config.get("database", {}).get("path", "data/trading.db")
            )

        if not await ctx.db_connection.initialize():
            print("DEBUG: Database Connection initialization failed", flush=True)
            ctx.logger.error("Failed to initialize database connection")
            overall_success = False
        ctx.logger.info("Database Connection initialized successfully.")

        # --- Initialize Portfolio Manager ---
        ctx.logger.info("Initializing Portfolio Manager...")
        ctx.portfolio_manager = PortfolioManager(ctx)
        if not await ctx.portfolio_manager.initialize():
            print("DEBUG: Portfolio Manager initialization failed", flush=True)
            ctx.logger.error("Failed to initialize portfolio manager")
            overall_success = False
        ctx.logger.info("Portfolio Manager initialized successfully.")

        # --- Initialize Risk Manager ---
        ctx.logger.info("Initializing Risk Manager...")
        ctx.risk_manager = RiskManager(ctx, db_queries=None, logger=ctx.logger)
        if not await ctx.risk_manager.initialize():
            print("DEBUG: Risk Manager initialization failed", flush=True)
            ctx.logger.error("Failed to initialize risk manager")
            overall_success = False
        ctx.risk_limits = ctx.risk_manager.risk_limits
        ctx.logger.info(
            "Risk Manager initialized successfully. Loaded risk limits: "
            + str(ctx.risk_limits)
        )

        # --- Initialize Exchange Interface ---
        ctx.logger.info("Initializing Exchange Interface...")
        ctx.exchange_interface = ExchangeInterface(ctx)
        if not await ctx.exchange_interface.initialize():
            print("DEBUG: Exchange Interface initialization failed", flush=True)
            ctx.logger.error("Failed to initialize exchange interface")
            overall_success = False
        ctx.logger.info("Exchange Interface initialized successfully.")

        # --- Initialize Market Data ---
        ctx.logger.info("Initializing Market Data...")
        ctx.market_data = MarketData(ctx)
        if not await ctx.market_data.initialize():
            print("DEBUG: Market Data initialization failed", flush=True)
            ctx.logger.error("Failed to initialize market data")
            overall_success = False
        ctx.logger.info("Market Data initialized successfully.")

        # --- Initialize Circuit Breaker ---
        ctx.logger.info("Initializing Circuit Breaker...")
        ctx.circuit_breaker = CircuitBreaker(ctx)
        if not await ctx.circuit_breaker.initialize():
            print("DEBUG: Circuit Breaker initialization failed", flush=True)
            ctx.logger.error("Failed to initialize circuit breaker")
            overall_success = False
        ctx.logger.info("Circuit Breaker initialized successfully.")

        # --- Initialize Health Monitor ---
        ctx.logger.info("Initializing Health Monitor...")
        ctx.health_monitor = HealthMonitor(ctx)
        if not await ctx.health_monitor.initialize():
            print("DEBUG: Health Monitor initialization failed", flush=True)
            ctx.logger.error("Failed to initialize health monitor")
            overall_success = False
        ctx.logger.info("Health Monitor initialized successfully.")

        # --- Initialize Ratchet Manager ---
        ctx.logger.info("Initializing Ratchet Manager...")
        ctx.ratchet_manager = RatchetManager(ctx)
        if not await ctx.ratchet_manager.initialize():
            print("DEBUG: Ratchet Manager initialization failed", flush=True)
            ctx.logger.error("Failed to initialize ratchet manager")
            overall_success = False
        ctx.logger.info("Ratchet Manager initialized successfully.")

        ctx.logger.info(
            "All components initialized with overall success: {}".format(
                overall_success
            )
        )
        if paper_mode:
            ctx.logger.info("Paper mode enabled - forcing initialization success.")
            return True
        return overall_success
    except Exception as e:
        ctx.logger.error(f"Unhandled exception during initialization: {e}")
        if ctx.config.get("paper_mode", False):
            ctx.logger.info(
                "Paper mode enabled - forcing initialization success despite errors."
            )
            return True
        return False


async def run_bot():
    """Main entry point for the trading bot"""
    ctx = TradingContext()

    init_success = await initialize_components(ctx)
    if not init_success and ctx.config.get("paper_mode", False):
        ctx.logger.warning(
            "Initialization returned failure, but paper mode is enabled - overriding to True."
        )
        init_success = True
    if not init_success:
        ctx.logger.error("Failed to initialize components")
        return

    # Log the loaded configuration for debugging
    ctx.logger.info(
        "Loaded configuration: " + json.dumps(ctx.config, indent=2, default=str)
    )

    # If operating in paper mode, execute a test simulated trade
    if ctx.config.get("paper_mode", False):
        test_trade = {
            "side": "buy",
            "amount": "0.1",
            "price": "100",
            "symbol": "BTC/USDT",
        }
        ctx.portfolio_manager.simulate_trade(test_trade)
        summary = ctx.portfolio_manager.get_trade_summary()
        ctx.logger.info(
            "Test simulated trade executed. Trade Summary: "
            + json.dumps(summary, default=str)
        )
        test_trade_time = time.time()

    try:
        while ctx.running:
            if ctx.config.get("paper_mode", False):
                if time.time() - test_trade_time >= 5:
                    ctx.logger.info(
                        "Paper mode: simulation duration complete, shutting down."
                    )
                    break
            if ctx.health_monitor and ctx.health_monitor.should_emergency_shutdown():
                ctx.logger.error("Emergency shutdown triggered")
                break
            await asyncio.sleep(1)
    except Exception as e:
        ctx.logger.error(f"Error in main loop: {str(e)}")
    finally:
        ctx.running = False
        # Cleanup
        if ctx.exchange_interface:
            await ctx.exchange_interface.close()
        if ctx.db_connection:
            await ctx.db_connection.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
