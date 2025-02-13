#!/usr/bin/env python3
"""
Module: execution/bot.py
Main trading bot orchestrator with paper/live trading support and proper risk management.
"""
import os
import json
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from config.config import load_config 
from risk.manager import RiskManager
from trading.portfolio import PortfolioManager
from execution.market_data import MarketData
from execution.exchange_interface import ExchangeInterface
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from database.database import DatabaseConnection
from utils.error_handler import handle_error_async

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

async def initialize_components(ctx: TradingContext) -> bool:
    """Initialize all components in the correct order"""
    try:
        # 1. Database first
        ctx.db_connection = DatabaseConnection(
            ctx.config.get("database", {}).get("path", "data/trading.db")
        )
        if not await ctx.db_connection.initialize():
            ctx.logger.error("Failed to initialize database connection")
            return False

        # 2. Risk Manager (needed by other components)
        ctx.risk_manager = RiskManager(ctx)
        if not await ctx.risk_manager.initialize():
            ctx.logger.error("Failed to initialize risk manager")
            return False

        # 3. Portfolio Manager
        ctx.portfolio_manager = PortfolioManager(ctx)
        if not await ctx.portfolio_manager.initialize():
            ctx.logger.error("Failed to initialize portfolio manager")
            return False

        # 4. Exchange Interface
        ctx.exchange_interface = ExchangeInterface(ctx)
        if not await ctx.exchange_interface.initialize():
            ctx.logger.error("Failed to initialize exchange interface")
            return False

        # 5. Market Data
        ctx.market_data = MarketData(ctx)
        if not await ctx.market_data.initialize():
            ctx.logger.error("Failed to initialize market data")
            return False

        # 6. Circuit Breaker
        ctx.circuit_breaker = CircuitBreaker(ctx)
        if not await ctx.circuit_breaker.initialize():
            ctx.logger.error("Failed to initialize circuit breaker")
            return False

        # 7. Health Monitor
        ctx.health_monitor = HealthMonitor(ctx)
        if not await ctx.health_monitor.initialize():
            ctx.logger.error("Failed to initialize health monitor")
            return False

        # 8. Ratchet Manager
        ctx.ratchet_manager = RatchetManager(ctx)
        if not await ctx.ratchet_manager.initialize():
            ctx.logger.error("Failed to initialize ratchet manager")
            return False

        ctx.logger.info("All components initialized successfully")
        return True

    except Exception as e:
        ctx.logger.error(f"Error during component initialization: {str(e)}")
        return False

async def run_bot():
    """Main entry point for the trading bot"""
    ctx = TradingContext()
    
    if not await initialize_components(ctx):
        ctx.logger.error("Failed to initialize components")
        return

    try:
        while ctx.running:
            if ctx.health_monitor.should_emergency_shutdown():
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