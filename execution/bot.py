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

from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from config.config import load_config 
from risk.manager import RiskManager
from trading.portfolio import PortfolioManager
from risk.validation import MarketDataValidation
from execution.market_data import MarketData
from signals.ga_synergy import generate_ga_signals
from utils.numeric_handler import NumericHandler
from utils.exceptions import InvalidOrderError
from execution.exchange_interface import ExchangeInterface
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from trading.circuit_breaker import CircuitBreaker
from trading.ratchet import RatchetManager
from database.queries import DatabaseQueries, QueryBuilder, execute_sql
from utils.error_handler import handle_error, handle_error_async
from execution.main_loop import main_loop

class TradingContext:
    """Trading context maintaining all component instances and state"""
    def __init__(self):
        # Initialize logging first
        self.logger = setup_logging(name="TradingBot", level="INFO")
        self.config = load_config()
        
        # Core components
        self.running = True
        self.exchange_interface = None
        self.market_data = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.circuit_breaker = None
        self.health_monitor = None
        self.ratchet_manager = None
        self.market_validator = None
         
        # State tracking
        self.last_health_check = 0
        self.last_metrics_update = 0
        self.db_queries = None

async def initialize_components(ctx: TradingContext) -> bool:
    """Initialize all trading bot components in correct order"""
    try:
        # Initialize database first with proper connection
        ctx.db_queries = DatabaseQueries(
            db_path=ctx.config.get("database", {}).get("path", "trading.db"),
            logger=ctx.logger
        )
        
        # Initialize database connection
        if not await ctx.db_queries.initialize():
            ctx.logger.error("Database initialization failed")
            return False
            
        # Initialize portfolio manager first as other components depend on its risk limits
        try:
            ctx.portfolio_manager = PortfolioManager(ctx)
            if not await ctx.portfolio_manager.initialize():
                ctx.logger.error("Portfolio manager initialization failed")
                return False
        except ValueError as e:
            ctx.logger.error(f"Portfolio initialization failed: {str(e)}")
            return False
            
        # Initialize risk manager next
        ctx.risk_manager = RiskManager(ctx)
        if not await ctx.risk_manager.initialize():
            ctx.logger.error("Risk manager initialization failed")
            return False
            
        # Initialize exchange interface
        ctx.exchange_interface = ExchangeInterface(ctx)
        if not await ctx.exchange_interface.initialize():
            ctx.logger.error("Exchange interface initialization failed")
            return False
            
        # Initialize market data with validated risk limits
        ctx.market_data = MarketData(ctx)
        if not await ctx.market_data.initialize():
            ctx.logger.error("Market data initialization failed")
            return False
        
        # Initialize monitoring components
        ctx.circuit_breaker = CircuitBreaker(ctx)
        if not await ctx.circuit_breaker.initialize():
            ctx.logger.error("Circuit breaker initialization failed")
            return False
            
        ctx.health_monitor = HealthMonitor(ctx)
        ctx.ratchet_manager = RatchetManager(ctx)
        if not await ctx.ratchet_manager.initialize():
            ctx.logger.error("Ratchet manager initialization failed")
            return False
            
        ctx.market_validator = MarketDataValidation(ctx.risk_manager.risk_limits, ctx.logger)
        
        ctx.logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        if ctx and ctx.logger:
            ctx.logger.error(f"Component initialization failed: {str(e)}")
        else:
            print(f"Critical initialization error: {str(e)}")
        return False

async def main():
    """Main entry point for trading bot"""
    ctx = TradingContext()
    
    try:
        init_success = await initialize_components(ctx)
        if not init_success:
            ctx.logger.error("Failed to initialize components")
            return
            
        ctx.logger.info("Starting main trading loop")
        await main_loop(ctx)
        
    except Exception as e:
        if ctx and ctx.logger:
            await handle_error_async(e, "main", logger=ctx.logger)
        else:
            print(f"Critical error in main: {str(e)}")
    finally:
        ctx.running = False

if __name__ == "__main__":
    print("execution module loaded.")
    asyncio.run(main())