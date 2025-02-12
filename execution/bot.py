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
from config import load_config

from risk.manager import RiskManager
from trading.portfolio import PortfolioManager
from risk.validation import MarketDataValidation
from execution.market_data import MarketData
from models.ga_signal import generate_ml_signals
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
        self.config = None
        self.logger = setup_logging(name="TradingBot", level="INFO")
        self.running = True
        self.exchange_interface = None
        self.market_data = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.circuit_breaker = None
        self.health_monitor = None
        self.ratchet_manager = None
        self.market_validator = None
        self.last_health_check = 0
        self.last_metrics_update = 0

async def initialize_components(ctx: TradingContext) -> bool:
    """Initialize all trading components in dependency order"""
    try:
        # Initialize exchange interface
        ctx.exchange_interface = ExchangeInterface(ctx)
        if not await ctx.exchange_interface.initialize():
            ctx.logger.error("Failed to initialize exchange interface")
            return False
        
        # Initialize market data
        ctx.market_data = MarketData(ctx)
        await ctx.market_data.initialize()
        
        # Initialize portfolio manager
        ctx.portfolio_manager = PortfolioManager(ctx.risk_manager.risk_limits, logger=ctx.logger)
        await ctx.portfolio_manager.initialize()
        
        # Initialize risk manager
        ctx.risk_manager = RiskManager(ctx)
        await ctx.risk_manager.initialize()
        
        # Initialize circuit breaker
        ctx.circuit_breaker = CircuitBreaker(ctx.db_queries, ctx.logger)
        await ctx.circuit_breaker.initialize()
        
        # Initialize ratchet manager
        ctx.ratchet_manager = RatchetManager(ctx)
        await ctx.ratchet_manager.initialize()
        
        # Initialize health monitor
        ctx.health_monitor = HealthMonitor(ctx)
        asyncio.create_task(ctx.health_monitor.start_monitoring())
        
        return True
    except Exception as e:
        await handle_error_async(e, "initialize_components", logger=ctx.logger)
        return False

async def main():
    ctx = TradingContext()
    init_success = await initialize_components(ctx)
    if not init_success:
        ctx.logger.error("Initialization failed. Exiting.")
        return
    
    await main_loop(ctx)

if __name__ == "__main__":
    asyncio.run(main())