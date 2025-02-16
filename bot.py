#!/usr/bin/env python3
import asyncio
import json
import logging
import os

from src.database.database import DatabaseConnection
from src.execution.exchange_interface import ExchangeInterface
from src.execution.market_data import MarketData

# Import core modules
from src.startup.system_init import SystemInitializer
from src.trading.portfolio import PortfolioManager
from src.utils.health_monitor import HealthMonitor


async def main():
    # Load configuration from config/config.json if available
    config_path = os.path.join("config", "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        config = {}

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("Bot")

    # Create a minimal context with required attributes
    class Context:
        pass

    ctx = Context()
    ctx.config = config
    ctx.logger = logger

    # Stub implementations for components
    # Initialize database connection (using a file 'bot.db' in the working directory)
    ctx.db_connection = DatabaseConnection("bot.db")

    # Initialize Exchange Interface
    ctx.exchange_interface = ExchangeInterface(ctx)

    # Initialize Market Data
    ctx.market_data = MarketData(ctx)

    # Initialize Portfolio Manager
    ctx.portfolio_manager = PortfolioManager(ctx)

    # For risk management, assign a stub (could be extended later)
    ctx.risk_manager = None

    # db_queries stub
    ctx.db_queries = None

    # Initialize components sequentially
    logger.info("Initializing Database Connection...")
    try:
        await ctx.db_connection.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize DB Connection: {e}")

    logger.info("Initializing Exchange Interface...")
    try:
        await ctx.exchange_interface.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Exchange Interface: {e}")

    logger.info("Initializing Market Data service...")
    try:
        await ctx.market_data.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Market Data: {e}")

    logger.info("Initializing Portfolio Manager...")
    try:
        await ctx.portfolio_manager.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Portfolio Manager: {e}")

    logger.info("Initializing Health Monitor...")
    health_monitor = HealthMonitor(ctx)
    try:
        await health_monitor.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Health Monitor: {e}")

    logger.info("Bot started successfully.")
    logger.info("Entering main loop. Press Ctrl+C to stop.")

    # Main loop: simply sleep and log periodically
    try:
        while True:
            await asyncio.sleep(5)
            logger.info("Bot running...")
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
