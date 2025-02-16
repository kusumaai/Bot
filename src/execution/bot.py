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
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

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

        # Task management
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._cleanup_lock = asyncio.Lock()

    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create and track a new task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self, signal_name: Optional[str] = None):
        """Gracefully shutdown all components and tasks."""
        if signal_name:
            self.logger.info(f"Received {signal_name}, initiating shutdown...")

        async with self._cleanup_lock:  # Prevent multiple simultaneous cleanups
            if not self.running:  # Already shutting down
                return

            self.running = False
            self._shutdown_event.set()

            # Cancel all running tasks
            if self._tasks:
                self.logger.info(f"Cancelling {len(self._tasks)} running tasks...")
                for task in self._tasks:
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks.clear()

            # Cleanup components in reverse initialization order
            components = [
                (self.ratchet_manager, "Ratchet Manager"),
                (self.health_monitor, "Health Monitor"),
                (self.circuit_breaker, "Circuit Breaker"),
                (self.market_data, "Market Data"),
                (self.exchange_interface, "Exchange Interface"),
                (self.portfolio_manager, "Portfolio Manager"),
                (self.risk_manager, "Risk Manager"),
                (self.db_connection, "Database Connection"),
            ]

            for component, name in components:
                if component:
                    try:
                        self.logger.info(f"Shutting down {name}...")
                        if hasattr(component, "close"):
                            await component.close()
                        elif hasattr(component, "shutdown"):
                            await component.shutdown()
                    except Exception as e:
                        self.logger.error(f"Error shutting down {name}: {e}")

            self.logger.info("Shutdown complete")


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
        from database.database import DatabaseConnection

        ctx.db_connection = DatabaseConnection(
            ctx.config.get("database", {}).get("path", "data/trading.db")
        )
        #
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


async def run_background_tasks(ctx: TradingContext):
    """Run background maintenance tasks."""
    try:
        while not ctx._shutdown_event.is_set():
            if ctx.health_monitor and ctx.health_monitor.should_emergency_shutdown():
                ctx.logger.error("Emergency shutdown triggered")
                await ctx.shutdown()
                break

            if ctx.ratchet_manager:
                try:
                    await ctx.ratchet_manager.ratchet()
                except Exception as e:
                    ctx.logger.error(f"Error in ratchet manager: {e}")
                    if not ctx.config.get("paper_mode", False):
                        await ctx.shutdown()
                        break

            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        ctx.logger.error(f"Error in background tasks: {e}")
        await ctx.shutdown()


async def setup_signal_handlers(ctx: TradingContext):
    """Setup signal handlers for graceful shutdown."""
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(ctx.shutdown(signal.Signals(s).name)),
            )
    except NotImplementedError:
        # Windows doesn't support SIGTERM
        pass


async def run_bot():
    """Main entry point for the trading bot with proper task management."""
    ctx = TradingContext()

    try:
        # Setup signal handlers
        await setup_signal_handlers(ctx)

        # Initialize components
        init_success = await initialize_components(ctx)
        if not init_success and not ctx.config.get("paper_mode", False):
            ctx.logger.error("Failed to initialize components")
            return

        # Log configuration
        ctx.logger.info(
            "Loaded configuration: " + json.dumps(ctx.config, indent=2, default=str)
        )

        # Run test trade in paper mode
        if ctx.config.get("paper_mode", False):
            test_trade = {
                "side": "buy",
                "amount": "0.1",
                "price": "100",
                "symbol": "BTC/USDT",
            }
            await ctx.portfolio_manager.simulate_trade(test_trade)
            summary = ctx.portfolio_manager.get_trade_summary()
            ctx.logger.info(
                f"Test simulated trade executed. Trade Summary: {json.dumps(summary, default=str)}"
            )

        # Start background tasks
        background_task = ctx.create_task(
            run_background_tasks(ctx), name="background_tasks"
        )

        # Wait for shutdown signal
        await ctx._shutdown_event.wait()

    except Exception as e:
        ctx.logger.error(f"Critical error in main loop: {e}")
    finally:
        # Ensure cleanup happens
        await ctx.shutdown()


def main():
    """Entry point with proper asyncio handling."""
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass  # Handled by signal handlers
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
