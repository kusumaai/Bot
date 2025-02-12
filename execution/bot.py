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
from risk.portfolio import PortfolioManager
from risk.validation import MarketDataValidation
from execution.market_data import MarketData
from models.ml_signal import generate_ml_signals
from signals.ga_synergy import generate_ga_signals
from trading.math import (
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_expected_value
)
from execution.exchange_interface import ExchangeInterface
from utils.health_monitor import HealthMonitor
from utils.logger import setup_logging
from trading.circuit_breaker import CircuitBreaker, CircuitBreakerState
from trading.ratchet import RatchetManager
from database.database import DBConnection
from utils.error_handler import handle_error


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

async def validate_signal(signal: Dict[str, Any], ctx: TradingContext) -> bool:
    """Validate trading signal meets minimum criteria"""
    try:
        if signal["ml_signal"]["strength"] < ctx.config["min_signal_strength"]:
            return False
        if signal["expected_value"] < ctx.config["min_expected_value"]:
            return False
        if signal["direction"] == "long" and signal["ml_signal"]["prediction"] < ctx.config["ml_long_threshold"]:
            return False
        if signal["direction"] == "short" and signal["ml_signal"]["prediction"] > ctx.config["ml_short_threshold"]:
            return False
        return True
    except Exception as e:
        handle_error(e, "validate_signal", logger=ctx.logger)
        return False

async def process_market_data(ctx: TradingContext, symbols: List[str]) -> Dict:
    """Process market data and generate trading signals"""
    try:
        signals = {}
        for symbol in symbols:
            market_data = await ctx.market_data.get_latest_candles(symbol)
            if not ctx.market_validator.validate_data(market_data):
                ctx.logger.warning(f"❌ Invalid market data for {symbol}")
                continue

            ml_signal = await generate_ml_signals(market_data, ctx.ml_signal)
            ga_signal = await generate_ga_signals(market_data, ctx.config["ga_settings"])
            
            equity = ctx.portfolio_manager.get_equity()
            kelly_fraction = calculate_kelly_fraction(
                ml_signal["win_rate"], 
                ml_signal["profit_ratio"]
            )

            # Calculate position size (max 10% of equity)
            max_position = equity * Decimal(str(ctx.config["max_position_pct"] / 100))
            position_size = min(
                calculate_position_size(
                    equity,
                    ctx.config["risk_factor"],
                    kelly_fraction
                ),
                max_position
            )

            entry_price = Decimal(str(market_data.iloc[-1]["close"]))
            tp_price = entry_price * (1 + Decimal(str(ctx.config["take_profit_pct"] / 100)))
            sl_price = entry_price * (1 - Decimal(str(ctx.config["stop_loss_pct"] / 100)))

            signals[symbol] = {
                "ml_signal": ml_signal,
                "ga_signal": ga_signal,
                "position_size": position_size,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "direction": "long" if ml_signal["prediction"] > ctx.config["ml_long_threshold"] else "short"
            }

            ctx.logger.info(
                f"📊 {symbol} Signal Generated:\n"
                f"Direction: {signals[symbol]['direction']}\n"
                f"Position Size: {position_size:.4f}\n"
                f"Entry: {entry_price:.2f}\n"
                f"TP: {tp_price:.2f}\n"
                f"SL: {sl_price:.2f}"
            )

        return signals

    except Exception as e:
        handle_error(e, "process_market_data", logger=ctx.logger)
        return {}

async def check_trade_closure(ctx: TradingContext, trade: Dict[str, Any]) -> Optional[str]:
    """Determine if and why a trade should be closed"""
    try:
        current_price = await ctx.exchange_interface.fetch_ticker(trade["symbol"])
        entry_time = datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
        hours_open = (datetime.utcnow() - entry_time).total_seconds() / 3600

        # Check max hold time
        if hours_open >= ctx.config["max_hold_hours"]:
            return "TO"

        # Check stop loss
        if trade["direction"] == "long" and current_price <= trade["sl_price"]:
            return "SL"
        elif trade["direction"] == "short" and current_price >= trade["sl_price"]:
            return "SL"

        # Check take profit
        if trade["direction"] == "long" and current_price >= trade["tp_price"]:
            return "TP"
        elif trade["direction"] == "short" and current_price <= trade["tp_price"]:
            return "TP"

        # Check ratchet stop
        ratchet_stop = ctx.ratchet_manager.get_stop_price(trade["id"])
        if ratchet_stop and (
            (trade["direction"] == "long" and current_price <= ratchet_stop) or
            (trade["direction"] == "short" and current_price >= ratchet_stop)
        ):
            return "SP"

        return None

    except Exception as e:
        handle_error(e, "check_trade_closure", logger=ctx.logger)
        return None

async def execute_trades(ctx: TradingContext, signals: Dict) -> None:
    """Execute new trades based on signals"""
    try:
        if not ctx.config.get("allow_new_trades", True):
            ctx.logger.info("🚫 New trades disabled - monitoring existing positions only")
            return

        for symbol, signal in signals.items():
            if not await validate_signal(signal, ctx):
                continue

            if not ctx.risk_manager.check_position_limits(symbol, signal["position_size"]):
                continue

            ctx.logger.info(f"🎯 Placing {signal['direction']} order for {symbol}")
            
            order = await ctx.exchange_interface.place_order(
                symbol=symbol,
                side=signal["direction"],
                amount=signal["position_size"],
                price=None  # Market order
            )

            if order:
                ctx.logger.info(f"✅ Order executed: {symbol} @ {order['price']}")
                await ctx.ratchet_manager.initialize_trade(
                    order["id"],
                    float(order["price"]),
                    signal["tp_price"],
                    signal["sl_price"]
                )

    except Exception as e:
        handle_error(e, "execute_trades", logger=ctx.logger)

async def manage_positions(ctx: TradingContext) -> None:
    """Manage open positions"""
    try:
        open_trades = await ctx.portfolio_manager.get_open_trades()
        
        for trade in open_trades:
            close_reason = await check_trade_closure(ctx, trade)
            
            if close_reason:
                ctx.logger.info(
                    f"🔒 Closing trade {trade['symbol']} - Reason: {close_reason}"
                )
                
                result = await ctx.exchange_interface.close_position(trade)
                if result:
                    await ctx.portfolio_manager.record_trade_closure(
                        trade["id"], 
                        close_reason,
                        await ctx.exchange_interface.fetch_ticker(trade["symbol"])
                    )

    except Exception as e:
        handle_error(e, "manage_positions", logger=ctx.logger)

async def main_loop(ctx: TradingContext) -> None:
    """Main trading loop"""
    ctx.logger.info(
        f"🚀 Starting trading bot\n"
        f"Mode: {'PAPER' if ctx.config['paper_mode'] else 'LIVE'}\n" 
        f"Exchanges: {', '.join(ctx.config['exchanges'])}\n"
        f"Trading Fees: {ctx.config['trading_fees']*100:.2f}%\n"
        f"Slippage: {ctx.config['slippage']*100:.2f}%\n"
        f"Max Position: {ctx.config['max_position_pct']}% of equity\n"
        f"Take Profit: {ctx.config['take_profit_pct']}%"
    )

    while ctx.running:
        try:
            current_time = time.time()
            
            # Health check
            if current_time - ctx.last_health_check >= 30:
                health_status = await ctx.health_monitor.check_system_health()
                if not health_status:
                    ctx.logger.error("🔴 System health check failed")
                    if health_status.get("critical", False):
                        ctx.circuit_breaker.trigger("Critical health check failure")
                    await asyncio.sleep(30)
                    continue
                ctx.last_health_check = current_time

            # Process market data and execute trades
            symbols = ctx.config["market_list"]
            signals = await process_market_data(ctx, symbols)
            
            if ctx.circuit_breaker.state == CircuitBreakerState.OPEN:
                ctx.logger.warning(f"⚡ Circuit breaker active")
                await asyncio.sleep(60)
                continue

            await execute_trades(ctx, signals)
            await manage_positions(ctx)

            # Performance logging
            if current_time - ctx.last_metrics_update >= 300:
                equity = ctx.portfolio_manager.get_equity()
                open_positions = len(await ctx.portfolio_manager.get_open_trades())
                daily_pnl = ctx.portfolio_manager.get_daily_pnl()
                
                ctx.logger.info(
                    f"📈 Performance Update:\n"
                    f"Equity: {equity:.2f} USDT\n"
                    f"Daily PnL: {daily_pnl:+.2%}\n"
                    f"Open Positions: {open_positions}"
                )
                ctx.last_metrics_update = current_time

            await asyncio.sleep(ctx.config["execution_interval"])

        except Exception as e:
            handle_error(e, "main_loop", logger=ctx.logger)
            await asyncio.sleep(10)

def main():
    """Entry point"""
    ctx = TradingContext()
    
    try:
        # Initialize required components in correct order
        ctx.config = load_config()
        
        # Initialize exchange interface and wait for connection
        ctx.exchange_interface = ExchangeInterface(ctx)
        if not asyncio.run(ctx.exchange_interface.initialize()):
            raise RuntimeError("Failed to initialize exchange connection")
        
        # Initialize remaining components in dependency order
        ctx.market_data = MarketData(ctx)
        ctx.portfolio_manager = PortfolioManager(ctx)
        ctx.risk_manager = RiskManager(ctx)
        ctx.circuit_breaker = CircuitBreaker(ctx)
        ctx.health_monitor = HealthMonitor(ctx)
        ctx.ratchet_manager = RatchetManager(ctx)
        
        # Start main loop
        asyncio.run(main_loop(ctx))
        
    except KeyboardInterrupt:
        ctx.logger.info("👋 Shutting down gracefully...")
        ctx.running = False
    except Exception as e:
        handle_error(e, "main", logger=ctx.logger)
    finally:
        if ctx.exchange_interface:
            asyncio.run(ctx.exchange_interface.close())
        ctx.logger.info("✨ Bot shutdown complete")

if __name__ == "__main__":
    main()