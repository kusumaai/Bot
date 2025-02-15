#!/usr /bin/env python3
# backtesting/backtester.py
"""
Module: backtesting/backtester.py
Comprehensive backtesting framework with proper risk management
"""
import asyncio
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# import the necessary libraries
import pandas as pd
import pandas_ta as ta

from database.database import DatabaseConnection
from indicators.indicators_pta import compute_indicators
from signals.evaluation import evaluate_rule
from signals.ga_synergy import generate_ga_signals
from signals.population import create_baseline_rule
from signals.trading_types import MarketState
from trading.math import (
    calculate_expected_value,
    calculate_kelly_fraction,
    calculate_position_size,
    calculate_trend_probability,
    estimate_volatility,
    predict_next_return,
)
from trading.ratchet import RatchetManager
from utils.error_handler import handle_error_async
from utils.exceptions import BackTestError
from utils.numeric_handler import NumericHandler


# dataclass for the backtest results
@dataclass
class BacktestResults:
    """Container for backtest results"""

    # Required fields (no defaults)
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series

    # Optional fields (with defaults)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Total Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate:.2%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Total Return: {self.total_return:.2%}"
        )


# dataclass for the backtest config
@dataclass
class BacktestConfig:
    warmup_periods: int = 233  # Was magic number
    position_size_limit: Decimal = Decimal("0.10")  # Was hardcoded 10%
    min_trade_size: Decimal = Decimal("0.01")
    commission: Decimal = Decimal("0.001")  # 0.1% commission
    stop_loss_pct: Decimal = Decimal("0.02")  # 2% stop loss
    take_profit_pct: Decimal = Decimal("0.03")  # 3% take profit


# class for the backtester
class Backtester:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger or logging.getLogger(__name__)
        self.initialized = False
        self.config = ctx.config
        self.nh = NumericHandler()
        self.db_connection = None

        # These will be set during initialization
        self.portfolio = None
        self.ratchet_manager = None
        self.current_balance = None
        self.trades = []

    # get the database connection for the backtester
    async def get_db_connection(self) -> Optional[DatabaseConnection]:
        """Get database connection from context or create new one"""
        if self.db_connection:
            return self.db_connection

        try:
            db_path = self.ctx.config.get("database", {}).get("path", "data/trading.db")
            self.db_connection = DatabaseConnection(db_path)
            return self.db_connection
        except Exception as e:
            await handle_error_async(e, "Backtester.get_db_connection", self.logger)
            return None

    # initialize the backtester
    async def initialize(self) -> bool:
        try:
            if self.initialized:
                return True

            # Initialize database connection first
            self.db_connection = await self.get_db_connection()
            if not self.db_connection:
                self.logger.error("Failed to obtain database connection")
                return False

            if (
                not self.ctx.portfolio_manager
                or not self.ctx.portfolio_manager.initialized
            ):
                self.logger.error("Portfolio manager must be initialized first")
                return False

            if not self.ctx.ratchet_manager or not self.ctx.ratchet_manager.initialized:
                self.logger.error("Ratchet manager must be initialized first")
                return False

            self.portfolio = self.ctx.portfolio_manager
            self.ratchet_manager = self.ctx.ratchet_manager
            self.current_balance = Decimal(
                str(self.config.get("initial_balance", "10000"))
            )

            self.initialized = True
            return True

        except Exception as e:
            await handle_error_async(e, "Backtester.initialize", self.logger)
            return False

    # execute the trade for the backtester
    async def _execute_trade(
        self,
        signal: Dict[str, Any],
        position_size: Decimal,
        entry_candle: pd.Series,
        future_data: pd.DataFrame,
        market_state: Any,
    ) -> Optional[Dict[str, Any]]:
        """Execute trade with comprehensive risk management"""
        try:
            if not self.initialized:
                await self.initialize()
                if not self.initialized:
                    raise BackTestError("Backtester not initialized")

            entry_price = self.nh.to_decimal(entry_candle["close"])
            entry_cost = (
                position_size
                * entry_price
                * self.config.get("commission", Decimal("0.001"))
            )

            # Calculate max position size based on current balance
            max_position = (
                self.current_balance * self.portfolio.risk_limits.max_position_size
            )
            position_size = min(position_size, max_position / entry_price)

            if position_size < self.portfolio.risk_limits.min_position_size:
                self.logger.error("Position size below minimum allowed")
                return None

            # Set stops using risk limits
            stop_loss = entry_price * (
                Decimal("1") - self.portfolio.risk_limits.risk_factor
            )
            take_profit = entry_price * (
                Decimal("1") + self.portfolio.risk_limits.risk_factor * Decimal("2")
            )

            trade_id = str(uuid.uuid4())
            await self.ratchet_manager.initialize_trade(
                trade_id, float(entry_price), float(take_profit), float(stop_loss)
            )

            return {
                "trade_id": trade_id,
                "entry_price": entry_price,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

        except Exception as e:
            await handle_error_async(e, "Backtester._execute_trade", self.logger)
            return None

    # run the backtest for the backtester
    async def run_backtest(self, historical_data: pd.DataFrame) -> None:
        """Run the backtesting simulation"""
        for index, candle in historical_data.iterrows():
            # Generate signal based on historical data
            signal = generate_ga_signals(candle)

            if not signal:
                continue

            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(signal)

            # Execute trade
            trade = await self._execute_trade(
                signal=signal,
                position_size=position_size,
                entry_candle=candle,
                future_data=historical_data.loc[index + 1 :],
                market_state=None,  # Placeholder for actual market state
            )

            # Optionally, implement trade tracking and exit conditions
            if trade:
                self.logger.info(
                    f"Trade placed: {trade['trade_id']} for {trade['entry_price']}"
                )

                # Implement trade exit logic based on future data
                # ...

        self.logger.info("Backtesting completed.")
        self.logger.info(f"Final Balance: {self.current_balance}")
        self.logger.info(f"Total Trades Executed: {len(self.trades)}")


# run the backtest for the backtester
def run_backtest(start_date: str, end_date: str, ctx: Any) -> BacktestResults:
    """Run backtest for specified period"""
    backtester = Backtester(ctx)
    data = backtester.load_data(start_date, end_date)

    if data.empty:
        ctx.logger.error("No data available for backtesting")
        return BacktestResults(0, 0.0, 0.0, 0.0, 0.0, 0.0, [], pd.Series([]))

    return backtester.simulate_trades(data)


# run the backtest for the backtester
if __name__ == "__main__":
    import json
    import logging
    import os
    from dataclasses import dataclass

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("TradingBot")

    # Load config.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "config", "config.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        config = {}

    # dataclass for the context of the backtester
    @dataclass
    class Context:
        logger: logging.Logger
        config: dict
        db_pool: str

    # Create context with actual config
    ctx = Context(
        logger=logger,
        config=config,
        db_pool=os.path.join(project_root, "data", "candles.db"),
    )

    try:
        # Run backtest for last month
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running backtest from {start_date} to {end_date}")
        results = run_backtest(start_date, end_date, ctx)

        print("\nBacktest Results:")
        print("=" * 50)
        print(results)

        # Additional analytics
        if results.trades:
            df_trades = pd.DataFrame(results.trades)
            print("\nTrade Statistics:")
            print("-" * 30)
            print(f"Average Trade PnL: {df_trades['pnl'].mean():.2f}")
            print(f"Average Win: {df_trades[df_trades['pnl'] > 0]['pnl'].mean():.2f}")
            print(f"Average Loss: {df_trades[df_trades['pnl'] < 0]['pnl'].mean():.2f}")
            print(f"Largest Win: {df_trades['pnl'].max():.2f}")
            print(f"Largest Loss: {df_trades['pnl'].min():.2f}")
            print(f"Total Fees Paid: {df_trades['fees_paid'].sum():.2f}")

            print("\nExit Reasons:")
            print(df_trades["exit_reason"].value_counts())

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise
