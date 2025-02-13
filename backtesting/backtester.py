#!/usr/bin/env python3
"""
Module: backtesting/backtester.py
Comprehensive backtesting framework with proper risk management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
import logging
import pandas_ta as ta
import sys
import asyncio

from database.database import DBConnection, execute_sql
from signals.ga_synergy import generate_ga_signals
from utils.error_handler import handle_error, handle_error_async
from trading.math import (
    calculate_expected_value,
    calculate_kelly_fraction,
    calculate_position_size,
    predict_next_return,
    calculate_trend_probability,
    estimate_volatility
)
from trading.ratchet import RatchetManager
from indicators.indicators_pta import compute_indicators
from signals.trading_types import MarketState
from signals.population import create_baseline_rule
from signals.evaluation import evaluate_rule
from utils.numeric import NumericHandler

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

@dataclass
class BacktestConfig:
    warmup_periods: int = 233  # Was magic number
    position_size_limit: Decimal = Decimal('0.10')  # Was hardcoded 10%
    min_trade_size: Decimal = Decimal('0.01')
    commission: Decimal = Decimal('0.001')  # 0.1% commission
    stop_loss_pct: Decimal = Decimal('0.02')  # 2% stop loss
    take_profit_pct: Decimal = Decimal('0.03')  # 3% take profit

class Backtester:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__)
        self.config = ctx.config
        self.portfolio = ctx.portfolio_manager
        self.ratchet_manager = ctx.ratchet_manager
        self.nh = NumericHandler()
        self.current_balance = Decimal(str(self.config.get("initial_balance", "10000")))
        self.trades = []

    def _execute_trade(
        self,
        signal: Dict[str, Any],
        position_size: Decimal,
        entry_candle: pd.Series,
        future_data: pd.DataFrame,
        market_state: Any
    ) -> Optional[Dict[str, Any]]:
        """Execute trade with comprehensive risk management"""
        
        entry_price = self.nh.to_decimal(entry_candle["close"])
        entry_cost = position_size * entry_price * self.config["commission"]
        
        # Calculate max position size based on current balance
        max_position = self.current_balance * self.config["position_size_limit"]
        position_size = min(position_size, max_position / entry_price)
        
        if position_size < Decimal(str(self.config["min_trade_size"])):
            self.logger.info("Trade size below minimum threshold. Skipping trade.")
            return None
        
        # Set stops using config values
        stop_loss = entry_price * (Decimal('1') - Decimal(str(self.config["stop_loss_pct"])))
        take_profit = entry_price * (Decimal('1') + Decimal(str(self.config["take_profit_pct"])))
        
        trade_id = str(uuid.uuid4())
        self.ratchet_manager.initialize_trade(trade_id, float(entry_price))
        
        trade = {
            "id": trade_id,
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.utcnow(),
            "status": "open"
        }
        
        # Add trade to portfolio
        added = asyncio.run(self.portfolio.add_position(
            symbol=signal["symbol"],
            size=position_size,
            entry_price=entry_price
        ))
        
        if not added:
            self.logger.warning(f"Failed to add position for {signal['symbol']}. Trade aborted.")
            return None
        
        self.logger.info(f"Trade executed: {trade}")
        self.current_balance -= entry_cost
        self.trades.append(trade)
        
        return trade

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
            trade = self._execute_trade(
                signal=signal,
                position_size=position_size,
                entry_candle=candle,
                future_data=historical_data.loc[index+1:],
                market_state=None  # Placeholder for actual market state
            )
            
            # Optionally, implement trade tracking and exit conditions
            if trade:
                self.logger.info(f"Trade placed: {trade['id']} for {trade['symbol']}")
                
                # Implement trade exit logic based on future data
                # ...
                
        self.logger.info("Backtesting completed.")
        self.logger.info(f"Final Balance: {self.current_balance}")
        self.logger.info(f"Total Trades Executed: {len(self.trades)}")


def run_backtest(start_date: str, end_date: str, ctx: Any) -> BacktestResults:
    """Run backtest for specified period"""
    backtester = Backtester(ctx)
    data = backtester.load_data(start_date, end_date)
    
    if data.empty:
        ctx.logger.error("No data available for backtesting")
        return BacktestResults(0, 0.0, 0.0, 0.0, 0.0, 0.0, [], pd.Series([]))
        
    return backtester.simulate_trades(data)


if __name__ == "__main__":
    import logging
    import json
    import os
    from dataclasses import dataclass
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("TradingBot")
    
    # Load config.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "config", "config.json")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config.json: {e}")
        config = {}
    
    @dataclass
    class Context:
        logger: logging.Logger
        config: dict
        db_pool: str
    
    # Create context with actual config
    ctx = Context(
        logger=logger,
        config=config,
        db_pool=os.path.join(project_root, "data", "candles.db")
    )
    
    try:
        # Run backtest for last month
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
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
            print(df_trades['exit_reason'].value_counts())
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise