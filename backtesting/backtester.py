# File: backtester.py
"""
Module: backtesting/backtester.py
Comprehensive backtesting framework for the trading system
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error
import uuid
import logging

@dataclass
class BacktestResults:
    """Container for backtest results"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series

class Backtester:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.initial_balance = ctx.config.get("initial_balance", 10000)
        self.commission = ctx.config.get("commission_rate", 0.001)
        
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        try:
            with DBConnection(self.ctx.db_pool) as conn:
                query = """
                    SELECT *
                    FROM candles
                    WHERE datetime BETWEEN ? AND ?
                    ORDER BY datetime ASC
                """
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
                
                if df.empty:
                    self.ctx.logger.warning("No data found for the specified date range")
                else:
                    self.ctx.logger.info(f"Loaded {len(df)} candles for backtesting")
                    
                return df
                
        except Exception as e:
            self.ctx.logger.error(f"Error loading backtest data: {e}")
            return pd.DataFrame()
        
    def record_backtest_trade(self, trade: Dict[str, Any], ctx: Any) -> None:
        """Record backtest trade to database in same format as live trades"""
        try:
            with DBConnection(ctx.db_pool) as conn:
                sql_ins = (
                    "INSERT INTO trades (id, symbol, timeframe, trade_source, direction, "
                    "entry_price, sl, tp, entry_time, exit_time, result, close_reason, "
                    "exchange, position_size) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params_ins = [
                    trade["id"],
                    trade["symbol"],
                    ctx.config["timeframe"],
                    "backtest",  # Mark source as backtest
                    trade["direction"],
                    trade["entry_price"],
                    trade.get("sl", 0),
                    trade.get("tp", 0),
                    trade["entry_time"],
                    trade["exit_time"],
                    trade["pnl"],
                    "closed",
                    trade["exchange"],
                    trade["position_size"]
                ]
                execute_sql(conn, sql_ins, params_ins)
                
        except Exception as e:
            handle_error(e, context="Backtester.record_backtest_trade", logger=ctx.logger)

    def prepare_market_state(self, data: pd.DataFrame) -> Any:
        """Prepare market state for signal generation"""
        from signals.trading_types import MarketState
        
        returns = np.log(data["close"] / data["close"].shift(1))
        returns_array = returns.dropna().values
        
        try:
            current_return = returns_array[-1]
        except IndexError:
            current_return = 0.0
            
        return MarketState(
            returns=returns_array,
            ar1_coef=returns.autocorr() if len(returns_array) > 1 else 0.0,
            current_return=current_return,
            volatility=returns.std() if len(returns_array) > 1 else 0.0,
            last_price=data["close"].iloc[-1],
            ema_short=data["EMA_8"].iloc[-1] if "EMA_8" in data.columns else 0.0,
            ema_long=data["EMA_21"].iloc[-1] if "EMA_21" in data.columns else 0.0
        )

    def simulate_trades(self, data: pd.DataFrame) -> BacktestResults:
        """Run backtest simulation with baseline GA rules"""
        balance = self.initial_balance
        positions = []
        equity_curve = []
        trades = []
        
        # Create baseline rules
        from signals.population import create_baseline_rule
        baseline_rule = create_baseline_rule()
        
        self.ctx.logger.info("Starting backtest with baseline GA rules")
        
        for symbol, group in data.groupby("symbol"):
            market_state = self.prepare_market_state(group)
            
            # Compute all indicators once
            from indicators.indicators_pta import compute_indicators
            prepared_data = compute_indicators(group.to_dict('records'), self.ctx)
            
            for i in range(len(prepared_data)):
                if i < 55:  # Skip initial periods for longer EMA
                    continue
                    
                window = prepared_data.iloc[max(0, i-100):i+1]
                current_candle = window.iloc[-1]
                
                # Evaluate using baseline rule
                from signals.evaluation import evaluate_rule
                signal = evaluate_rule(baseline_rule, current_candle.to_dict(), market_state)
                
                if signal:  # If we get a valid signal
                    # Calculate position size and execute trade...
                    pos_size = self._calculate_position_size(
                        balance,
                        {"probability": 0.6},  # Conservative default probability
                        current_candle["close"]
                    )
                    
                    trade_result = self._execute_trade(
                        {
                            "symbol": symbol,
                            "direction": signal,
                            "entry_price": current_candle["close"],
                            "exchange": "backtest"
                        },
                        pos_size,
                        current_candle,
                        prepared_data.iloc[i+1:],
                        market_state
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        balance += trade_result["pnl"]
                        self.ctx.logger.info(f"Backtest trade executed: {trade_result['symbol']} {trade_result['direction']} PnL: {trade_result['pnl']:.2f}")
                
                equity_curve.append(balance)
        
        results = self._calculate_metrics(trades, pd.Series(equity_curve))
        self.ctx.logger.info(f"Backtest completed with {len(trades)} trades. Final balance: {balance:.2f}")
        return results

    def _calculate_position_size(
        self,
        balance: float,
        signal: Dict[str, Any],
        price: float
    ) -> float:
        """Calculate position size using Kelly criterion and volatility scaling"""
        from trading.math import (
            calculate_kelly_fraction,
            calculate_position_size,
            estimate_volatility
        )
        
        probability = signal.get("probability", 0.6)  # Conservative default
        kelly = calculate_kelly_fraction(
            probability,
            signal.get("win_target", price * 0.02),  # Default 2% target
            signal.get("loss_target", -price * 0.01)  # Default 1% stop
        )
        
        return calculate_position_size(
            balance,
            kelly * self.ctx.config.get("kelly_scaling", 0.5),
            price,
            estimate_volatility(np.array([1.0])),  # Placeholder volatility
            self.ctx.config.get("risk_factor", 0.1)
        )

    def _execute_trade(
        self,
        signal: Dict[str, Any],
        position_size: float,
        entry_candle: pd.Series,
        future_data: pd.DataFrame,
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute trade and track results with database recording"""
        entry_price = entry_candle["close"]
        entry_cost = position_size * entry_price * self.commission
        entry_time = pd.to_datetime(entry_candle["datetime"])
        
        # Set stops
        stop_loss = entry_price * (1 - self.ctx.config.get("stop_loss_pct", 0.02))
        take_profit = entry_price * (1 + self.ctx.config.get("take_profit_pct", 0.03))
        
        # Track trade
        for i, candle in future_data.iterrows():
            # Check stops
            if signal["direction"] == "long":
                if candle["low"] <= stop_loss:
                    trade_result = self._close_trade(
                        signal, position_size, entry_price,
                        stop_loss, entry_cost, candle["datetime"], entry_time
                    )
                    if trade_result:
                        self.record_backtest_trade(trade_result, self.ctx)
                    return trade_result
                elif candle["high"] >= take_profit:
                    trade_result = self._close_trade(
                        signal, position_size, entry_price,
                        take_profit, entry_cost, candle["datetime"], entry_time
                    )
                    if trade_result:
                        self.record_backtest_trade(trade_result, self.ctx)
                    return trade_result
            else:  # short
                if candle["high"] >= stop_loss:
                    trade_result = self._close_trade(
                        signal, position_size, entry_price,
                        stop_loss, entry_cost, candle["datetime"], entry_time
                    )
                    if trade_result:
                        self.record_backtest_trade(trade_result, self.ctx)
                    return trade_result
                elif candle["low"] <= take_profit:
                    trade_result = self._close_trade(
                        signal, position_size, entry_price,
                        take_profit, entry_cost, candle["datetime"], entry_time
                    )
                    if trade_result:
                        self.record_backtest_trade(trade_result, self.ctx)
                    return trade_result
                    
        return None

    def _close_trade(
        self,
        signal: Dict[str, Any],
        position_size: float,
        entry_price: float,
        exit_price: float,
        entry_cost: float,
        exit_time: str,
        entry_time: str
    ) -> Dict[str, Any]:
        """Calculate trade result and format for database"""
        exit_cost = position_size * exit_price * self.commission
        total_cost = entry_cost + exit_cost
        
        if signal["direction"] == "long":
            pnl = position_size * (exit_price - entry_price) - total_cost
        else:
            pnl = position_size * (entry_price - exit_price) - total_cost
            
        return {
            "id": str(uuid.uuid4()),
            "symbol": signal.get("symbol", ""),
            "direction": signal["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": pnl,
            "return": pnl / (position_size * entry_price),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "sl": signal.get("sl", 0),
            "tp": signal.get("tp", 0),
            "exchange": signal.get("exchange", "backtest"),
            "trade_source": "backtest"
        }

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]], 
        equity_curve: pd.Series
    ) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        if not trades:
            return BacktestResults(
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                trades=[],
                equity_curve=equity_curve
            )
            
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = winning_trades / total_trades
        
        # Profit metrics
        gross_profits = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Risk metrics
        returns = pd.Series([t["return"] for t in trades])
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Drawdown
        rolling_equity = equity_curve.cummax()
        drawdowns = (rolling_equity - equity_curve) / rolling_equity
        max_drawdown = drawdowns.max()
        
        total_return = (equity_curve.iloc[-1] - self.initial_balance) / self.initial_balance
        
        return BacktestResults(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            trades=trades,
            equity_curve=equity_curve
        )

def run_backtest(start_date: str, end_date: str, ctx: Any) -> BacktestResults:
    """Run backtest for specified period"""
    backtester = Backtester(ctx)
    data = backtester.load_data(start_date, end_date)
    return backtester.simulate_trades(data)

if __name__ == "__main__":
    import logging
    import json
    import os
    from dataclasses import dataclass
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
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
    
    # Run backtest for last year
    results = run_backtest(
        "2023-01-01",
        "2024-01-01",
        ctx
    )
    
    print("\nBacktest Results:")
    print("=" * 50)
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Total Return: {results.total_return:.2%}")
    print("=" * 50)