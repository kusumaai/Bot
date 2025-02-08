#!/usr/bin/env python3
"""
Module: backtesting/backtester.py
Comprehensive backtesting framework for the trading system
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

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
        with self.ctx.db_pool.connection() as conn:
            query = """
                SELECT *
                FROM candles
                WHERE datetime BETWEEN ? AND ?
                ORDER BY datetime ASC
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            
        return df
        
    def simulate_trades(self, data: pd.DataFrame) -> BacktestResults:
        """Run backtest simulation"""
        balance = self.initial_balance
        positions = []
        equity_curve = []
        trades = []
        
        # Generate signals and execute trades
        for symbol, group in data.groupby("symbol"):
            market_state = self._prepare_market_state(group)
            
            for i in range(len(group)):
                candle = group.iloc[i]
                
                # Generate signals
                ml_signals = self._generate_ml_signals(group[:i+1], market_state)
                ga_signals = self._generate_ga_signals(group[:i+1], market_state)
                
                # Combine signals
                signals = self._combine_signals(ml_signals, ga_signals)
                
                # Process signals
                for signal in signals:
                    if self._validate_signal(signal):
                        # Calculate position size
                        pos_size = self._calculate_position_size(
                            balance,
                            signal,
                            candle["close"]
                        )
                        
                        # Execute trade
                        trade_result = self._execute_trade(
                            signal,
                            pos_size,
                            candle,
                            group[i:],
                            market_state
                        )
                        
                        if trade_result:
                            trades.append(trade_result)
                            balance += trade_result["pnl"]
                            
                equity_curve.append(balance)
                
        # Calculate metrics
        results = self._calculate_metrics(trades, pd.Series(equity_curve))
        return results
        
    def _prepare_market_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare market state for signal generation"""
        returns = np.log(data["close"] / data["close"].shift(1))
        
        return {
            "returns": returns.dropna().values,
            "ar1_coef": returns.autocorr(),
            "volatility": returns.std(),
            "current_price": data["close"].iloc[-1]
        }
        
    def _generate_ml_signals(
        self,
        data: pd.DataFrame,
        market_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate ML-based signals"""
        from models.ml_signal import generate_ml_signals
        return generate_ml_signals(data, self.ctx)
        
    def _generate_ga_signals(
        self,
        data: pd.DataFrame,
        market_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate GA-based signals"""
        from signals.ga_synergy import generate_ga_signals
        return generate_ga_signals(
            data.to_dict("records"),
            market_state,
            self.ctx
        )
        
    def _combine_signals(
        self,
        ml_signals: List[Dict[str, Any]],
        ga_signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine ML and GA signals"""
        from signals.signal_utils import combine_signals
        return combine_signals(ml_signals, ga_signals, self.ctx)
        
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        from signals.signal_utils import validate_signal
        return validate_signal(signal, self.ctx)
        
    def _calculate_position_size(
        self,
        balance: float,
        signal: Dict[str, Any],
        price: float
    ) -> float:
        """Calculate position size using Kelly criterion"""
        from trading.math import (
            calculate_kelly_fraction,
            calculate_position_size
        )
        
        kelly = calculate_kelly_fraction(
            signal["probability"],
            signal.get("win_target", 0),
            signal.get("loss_target", 0)
        )
        
        return calculate_position_size(
            balance,
            kelly,
            price,
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
        """Execute trade and track results"""
        entry_price = entry_candle["close"]
        entry_cost = position_size * entry_price * self.commission
        
        # Set stops
        stop_loss = entry_price * (1 - self.ctx.config.get("stop_loss_pct", 0.02))
        take_profit = entry_price * (1 + self.ctx.config.get("take_profit_pct", 0.03))
        
        # Track trade
        for i, candle in future_data.iterrows():
            # Check stops
            if signal["direction"] == "long":
                if candle["low"] <= stop_loss:
                    return self._close_trade(
                        signal, position_size, entry_price,
                        stop_loss, entry_cost, candle["datetime"]
                    )
                elif candle["high"] >= take_profit:
                    return self._close_trade(
                        signal, position_size, entry_price,
                        take_profit, entry_cost, candle["datetime"]
                    )
            else:  # short
                if candle["high"] >= stop_loss:
                    return self._close_trade(
                        signal, position_size, entry_price,
                        stop_loss, entry_cost, candle["datetime"]
                    )
                elif candle["low"] <= take_profit:
                    return self._close_trade(
                        signal, position_size, entry_price,
                        take_profit, entry_cost, candle["datetime"]
                    )
                    
        return None
        
    def _close_trade(
        self,
        signal: Dict[str, Any],
        position_size: float,
        entry_price: float,
        exit_price: float,
        entry_cost: float,
        exit_time: str
    ) -> Dict[str, Any]:
        """Calculate trade result"""
        exit_cost = position_size * exit_price * self.commission
        total_cost = entry_cost + exit_cost
        
        if signal["direction"] == "long":
            pnl = position_size * (exit_price - entry_price) - total_cost
        else:
            pnl = position_size * (entry_price - exit_price) - total_cost
            
        return {
            "entry_time": signal.get("entry_time", ""),
            "exit_time": exit_time,
            "direction": signal["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": pnl,
            "return": pnl / (position_size * entry_price)
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
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
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

def run_backtest(
    start_date: str,
    end_date: str,
    ctx: Any
) -> BacktestResults:
    """Run backtest for specified period"""
    backtester = Backtester(ctx)
    data = backtester.load_data(start_date, end_date)
    return backtester.simulate_trades(data)

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    class DummyContext:
        def __init__(self):
            self.logger = logging.getLogger()
            self.config = {
                "initial_balance": 10000,
                "commission_rate": 0.001,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "risk_factor": 0.1
            }
    
    ctx = DummyContext()
    results = run_backtest(
        "2023-01-01",
        "2023-12-31",
        ctx
    )
    
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Total Return: {results.total_return:.2%}")