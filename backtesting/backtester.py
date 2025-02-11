#!/usr/bin/env python3
"""
Module: backtesting/backtester.py
Comprehensive backtesting framework with proper risk management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from database.database import DBConnection, execute_sql
from utils.error_handler import handle_error
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
        self.max_hold_bars = int((ctx.config.get("max_hold_hours", 8) * 60) / 
                                ctx.config.get("timeframe_minutes", 15))
        self.ratchet_manager = RatchetManager(ctx)

    def _format_datetime(self, dt) -> str:
        """Convert datetime objects to string format"""
        if isinstance(dt, pd.Timestamp):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return str(dt)

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
            handle_error(e, "Backtester.load_data", logger=self.ctx.logger)
            return pd.DataFrame()

    def record_backtest_trade(self, trade: Dict[str, Any]) -> None:
        """Record backtest trade to database"""
        try:
            with DBConnection(self.ctx.db_pool) as conn:
                sql_ins = (
                    "INSERT INTO trades (id, symbol, timeframe, trade_source, direction, "
                    "entry_price, sl, tp, entry_time, close_time, result, close_reason, "
                    "exchange, position_size) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params_ins = [
                    trade["id"],
                    trade["symbol"],
                    self.ctx.config.get("timeframe", "15m"),
                    "backtest",
                    trade["direction"],
                    trade["entry_price"],
                    trade.get("sl", 0),
                    trade.get("tp", 0),
                    self._format_datetime(trade["entry_time"]),
                    self._format_datetime(trade["exit_time"]),
                    trade["pnl"],
                    trade.get("exit_reason", "closed"),
                    trade["exchange"],
                    trade["position_size"]
                ]
                execute_sql(conn, sql_ins, params_ins)
                
        except Exception as e:
            handle_error(e, "Backtester.record_backtest_trade", logger=self.ctx.logger)

    def prepare_market_state(self, data: pd.DataFrame) -> Any:
        """Prepare market state for signal generation"""
        from signals.trading_types import MarketState
        
        if data.empty:
            return None
            
        returns = np.log(data["close"] / data["close"].shift(1))
        returns_array = returns.dropna().values
        
        try:
            current_return = returns_array[-1]
        except IndexError:
            current_return = 0.0
            
        volatility = estimate_volatility(returns_array)
            
        return MarketState(
            returns=returns_array,
            ar1_coef=returns.autocorr() if len(returns_array) > 1 else 0.0,
            current_return=current_return,
            volatility=volatility,
            last_price=data["close"].iloc[-1],
            ema_short=data["EMA_8"].iloc[-1] if "EMA_8" in data.columns else 0.0,
            ema_long=data["EMA_21"].iloc[-1] if "EMA_21" in data.columns else 0.0,
            ctx=self.ctx
        )

    def _calculate_position_size(
            self,
            balance: float,
            signal: Dict[str, Any],
            price: float,
            volatility: float
        ) -> float:
        """Calculate position size with strict risk management"""
        probability = signal.get("probability", 0.6)
        kelly = calculate_kelly_fraction(probability, price * 0.02, price * 0.01)
        
        # Enforce strict Kelly limit
        kelly = min(kelly, 0.2)  # Max 20% Kelly
        
        position = calculate_position_size(
            balance,
            kelly * self.ctx.config.get("kelly_scaling", 0.5),
            price,
            volatility,
            self.ctx.config.get("risk_factor", 0.1)
        )
        
        # Apply strict 10% position limit
        max_position = balance * 0.10  # Max 10% of balance
        return min(position, max_position / price)

    def _execute_trade(
            self,
            signal: Dict[str, Any],
            position_size: float,
            entry_candle: pd.Series,
            future_data: pd.DataFrame,
            market_state: Any
        ) -> Optional[Dict[str, Any]]:
        """Execute trade with comprehensive risk management"""
        entry_price = entry_candle["close"]
        entry_cost = position_size * entry_price * self.commission
        entry_time = pd.to_datetime(entry_candle["datetime"])
        
        # Enforce strict 10% position size limit
        max_position = self.initial_balance * 0.10  # Max 10% of balance
        position_size = min(position_size, max_position / entry_price)
        
        # Set strict stops
        stop_loss = entry_price * (1 - self.ctx.config.get("stop_loss_pct", 0.02))
        take_profit = entry_price * (1 + self.ctx.config.get("take_profit_pct", 0.03))
        
        trade_id = str(uuid.uuid4())
        self.ratchet_manager.initialize_trade(trade_id, entry_price)
        
        max_adverse_excursion = 0
        max_favorable_excursion = 0
        
        try:
            for i, candle in future_data.iterrows():
                # Update excursions
                if signal["direction"] == "long":
                    adverse_excursion = (entry_price - min(candle["low"], candle["close"])) / entry_price
                    favorable_excursion = (max(candle["high"], candle["close"]) - entry_price) / entry_price
                else:
                    adverse_excursion = (max(candle["high"], candle["close"]) - entry_price) / entry_price
                    favorable_excursion = (entry_price - min(candle["low"], candle["close"])) / entry_price
                
                max_adverse_excursion = max(max_adverse_excursion, adverse_excursion)
                max_favorable_excursion = max(max_favorable_excursion, favorable_excursion)
                
                # Check stops
                hit_stop = False
                exit_price = None
                exit_reason = None
                
                # Stop loss check
                if signal["direction"] == "long":
                    if candle["low"] <= stop_loss:
                        hit_stop = True
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                else:
                    if candle["high"] >= stop_loss:
                        hit_stop = True
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                
                # Take profit check
                if signal["direction"] == "long":
                    if candle["high"] >= take_profit:
                        hit_stop = True
                        exit_price = take_profit
                        exit_reason = "take_profit"
                else:
                    if candle["low"] <= take_profit:
                        hit_stop = True
                        exit_price = take_profit
                        exit_reason = "take_profit"
                
                # Ratchet stop check
                ratchet_update = self.ratchet_manager.update_price(trade_id, candle["close"])
                if ratchet_update:
                    new_stop, reason = ratchet_update
                    if (signal["direction"] == "long" and candle["low"] <= new_stop) or \
                       (signal["direction"] == "short" and candle["high"] >= new_stop):
                        hit_stop = True
                        exit_price = new_stop
                        exit_reason = f"ratchet_{reason}"
                
                # Max hold time check
                current_time = pd.to_datetime(candle["datetime"])
                if (current_time - entry_time).total_seconds() >= \
                   (self.ctx.config.get("max_hold_hours", 8) * 3600):
                    hit_stop = True
                    exit_price = candle["close"]
                    exit_reason = "max_hold_time"
                
                # Emergency stop check
                if max_adverse_excursion >= abs(self.ctx.config.get("emergency_stop_pct", -3) / 100):
                    hit_stop = True
                    exit_price = candle["close"]
                    exit_reason = "emergency_stop"
                
                if hit_stop:
                    return self._close_trade(
                        signal, position_size, entry_price, exit_price,
                        entry_cost, candle["datetime"], entry_time,
                        exit_reason, max_adverse_excursion, max_favorable_excursion
                    )
                    
        finally:
            self.ratchet_manager.remove_trade(trade_id)
        
        return None

    def _close_trade(
            self,
            signal: Dict[str, Any],
            position_size: float,
            entry_price: float,
            exit_price: float,
            entry_cost: float,
            exit_time: str,
            entry_time: str,
            exit_reason: str,
            max_adverse_excursion: float,
            max_favorable_excursion: float
        ) -> Dict[str, Any]:
        """Calculate trade result with fees"""
        exit_cost = position_size * exit_price * self.commission
        total_fees = entry_cost + exit_cost
        
        if signal["direction"] == "long":
            raw_pnl = (exit_price - entry_price) * position_size
        else:
            raw_pnl = (entry_price - exit_price) * position_size
            
        net_pnl = raw_pnl - total_fees
        
        # Sanity check on PnL
        if abs(net_pnl) > self.initial_balance:
            self.ctx.logger.warning(f"Excessive PnL detected: {net_pnl}. Capping gains/losses.")
            net_pnl = self.initial_balance if net_pnl > 0 else -self.initial_balance
            
        return {
            "id": str(uuid.uuid4()),
            "symbol": signal.get("symbol", ""),
            "direction": signal["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": net_pnl,
            "return": net_pnl / (position_size * entry_price) if position_size > 0 else 0,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "sl": signal.get("sl", 0),
            "tp": signal.get("tp", 0),
            "exchange": signal.get("exchange", "backtest"),
            "trade_source": "backtest",
            "exit_reason": exit_reason,
            "fees_paid": total_fees,
            "max_adverse_excursion": max_adverse_excursion,
            "max_favorable_excursion": max_favorable_excursion
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
            
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        gross_profits = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        returns = pd.Series([t["return"] for t in trades])
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        rolling_equity = equity_curve.cummax()
        drawdowns = (rolling_equity - equity_curve) / rolling_equity
        max_drawdown = drawdowns.max() if not drawdowns.empty else 0.0
        
        total_return = (equity_curve.iloc[-1] - self.initial_balance) / self.initial_balance \
            if not equity_curve.empty else 0.0
        
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

    def simulate_trades(self, data: pd.DataFrame) -> BacktestResults:
        """Run backtest simulation with comprehensive risk management"""
        balance = self.initial_balance
        equity_curve = [balance]
        trades = []
        
        # Create baseline rules
        from signals.population import create_baseline_rule
        baseline_rule = create_baseline_rule()
        
        self.ctx.logger.info("Starting backtest simulation...")
        
        for symbol, group in data.groupby("symbol"):
            self.ctx.logger.info(f"Processing {symbol}")
            market_state = self.prepare_market_state(group)
            
            if market_state is None:
                continue
            
            prepared_data = compute_indicators(group.to_dict('records'), self.ctx)
            if prepared_data.empty:
                continue
                
            for i in range(len(prepared_data)):
                if i < 233:  # Skip initial periods for longer EMAs
                    continue
                    
                window = prepared_data.iloc[max(0, i-100):i+1]
                current_candle = window.iloc[-1]
                
                # Evaluate trading rule
                from signals.evaluation import evaluate_rule
                signal = evaluate_rule(baseline_rule, current_candle.to_dict(), market_state)
                
                if signal:
                    pos_size = self._calculate_position_size(
                        balance,
                        {"probability": 0.6},
                        current_candle["close"],
                        market_state.volatility
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
                        equity_curve.append(balance)
                        
                        self.record_backtest_trade(trade_result)
                        
                        self.ctx.logger.info(
                            f"Trade completed: {symbol} {trade_result['direction']} "
                            f"PnL: {trade_result['pnl']:.2f} "
                            f"Balance: {balance:.2f} "
                            f"Exit: {trade_result['exit_reason']}"
                        )
                    
                elif i % 1000 == 0:  # Periodic equity tracking even without trades
                    equity_curve.append(balance)
        
        results = self._calculate_metrics(trades, pd.Series(equity_curve))
        
        self.ctx.logger.info(f"\nBacktest Summary:")
        self.ctx.logger.info(f"Total Trades: {results.total_trades}")
        self.ctx.logger.info(f"Win Rate: {results.win_rate:.2%}")
        self.ctx.logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        self.ctx.logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        self.ctx.logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        self.ctx.logger.info(f"Total Return: {results.total_return:.2%}")
        
        return results


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
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Total Return: {results.total_return:.2%}")
        print("=" * 50)
        
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