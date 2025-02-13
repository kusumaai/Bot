#!/usr/bin/env python3
"""
Module: utils/helpers.py
Common helper functions for trading system
"""

import numpy as np
from typing import Any, List, Dict, Tuple
from decimal import Decimal
import pandas as pd
from datetime import datetime, timedelta
import json

from database.connection import DatabaseConnection
from database.database import execute_sql
from utils.error_handler import handle_error

async def vol_estimate(symbol: str, ctx: Any) -> Decimal:
    """
    Estimate the volatility for a given symbol using historical candle data.

    Args:
        symbol (str): The trading pair symbol (e.g., "BTC/USDT")
        ctx (Any): Global context containing configuration and DB connection

    Returns:
        Decimal: The estimated volatility (as decimal). Returns 0 if insufficient data.
    """
    try:
        limit = ctx.config.get("vol_estimate_limit", 100)
        async with DatabaseConnection(ctx.db_pool) as conn:
            rows = await conn.execute_sql(
                """
                SELECT close
                FROM candles
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [symbol, limit]
            )

        if len(rows) < 2:
            return Decimal("0")

        prices = [Decimal(str(r["close"])) for r in rows if r["close"] is not None]
        if len(prices) < 2:
            return Decimal("0")

        prices.reverse()
        arr = np.array([float(p) for p in prices])
        log_returns = np.diff(np.log(arr))
        return Decimal(str(float(np.std(log_returns))))

    except Exception as e:
        handle_error(e, "helpers.vol_estimate", logger=ctx.logger)
        return Decimal("0")

def calculate_correlation(
    symbol1: str,
    symbol2: str,
    ctx: Any,
    lookback_hours: int = 24
) -> Decimal:
    """
    Calculate price correlation between two symbols.

    Args:
        symbol1 (str): First symbol
        symbol2 (str): Second symbol
        ctx (Any): Global context
        lookback_hours (int): Hours of historical data to use

    Returns:
        Decimal: Correlation coefficient (-1 to 1). Returns 0 if error.
    """
    try:
        lookback = datetime.now() - timedelta(hours=lookback_hours)
        with DatabaseConnection(ctx.db_pool) as conn:
            # Get prices for both symbols
            prices = {}
            for symbol in [symbol1, symbol2]:
                rows = execute_sql(
                    conn,
                    """
                    SELECT timestamp, close
                    FROM candles
                    WHERE symbol = ?
                    AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    [symbol, lookback.timestamp()]
                )
                if len(rows) < 2:
                    return Decimal("0")
                    
                prices[symbol] = pd.DataFrame(rows).set_index('timestamp')

        # Align timestamps and calculate correlation
        df = pd.merge(
            prices[symbol1],
            prices[symbol2],
            left_index=True,
            right_index=True,
            suffixes=('_1', '_2')
        )

        if len(df) < 2:
            return Decimal("0")

        correlation = df['close_1'].corr(df['close_2'])
        return Decimal(str(correlation))

    except Exception as e:
        handle_error(e, "helpers.calculate_correlation", logger=ctx.logger)
        return Decimal("0")

def calculate_metrics(
    returns: List[Decimal],
    risk_free_rate: Decimal = Decimal("0.02")
) -> Dict[str, Decimal]:
    """
    Calculate trading performance metrics.

    Args:
        returns (List[Decimal]): List of period returns
        risk_free_rate (Decimal): Annual risk-free rate

    Returns:
        Dict[str, Decimal]: Dictionary of metrics
    """
    try:
        if not returns:
            return {
                "sharpe_ratio": Decimal("0"),
                "sortino_ratio": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "win_rate": Decimal("0"),
                "profit_factor": Decimal("0")
            }

        returns_arr = np.array([float(r) for r in returns])
        
        # Sharpe ratio
        excess_returns = returns_arr - float(risk_free_rate) / 252  # Daily
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino = (
            np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0 else 0
        )
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate and profit factor
        wins = sum(1 for r in returns if r > 0)
        win_rate = Decimal(str(wins / len(returns)))
        
        gains = sum(float(r) for r in returns if r > 0)
        losses = abs(sum(float(r) for r in returns if r < 0))
        profit_factor = Decimal(str(gains / losses if losses > 0 else gains))

        return {
            "sharpe_ratio": Decimal(str(sharpe)),
            "sortino_ratio": Decimal(str(sortino)),
            "max_drawdown": Decimal(str(max_drawdown)),
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }

    except Exception as e:
        handle_error(e, "helpers.calculate_metrics")
        return {
            "sharpe_ratio": Decimal("0"),
            "sortino_ratio": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "win_rate": Decimal("0"),
            "profit_factor": Decimal("0")
        }

def format_trade_log(
    trade: Dict[str, Any],
    include_signals: bool = False
) -> Dict[str, Any]:
    """
    Format trade data for logging.

    Args:
        trade (Dict[str, Any]): Raw trade data
        include_signals (bool): Whether to include signal data

    Returns:
        Dict[str, Any]: Formatted trade log
    """
    try:
        formatted = {
            "trade_id": trade.get("trade_id", ""),
            "symbol": trade.get("symbol", ""),
            "direction": trade.get("direction", ""),
            "entry_time": datetime.fromtimestamp(trade["entry_time"]).isoformat(),
            "entry_price": float(trade["entry_price"]),
            "position_size": float(trade["position_size"]),
            "exit_price": float(trade.get("exit_price", 0)),
            "exit_time": (
                datetime.fromtimestamp(trade["exit_time"]).isoformat()
                if trade.get("exit_time") else None
            ),
            "pnl": float(trade.get("pnl", 0)),
            "pnl_pct": float(trade.get("pnl_pct", 0)),
            "status": trade.get("status", "open"),
            "exit_reason": trade.get("exit_reason", "")
        }

        if include_signals:
            formatted["signals"] = {
                "ml_probability": float(trade.get("ml_probability", 0)),
                "ga_score": float(trade.get("ga_score", 0)),
                "combined_signal": float(trade.get("combined_signal", 0))
            }

        return formatted

    except Exception as e:
        handle_error(e, "helpers.format_trade_log")
        return {"error": str(e)}

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate trading configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation errors)
    """
    try:
        errors = []
        
        # Required fields
        required = [
            "exchange", "timeframe", "market_list",
            "risk_factor", "max_positions", "emergency_stop_pct"
        ]
        
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Numeric validations
        numeric_fields = {
            "risk_factor": (0, 1),
            "max_positions": (1, 100),
            "emergency_stop_pct": (-100, 0)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                try:
                    val = Decimal(str(config[field]))
                    if val < Decimal(str(min_val)) or val > Decimal(str(max_val)):
                        errors.append(
                            f"{field} must be between {min_val} and {max_val}"
                        )
                except:
                    errors.append(f"Invalid numeric value for {field}")

        # Market list validation
        if "market_list" in config:
            if not isinstance(config["market_list"], list):
                errors.append("market_list must be a list")
            elif not all(isinstance(m, str) for m in config["market_list"]):
                errors.append("market_list must contain strings")

        return len(errors) == 0, errors

    except Exception as e:
        handle_error(e, "helpers.validate_config")
        return False, [str(e)]
