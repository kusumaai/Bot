#!/usr/bin/env python3
"""
signals/trading_types.py - Core data types for trading system
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class TradeMetrics:
    """Performance metrics for a set of trades"""
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    exposure_ratio: float = 0.0
    avg_trade_return: float = 0.0
    total_trades: int = 0
    consecutive_losses: int = 0
    kelly_fraction: float = 0.0
    avg_leverage: float = 0.0

@dataclass
class MarketState:
    """Current market conditions and metrics"""
    returns: np.ndarray
    ar1_coef: float
    current_return: float
    volatility: float
    last_price: float
    ema_short: float
    ema_long: float
    ctx: any 

@dataclass
class TradingRule:
    """Trading strategy definition"""
    buy_conditions: List[Dict[str, Any]]
    sell_conditions: List[Dict[str, Any]]
    fitness: float = 0.0

@dataclass
class SimulationResult:
    """Results from strategy simulation"""
    fitness: float
    metrics: TradeMetrics
    trades: List[Dict[str, Any]]
    warnings: List[str]