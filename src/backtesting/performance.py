#!/usr/bin/env python3
#backtesting/performance.py
"""
Module: backtesting/performance.py
Performance metrics for backtesting results
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict

from backtesting.backtester import BacktestResults

#dataclass for the performance stats
@dataclass
class PerformanceStats:
    # Required fields (no defaults)
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_trades: int
    win_rate: float
    
    # Optional fields (with defaults)
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trades_per_day: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict) 

#class for the performance metrics
class PerformanceMetrics:
    def __init__(self, results: BacktestResults):
        self.results = results
        self.metrics = {}
        self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""
        pass

    def get_metrics(self):
        """Get calculated performance metrics"""
        return self.metrics

#calculate the performance metrics
def calculate_performance_metrics(results: BacktestResults):
    """Calculate performance metrics"""
    metrics = PerformanceMetrics(results)
    metrics.calculate_metrics()
    return metrics.get_metrics()
