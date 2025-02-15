#! /usr/bin/env python3
#src/monitoring/metrics.py
"""
Module: src.monitoring
Provides metrics management.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

#trading metrics class that defines the trading metrics
@dataclass
class TradingMetrics:
    # Required fields (no defaults)
    symbol: str
    timestamp: datetime
    trade_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    
    # Optional fields (with defaults)
    drawdown: float = 0.0
    volatility: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    
#metrics manager class that manages the metrics
class MetricsManager:
    def __init__(self):
        self.metrics = []
        
    def add_metric(self, metric: TradingMetrics):
        self.metrics.append(metric)
        
    def get_metrics(self):
        return self.metrics
    
#metrics types
class MetricsType:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def get_metrics(self):
        pass
    

