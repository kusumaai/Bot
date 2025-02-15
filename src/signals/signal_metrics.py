#! /usr/bin/env python3
#src/signals/signal_metrics.py
"""
Module: src.signals
Provides signal performance metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, Any
from decimal import Decimal

@dataclass
class SignalPerformance:
    # Required fields (no defaults)
    signal_id: str
    symbol: str
    direction: str
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    
    # Optional fields (with defaults)
    hold_time: float = 0.0
    max_adverse_excursion: Decimal = Decimal('0')
    max_favorable_excursion: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
#signal performance manager class that manages the signal performance
class SignalPerformanceManager:
    def __init__(self):
        self.performances = []
        
    def add_performance(self, performance: SignalPerformance):
        self.performances.append(performance)
        
    def get_performances(self):
        return self.performances
    
#signal performance types
class SignalPerformanceType:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
                            
    def get_performance(self):
        pass    