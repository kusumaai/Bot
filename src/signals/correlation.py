#! /usr/bin/env python3
#src/signals/correlation.py
"""
Module: src.signals
Provides correlation metrics.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from datetime import datetime

@dataclass
class CorrelationMetrics:
    # Required fields (no defaults)
    symbol_pair: Tuple[str, str]
    correlation: float
    significance: float
    sample_size: int
    
    # Optional fields (with defaults)
    lookback_days: int = 30
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
#correlation manager class that manages the correlation
class CorrelationManager:
    def __init__(self):
        self.correlations = []
        
    def add_correlation(self, correlation: CorrelationMetrics):
        self.correlations.append(correlation)
        
    def get_correlations(self):
        return self.correlations
    
#correlation types
class CorrelationType:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def get_correlation(self):
        pass
    
