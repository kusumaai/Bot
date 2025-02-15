#! /usr/bin/env python3
#src/risk/exposure.py
"""
Module: src.risk
Provides exposure management.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict


@dataclass
class ExposureMetrics:
    # Required fields (no defaults)
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    leverage: Decimal
    
    # Optional fields (with defaults)
    long_exposure: Decimal = Decimal('0')
    short_exposure: Decimal = Decimal('0')
    sector_exposure: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
#exposure manager class that manages the exposure
class ExposureManager:
    def __init__(self):
        self.exposure = []
        
    def add_exposure(self, exposure: ExposureMetrics):
        self.exposure.append(exposure)  
        
    def get_exposure(self):
        return self.exposure
    
#exposure types
class ExposureType:     
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def get_exposure(self):
        pass