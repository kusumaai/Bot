#!/usr/bin/env python3
"""
Base type definitions and shared interfaces
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from decimal import Decimal
import time
from datetime import datetime

@dataclass
class BaseValidationResult:
    """Base validation result structure"""
    is_valid: bool
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketState:
    """Core market state data"""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict) 