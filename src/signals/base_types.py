#! /usr/bin/env python3
#src/signals/base_types.py
"""
Module: src.signals
Provides base signal types.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
import time

@dataclass
class BaseSignal:
    """Base signal structure shared by GA and ML signals"""
    symbol: str
    direction: str
    strength: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict) 