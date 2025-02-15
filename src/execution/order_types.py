#! /usr/bin/env python3
#src/execution/order_types.py
"""
Module: src.execution
Provides order types management.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from decimal import Decimal

@dataclass
class OrderDetails:
    # Required fields (no defaults)
    symbol: str
    side: str
    order_type: str
    size: Decimal
    price: Decimal
    
    # Optional fields (with defaults)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_in_force: str = 'GTC'
    post_only: bool = False
    reduce_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
    
@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    
    

    # ... rest of the class needs implementation ... 
    #TODO: Implement the class and the methods
