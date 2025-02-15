#! /usr/bin/env python3
#src/trading/position_info.py
"""
Module: src.trading
Provides position information functionality.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any
#position info class
@dataclass
class PositionInfo:
    # Required fields (no defaults)
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    timestamp: int
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    strategy: str
    
    # Optional fields (with defaults)
    metadata: Dict[str, Any] = field(default_factory=dict)
    #default constructor
    def __init__(self):
        """
        Purpose: Initialize the metadata field.
        """
        self.metadata = {}
    #update the position price
    def update_price(self, current_price: Decimal):
        """
        Purpose: Update the position price.
        """
        self.current_price = current_price
    #calculate the unrealized pnl
    def calculate_unrealized_pnl(self):
        """
        Purpose: Calculate the unrealized pnl.
        """
        self.unrealized_pnl = self.current_price - self.entry_price * self.size
    #calculate the realized pnl
    def calculate_realized_pnl(self):
        """
        Purpose: Calculate the realized pnl.
        """ 
        self.realized_pnl = self.current_price - self.entry_price * self.size
    #end calculate_realized_pnl

    #calculate the pnl percentage
    def calculate_pnl_percentage(self):
        """
        Purpose: Calculate the pnl percentage.
        """     
        self.pnl_percentage = self.pnl / self.entry_price * 100
