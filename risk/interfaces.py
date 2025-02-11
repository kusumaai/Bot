#!/usr/bin/env python3
"""
Module: risk/interfaces.py
Core interfaces for risk management components
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from risk.constants import (
    PRICE_PRECISION,
    SIZE_PRECISION,
    PNL_PRECISION,
    PERCENTAGE_PRECISION
)

@dataclass
class TradeStats:
    """Statistics for a single trade"""
    entry_price: Decimal
    current_price: Decimal
    size: Decimal
    unrealized_pnl: Decimal
    max_adverse_excursion: Decimal
    max_favorable_excursion: Decimal
    duration: float  # seconds
    direction: str

class IPortfolio(ABC):
    """Interface for portfolio management"""
    
    @abstractmethod
    def calculate_value(self) -> Decimal:
        """Calculate total portfolio value including open positions"""
        pass
        
    @abstractmethod
    def calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak value"""
        pass
        
    @abstractmethod
    def get_position_exposure(self, symbol: str) -> Decimal:
        """Get current exposure for a specific position"""
        pass
        
    @abstractmethod
    def get_total_exposure(self) -> Decimal:
        """Get total portfolio exposure across all positions"""
        pass

class IRiskManager(ABC):
    """Interface for risk management"""
    
    @abstractmethod
    def validate_new_position(
        self, 
        symbol: str,
        size: Decimal,
        price: Decimal,
        direction: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if a new position can be opened.
        
        Returns:
            Tuple of (is_valid: bool, reason: Optional[str])
        """
        pass
        
    @abstractmethod
    def update_position_risk(
        self,
        symbol: str,
        current_price: Decimal,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Update risk metrics for a position and return any triggered alerts.
        
        Returns:
            List of alert dictionaries
        """
        pass
        
    @abstractmethod
    def get_position_size_limits(
        self,
        symbol: str,
        price: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """
        Get minimum and maximum allowed position sizes.
        
        Returns:
            Tuple of (min_size: Decimal, max_size: Decimal)
        """
        pass

class IPositionManager(ABC):
    """Interface for position management"""
    
    @abstractmethod
    def open_position(
        self,
        symbol: str,
        direction: str,
        size: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal
    ) -> bool:
        """Open a new position with risk parameters"""
        pass
        
    @abstractmethod
    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        reason: str
    ) -> Optional[Dict[str, Any]]:
        """Close an existing position and return trade summary"""
        pass
        
    @abstractmethod
    def update_position_stops(
        self,
        symbol: str,
        new_stop_loss: Optional[Decimal] = None,
        new_take_profit: Optional[Decimal] = None
    ) -> bool:
        """Update stop loss and/or take profit levels"""
        pass

class IRiskLimits(ABC):
    """Interface for risk limits"""
    
    @abstractmethod
    def validate_position_size(self, size: Decimal, total_value: Decimal) -> bool:
        """Validate position size against limits"""
        pass
        
    @abstractmethod
    def validate_daily_loss(self, daily_pnl: Decimal, total_value: Decimal) -> bool:
        """Validate daily loss against limits"""
        pass
        
    @abstractmethod
    def validate_drawdown(self, drawdown: Decimal) -> bool:
        """Validate drawdown against limits"""
        pass
        
    @abstractmethod
    def validate_leverage(self, total_exposure: Decimal, total_value: Decimal) -> bool:
        """Validate leverage against limits"""
        pass

class IHealthMonitor(ABC):
    """Interface for system health monitoring"""
    
    @abstractmethod
    def check_system_health(self) -> Tuple[bool, List[str]]:
        """
        Check overall system health.
        
        Returns:
            Tuple of (is_healthy: bool, warnings: List[str])
        """
        pass
        
    @abstractmethod
    def log_health_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log current health metrics"""
        pass
        
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary"""
        pass 