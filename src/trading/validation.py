#! /usr/bin/env python3
#src/trading/validation.py
"""
Module: src.trading
Provides validation functionality.
"""
from dataclasses import dataclass
from typing import Dict, List
from bot_types.base_types import (
    Position,
    ValidationResult,
    Validatable
)
from risk.limits import RiskLimits
import numpy as np                              
#portfolio validator class that validates the portfolio
@dataclass
class PortfolioValidator(Validatable):
    """Centralized portfolio validation"""
    risk_limits: RiskLimits
    #validate the portfolio
    def validate_portfolio(self, portfolio: Dict[str, Position]) -> ValidationResult:
        """Validate portfolio state"""
        try:
            # Portfolio-wide validations
            validations = [
                (len(portfolio) <= self.risk_limits.max_positions,
                 "Maximum positions limit reached"),
                (self._validate_total_exposure(portfolio),
                 "Total exposure exceeds limits"),
                (self._validate_sector_diversification(portfolio),
                 "Sector diversification requirements not met"),
                (self._validate_correlation_matrix(portfolio),
                 "Portfolio correlation limits exceeded")
            ]
            #validate the portfolio
            for condition, message in validations:
                if not condition:
                    return ValidationResult(is_valid=False, error_message=message)
                    
            # Validate individual positions
            for symbol, position in portfolio.items():
                position_validation = self._validate_position(position)
                if not position_validation.is_valid:
                    return position_validation
                    
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Portfolio validation failed: {str(e)}"
            ) 
    #validate the total exposure
    def _validate_total_exposure(self, portfolio: Dict[str, Position]) -> bool:
        """Validate total exposure"""
        try:
            total_exposure = sum(position.size for position in portfolio.values())
            return total_exposure <= self.risk_limits.max_exposure
        except Exception as e:
            return False
    #validate the sector diversification    
    def _validate_sector_diversification(self, portfolio: Dict[str, Position]) -> bool:
        """Validate sector diversification"""
        try:
            sectors = set(position.sector for position in portfolio.values())
            return len(sectors) <= self.risk_limits.max_sectors
        except Exception as e:
            return False    
    #validate the correlation matrix
    def _validate_correlation_matrix(self, portfolio: Dict[str, Position]) -> bool:
        """Validate correlation matrix"""
        try:
            correlation_matrix = self._calculate_correlation_matrix(portfolio)
            return all(correlation_matrix[i][j] <= self.risk_limits.max_correlation for i in range(len(portfolio)) for j in range(i))
        except Exception as e:
            return False
    #calculate the correlation matrix
    def _calculate_correlation_matrix(self, portfolio: Dict[str, Position]) -> List[List[float]]:
        """Calculate correlation matrix"""
        try:
            prices = [position.current_price for position in portfolio.values()]
            return np.corrcoef(prices)
        except Exception as e:
            return []   
    #validate the position
    def _validate_position(self, position: Position) -> ValidationResult:
        """Validate individual position"""
        try:
            if not position.symbol:
                return ValidationResult(is_valid=False, error_message="Position must have a symbol")
            if position.size <= 0:
                return ValidationResult(is_valid=False, error_message="Position size must be positive")
            if position.current_price <= 0:
                return ValidationResult(is_valid=False, error_message="Current price must be positive")
            return ValidationResult(is_valid=True)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Position validation failed: {str(e)}"
            )   
    #validate the position size
    def _validate_position_size(self, position: Position) -> bool:
        """Validate position size"""
        try:
            return position.size <= self.risk_limits.max_position_size
        except Exception as e:
            return False    
    #validate the position price
    def _validate_position_price(self, position: Position) -> bool:
        """Validate position price"""
        try:
            return position.current_price > 0
        except Exception as e:
            return False        
    #validate the position symbol
    def _validate_position_symbol(self, position: Position) -> bool:
        """Validate position symbol"""
        try:
            return position.symbol is not None
        except Exception as e:
            return False            
    #validate the position sector
    def _validate_position_sector(self, position: Position) -> bool:
        """Validate position sector"""
        try:
            return position.sector is not None
        except Exception as e:
            return False                
    #validate the position type
    def _validate_position_type(self, position: Position) -> bool:
        """Validate position type"""
        try:
            return position.type is not None
        except Exception as e:
            return False
