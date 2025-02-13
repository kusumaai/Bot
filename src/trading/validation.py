from dataclasses import dataclass
from typing import Dict
from bot_types.base_types import (
    Position,
    ValidationResult,
    Validatable
)
from risk.limits import RiskLimits

@dataclass
class PortfolioValidator(Validatable):
    """Centralized portfolio validation"""
    risk_limits: RiskLimits
    
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