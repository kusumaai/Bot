from dataclasses import dataclass
from typing import Any, Dict
from src.bot_types.base_types import Validatable, ValidationResult


@dataclass
class ConfigValidator(Validatable):
    """Centralized configuration validation"""
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration parameters"""
        try:
            # Validate required sections
            required_sections = ['risk', 'trading', 'execution']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required config sections: {missing_sections}"
                )
            
            # Validate risk limits
            risk_validation = self._validate_risk_config(config['risk'])
            if not risk_validation.is_valid:
                return risk_validation
                
            # Validate trading parameters
            trading_validation = self._validate_trading_config(config['trading'])
            if not trading_validation.is_valid:
                return trading_validation
                
            # Validate execution parameters
            execution_validation = self._validate_execution_config(config['execution'])
            if not execution_validation.is_valid:
                return execution_validation
                
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Configuration validation failed: {str(e)}"
            ) 