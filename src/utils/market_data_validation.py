#! /usr/bin/env python3
#src/utils/market_data_validation.py
"""
Module: src.utils
Provides market data validation functionality.
"""
#import required modules
from decimal import Decimal
from typing import Dict, Any, Tuple, Optional
from src.utils.exceptions import ValidationError
from src.utils.error_handler import handle_error_async
#market data validation class that validates trade parameters and market correlations
class MarketDataValidation: 
    def __init__(self, risk_limits, logger):
        """Initialize the MarketDataValidation class."""
        self.risk_limits = risk_limits
        self.logger = logger
    #validate trade parameters by checking required fields, side, amount and price 
    async def validate_trade_parameters(self, trade_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates trade parameters."""
        try:
            # Required fields
            required_fields = ['symbol', 'side', 'amount', 'price']
            for field in required_fields:
                if field not in trade_params:
                    return False, f"Missing required field: {field}"
            
            # Validate side
            if trade_params['side'] not in ['buy', 'sell']:
                return False, f"Invalid trade side: {trade_params['side']}"
            
            # Validate amount and price
            if trade_params['amount'] <= Decimal('0'):
                return False, "Trade amount must be positive"
            if trade_params['price'] <= Decimal('0'):
                return False, "Trade price must be positive"
            
            # Additional validations based on risk limits
            # Example: Check if amount exceeds max position size
            if trade_params['amount'] > self.risk_limits.max_position_size:
                return False, "Trade amount exceeds maximum position size"
            
            return True, None
        except KeyError as e:
            await handle_error_async(e, "validate_trade_parameters", self.logger)
            return False, f"Missing trade parameter: {e}"
        except Exception as e:
            await handle_error_async(e, "validate_trade_parameters", self.logger)
            return False, f"Error validating trade parameters: {e}"

    async def validate_correlation(self, symbol: str, correlations: Dict[str, Decimal]) -> Tuple[bool, Optional[str]]:
        """Validates market correlations."""
        try:
            for correlated_symbol, correlation in correlations.items():
                if correlation > self.risk_limits.max_correlation:
                    return False, f"Correlation for {correlated_symbol} exceeds maximum allowed"
            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_correlation", self.logger)
            return False, f"Error validating correlations: {e}"
    #validate market liquidity by checking if the liquidity is below the minimum required
    async def validate_liquidity(self, symbol: str, liquidity: Decimal) -> Tuple[bool, Optional[str]]:
        """Validates market liquidity."""
        try:
            if liquidity < self.risk_limits.min_liquidity:
                return False, f"Liquidity for {symbol} is below minimum required"
            return True, None
        except Exception as e:
            await handle_error_async(e, "validate_liquidity", self.logger)
            return False, f"Error validating liquidity: {e}"
