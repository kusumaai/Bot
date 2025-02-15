#! /usr/bin/env python3
#src/utils/data_validator.py
"""
Module: src.utils
Provides data validation functionality.
"""
from typing import Tuple, Union, Optional
from decimal import Decimal
import logging
import pandas as pd
from utils.numeric_handler import NumericHandler
from utils.exceptions import MarketDataValidationError
#data validator class that validates the data
class DataValidator:
    """Validates trade and order parameters to ensure integrity."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.nh = NumericHandler()
    #validate the order parameters
    def validate_order_params(
        self,
        symbol: str,
        side: str,
        amount: Union[str, float, Decimal],
        price: Optional[Union[str, float, Decimal]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate order parameters."""
        try:
            if side not in ['buy', 'sell']:
                return False, f"Invalid side: {side}"
            
            amount = self.nh.to_decimal(amount)
            if amount is None or amount <= Decimal('0'):
                return False, "Amount must be positive"
            
            if price is not None:
                price = self.nh.to_decimal(price)
                if price is None or price <= Decimal('0'):
                    return False, "Price must be positive"
            
            # Additional validations can be added here (e.g., min/max trade sizes)
            return True, None
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False, str(e)
    #validate the market data
    def validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data."""
        try:
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing required fields in market data.")
                return False

            if df.empty or len(df) < 2:
                return False

            return not (df.isnull().any().any() or (df < 0).any().any())
        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            return False
    #validate the trade parameters  
    def validate_trade_parameters(self, data: dict):
        required_fields = ['symbol', 'amount', 'price', 'side']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return True 