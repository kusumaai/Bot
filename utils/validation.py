from typing import Tuple, Optional, Union, List, Dict, Any
import logging
import pandas as pd
from decimal import Decimal
from utils.numeric_handler import NumericHandler
from utils.error_handler import handle_error, ValidationError
from utils.exceptions import MarketDataValidationError
from typing import Dict, Any
from dataclasses import dataclass, field

class DataValidator:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.nh = NumericHandler()

    def validate_order_params(
        self,
        symbol: str,
        side: str,
        amount: Union[str, float, Decimal],
        price: Optional[Union[str, float, Decimal]] = None
    ) -> Tuple[bool, Optional[str]]:
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

            return True, None

        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False, str(e)

    def validate_market_data(self, df: pd.DataFrame) -> bool:
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

@dataclass
class ValidationResult:
    # Required fields (no defaults)
    is_valid: bool
    validation_type: str
    timestamp: float
    
    # Optional fields (with defaults)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 