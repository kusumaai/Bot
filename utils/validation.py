from typing import Tuple, Optional, Union
import logging
import pandas as pd
from decimal import Decimal
from utils.numeric import NumericHandler

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
            if amount <= Decimal('0'):
                return False, "Amount must be positive"

            if price is not None:
                price = self.nh.to_decimal(price)
                if price <= Decimal('0'):
                    return False, "Price must be positive"

            return True, None

        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False, str(e)

    def validate_market_data(self, df: pd.DataFrame) -> bool:
        try:
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return False

            if df.empty or len(df) < 2:
                return False

            return not (df.isnull().any().any() or (df < 0).any().any())

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            return False 