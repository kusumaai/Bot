import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from src.risk.limits import RiskLimits
from src.utils.exceptions import ValidationError
from src.utils.numeric_handler import NumericHandler

logger = logging.getLogger(__name__)


class MarketDataValidation:
    """Validates market data and trade parameters."""

    def __init__(
        self, risk_limits: RiskLimits, logger: Optional[logging.Logger] = None
    ):
        self.risk_limits = risk_limits
        self.logger = logger or logging.getLogger(__name__)
        self._validation_history: List[Dict[str, Any]] = []
        self._volatility_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._cache_timeout = 300  # 5 minutes
        self.nh = NumericHandler()
        self._consecutive_failures = 0
        self.MAX_CONSECUTIVE_FAILURES = 3

    def validate_trade_parameters(self, trade_params: Dict[str, Any]) -> bool:
        """
        Validate trade parameters.

        Args:
            trade_params: Dictionary containing trade parameters:
                - symbol: Trading symbol
                - side: Trade side (buy/sell)
                - amount: Trade amount
                - price: Trade price

        Returns:
            bool: True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        # Validate required fields
        required_fields = ["symbol", "side", "amount", "price"]
        for field in required_fields:
            if field not in trade_params:
                raise ValidationError(f"Missing required field: {field}")

        # Validate trade side
        side = str(trade_params["side"]).lower()
        if side not in ["buy", "sell"]:
            raise ValidationError(f"Invalid trade side: {side}")

        # Validate amount and price
        try:
            amount = Decimal(str(trade_params["amount"]))
            if amount <= Decimal("0"):
                raise ValidationError("Trade amount must be positive")
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid trade amount: {str(e)}")

        try:
            price = Decimal(str(trade_params["price"]))
            if price <= Decimal("0"):
                raise ValidationError("Trade price must be positive")
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid trade price: {str(e)}")

        return True

    def validate_correlation(
        self, symbol: str, correlations: Dict[str, Decimal]
    ) -> bool:
        """
        Validate market correlations.

        Args:
            symbol: Trading symbol
            correlations: Dictionary of symbol correlations

        Returns:
            bool: True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        max_correlation = Decimal("0.7")
        for corr_symbol, correlation in correlations.items():
            if correlation > max_correlation:
                raise ValidationError(
                    f"Correlation for {corr_symbol} exceeds maximum allowed: {correlation}"
                )
        return True

    def validate_liquidity(self, symbol: str, liquidity: Decimal) -> bool:
        """
        Validate market liquidity.

        Args:
            symbol: Trading symbol
            liquidity: Market liquidity value

        Returns:
            bool: True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        min_liquidity = Decimal("100000")
        if liquidity < min_liquidity:
            raise ValidationError(
                f"Liquidity for {symbol} is below minimum required: {liquidity}"
            )
        return True

    def validate_volatility(self, symbol: str, volatility: Decimal) -> bool:
        """
        Validate market volatility.

        Args:
            symbol: Trading symbol
            volatility: Market volatility value

        Returns:
            bool: True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        try:
            max_volatility = self.risk_limits.max_volatility
            if volatility > max_volatility:
                raise ValidationError(
                    f"Volatility for {symbol} exceeds maximum allowed: {volatility} > {max_volatility}"
                )

            # Update volatility cache
            self._volatility_cache[symbol] = (volatility, datetime.now())
            return True
        except Exception as e:
            error_msg = f"Volatility validation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self._validation_history

    def clear_validation_history(self) -> None:
        """Clear validation history."""
        self._validation_history = []

    def get_consecutive_failures(self) -> int:
        """Get number of consecutive validation failures."""
        return self._consecutive_failures

    def reset_consecutive_failures(self) -> None:
        """Reset consecutive failures counter."""
        self._consecutive_failures = 0

    def get_cached_volatility(self, symbol: str) -> Optional[Decimal]:
        """
        Get cached volatility value if not expired.

        Args:
            symbol: Trading symbol

        Returns:
            Optional[Decimal]: Cached volatility or None if expired/not found
        """
        if symbol not in self._volatility_cache:
            return None

        volatility, timestamp = self._volatility_cache[symbol]
        if datetime.now() - timestamp > timedelta(seconds=self._cache_timeout):
            del self._volatility_cache[symbol]
            return None

        return volatility

    def add_validation_record(self, record: Dict[str, Any]) -> None:
        """
        Add a validation record to history.

        Args:
            record: Validation record to add
        """
        record["timestamp"] = datetime.now()
        self._validation_history.append(record)
        # Keep only last 1000 records
        if len(self._validation_history) > 1000:
            self._validation_history = self._validation_history[-1000:]
