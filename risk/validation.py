#!/usr/bin/env python3
"""
Module: risk/validation.py
Market data and risk validation utilities
"""

from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from utils.numeric import NumericHandler

from utils.error_handler import handle_error

@dataclass
class MarketDataValidation:
    """Market data validation result"""
    timestamp: float
    symbol: str
    price: Decimal
    volume: Decimal
    is_valid: bool
    error_message: Optional[str] = None

class MarketDataValidation:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.nh = NumericHandler()
        self.validation_cache: Dict[str, Dict] = {}
        self.stale_threshold = timedelta(minutes=5)
        
    async def validate_market_data(self, 
                                 symbol: str, 
                                 data: Dict) -> bool:
        """Comprehensive market data validation"""
        try:
            # Check for required fields
            required_fields = ['price', 'volume', 'timestamp']
            if not all(field in data for field in required_fields):
                return False
                
            # Validate timestamp freshness
            data_time = datetime.fromtimestamp(data['timestamp'])
            if datetime.utcnow() - data_time > self.stale_threshold:
                return False
                
            # Price sanity checks
            price = self.nh.to_decimal(data['price'])
            if price <= Decimal('0'):
                return False
                
            # Volume checks
            volume = self.nh.to_decimal(data['volume'])
            if volume <= Decimal('0'):
                return False
                
            # Volatility check
            if not await self._check_volatility(symbol, price):
                return False
                
            # Cache valid data
            self.validation_cache[symbol] = {
                'last_valid_price': price,
                'last_valid_time': datetime.utcnow()
            }
            
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"Market data validation failed: {e}")
            return False
            
    async def _check_volatility(self, 
                              symbol: str, 
                              current_price: Decimal) -> bool:
        """Check if price movement is within acceptable range"""
        if symbol not in self.validation_cache:
            return True
            
        last_price = self.validation_cache[symbol]['last_valid_price']
        price_change = abs(current_price - last_price) / last_price
        
        # Reject if price change > 10% in one update
        return price_change <= Decimal('0.10')

def validate_market_data(data: Dict[str, Any]) -> MarketDataValidation:
    """Validate market data freshness and integrity"""
    try:
        timestamp = float(data.get("timestamp", 0))
        if time.time() - timestamp > 5:  # 5 second staleness check
            return MarketDataValidation(
                timestamp=timestamp,
                symbol=data.get("symbol", ""),
                price=Decimal(0),
                volume=Decimal(0),
                is_valid=False,
                error_message="Stale data"
            )

        # Validate required fields
        required_fields = ["symbol", "price", "volume"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return MarketDataValidation(
                timestamp=timestamp,
                symbol=data.get("symbol", ""),
                price=Decimal(0),
                volume=Decimal(0),
                is_valid=False,
                error_message=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # Validate numeric values
        try:
            price = Decimal(str(data["price"]))
            volume = Decimal(str(data["volume"]))
            if price <= 0 or volume < 0:
                return MarketDataValidation(
                    timestamp=timestamp,
                    symbol=data["symbol"],
                    price=Decimal(0),
                    volume=Decimal(0),
                    is_valid=False,
                    error_message="Invalid price or volume"
                )
        except (ValueError, TypeError):
            return MarketDataValidation(
                timestamp=timestamp,
                symbol=data.get("symbol", ""),
                price=Decimal(0),
                volume=Decimal(0),
                is_valid=False,
                error_message="Invalid numeric values"
            )

        return MarketDataValidation(
            timestamp=timestamp,
            symbol=data["symbol"],
            price=price,
            volume=volume,
            is_valid=True
        )

    except Exception as e:
        handle_error(e, "validate_market_data", logger=None)
        return MarketDataValidation(
            timestamp=0,
            symbol=data.get("symbol", ""),
            price=Decimal(0),
            volume=Decimal(0),
            is_valid=False,
            error_message=str(e)
        )

def validate_risk_parameters(params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate risk parameters against limits"""
    try:
        required_params = [
            ("position_size", Decimal("0.5")),  # Max 50% position size
            ("leverage", Decimal("3")),         # Max 3x leverage
            ("stop_loss_pct", Decimal("0.1")),  # Max 10% stop loss
            ("take_profit_pct", Decimal("0.3")) # Max 30% take profit
        ]

        for param, max_value in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
            
            try:
                value = Decimal(str(params[param]))
                if value <= 0 or value > max_value:
                    return False, f"Invalid {param}: {value} (max: {max_value})"
            except (ValueError, TypeError):
                return False, f"Invalid numeric value for {param}"

        return True, None

    except Exception as e:
        handle_error(e, "validate_risk_parameters", logger=None)
        return False, str(e)

def validate_position_correlation(
    positions: List[Dict[str, Any]], 
    new_symbol: str,
    max_correlation: Decimal
) -> Tuple[bool, Optional[str]]:
    """Validate position correlation limits"""
    try:
        if not positions:
            return True, None

        # Simple correlation check based on direction
        long_count = sum(1 for p in positions if p["direction"] == "long")
        short_count = len(positions) - long_count
        
        # Avoid too many positions in same direction
        max_same_direction = int(len(positions) * float(max_correlation))
        if long_count > max_same_direction or short_count > max_same_direction:
            return False, "Too many positions in same direction"

        return True, None

    except Exception as e:
        handle_error(e, "validate_position_correlation", logger=None)
        return False, str(e)

def validate_portfolio_limits(
    portfolio_value: Decimal,
    total_exposure: Decimal,
    max_leverage: Decimal,
    drawdown: Decimal,
    max_drawdown: Decimal
) -> Tuple[bool, Optional[str]]:
    """Validate portfolio-wide risk limits"""
    try:
        # Check leverage
        if portfolio_value > 0:
            current_leverage = total_exposure / portfolio_value
            if current_leverage > max_leverage:
                return False, f"Leverage {current_leverage} exceeds maximum {max_leverage}"

        # Check drawdown
        if drawdown > max_drawdown:
            return False, f"Drawdown {drawdown} exceeds maximum {max_drawdown}"

        return True, None

    except Exception as e:
        handle_error(e, "validate_portfolio_limits", logger=None)
        return False, str(e)
