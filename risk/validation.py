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
import numpy as np
import pandas as pd
import logging

from utils.error_handler import handle_error, ValidationError
from .limits import RiskLimits

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
    """Validate market data quality and trading conditions"""
    
    def __init__(self, limits: RiskLimits, logger: logging.Logger):
        self.limits = limits
        self.logger = logger
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
            self.logger.error(f"Market data validation failed: {e}")
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

    def validate_candle_data(
        self,
        candles: List[Dict[str, Any]],
        min_candles: int = 20
    ) -> bool:
        """
        Validate candle data quality
        
        Args:
            candles: List of candle dictionaries
            min_candles: Minimum required candles
        
        Returns:
            bool: True if data is valid
        
        Raises:
            ValidationError: If data quality checks fail
        """
        if len(candles) < min_candles:
            raise ValidationError(
                f"Insufficient candle data. Got {len(candles)}, need {min_candles}"
            )
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(candles)
        
        # Check for missing values
        if df.isnull().any().any():
            raise ValidationError("Found missing values in candle data")
        
        # Validate price consistency
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_prices.any():
            raise ValidationError(
                f"Found {invalid_prices.sum()} candles with invalid price levels"
            )
        
        # Check for zero volumes
        if (df['volume'] <= 0).any():
            raise ValidationError("Found candles with zero or negative volume")
        
        return True
    
    def validate_liquidity(
        self,
        volume: Decimal,
        price: Decimal,
        timeframe_minutes: int = 15
    ) -> bool:
        """
        Validate if market has sufficient liquidity
        
        Args:
            volume: Volume in base currency
            price: Current price
            timeframe_minutes: Candle timeframe in minutes
        """
        daily_volume = volume * (1440 / timeframe_minutes)  # Extrapolate to 24h
        daily_volume_usd = daily_volume * price
        
        if daily_volume_usd < self.limits.min_liquidity:
            raise ValidationError(
                f"Insufficient liquidity. "
                f"Need ${self.limits.min_liquidity}, got ${daily_volume_usd}"
            )
        
        return True
    
    def calculate_volatility(
        self,
        prices: List[Decimal],
        window: int = 20
    ) -> Decimal:
        """Calculate rolling volatility"""
        returns = pd.Series(prices).pct_change().dropna()
        volatility = returns.rolling(window).std().iloc[-1]
        return Decimal(str(volatility))
    
    def validate_volatility(
        self,
        prices: List[Decimal],
        custom_threshold: Optional[Decimal] = None
    ) -> bool:
        """
        Validate if market volatility is within acceptable limits
        
        Args:
            prices: List of historical prices
            custom_threshold: Optional custom volatility threshold
        """
        volatility = self.calculate_volatility(prices)
        threshold = custom_threshold or self.limits.max_volatility
        
        if volatility > threshold:
            raise ValidationError(
                f"Excessive volatility. "
                f"Maximum {threshold}, current {volatility}"
            )
        
        return True
    
    def validate_correlation(
        self,
        symbol: str,
        correlations: Dict[str, Decimal]
    ) -> bool:
        """
        Validate correlation with existing positions
        
        Args:
            symbol: Symbol to check
            correlations: Dictionary of correlations with other assets
        """
        for other_symbol, correlation in correlations.items():
            if abs(correlation) > self.limits.max_correlation:
                raise ValidationError(
                    f"Excessive correlation between {symbol} and {other_symbol}: "
                    f"{correlation} > {self.limits.max_correlation}"
                )
        
        return True

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
