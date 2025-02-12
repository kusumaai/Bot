#!/usr/bin/env python3
"""
Module: risk/manager.py
Core risk management functionality
"""

from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging
import asyncio

from utils.error_handler import handle_error, handle_error_async
from .position import Position
from .limits import RiskLimits
from .portfolio import PortfolioManager
from .validation import validate_market_data
# Remove portfolio import, use type hints instead
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .portfolio import PortfolioManager
from utils.numeric import NumericHandler

class RiskManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self._lock = asyncio.Lock()
        self.position_limits = self._load_position_limits()
        self.risk_limits = {
            'max_position_size': Decimal(str(ctx.config.get('max_position_pct', 10))) / Decimal('100'),
            'max_drawdown': Decimal(str(ctx.config.get('max_drawdown', 10))) / Decimal('100'),
            'max_daily_loss': Decimal(str(ctx.config.get('max_daily_loss', 3))) / Decimal('100')
        }
        self.portfolio = PortfolioManager(self.risk_limits)
        self.nh = NumericHandler()  # Use our new numeric handler
        
    def validate_position(self, symbol: str, size: Decimal, price: Decimal) -> Tuple[bool, Optional[str]]:
        """Validate a new position against all risk limits"""
        try:
            # Check max positions
            if len(self.portfolio.positions) >= self.risk_limits['max_position_size']:
                return False, "Maximum positions limit reached"
                
            # Check position size
            position_value = size * price
            portfolio_value = self.portfolio.calculate_portfolio_value()
            if position_value / portfolio_value > self.risk_limits['max_position_size']:
                return False, "Position size exceeds maximum allowed"
                
            # Check leverage
            total_exposure = sum(
                p.size * p.current_price 
                for p in self.portfolio.positions.values()
            ) + position_value
            
            if total_exposure / portfolio_value > self.risk_limits['max_position_size']:
                return False, "Leverage limit exceeded"
                
            # Check drawdown
            if self.portfolio.calculate_drawdown() > self.risk_limits['max_drawdown']:
                return False, "Maximum drawdown reached"
                
            # Check correlation if multiple positions
            if len(self.portfolio.positions) > 0:
                correlation = self._calculate_position_correlation(symbol)
                if correlation > self.risk_limits['max_position_size']:
                    return False, "Position correlation too high"
                
            return True, None
            
        except Exception as e:
            handle_error(e, "RiskManager.validate_position", logger=self.logger)
            return False, f"Error validating position: {str(e)}"

    def calculate_position_params(
        self, 
        signal: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position parameters with risk limits"""
        try:
            with self.portfolio.lock:
                if not validate_market_data(market_data):
                    return {"size": Decimal(0)}
                    
                current_price = Decimal(str(market_data["current_price"]))
                
                # Calculate expected value
                ev = self._calculate_expected_value(signal, market_data)
                if ev <= 0:
                    return {"size": Decimal(0)}
                
                # Calculate position size with Kelly
                size = self._calculate_kelly_position_size(
                    signal.get("probability", Decimal("0.5")),
                    ev,
                    current_price
                )
                
                # Apply risk factor scaling
                size *= self.risk_limits['max_position_size']
                
                # Calculate stops based on volatility
                atr = Decimal(str(market_data.get("ATR_14", self.risk_limits['max_position_size'])))
                stop_loss = current_price * (Decimal(1) - atr)
                take_profit = current_price * (Decimal(1) + atr * Decimal(2))
                
                return {
                    "size": size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": self.risk_limits['max_position_size']
                }
                
        except Exception as e:
            handle_error(e, "RiskManager.calculate_position_params", logger=self.logger)
            return {"size": Decimal(0)}

    def update_positions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update all positions and return any that need closing"""
        try:
            positions_to_close = []
            
            for symbol, position in self.portfolio.positions.items():
                if symbol not in market_data:
                    continue
                    
                current_price = Decimal(str(market_data[symbol]["current_price"]))
                position.update(current_price)
                
                # Check stop conditions
                if self._check_stop_conditions(position):
                    positions_to_close.append({
                        "symbol": symbol,
                        "reason": "stop_loss",
                        "position": position
                    })
                    
                # Check max adverse excursion
                if position.max_adverse_excursion < -self.risk_limits['max_position_size']:
                    positions_to_close.append({
                        "symbol": symbol,
                        "reason": "max_adverse_excursion",
                        "position": position
                    })
                    
            return positions_to_close
            
        except Exception as e:
            handle_error(e, "RiskManager.update_positions", logger=self.logger)
            return []

    def _calculate_position_correlation(self, new_symbol: str) -> Decimal:
        """Calculate correlation between positions"""
        # Implementation based on price correlation
        return Decimal(0)  # Placeholder

    def _calculate_expected_value(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Decimal:
        """Calculate trade expected value"""
        try:
            probability = Decimal(str(signal.get("probability", 0.5)))
            potential_profit = Decimal(str(signal.get("potential_profit", 0)))
            potential_loss = Decimal(str(signal.get("potential_loss", 0)))
            
            return (probability * potential_profit) - ((1 - probability) * potential_loss)
            
        except Exception as e:
            handle_error(e, "RiskManager._calculate_expected_value", logger=self.logger)
            return Decimal(0)

    def _calculate_kelly_position_size(
        self,
        probability: Decimal,
        expected_value: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate position size using Kelly Criterion"""
        try:
            kelly_fraction = probability - ((1 - probability) / (expected_value / price))
            kelly_fraction = max(Decimal(0), min(kelly_fraction, Decimal(1)))
            
            # Apply Kelly scaling factor
            return kelly_fraction * self.risk_limits['max_position_size']
            
        except Exception as e:
            handle_error(e, "RiskManager._calculate_kelly_position_size", logger=self.logger)
            return Decimal(0)

    def _check_stop_conditions(self, position: Position) -> bool:
        """Check if position should be stopped out"""
        return (
            (position.direction == "long" and position.current_price <= position.stop_loss) or
            (position.direction == "short" and position.current_price >= position.stop_loss)
        )

    async def calculate_position_size(self, signal: Dict) -> Decimal:
        """Calculate safe position size with all risk checks"""
        try:
            price = self.nh.to_decimal(signal['price'])
            account_size = await self.get_account_size()
            risk_factor = self.nh.percentage_to_decimal(self.risk_limits['max_position_size'])
            
            position_size = account_size * risk_factor
            return min(
                position_size,
                self.risk_limits['max_position_size'] * account_size
            )
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return Decimal('0')

    async def validate_order(self, order: Dict[str, Any]) -> bool:
        async with self._lock:
            try:
                position_size = Decimal(str(order['amount']))
                symbol = order['symbol']
                
                if not await self._check_position_limits(symbol, position_size):
                    return False
                    
                if not await self._check_risk_limits(position_size):
                    return False
                    
                return True
                
            except Exception as e:
                self.logger.error(f"Order validation failed: {e}")
                return False

    async def _check_position_limits(self, symbol: str, size: Decimal) -> bool:
        limits = self.position_limits.get(symbol, {})
        min_size = Decimal(str(limits.get('min_qty', 0)))
        max_size = Decimal(str(limits.get('max_qty', float('inf'))))
        return min_size <= size <= max_size 