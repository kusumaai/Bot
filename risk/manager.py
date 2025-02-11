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

from utils.error_handler import handle_error
from .position import Position
from .limits import RiskLimits
from .portfolio import PortfolioManager
from .validation import validate_market_data
# Remove portfolio import, use type hints instead
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .portfolio import PortfolioManager

class RiskManager:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.logger = ctx.logger
        self.limits = RiskLimits.from_config(ctx.config)
        if not self.limits:
            raise ValueError("Failed to initialize risk limits")
        self.portfolio = PortfolioManager(self.limits)
        
    def validate_position(self, symbol: str, size: Decimal, price: Decimal) -> Tuple[bool, Optional[str]]:
        """Validate a new position against all risk limits"""
        try:
            # Check max positions
            if len(self.portfolio.positions) >= self.limits.max_positions:
                return False, "Maximum positions limit reached"
                
            # Check position size
            position_value = size * price
            portfolio_value = self.portfolio.calculate_portfolio_value()
            if position_value / portfolio_value > self.limits.max_position_size:
                return False, "Position size exceeds maximum allowed"
                
            # Check leverage
            total_exposure = sum(
                p.size * p.current_price 
                for p in self.portfolio.positions.values()
            ) + position_value
            
            if total_exposure / portfolio_value > self.limits.max_leverage:
                return False, "Leverage limit exceeded"
                
            # Check drawdown
            if self.portfolio.calculate_drawdown() > self.limits.max_drawdown:
                return False, "Maximum drawdown reached"
                
            # Check correlation if multiple positions
            if len(self.portfolio.positions) > 0:
                correlation = self._calculate_position_correlation(symbol)
                if correlation > self.limits.max_correlation:
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
                size *= self.limits.risk_factor
                
                # Calculate stops based on volatility
                atr = Decimal(str(market_data.get("ATR_14", self.limits.emergency_stop_pct)))
                stop_loss = current_price * (Decimal(1) - atr)
                take_profit = current_price * (Decimal(1) + atr * Decimal(2))
                
                return {
                    "size": size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "trailing_stop": self.limits.trailing_stop_pct
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
                if position.max_adverse_excursion < -self.limits.max_adverse_pct:
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
            return kelly_fraction * self.limits.kelly_scaling
            
        except Exception as e:
            handle_error(e, "RiskManager._calculate_kelly_position_size", logger=self.logger)
            return Decimal(0)

    def _check_stop_conditions(self, position: Position) -> bool:
        """Check if position should be stopped out"""
        return (
            (position.direction == "long" and position.current_price <= position.stop_loss) or
            (position.direction == "short" and position.current_price >= position.stop_loss)
        ) 