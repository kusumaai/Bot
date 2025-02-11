from decimal import Decimal
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from .position import Position
from .limits import RiskLimits
from .portfolio import PortfolioManager
from .validation import validate_market_data
# Remove portfolio import, use type hints instead
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .portfolio import PortfolioManager

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.limits = RiskLimits.from_config(config)
        self.portfolio = PortfolioManager(self.limits)
        
    def validate_position(self, symbol: str, size: Decimal, price: Decimal) -> bool:
        """Validate a new position against all risk limits"""
        # Check max positions
        if len(self.portfolio.positions) >= self.limits.max_positions:
            return False
            
        # Check position size
        position_value = size * price
        portfolio_value = self.portfolio.calculate_portfolio_value()
        if position_value / portfolio_value > self.limits.max_position_size:
            return False
            
        # Check leverage
        total_exposure = sum(
            p.size * p.current_price 
            for p in self.portfolio.positions.values()
        ) + position_value
        
        if total_exposure / portfolio_value > self.limits.max_leverage:
            return False
            
        # Check drawdown
        if self.portfolio.calculate_drawdown() > self.limits.max_drawdown:
            return False
            
        return True
        
    def calculate_position_params(self, signal: Dict[str, Any], 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position parameters with risk limits"""
        with self.portfolio.lock:
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
            
            return {
                "size": size,
                "stop_loss": current_price * (1 - self.limits.emergency_stop_pct),
                "take_profit": current_price * (1 + self.limits.emergency_stop_pct * 2)
            }
        
    def update_positions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update all positions and return any that need closing"""
        positions_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            current_price = Decimal(str(market_data[symbol]["current_price"]))
            position.update(current_price)
            
            # Check stop conditions
            if (position.direction == "long" and current_price <= position.stop_loss) or \
               (position.direction == "short" and current_price >= position.stop_loss):
                positions_to_close.append({
                    "symbol": symbol,
                    "reason": "stop_loss",
                    "position": position
                })
                
        return positions_to_close 