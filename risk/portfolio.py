from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any
import time
from .limits import RiskLimits

class PortfolioManager:
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.positions: Dict[str, Position] = {}
        self.balance = Decimal(0)
        self.peak_balance = Decimal(0)
        self._portfolio_value: Decimal = Decimal(0)
        self._last_update: float = 0
        self._update_interval: float = 0.1  # 100ms
        
    def calculate_portfolio_value(self) -> Decimal:
        now = time.time()
        if now - self._last_update >= self._update_interval:
            with self.lock:
                self._portfolio_value = self.balance + sum(
                    pos.unrealized_pnl for pos in self.positions.values()
                )
                self._last_update = now
        return self._portfolio_value
        
    def calculate_drawdown(self) -> Decimal:
        portfolio_value = self.calculate_portfolio_value()
        return (self.peak_balance - portfolio_value) / self.peak_balance 