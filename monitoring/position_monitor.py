from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import asyncio
import logging

from risk.limits import RiskLimits
from risk.manager import RiskManager
from risk.validation import MarketDataValidation

@dataclass
class PositionAlert:
    symbol: str
    alert_type: str
    message: str
    timestamp: float
    severity: str

class PositionMonitor:
    def __init__(self, risk_manager: RiskManager, logger: Optional[logging.Logger] = None):
        self.risk_manager = risk_manager
        self.alerts: List[PositionAlert] = []
        self.logger = logger or logging.getLogger(__name__)
        
    async def monitor_positions(self):
        while True:
            try:
                for symbol, position in self.risk_manager.portfolio.positions.items():
                    # Check duration
                    if time.time() - position.entry_time > 24*3600:  # 24h
                        alert = PositionAlert(
                            symbol=symbol,
                            alert_type="DURATION",
                            message=f"Position held > 24h",
                            timestamp=time.time(),
                            severity="WARNING"
                        )
                        self.alerts.append(alert)
                        self.logger.warning(alert.message)
                    
                    # Check drawdown
                    if position.max_adverse_excursion < Decimal("-0.1"):  # 10%
                        alert = PositionAlert(
                            symbol=symbol,
                            alert_type="DRAWDOWN",
                            message=f"Position MAE > 10%",
                            timestamp=time.time(),
                            severity="CRITICAL"
                        )
                        self.alerts.append(alert)
                        self.logger.critical(alert.message)
                
                await asyncio.sleep(1)  # 1s check interval
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Back off on error 