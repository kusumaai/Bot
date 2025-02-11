from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any

@dataclass(frozen=True)
class RiskLimits:
    max_position_size: Decimal
    max_positions: int
    max_leverage: Decimal
    max_correlation: float
    max_drawdown: Decimal
    emergency_stop_pct: Decimal
    max_daily_loss: Decimal
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RiskLimits':
        return cls(
            max_position_size=Decimal(str(config.get("max_position_size", 0.1))),
            max_positions=config.get("max_positions", 3),
            max_leverage=Decimal(str(config.get("max_leverage", 2.0))),
            max_correlation=config.get("max_correlation", 0.7),
            max_drawdown=Decimal(str(config.get("max_drawdown_pct", 10))) / 100,
            emergency_stop_pct=Decimal(str(config.get("emergency_stop_pct", 5))) / 100,
            max_daily_loss=Decimal(str(config.get("max_daily_loss", 3))) / 100
        )