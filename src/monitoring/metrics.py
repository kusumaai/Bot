from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class TradingMetrics:
    # Required fields (no defaults)
    symbol: str
    timestamp: datetime
    trade_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    
    # Optional fields (with defaults)
    drawdown: float = 0.0
    volatility: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)