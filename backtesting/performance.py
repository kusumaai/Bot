from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict


@dataclass
class PerformanceStats:
    # Required fields (no defaults)
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_trades: int
    win_rate: float
    
    # Optional fields (with defaults)
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    trades_per_day: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict) 