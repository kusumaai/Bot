#!/usr/bin/env python3
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

#dataclass for the trade result
@dataclass
class TradeResult:
    # Required fields (no defaults)
    symbol: str
    entry_time: datetime
    entry_price: Decimal
    direction: str
    size: Decimal
    
    # Optional fields (with defaults)
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    pnl: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)

#dataclass for the backtest results
@dataclass
class BacktestResults:
    # Required fields (no defaults)
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    
    # Optional fields (with defaults)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    #string representation of the backtest results
    def __str__(self) -> str:
        return (
            f"Total Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate:.2%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
        ) 
