#!/usr/bin/env python3
"""
signals/trading_types.py - Core data types for trading system
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from decimal import Decimal
import numpy as np
import time

@dataclass
class TradeMetrics:
    """Performance metrics for a set of trades"""
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    exposure_ratio: Decimal = Decimal("0")
    avg_trade_return: Decimal = Decimal("0")
    total_trades: int = 0
    consecutive_losses: int = 0
    kelly_fraction: Decimal = Decimal("0")
    avg_leverage: Decimal = Decimal("0")
    peak_value: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    last_update_time: float = field(default_factory=time.time)

@dataclass
class MarketState:
    """Current market conditions and metrics"""
    returns: np.ndarray
    ar1_coef: float
    current_return: float
    volatility: float
    last_price: Decimal
    ema_short: Decimal
    ema_long: Decimal
    atr: Decimal = Decimal("0")
    rsi: Decimal = Decimal("50")
    bb_width: Decimal = Decimal("0")
    volume_ma_ratio: Decimal = Decimal("1")
    trend_strength: Decimal = Decimal("0")
    timestamp: float = field(default_factory=time.time)
    ctx: Any = None

    def is_valid(self) -> bool:
        """Check if market state is valid and recent"""
        try:
            if time.time() - self.timestamp > 300:  # 5 minute staleness check
                return False
            return (
                len(self.returns) > 0 and
                self.last_price > 0 and
                self.volatility >= 0
            )
        except Exception:
            return False

@dataclass
class TradingRule:
    """Trading strategy definition"""
    buy_conditions: List[Dict[str, Any]]
    sell_conditions: List[Dict[str, Any]]
    fitness: Decimal = Decimal("0")
    generation: int = 0
    parent_fitness: Decimal = Decimal("0")
    mutation_rate: Decimal = Decimal("0.1")
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Validate trading rule structure"""
        try:
            if not self.buy_conditions or not self.sell_conditions:
                return False
                
            for condition in self.buy_conditions + self.sell_conditions:
                if not all(k in condition for k in ["indicator", "op", "ref"]):
                    return False
                    
            return True
        except Exception:
            return False

@dataclass
class SimulationResult:
    """Results from strategy simulation"""
    fitness: Decimal
    metrics: TradeMetrics
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]
    equity_curve: List[Decimal]
    drawdown_curve: List[Decimal]
    exposure_curve: List[Decimal]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Check if simulation results are valid"""
        try:
            return (
                self.fitness >= 0 and
                len(self.trades) > 0 and
                len(self.equity_curve) > 0 and
                len(self.equity_curve) == len(self.drawdown_curve) and
                self.end_time > self.start_time
            )
        except Exception:
            return False

@dataclass
class SignalMetadata:
    """Enhanced signal metadata"""
    probability: Decimal
    expected_value: Decimal
    kelly_fraction: Decimal
    predicted_return: Decimal
    entry_price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    leverage: Decimal = Decimal("1")
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Validate signal metadata"""
        try:
            return (
                self.probability > 0 and
                self.probability <= 1 and
                self.entry_price > 0 and
                self.leverage > 0 and
                time.time() - self.timestamp < 300  # 5 minute staleness check
            )
        except Exception:
            return False