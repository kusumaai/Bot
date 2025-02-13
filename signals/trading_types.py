#!/usr/bin/env python3
"""
signals/trading_types.py - Core data types for trading system
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from decimal import Decimal
import numpy as np
import time
from datetime import datetime
from bot_types.base_types import MarketState, ValidationResult
from bot_types.base_types import Validatable

@dataclass
class TradeMetrics:
    """Performance metrics for a set of trades"""
    # Required fields (no defaults)
    symbol: str
    timestamp: datetime
    trade_count: int
    
    # Optional fields (with defaults)
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    exposure_ratio: Decimal = Decimal("0")
    avg_trade_return: Decimal = Decimal("0")
    consecutive_losses: int = 0
    kelly_fraction: Decimal = Decimal("0")
    avg_leverage: Decimal = Decimal("0")
    peak_value: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    last_update_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketState:
    """Current market conditions and metrics"""
    # Required fields (no defaults)
    returns: np.ndarray
    ar1_coef: float
    current_return: float
    volatility: float
    last_price: Decimal
    ema_short: Decimal
    ema_long: Decimal
    
    # Optional fields (with defaults)
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
        except Exception as e:
            return False

@dataclass
class TradingRule:
    """Trading rule configuration and performance tracking"""
    # Required fields (no defaults)
    id: str
    conditions: Dict[str, Any]
    weights: Dict[str, float]
    
    # Optional fields (with defaults)
    performance: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
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
class SignalMetadata(Validatable):
    """Enhanced signal metadata"""
    # Required fields (no defaults)
    timeframe: str
    market_state: Dict[str, Any]
    model_version: str
    probability: Decimal
    expected_value: Decimal
    kelly_fraction: Decimal
    predicted_return: Decimal
    entry_price: Decimal
    
    # Optional fields (with defaults)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    leverage: Decimal = Decimal("1")
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate signal metadata"""
        try:
            if self.probability < Decimal('0') or self.probability > Decimal('1'):
                return ValidationResult(
                    is_valid=False,
                    error_message="Probability must be between 0 and 1"
                )
                
            if self.leverage < Decimal('1'):
                return ValidationResult(
                    is_valid=False,
                    error_message="Leverage cannot be less than 1"
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False, 
                error_message=f"Signal validation failed: {str(e)}"
            )

@dataclass
class Signal:
    symbol: str
    signal_type: str
    direction: str
    strength: Decimal
    timestamp: datetime
    metadata: SignalMetadata