#!/usr/bin/env python3
"""
Unit tests for risk management system
"""
import pytest
from decimal import Decimal
from typing import Dict, Any
from risk.manager import RiskManager
from risk.position import Position
from risk.limits import RiskLimits

def test_position_size_calculation(test_context):
    """Test position size calculation with risk factors"""
    # Initialize risk manager with test config
    rm = RiskManager(test_context)
    
    # Test basic calculation
    signal = {
        "symbol": "BTC/USDT",
        "direction": "long",
        "probability": Decimal("0.6"),
        "entry_price": Decimal("50000")
    }
    
    market_data = {
        "symbol": "BTC/USDT",
        "current_price": Decimal("50000"),
        "volatility": Decimal("0.2"),
        "volume": Decimal("1000"),
        "timestamp": 1600000000000
    }
    
    result = rm.calculate_position_size(signal, market_data)
    assert isinstance(result, Decimal)
    assert result > 0
    assert result <= test_context.config["max_position_size"]

def test_risk_limit_validation(test_context):
    """Test risk limit validation logic"""
    rm = RiskManager(test_context)
    
    # Test position count limit
    positions = [
        Position("BTC/USDT", "long", Decimal("1"), Decimal("50000")),
        Position("ETH/USDT", "long", Decimal("10"), Decimal("3000")),
        Position("SOL/USDT", "long", Decimal("100"), Decimal("100"))
    ]
    
    for pos in positions:
        rm.portfolio.add_position(pos)
    
    # Should reject new position due to max positions limit
    is_valid, reason = rm.validate_new_position(
        "ADA/USDT", 
        Decimal("1000"), 
        Decimal("2")
    )
    assert not is_valid
    assert "maximum positions" in reason.lower()

def test_drawdown_protection(test_context):
    """Test drawdown protection mechanisms"""
    rm = RiskManager(test_context)
    
    # Simulate drawdown
    rm.portfolio.peak_value = Decimal("10000")
    rm.portfolio.current_value = Decimal("8000")  # 20% drawdown
    
    # Should reject new position due to drawdown
    is_valid, reason = rm.validate_new_position(
        "BTC/USDT",
        Decimal("0.1"),
        Decimal("50000")
    )
    assert not is_valid
    assert "drawdown" in reason.lower()

def test_correlation_limits(test_context):
    """Test position correlation limits"""
    rm = RiskManager(test_context)
    
    # Add correlated position
    eth_pos = Position("ETH/USDT", "long", Decimal("10"), Decimal("3000"))
    rm.portfolio.add_position(eth_pos)
    
    # Test highly correlated asset
    is_valid, reason = rm.validate_new_position(
        "WETH/USDT",  # Wrapped ETH should be highly correlated
        Decimal("10"),
        Decimal("3000")
    )
    assert not is_valid
    assert "correlation" in reason.lower()

def test_leverage_limits(test_context):
    """Test leverage limit enforcement"""
    rm = RiskManager(test_context)
    
    # Add leveraged position
    pos = Position(
        "BTC/USDT",
        "long",
        Decimal("2"),  # 2x leverage
        Decimal("50000"),
        leverage=Decimal("2")
    )
    rm.portfolio.add_position(pos)
    
    # Test leverage limit
    is_valid, reason = rm.validate_new_position(
        "ETH/USDT",
        Decimal("10"),
        Decimal("3000"),
        leverage=Decimal("2")
    )
    assert not is_valid
    assert "leverage" in reason.lower()

def test_emergency_stop(test_context):
    """Test emergency stop loss mechanism"""
    rm = RiskManager(test_context)
    
    # Simulate large drawdown
    rm.portfolio.peak_value = Decimal("10000")
    rm.portfolio.current_value = Decimal("9500")
    
    # Add position
    pos = Position("BTC/USDT", "long", Decimal("1"), Decimal("50000"))
    rm.portfolio.add_position(pos)
    
    # Update with loss exceeding emergency stop
    rm.update_position_status(
        "BTC/USDT",
        Decimal("47500"),  # 5% loss
        timestamp=1600000000000
    )
    
    # Should trigger emergency stop
    assert rm.emergency_stop_triggered
    assert len(rm.portfolio.positions) == 0

def test_kelly_sizing(test_context):
    """Test Kelly Criterion position sizing"""
    rm = RiskManager(test_context)
    
    signal = {
        "symbol": "BTC/USDT",
        "direction": "long",
        "probability": Decimal("0.6"),
        "entry_price": Decimal("50000"),
        "stop_loss": Decimal("48000"),
        "take_profit": Decimal("54000")
    }
    
    market_data = {
        "symbol": "BTC/USDT",
        "current_price": Decimal("50000"),
        "volatility": Decimal("0.2"),
        "volume": Decimal("1000"),
        "timestamp": 1600000000000
    }
    
    size = rm.calculate_position_size(signal, market_data)
    
    # Kelly size should be scaled by kelly_scaling factor
    assert size <= test_context.config["kelly_scaling"]
    assert size > 0

def test_risk_factor_adjustment(test_context):
    """Test risk factor adjustment based on performance"""
    rm = RiskManager(test_context)
    
    # Simulate good performance
    rm.portfolio.win_rate = Decimal("0.65")
    rm.portfolio.profit_factor = Decimal("2.0")
    rm.portfolio.sharpe_ratio = Decimal("2.5")
    
    initial_risk = rm.current_risk_factor
    rm.adjust_risk_factors()
    
    # Risk factor should increase with good performance
    assert rm.current_risk_factor > initial_risk
    assert rm.current_risk_factor <= test_context.config["risk_factor"]

def test_position_correlation(test_context):
    """Test position correlation calculation"""
    rm = RiskManager(test_context)
    
    # Add positions
    positions = [
        Position("BTC/USDT", "long", Decimal("1"), Decimal("50000")),
        Position("ETH/USDT", "long", Decimal("10"), Decimal("3000")),
        Position("SOL/USDT", "short", Decimal("100"), Decimal("100"))
    ]
    
    for pos in positions:
        rm.portfolio.add_position(pos)
    
    # Calculate correlations
    correlations = rm.calculate_position_correlations()
    
    assert isinstance(correlations, dict)
    assert all(isinstance(v, Decimal) for v in correlations.values())
    assert all(-1 <= v <= 1 for v in correlations.values()) 