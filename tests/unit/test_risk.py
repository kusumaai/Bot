#! /usr/bin/env python3
# tests/unit/test_risk.py
"""
Module: tests.unit
Provides unit testing functionality for the risk module.
"""
import logging
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from database.queries import DatabaseQueries
from risk.limits import RiskLimits
from risk.manager import RiskManager
from risk.validation import MarketDataValidation
from trading.position import Position
from utils.error_handler import RiskError, ValidationError
from utils.numeric_handler import NumericHandler


@pytest.fixture
def risk_limits():
    """Provide test risk limits"""
    return RiskLimits.from_config(
        {
            "max_position_size": "0.1",
            "min_position_size": "0.01",
            "max_positions": 3,
            "max_leverage": "2.0",
            "max_drawdown": "0.1",
            "max_daily_loss": "0.03",
            "emergency_stop_pct": "-3.0",
            "risk_factor": "0.01",
            "kelly_scaling": "0.5",
            "max_correlation": "0.7",
            "max_sector_exposure": "0.3",
            "max_volatility": "0.05",
            "min_liquidity": "100000",
        }
    )


@pytest.fixture
def sample_candles():
    """Provide sample market data"""
    return [
        {
            "timestamp": int((datetime.utcnow() - timedelta(minutes=i)).timestamp()),
            "open": 35000.0 + i,
            "high": 35100.0 + i,
            "low": 34900.0 + i,
            "close": 35050.0 + i,
            "volume": 10.0,
        }
        for i in range(20)
    ]


@pytest.mark.asyncio
async def test_risk_limits_validation(risk_limits):
    """Test risk limits validation for position sizes and exposures."""
    # Test position size within limits
    assert risk_limits.validate_position_size(Decimal("0.05"))
    assert not risk_limits.validate_position_size(Decimal("0.2"))
    assert risk_limits.validate_position_size(Decimal("0.01"))
    assert not risk_limits.validate_position_size(Decimal("0.0"))

    # Test maximum number of positions
    portfolio_size = 2
    assert risk_limits.validate_max_positions(portfolio_size)
    portfolio_size = 4
    assert not risk_limits.validate_max_positions(portfolio_size)

    # Test leverage limits
    assert risk_limits.validate_leverage(Decimal("1.5"))
    assert not risk_limits.validate_leverage(Decimal("2.5"))

    # Test drawdown limits
    assert risk_limits.validate_drawdown(Decimal("0.05"))
    assert not risk_limits.validate_drawdown(Decimal("0.15"))

    # Test daily loss limits
    assert risk_limits.validate_daily_loss(Decimal("0.02"))
    assert not risk_limits.validate_daily_loss(Decimal("0.05"))

    # Test emergency stop
    assert not risk_limits.emergency_stop_triggered(Decimal("0.0"))
    assert risk_limits.emergency_stop_triggered(Decimal("-3.5"))


@pytest.mark.asyncio
async def test_position_size_calculation(risk_limits, db_queries, logger):
    """Test calculation of position sizes based on risk factors."""
    rm = RiskManager(risk_limits, db_queries, logger)

    symbol = "BTC/USDT"
    account_size = Decimal("10000")
    expected_risk = account_size * risk_limits.risk_factor  # 100
    # Example: Kelly Fraction = (prob * (b + 1) - 1) / b
    signal = {
        "probability": Decimal("0.6"),
        "odds": Decimal("1.5"),  # Implied by strategy
    }
    kelly_fraction = rm.calculate_kelly_fraction(signal["probability"], signal["odds"])
    expected_position_size = (expected_risk * kelly_fraction) / Decimal(
        "35000"
    )  # Example entry price
    calculated_size = rm.calculate_position_size(signal, Decimal("35000"))

    assert isinstance(calculated_size, Decimal)
    assert calculated_size <= risk_limits.max_position_size
    assert calculated_size >= risk_limits.min_position_size


@pytest.mark.asyncio
async def test_risk_metrics_validation(risk_limits, db_queries, logger):
    """Test validation of risk metrics such as drawdown and daily loss."""
    rm = RiskManager(risk_limits, db_queries, logger)

    # Simulate acceptable metrics
    rm.current_drawdown = Decimal("0.05")
    rm.daily_loss = Decimal("0.02")
    assert await rm.validate_risk_metrics() is True

    # Simulate exceeding drawdown
    rm.current_drawdown = Decimal("0.15")
    with pytest.raises(RiskError):
        await rm.validate_risk_metrics()

    # Reset and simulate exceeding daily loss
    rm.current_drawdown = Decimal("0.05")
    rm.daily_loss = Decimal("0.05")
    with pytest.raises(RiskError):
        await rm.validate_risk_metrics()

    # Simulate emergency stop
    rm.current_drawdown = Decimal("-3.5")
    with pytest.raises(RiskError):
        await rm.validate_risk_metrics()


@pytest.mark.asyncio
async def test_correlation_validation(risk_limits, db_queries, logger):
    """Test correlation validation between different positions."""
    rm = RiskManager(risk_limits, db_queries, logger)

    # Add positions with acceptable correlation
    rm.portfolio.add_position(
        Position("BTC/USDT", "long", Decimal("1"), Decimal("50000"))
    )
    rm.portfolio.add_position(
        Position("ETH/USDT", "long", Decimal("10"), Decimal("3000"))
    )

    correlations = {"ETH/USDT": Decimal("0.5"), "SOL/USDT": Decimal("0.6")}
    validation = rm.validate_correlation("BTC/USDT", correlations)
    assert validation is True

    # Add a position with high correlation
    correlations["XRP/USDT"] = Decimal("0.8")
    validation = rm.validate_correlation("BTC/USDT", correlations)
    assert validation is False


@pytest.mark.asyncio
async def test_risk_manager_portfolio_management(risk_limits, db_queries, logger):
    """Test adding and removing positions in the portfolio."""
    rm = RiskManager(risk_limits, db_queries, logger)

    # Initially, no positions
    assert len(rm.portfolio.positions) == 0

    # Add a position
    pos = Position("BTC/USDT", "long", Decimal("0.05"), Decimal("50000"))
    success = rm.portfolio.add_position(pos)
    assert success is True
    assert len(rm.portfolio.positions) == 1

    # Attempt to add another position exceeding max positions
    for _ in range(risk_limits.max_positions):
        pos = Position("ETH/USDT", "short", Decimal("0.02"), Decimal("3000"))
        rm.portfolio.add_position(pos)

    # Now, adding one more should fail
    pos_extra = Position("SOL/USDT", "long", Decimal("0.03"), Decimal("100"))
    with pytest.raises(RiskError):
        rm.portfolio.add_position(pos_extra)
