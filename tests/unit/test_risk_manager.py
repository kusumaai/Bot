#! /usr/bin/env python3
#test risk manager
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from decimal import Decimal
import logging
from src.risk.manager import RiskManager, PositionInfo
from src.risk.limits import RiskLimits
from src.database.queries import DatabaseQueries
from src.utils.error_handler import RiskError
from src.utils.exceptions import MathError, RiskCalculationError

#risk limits for the risk manager tests
@pytest.fixture
def risk_limits():
    """Provide test risk limits"""
    return RiskLimits(
        max_value=Decimal('1000'),
        max_correlation=Decimal('0.75'),
        min_liquidity=Decimal('10000'),
        max_position_size=Decimal('500'),
        min_position_size=Decimal('10'),
        risk_factor=Decimal('1')  # Assuming '1' represents 1%
    )

#database queries for the risk manager tests
@pytest.fixture
async def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return DatabaseQueries(AsyncMock())

#logger for the risk manager tests
@pytest.fixture
def logger():
    """Provide a logger fixture."""
    return logging.getLogger("TestRiskManager")

#test calculate kelly fraction for the risk manager tests
@pytest.mark.asyncio
async def test_calculate_kelly_fraction(risk_manager):
    """Test Kelly fraction calculation."""
    probability = Decimal('0.6')
    odds = Decimal('1.5')
    loss_target = Decimal('0.05')
    kelly = risk_manager.calculate_kelly_fraction(probability, odds, loss_target)
    expected_kelly = (probability * (odds + 1) - 1) / odds
    assert kelly == expected_kelly.quantize(Decimal('0.0001'))

#test calculate position size for the risk manager tests
@pytest.mark.asyncio
async def test_calculate_position_size(risk_limits, db_queries, logger):
    """Test position size calculation based on risk factors."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    signal = {
        "probability": Decimal('0.6'),
        "odds": Decimal('1.5'),
        "price": Decimal('50000')
    }
    
    # Mock portfolio.get_total_value()
    rm.portfolio.get_total_value = MagicMock(return_value=Decimal('10000'))
    
    calculated_size = await rm.calculate_position_size(signal, Decimal('50000'))
    expected_size = (Decimal('10000') * (Decimal('1') / Decimal('100'))).quantize(Decimal('0.0001'))  # 100 / 50000 = 0.0020
    
    assert isinstance(calculated_size, Decimal)
    assert calculated_size == expected_size
    assert calculated_size <= risk_limits.max_position_size
    assert calculated_size >= risk_limits.min_position_size

#test calculate position size for the risk manager tests
@pytest.mark.asyncio
async def test_calculate_position_size_invalid_price(risk_limits, db_queries, logger):
    """Test position size calculation with invalid price."""
    rm = RiskManager(risk_limits, db_queries, logger)
    #signal for the risk manager tests
    signal = {
        "probability": Decimal('0.6'),
        "odds": Decimal('1.5'),
        "price": 'invalid_price'  # Invalid price
    }
    
    # Mock portfolio.get_total_value()
    rm.portfolio.get_total_value = MagicMock(return_value=Decimal('10000'))
    
    calculated_size = await rm.calculate_position_size(signal, Decimal('50000'))
    assert calculated_size == Decimal('0')

#test validate risk metrics within limits for the risk manager tests
@pytest.mark.asyncio
async def test_validate_risk_metrics_within_limits(risk_limits, db_queries, logger):
    """Test that validate_risk_metrics passes when within limits."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    rm.current_drawdown = Decimal('0.05')  # Within max_drawdown
    rm.daily_loss = Decimal('0.02')        # Within max_daily_loss
    with pytest.raises(RiskError):
        await rm.validate_risk_metrics()  # Assuming this should pass without error
    
    # Adjust test to not raise using the corrected implementation
    try:
        await rm.validate_risk_metrics()
    except RiskError:
        pytest.fail("validate_risk_metrics raised RiskError unexpectedly!")

#test validate risk metrics exceed drawdown for the risk manager tests
@pytest.mark.asyncio
async def test_validate_risk_metrics_exceed_drawdown(risk_limits, db_queries, logger):
    """Test that validate_risk_metrics raises RiskError when drawdown is exceeded."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    rm.current_drawdown = Decimal('0.15')  # Exceeds max_drawdown
    rm.daily_loss = Decimal('0.02')        # Within max_daily_loss
    
    with pytest.raises(RiskError, match="Drawdown exceeds maximum allowed"):
        await rm.validate_risk_metrics()

#test validate risk metrics exceed daily loss for the risk manager tests
@pytest.mark.asyncio
async def test_validate_risk_metrics_exceed_daily_loss(risk_limits, db_queries, logger):
    """Test that validate_risk_metrics raises RiskError when daily loss is exceeded."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    rm.current_drawdown = Decimal('0.05')  # Within max_drawdown
    rm.daily_loss = Decimal('0.05')        # Exceeds max_daily_loss
    
    with pytest.raises(RiskError, match="Daily loss exceeds maximum allowed"):
        await rm.validate_risk_metrics()

#test emergency stop triggered for the risk manager tests
@pytest.mark.asyncio
async def test_emergency_stop_triggered(risk_limits, db_queries, logger):
    """Test that emergency stop is triggered correctly."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    # Simulate emergency drawdown
    rm.current_drawdown = Decimal('3.5')  # Beyond emergency_stop_pct
    
    with pytest.raises(RiskError, match="Emergency stop triggered"):
        await rm.validate_risk_metrics() 

#test mock portfolio manager for the risk manager tests 
@pytest.fixture
def mock_portfolio_manager():
    """Provide a mocked PortfolioManager."""
    mock_portfolio = MagicMock(spec=PortfolioManager)
    mock_portfolio.get_total_positions.return_value = 2
    mock_portfolio.current_drawdown = Decimal('0.05')
    return mock_portfolio

#test risk manager for the risk manager tests   
@pytest.fixture
def risk_manager(risk_limits, mock_portfolio_manager, logger):
    """Provide a RiskManager instance with mocked dependencies."""
    ctx = MagicMock()
    ctx.logger = logger
    ctx.portfolio_manager = mock_portfolio_manager
    ctx.config = {
        "position_limits": {
            "max_position_size": "0.1",
            "min_position_size": "0.01",
            "max_positions": 3,
            "max_leverage": "2.0",
            "max_drawdown": "0.1",
            "max_daily_loss": "0.03",
            "emergency_stop_pct": "3.0",
            "risk_factor": "0.01",
            "kelly_scaling": "0.5",
            "max_correlation": "0.7",
            "max_sector_exposure": "0.3",
            "max_volatility": "0.05",
            "min_liquidity": "100000"
        }
    }
    risk_mgr = RiskManager(ctx)
    asyncio.run(risk_mgr.initialize())
    return risk_mgr


@pytest.mark.asyncio
async def test_validate_trade_within_limits(risk_manager):
    """Test validating a trade within risk limits."""
    is_valid, error = await risk_manager.validate_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.05'),
        price=Decimal('50000')
    )
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_trade_below_min_size(risk_manager):
    """Test validating a trade below minimum position size."""
    is_valid, error = await risk_manager.validate_trade(
        symbol='BTC/USDT',
        side='buy',
        amount=Decimal('0.005'),
        price=Decimal('50000')
    )
    assert is_valid is False
    assert error == "Trade amount below minimum position size."


@pytest.mark.asyncio
async def test_validate_trade_exceeds_max_positions(risk_manager, mock_portfolio_manager):
    """Test validating a trade that exceeds the maximum number of positions."""
    mock_portfolio_manager.get_total_positions.return_value = 3
    is_valid, error = await risk_manager.validate_trade(
        symbol='ETH/USDT',
        side='sell',
        amount=Decimal('0.05'),
        price=Decimal('3000')
    )
    assert is_valid is False
    assert error == "Maximum number of positions reached." 