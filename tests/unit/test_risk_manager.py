from unittest.mock import AsyncMock
import pytest
from decimal import Decimal
import logging

from risk.manager import RiskManager
from risk.limits import RiskLimits
from database.queries import DatabaseQueries
from utils.error_handler import RiskError


@pytest.fixture
def risk_limits():
    """Provide test risk limits"""
    return RiskLimits.from_config({
        'max_position_size': '0.1',
        'min_position_size': '0.01',
        'max_positions': 3,
        'max_leverage': '2.0',
        'max_drawdown': '0.1',
        'max_daily_loss': '0.03',
        'emergency_stop_pct': '-3.0',
        'risk_factor': '0.01',
        'kelly_scaling': '0.5',
        'max_correlation': '0.7',
        'max_sector_exposure': '0.3',
        'max_volatility': '0.05',
        'min_liquidity': '100000'
    })


@pytest.fixture
async def db_queries():
    """Provide a mocked DatabaseQueries instance."""
    return DatabaseQueries(AsyncMock())


@pytest.fixture
def logger():
    """Provide a logger fixture."""
    return logging.getLogger("TestRiskManager")


@pytest.mark.asyncio
async def test_calculate_kelly_fraction():
    """Test Kelly fraction calculation."""
    from trading.math import calculate_kelly_fraction
    probability = Decimal('0.6')
    odds = Decimal('1.5')
    
    kelly = calculate_kelly_fraction(probability, odds)
    expected = (probability * (odds + 1) - 1) / odds  # (0.6 * 2.5 -1)/1.5 = (1.5 -1)/1.5 = 0.333...
    assert kelly == Decimal('0.3333333333333333')


@pytest.mark.asyncio
async def test_calculate_position_size(risk_limits, db_queries, logger):
    """Test position size calculation based on risk factors."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    symbol = "BTC/USDT"
    account_size = Decimal('10000')
    expected_risk = account_size * risk_limits.risk_factor  # 100
    
    signal = {
        "probability": Decimal('0.6'),
        "odds": Decimal('1.5'),
        "entry_price": Decimal('50000')
    }
    
    kelly_fraction = (signal['probability'] * (signal['odds'] + 1) - 1) / signal['odds']  # 0.333...
    calculated_size = rm.calculate_position_size(signal, Decimal('50000'))
    expected_size = (expected_risk * Decimal(kelly_fraction)).quantize(Decimal('0.0001')) / signal['entry_price']
    
    assert calculated_size == expected_size
    assert calculated_size <= risk_limits.max_position_size
    assert calculated_size >= risk_limits.min_position_size


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


@pytest.mark.asyncio
async def test_validate_risk_metrics_exceed_drawdown(risk_limits, db_queries, logger):
    """Test that validate_risk_metrics raises RiskError when drawdown is exceeded."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    rm.current_drawdown = Decimal('0.15')  # Exceeds max_drawdown
    rm.daily_loss = Decimal('0.02')        # Within max_daily_loss
    
    with pytest.raises(RiskError, match="Drawdown exceeds maximum allowed"):
        await rm.validate_risk_metrics()


@pytest.mark.asyncio
async def test_validate_risk_metrics_exceed_daily_loss(risk_limits, db_queries, logger):
    """Test that validate_risk_metrics raises RiskError when daily loss is exceeded."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    rm.current_drawdown = Decimal('0.05')  # Within max_drawdown
    rm.daily_loss = Decimal('0.05')        # Exceeds max_daily_loss
    
    with pytest.raises(RiskError, match="Daily loss exceeds maximum allowed"):
        await rm.validate_risk_metrics()


@pytest.mark.asyncio
async def test_emergency_stop_triggered(risk_limits, db_queries, logger):
    """Test that emergency stop is triggered correctly."""
    rm = RiskManager(risk_limits, db_queries, logger)
    
    # Simulate emergency drawdown
    rm.current_drawdown = Decimal('-3.5')  # Beyond emergency_stop_pct
    
    with pytest.raises(RiskError, match="Emergency stop triggered"):
        await rm.validate_risk_metrics() 