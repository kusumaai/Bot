import pytest
from decimal import Decimal
import logging
from datetime import datetime, timedelta

from risk.limits import RiskLimits
from risk.validation import MarketDataValidation
from risk.manager import RiskManager, PositionInfo
from utils.error_handler import ValidationError, RiskError

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
def sample_candles():
    """Provide sample market data"""
    return [
        {
            'timestamp': int((datetime.utcnow() - timedelta(minutes=i)).timestamp()),
            'open': 35000.0 + i,
            'high': 35100.0 + i,
            'low': 34900.0 + i,
            'close': 35050.0 + i,
            'volume': 10.0
        } for i in range(20)
    ]

async def test_risk_limits_validation(risk_limits):
    """Test risk limits validation"""
    # Test position size validation
    assert risk_limits.validate_position_size(Decimal('0.05'))
    assert not risk_limits.validate_position_size(Decimal('0.2'))
    
    # Test exposure validation
    assert risk_limits.validate_total_exposure(2)
    assert not risk_limits.validate_total_exposure(3)
    
    # Test drawdown validation
    assert risk_limits.validate_drawdown(Decimal('-0.05'))
    assert not risk_limits.validate_drawdown(Decimal('-0.15'))
    
    # Test emergency stop
    assert risk_limits.validate_emergency_stop(Decimal('-0.02'))
    assert not risk_limits.validate_emergency_stop(Decimal('-0.04'))

async def test_market_data_validation(risk_limits, sample_candles, logger):
    """Test market data validation"""
    validator = MarketDataValidation(risk_limits, logger)
    
    # Test valid data
    assert await validator.validate_candle_data(sample_candles)
    
    # Test insufficient data
    with pytest.raises(ValidationError):
        await validator.validate_candle_data(sample_candles[:5])
    
    # Test invalid prices
    invalid_candles = sample_candles.copy()
    invalid_candles[0]['high'] = invalid_candles[0]['low'] - 100
    with pytest.raises(ValidationError):
        await validator.validate_candle_data(invalid_candles)
    
    # Test liquidity validation
    assert validator.validate_liquidity(
        volume=Decimal('100'),
        price=Decimal('35000')
    )
    
    with pytest.raises(ValidationError):
        validator.validate_liquidity(
            volume=Decimal('1'),
            price=Decimal('35000')
        )

async def test_risk_manager(risk_limits, db_queries, logger, sample_candles):
    """Test risk manager functionality"""
    risk_manager = RiskManager(risk_limits, db_queries, logger)
    
    # Test position validation
    valid_size = Decimal('0.05')
    symbol = "BTC/USDT"
    
    assert await risk_manager.validate_new_position(
        symbol=symbol,
        direction='long',
        size=valid_size,
        price=Decimal('35000'),
        candles=sample_candles
    )
    
    # Test position size calculation
    account_size = Decimal('100000')
    size = await risk_manager.calculate_position_size(
        symbol=symbol,
        account_size=account_size
    )
    assert size <= risk_limits.max_position_size
    assert size >= risk_limits.min_position_size
    
    # Test risk metrics
    daily_pnl = await risk_manager._get_daily_pnl()
    assert isinstance(daily_pnl, Decimal)

async def test_correlation_validation(risk_limits, db_queries, logger, sample_candles):
    """Test correlation validation"""
    risk_manager = RiskManager(risk_limits, db_queries, logger)
    
    # Create test correlations
    correlations = {
        'ETH/USDT': Decimal('0.5'),  # Acceptable correlation
        'SOL/USDT': Decimal('0.8')   # Too high correlation
    }
    
    # Test correlation validation
    symbol = "BTC/USDT"
    validator = MarketDataValidation(risk_limits, logger)
    
    assert validator.validate_correlation(symbol, {'ETH/USDT': Decimal('0.5')})
    
    with pytest.raises(ValidationError):
        validator.validate_correlation(symbol, {'SOL/USDT': Decimal('0.8')})

async def test_emergency_conditions(risk_limits, db_queries, logger):
    """Test emergency risk conditions"""
    risk_manager = RiskManager(risk_limits, db_queries, logger)
    
    # Simulate emergency conditions
    async def mock_daily_pnl():
        return Decimal('-3.5')  # Beyond emergency stop
    
    risk_manager._get_daily_pnl = mock_daily_pnl
    
    with pytest.raises(RiskError):
        await risk_manager._validate_risk_metrics()