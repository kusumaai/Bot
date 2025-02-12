import logging
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from risk.validation import MarketDataValidation
from risk.limits import RiskLimits
from utils.error_handler import ValidationError


@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config({
        'max_correlation': '0.7',
        'min_liquidity': '100000',
        'max_volatility': '0.05',
        'max_sector_exposure': '0.3'
    })


@pytest.fixture
def logger():
    """Provide a logger fixture."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def market_data_validation(risk_limits, logger):
    """Provide a MarketDataValidation instance."""
    return MarketDataValidation(risk_limits, logger)


def test_validate_trade_parameters_valid(market_data_validation):
    """Test validating trade parameters with valid data."""
    symbol = "BTC/USDT"
    trade_params = {
        'symbol': symbol,
        'side': 'buy',
        'amount': Decimal('0.1'),
        'price': Decimal('50000')
    }
    
    assert market_data_validation.validate_trade_parameters(trade_params) is True


def test_validate_trade_parameters_invalid_side(market_data_validation):
    """Test validating trade parameters with invalid side."""
    symbol = "BTC/USDT"
    trade_params = {
        'symbol': symbol,
        'side': 'hold',
        'amount': Decimal('0.1'),
        'price': Decimal('50000')
    }
    
    with pytest.raises(ValidationError, match="Invalid trade side: hold"):
        market_data_validation.validate_trade_parameters(trade_params)


def test_validate_trade_parameters_negative_amount(market_data_validation):
    """Test validating trade parameters with negative amount."""
    symbol = "BTC/USDT"
    trade_params = {
        'symbol': symbol,
        'side': 'sell',
        'amount': Decimal('-0.1'),
        'price': Decimal('50000')
    }
    
    with pytest.raises(ValidationError, match="Trade amount must be positive"):
        market_data_validation.validate_trade_parameters(trade_params)


def test_validate_trade_parameters_invalid_price(market_data_validation):
    """Test validating trade parameters with invalid price."""
    symbol = "BTC/USDT"
    trade_params = {
        'symbol': symbol,
        'side': 'buy',
        'amount': Decimal('0.1'),
        'price': Decimal('-50000')
    }
    
    with pytest.raises(ValidationError, match="Trade price must be positive"):
        market_data_validation.validate_trade_parameters(trade_params)


def test_validate_trade_with_high_correlation(market_data_validation):
    """Test validating trade with high market correlation."""
    symbol = "BTC/USDT"
    correlations = {"ETH/USDT": Decimal('0.8')}
    
    with pytest.raises(ValidationError, match="Correlation for ETH/USDT exceeds maximum allowed"):
        market_data_validation.validate_correlation(symbol, correlations)


def test_validate_trade_with_low_liquidity(market_data_validation):
    """Test validating trade with insufficient liquidity."""
    symbol = "BTC/USDT"
    liquidity = Decimal('50000')
    
    with pytest.raises(ValidationError, match="Liquidity for BTC/USDT is below minimum required"):
        market_data_validation.validate_liquidity(symbol, liquidity) 