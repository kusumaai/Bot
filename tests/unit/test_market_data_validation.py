import pytest
from decimal import Decimal

from risk.limits import RiskLimits
from risk.validation import MarketDataValidation
from utils.error_handler import ValidationError
import logging
from unittest.mock import MagicMock


@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config({
        'max_correlation': Decimal('0.7'),
        'min_liquidity': Decimal('100000'),
        'max_volatility': Decimal('0.05'),
        'max_sector_exposure': Decimal('0.3')
    })


@pytest.fixture
def logger():
    """Provide a logger fixture."""
    return logging.getLogger("TestMarketDataValidation")


@pytest.fixture
def market_data_validation(risk_limits, logger):
    """Provide a MarketDataValidation instance."""
    return MarketDataValidation(risk_limits, logger)


def test_validate_trade_parameters_valid(market_data_validation):
    """Test validation of valid trade parameters."""
    trade_params = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': Decimal('50000'),
        'amount': Decimal('0.1')
    }
    is_valid, error = market_data_validation.validate_trade_parameters(trade_params)
    assert is_valid is True
    assert error is None


def test_validate_trade_parameters_invalid_side(market_data_validation):
    """Test validation fails with invalid trade side."""
    trade_params = {
        'symbol': 'BTC/USDT',
        'side': 'hold',  # Invalid side
        'price': Decimal('50000'),
        'amount': Decimal('0.1')
    }
    is_valid, error = market_data_validation.validate_trade_parameters(trade_params)
    assert is_valid is False
    assert error == "Trade side must be 'buy' or 'sell'"


def test_validate_correlation_exceeds_limit(market_data_validation):
    """Test validation fails when market correlation exceeds limit."""
    symbol = "BTC/USDT"
    correlations = {"ETH/USDT": Decimal('0.8')}

    is_valid, error = market_data_validation.validate_correlation(symbol, correlations)
    assert is_valid is False
    assert error == "Correlation for ETH/USDT exceeds maximum allowed 0.700000."


def test_validate_liquidity_below_minimum(market_data_validation):
    """Test validation fails when liquidity is below minimum."""
    symbol = "BTC/USDT"
    liquidity = Decimal('50000')  # Below min_liquidity of 100000

    is_valid, error = market_data_validation.validate_liquidity(symbol, liquidity)
    assert is_valid is False
    assert error == "Liquidity for BTC/USDT is below minimum required 100000."


def test_validate_volatility_within_limit(market_data_validation):
    """Test validation passes when volatility is within limit."""
    symbol = "ETH/USDT"
    volatility = Decimal('0.04')  # Below max_volatility of 0.05

    is_valid, error = market_data_validation.validate_volatility(symbol, volatility)
    assert is_valid is True
    assert error is None


def test_validate_volatility_exceeds_limit(market_data_validation):
    """Test validation fails when volatility exceeds limit."""
    symbol = "ETH/USDT"
    volatility = Decimal('0.06')  # Above max_volatility of 0.05

    is_valid, error = market_data_validation.validate_volatility(symbol, volatility)
    assert is_valid is False
    assert error == "Volatility for ETH/USDT exceeds maximum allowed 0.050000."


def test_validate_sector_exposure_within_limit(market_data_validation):
    """Test sector exposure validation within limit."""
    symbol = "BTC/USDT"
    sector_exposure = Decimal('0.25')  # Below max_sector_exposure of 0.3

    is_valid, error = market_data_validation.validate_sector_exposure(symbol, sector_exposure)
    assert is_valid is True
    assert error is None


def test_validate_sector_exposure_exceeds_limit(market_data_validation):
    """Test sector exposure validation exceeds limit."""
    symbol = "BTC/USDT"
    sector_exposure = Decimal('0.35')  # Above max_sector_exposure of 0.3

    is_valid, error = market_data_validation.validate_sector_exposure(symbol, sector_exposure)
    assert is_valid is False
    assert error == "Sector exposure for BTC/USDT exceeds maximum allowed 0.300000." 