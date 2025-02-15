#! /usr/bin/env python3
import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.risk.limits import RiskLimits
from src.risk.validation import MarketDataValidation
from src.utils.error_handler import ValidationError


# risk limits
@pytest.fixture
def risk_limits():
    """Provide test risk limits."""
    return RiskLimits.from_config(
        {
            "max_correlation": Decimal("0.7"),
            "min_liquidity": Decimal("100000"),
            "max_volatility": Decimal("0.05"),
            "max_sector_exposure": Decimal("0.3"),
        }
    )


# logger
@pytest.fixture
def logger():
    """Provide a logger fixture."""
    return logging.getLogger("TestMarketDataValidation")


# market data validation
@pytest.fixture
def market_data_validation(risk_limits, logger):
    """Provide a MarketDataValidation instance."""
    return MarketDataValidation(risk_limits, logger)


# test validate trade parameters valid
@pytest.mark.asyncio
async def test_validate_trade_parameters_valid(market_data_validation):
    """Test validation of valid trade parameters."""
    trade_params = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": Decimal("50000"),
        "amount": Decimal("0.1"),
    }
    is_valid, error = await market_data_validation.validate_trade_parameters(
        trade_params
    )
    assert is_valid is True
    assert error is None


# test validate trade parameters invalid price
@pytest.mark.asyncio
async def test_validate_trade_parameters_invalid_price(market_data_validation):
    """Test validation fails with invalid price."""
    trade_params = {
        "symbol": "ETH/USDT",
        "side": "sell",
        "price": Decimal("-3000"),
        "amount": Decimal("1"),
    }
    is_valid, error = await market_data_validation.validate_trade_parameters(
        trade_params
    )
    assert is_valid is False
    assert error == "Trade price must be positive"


# test validate trade parameters invalid amount
@pytest.mark.asyncio
async def test_validate_trade_parameters_invalid_amount(market_data_validation):
    """Test validation fails with invalid amount."""
    trade_params = {
        "symbol": "ETH/USDT",
        "side": "sell",
        "price": Decimal("3000"),
        "amount": Decimal("-1"),
    }
    is_valid, error = await market_data_validation.validate_trade_parameters(
        trade_params
    )
    assert is_valid is False
    assert error == "Trade amount must be positive"


# test validate correlation exceeds limit
def test_validate_correlation_exceeds_limit(market_data_validation):
    """Test validation fails when market correlation exceeds limit."""
    symbol = "BTC/USDT"
    correlations = {"ETH/USDT": Decimal("0.8")}

    is_valid, error = market_data_validation.validate_correlation(symbol, correlations)
    assert is_valid is False
    assert error == "Correlation for ETH/USDT exceeds maximum allowed 0.700000."


# test validate liquidity below minimum
def test_validate_liquidity_below_minimum(market_data_validation):
    """Test validation fails when liquidity is below minimum."""
    symbol = "BTC/USDT"
    liquidity = Decimal("50000")  # Below min_liquidity of 100000

    is_valid, error = market_data_validation.validate_liquidity(symbol, liquidity)
    assert is_valid is False
    assert error == "Liquidity for BTC/USDT is below minimum required 100000."


# test validate volatility within limit
def test_validate_volatility_within_limit(market_data_validation):
    """Test validation passes when volatility is within limit."""
    symbol = "ETH/USDT"
    volatility = Decimal("0.04")  # Below max_volatility of 0.05

    is_valid, error = market_data_validation.validate_volatility(symbol, volatility)
    assert is_valid is True
    assert error is None


# test validate volatility exceeds limit
def test_validate_volatility_exceeds_limit(market_data_validation):
    """Test validation fails when volatility exceeds limit."""
    symbol = "ETH/USDT"
    volatility = Decimal("0.06")  # Above max_volatility of 0.05

    is_valid, error = market_data_validation.validate_volatility(symbol, volatility)
    assert is_valid is False
    assert error == "Volatility for ETH/USDT exceeds maximum allowed 0.050000."


# test validate sector exposure within limit
def test_validate_sector_exposure_within_limit(market_data_validation):
    """Test sector exposure validation within limit."""
    symbol = "BTC/USDT"
    sector_exposure = Decimal("0.25")  # Below max_sector_exposure of 0.3

    is_valid, error = market_data_validation.validate_sector_exposure(
        symbol, sector_exposure
    )
    assert is_valid is True
    assert error is None


# test validate sector exposure exceeds limit
def test_validate_sector_exposure_exceeds_limit(market_data_validation):
    """Test sector exposure validation exceeds limit."""
    symbol = "BTC/USDT"
    sector_exposure = Decimal("0.35")  # Above max_sector_exposure of 0.3

    is_valid, error = market_data_validation.validate_sector_exposure(
        symbol, sector_exposure
    )
    assert is_valid is False
    assert error == "Sector exposure for BTC/USDT exceeds maximum allowed 0.300000."
